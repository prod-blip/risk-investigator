"""Atomic SQLite checkpoints and short leases for one active worker per run."""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


class RunStore:
    def __init__(self, database: Path):
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )""")

    def create(self, state: dict) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO runs (run_id, state_json, updated_at) VALUES (?, ?, ?)",
                               (state["run_id"], json.dumps(state, allow_nan=False), time.time()))

    def load(self, run_id: str) -> dict:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Run not found in the selected state database")
        return json.loads(row[0])

    @contextmanager
    def claim(self, run_id: str):
        owner = str(uuid4())
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner, lease_until FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Run not found")
            if row[0] is not None and row[1] > time.time():
                raise ValueError("Run has an active worker; retry after its lease expires (at most 120 seconds)")
            connection.execute("UPDATE runs SET owner = ?, lease_until = ? WHERE run_id = ?",
                               (owner, time.time() + 120, run_id))
        try:
            yield owner
        finally:
            with closing(sqlite3.connect(self.database)) as connection, connection:
                connection.execute("UPDATE runs SET owner = NULL, lease_until = 0 WHERE run_id = ? AND owner = ?",
                                   (run_id, owner))

    def save(self, state: dict, owner: str) -> None:
        encoded = json.dumps(state, allow_nan=False)
        if len(encoded.encode()) > 10_000_000:
            raise ValueError("Run checkpoint exceeds the 10 MB limit")
        with closing(sqlite3.connect(self.database)) as connection, connection:
            cursor = connection.execute(
                "UPDATE runs SET state_json = ?, updated_at = ?, lease_until = ? WHERE run_id = ? AND owner = ?",
                (encoded, time.time(), time.time() + 120, state["run_id"], owner),
            )
            if cursor.rowcount != 1:
                raise ValueError("Run worker lost its lease; refusing to overwrite another checkpoint")
