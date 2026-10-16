"""Timestamped synthetic outcomes, without scenario causes or grader metadata."""

from contextlib import closing
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Sequence

from .time_utils import normalize_time

if TYPE_CHECKING:
    from .simulation import Experiment


def write_outcomes(experiments: Sequence["Experiment"], destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(destination)) as connection, connection:
            connection.execute("""
                CREATE TABLE outcomes (
                    transaction_id TEXT PRIMARY KEY,
                    is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
                    available_at TEXT NOT NULL
                )
            """)
            connection.executemany(
                "INSERT INTO outcomes VALUES (?, ?, ?)",
                ((label.transaction_id, int(label.is_fraud), normalize_time(label.available_at))
                 for experiment in experiments for label in experiment.labels),
            )
            connection.execute("CREATE INDEX outcomes_by_availability ON outcomes(available_at)")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


class OutcomeReader:
    def __init__(self, database: Path):
        self.database = database.resolve()

    def available(self, transaction_ids: Sequence[str], as_of: str) -> dict[str, dict]:
        cutoff = normalize_time(as_of)
        result = {}
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            for offset in range(0, len(transaction_ids), 500):
                chunk = transaction_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT transaction_id, is_fraud, available_at FROM outcomes "
                    f"WHERE transaction_id IN ({placeholders}) AND available_at <= ?",
                    (*chunk, cutoff),
                ):
                    result[row["transaction_id"]] = {
                        "is_fraud": bool(row["is_fraud"]),
                        "available_at": row["available_at"],
                    }
        return result
