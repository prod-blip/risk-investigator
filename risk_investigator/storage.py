"""Keep investigation evidence separate from private benchmark answers."""

from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Sequence

from .simulation import Experiment

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY,
    start_at TEXT NOT NULL,
    boundary_at TEXT NOT NULL,
    end_at TEXT NOT NULL
);
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    period TEXT NOT NULL CHECK (period IN ('before', 'after')),
    occurred_at TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    payment_instrument_id TEXT NOT NULL,
    country TEXT NOT NULL,
    channel TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    account_age_days INTEGER NOT NULL,
    velocity_1h INTEGER NOT NULL,
    device_status TEXT,
    device_observed_at TEXT,
    source_device_status TEXT NOT NULL,
    feature_version TEXT NOT NULL
);
CREATE TABLE decisions (
    transaction_id TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
    score INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('approve', 'decline')),
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    decline_threshold INTEGER NOT NULL,
    triggered_rules TEXT NOT NULL,
    contributions TEXT NOT NULL,
    input_snapshot TEXT NOT NULL
);
CREATE TABLE changes (
    change_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    occurred_at TEXT NOT NULL,
    component TEXT NOT NULL,
    previous_version TEXT NOT NULL,
    current_version TEXT NOT NULL,
    details TEXT NOT NULL
);
CREATE INDEX transactions_by_incident_time ON transactions(incident_id, occurred_at);
CREATE VIEW decision_evidence AS
SELECT t.*, d.score, d.action, d.model_version, d.policy_version,
       d.decline_threshold, d.triggered_rules, d.contributions, d.input_snapshot
FROM transactions t JOIN decisions d USING (transaction_id);
PRAGMA user_version = 1;
"""


def write_experiments(
    experiments: Sequence[Experiment], output_dir: Path
) -> tuple[Path, Path]:
    if not experiments:
        raise ValueError("At least one experiment is required")
    output_dir = output_dir.resolve()
    database = output_dir / "observations.sqlite"
    evaluation = output_dir / "evaluation-private.json"
    if database.exists() or evaluation.exists():
        raise FileExistsError(
            f"Output already exists in {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents accidentally replacing an existing file.
    with database.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executescript(SCHEMA)
            for experiment in experiments:
                connection.execute(
                    "INSERT INTO incidents VALUES (?, ?, ?, ?)",
                    (experiment.incident_id, experiment.transactions[0].occurred_at,
                     experiment.boundary_at, experiment.transactions[-1].occurred_at),
                )
                for transaction, decision in zip(
                    experiment.transactions, experiment.decisions, strict=True
                ):
                    t = asdict(transaction)
                    connection.execute(
                        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        tuple(t.values()),
                    )
                    connection.execute(
                        "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (decision.transaction_id, decision.score, decision.action,
                         decision.model_version, decision.policy_version,
                         decision.decline_threshold, json.dumps(decision.triggered_rules),
                         json.dumps(decision.contributions, sort_keys=True),
                         json.dumps(t, sort_keys=True)),
                    )
                for change in experiment.changes:
                    connection.execute(
                        "INSERT INTO changes VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (change.change_id, change.incident_id, change.occurred_at,
                         change.component, change.previous_version, change.current_version,
                         json.dumps(change.details, sort_keys=True)),
                    )
        answers = {
            "synthetic_data": True,
            "purpose": "PRIVATE grading truth. Do not expose to the investigator.",
            "experiments": [{
                "incident_id": e.incident_id,
                "expected_cause": e.scenario,
                "seed": e.seed,
                "evaluation_at": e.evaluation_at,
                "labels": [asdict(label) for label in e.labels],
            } for e in experiments],
        }
        with evaluation.open("x") as handle:
            json.dump(answers, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        database.unlink(missing_ok=True)
        raise
    return database, evaluation


def summarize(database: Path) -> list[dict[str, object]]:
    # Read-only URI: inspecting a missing path must not create an empty database.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT incident_id, period, COUNT(*) AS transactions,
                   SUM(action = 'decline') AS declines,
                   SUM(device_status IS NULL) AS missing_device_signals,
                   ROUND(100.0 * SUM(action = 'decline') / COUNT(*), 2) AS decline_rate_pct
            FROM decision_evidence
            GROUP BY incident_id, period
            ORDER BY incident_id, CASE period WHEN 'before' THEN 0 ELSE 1 END
        """).fetchall()
    return [dict(row) for row in rows]
