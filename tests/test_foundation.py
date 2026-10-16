from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from risk_investigator.engine import decide
from risk_investigator.models import Policy, Transaction
from risk_investigator.simulation import SCENARIOS, simulate
from risk_investigator.storage import summarize, write_experiments


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {name: simulate(name) for name in SCENARIOS}

    def test_same_seed_reproduces_transactions_decisions_and_truth(self):
        for scenario, experiment in self.experiments.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(experiment, simulate(scenario))
                self.assertNotEqual(experiment, simulate(scenario, seed=43))

    def test_background_traffic_and_labels_match_between_non_attack_scenarios(self):
        broken = self.experiments["broken_signal"]
        restrictive = self.experiments["restrictive_rule"]
        for a, b, la, lb in zip(broken.transactions, restrictive.transactions,
                                broken.labels, restrictive.labels, strict=True):
            self.assertEqual((a.amount_cents, a.channel, a.account_age_days,
                              a.source_device_status, la.is_fraud),
                             (b.amount_cents, b.channel, b.account_age_days,
                              b.source_device_status, lb.is_fraud))

    def test_fraud_labels_are_unavailable_at_decision_time(self):
        for experiment in self.experiments.values():
            for transaction, label in zip(experiment.transactions, experiment.labels, strict=True):
                self.assertGreater(datetime.fromisoformat(label.available_at),
                                   datetime.fromisoformat(transaction.occurred_at))
                self.assertNotIn("is_fraud", asdict(transaction))
                self.assertLessEqual(label.available_at, experiment.evaluation_at)

    def test_broken_signal_affects_only_mobile_after_and_restoration_recovers_decision(self):
        experiment = self.experiments["broken_signal"]
        recovered_legitimate = 0
        for t, d, label in zip(experiment.transactions, experiment.decisions,
                               experiment.labels, strict=True):
            affected = t.period == "after" and t.channel == "mobile_web"
            self.assertEqual(t.device_status is None, affected)
            if affected:
                self.assertIn("DEVICE_SIGNAL_MISSING", d.triggered_rules)
                restored = replace(t, device_status=t.source_device_status)
                recovered_legitimate += not label.is_fraud and decide(restored).action == "approve"
            self.assertEqual(d.policy_version, "rules-v1")
        self.assertGreater(recovered_legitimate, 0)

    def test_restrictive_rule_increases_legitimate_declines_without_changing_features(self):
        experiment = self.experiments["restrictive_rule"]
        additional_legitimate_declines = 0
        for t, d, label in zip(experiment.transactions, experiment.decisions,
                               experiment.labels, strict=True):
            self.assertIsNotNone(t.device_status)
            if t.period == "after":
                self.assertEqual(d.policy_version, "rules-v2")
                additional_legitimate_declines += (
                    not label.is_fraud and d.action == "decline" and decide(t).action == "approve"
                )
        self.assertGreater(additional_legitimate_declines, 0)

    def test_emerging_attack_increases_fraud_and_has_approved_linked_transactions(self):
        experiment = self.experiments["emerging_fraud"]
        fraud_counts = {"before": 0, "after": 0}
        approved_fraud_instruments = []
        for t, d, label in zip(experiment.transactions, experiment.decisions,
                               experiment.labels, strict=True):
            fraud_counts[t.period] += label.is_fraud
            if t.period == "after" and label.is_fraud and d.action == "approve":
                approved_fraud_instruments.append(t.payment_instrument_id)
            self.assertEqual(d.policy_version, "rules-v1")
            self.assertEqual(t.feature_version, "device-parser-v1")
        self.assertGreater(fraud_counts["after"], fraud_counts["before"])
        self.assertGreater(len(approved_fraud_instruments), 0)
        self.assertLess(len(set(approved_fraud_instruments)), len(approved_fraud_instruments))

    def test_evidence_database_excludes_grading_answers_and_snapshots_reproduce_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            database, private = write_experiments(list(self.experiments.values()), Path(directory))
            with closing(sqlite3.connect(database)) as connection:
                objects = connection.execute("SELECT name, sql FROM sqlite_master").fetchall()
                schema = str(objects)
                for forbidden in ("is_fraud", "expected_cause", "evaluation_at"):
                    self.assertNotIn(forbidden, schema)
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                rows = connection.execute(
                    "SELECT input_snapshot, policy_version, decline_threshold, action, score FROM decisions"
                ).fetchall()
                self.assertEqual(len(rows), 3000)
                for snapshot, version, threshold, action, score in rows:
                    decision = decide(Transaction(**json.loads(snapshot)), Policy(version, threshold))
                    self.assertEqual((decision.action, decision.score), (action, score))
            answers = json.loads(private.read_text())
            self.assertEqual(len(answers["experiments"]), 3)
            self.assertEqual({e["expected_cause"] for e in answers["experiments"]}, set(SCENARIOS))
            summary = summarize(database)
            self.assertEqual(len(summary), 6)
            self.assertTrue(all(row["transactions"] == 500 for row in summary))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            database, private = write_experiments([self.experiments["broken_signal"]], Path(directory))
            original = (database.read_bytes(), private.read_bytes())
            with self.assertRaises(FileExistsError):
                write_experiments([self.experiments["emerging_fraud"]], Path(directory))
            self.assertEqual(original, (database.read_bytes(), private.read_bytes()))

    def test_inspect_missing_database_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                summarize(path)
            self.assertFalse(path.exists())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate("unknown")
        with self.assertRaises(ValueError):
            simulate("broken_signal", transactions_per_period=0)
        with self.assertRaises(ValueError):
            Policy(decline_threshold=0)
        with self.assertRaises(ValueError):
            replace(self.experiments["broken_signal"].transactions[0], amount_cents=-1)


if __name__ == "__main__":
    unittest.main()
