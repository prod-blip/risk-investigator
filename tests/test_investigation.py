from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from risk_investigator.baseline import investigate, render_markdown
from risk_investigator.engine import decide
from risk_investigator.models import Policy
from risk_investigator.outcomes import OutcomeReader, write_outcomes
from risk_investigator.simulation import Change, SCENARIOS, simulate
from risk_investigator.storage import write_experiments
from risk_investigator.time_utils import normalize_time
from risk_investigator.tools import InvestigationTools, ToolError

EARLY = "2026-01-13T00:00:00Z"
MATURE = "2026-01-27T00:00:00Z"


class InvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.experiments = {scenario: simulate(scenario) for scenario in SCENARIOS}
        cls.database, private = write_experiments(list(cls.experiments.values()), cls.root)
        cls.outcomes = write_outcomes(list(cls.experiments.values()), cls.root / "outcomes.sqlite")
        # The baseline must work when private causes/labels aren't even present.
        private.unlink()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tools(self, scenario="broken_signal", as_of=MATURE, **kwargs):
        return InvestigationTools(self.database, self.experiments[scenario].incident_id,
                                  as_of=as_of, outcomes_database=self.outcomes, **kwargs)

    @contextmanager
    def custom(self, experiment):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, private = write_experiments([experiment], root)
            private.unlink()
            yield InvestigationTools(database, experiment.incident_id, as_of=MATURE), database

    def test_known_cases_diagnosed_from_evidence_without_private_file(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                report = investigate(self.tools(scenario))
                self.assertEqual(report["diagnosis"], scenario)
                self.assertEqual(report["status"], "supported")
                self.assertTrue(report["human_review_required"])
                self.assertEqual(report["tool_calls"], 7)
                evidence_ids = {e["evidence_id"] for e in report["evidence"]}
                for finding in report["findings"]:
                    self.assertTrue(set(finding["evidence_ids"]).issubset(evidence_ids))

    def test_attack_is_suspected_before_labels_and_confirmed_after_availability(self):
        early = investigate(self.tools("emerging_fraud", as_of=EARLY))
        late = investigate(self.tools("emerging_fraud"))
        self.assertEqual(early["status"], "suspected")
        self.assertEqual(early["diagnosis"], "suspected_emerging_fraud")
        self.assertEqual(late["status"], "supported")
        self.assertTrue(all(v is None for v in early["response_test"]["labelled_outcome_delta"].values()))

    def test_future_outcomes_hidden_and_exact_availability_included(self):
        ids = [t.transaction_id for t in self.experiments["broken_signal"].transactions]
        reader = OutcomeReader(self.outcomes)
        self.assertEqual(reader.available(ids, EARLY), {})
        visible = reader.available(ids, "2026-01-26T09:00:00Z")
        self.assertEqual(set(visible), {ids[0]})
        self.assertEqual(visible, reader.available(ids, "2026-01-26T14:30:00+05:30"))
        self.assertEqual(len(reader.available(ids, MATURE)), 1000)

    def test_no_outcome_store_preserves_unknown_tradeoffs(self):
        tools = InvestigationTools(self.database, self.experiments["broken_signal"].incident_id, as_of=MATURE)
        report = investigate(tools)
        self.assertGreater(report["response_test"]["newly_approved"], 0)
        self.assertEqual(report["response_test"]["baseline_outcomes"]["label_coverage"], 0)
        self.assertTrue(all(v is None for v in report["response_test"]["labelled_outcome_delta"].values()))
        self.assertIn("trade-offs are unknown", render_markdown(report))

    def test_cohort_and_deployment_queries_respect_investigation_time(self):
        cutoff = "2026-01-12T10:00:00Z"
        tools = self.tools(as_of=cutoff)
        self.assertTrue(all(row["occurred_at"] <= tools.as_of for row in tools.rows))
        self.assertEqual(tools.call("deployment_history")["data"]["changes"], [])
        report = investigate(self.tools(as_of=cutoff))
        self.assertEqual(report["status"], "inconclusive")
        self.assertIsNone(report["diagnosis"])
        self.assertEqual(report["tool_calls"], 1)

    def test_signal_and_policy_replays_recover_labelled_legitimate_payments(self):
        for scenario, response in (("broken_signal", "restore_device_signal"),
                                   ("restrictive_rule", "previous_threshold")):
            with self.subTest(scenario=scenario):
                result = self.tools(scenario).call("replay", response=response)["data"]
                self.assertGreater(result["newly_approved"], 0)
                self.assertLess(result["labelled_outcome_delta"]["legitimate_declined"], 0)
                self.assertEqual(result["labelled_outcome_delta"]["fraud_declined"], 0)

    def test_linked_response_catches_additional_labelled_fraud(self):
        replay = self.tools("emerging_fraud").call("replay", response="linked_activity_control")["data"]
        self.assertGreater(replay["labelled_outcome_delta"]["fraud_declined"], 0)
        self.assertLess(replay["labelled_outcome_delta"]["approved_fraud_amount_cents"], 0)

    def test_linked_replay_uses_prefix_counts_not_future_accounts(self):
        experiment = simulate("emerging_fraud", transactions_per_period=50)
        before = tuple(t for t in experiment.transactions if t.period == "before")
        after = tuple(replace(t, payment_instrument_id="pi_prefix_test", device_status="new",
                              source_device_status="new", account_age_days=5,
                              velocity_1h=1, amount_cents=5000)
                      for t in experiment.transactions if t.period == "after")[:6]
        transactions = before + after
        experiment = replace(experiment, transactions=transactions,
                             decisions=tuple(decide(t) for t in transactions), changes=())
        with self.custom(experiment) as (tools, _):
            replay = tools.call("replay", response="linked_activity_control", min_accounts=5)["data"]
            self.assertEqual(replay["newly_declined"], 2)
            self.assertEqual(replay["source_refs"], [f"decision:{t.transaction_id}" for t in after[4:]])

    def test_multiple_supported_mechanisms_require_further_investigation(self):
        experiment = self.experiments["broken_signal"]
        decisions = tuple(decide(t, Policy("rules-v2", 25)) if t.period == "after" else decide(t)
                          for t in experiment.transactions)
        change = Change("chg_combined", experiment.incident_id, experiment.boundary_at,
                        "decision_policy", "rules-v1", "rules-v2",
                        {"decline_threshold": {"before": 70, "after": 25}})
        experiment = replace(experiment, decisions=decisions, changes=experiment.changes + (change,))
        with self.custom(experiment) as (tools, _):
            report = investigate(tools)
            self.assertEqual(set(report["candidate_causes"]), {"broken_signal", "restrictive_rule"})
            self.assertEqual(report["status"], "inconclusive")
            self.assertIsNone(report["response_test"])

    def test_healthy_fixture_does_not_force_a_diagnosis(self):
        experiment = self.experiments["broken_signal"]
        transactions = tuple(replace(t, device_status=t.source_device_status,
                                     feature_version="device-parser-v1",
                                     device_observed_at=t.occurred_at)
                             for t in experiment.transactions)
        experiment = replace(experiment, transactions=transactions,
                             decisions=tuple(decide(t) for t in transactions), changes=())
        with self.custom(experiment) as (tools, _):
            report = investigate(tools)
            self.assertEqual(report["status"], "no_material_change")
            self.assertIsNone(report["diagnosis"])

    def test_unreproducible_decision_blocks_response_endorsement(self):
        with self.custom(self.experiments["broken_signal"]) as (_, database):
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("UPDATE decisions SET score = score + 1")
            tools = InvestigationTools(database, self.experiments["broken_signal"].incident_id, as_of=MATURE)
            report = investigate(tools)
            self.assertEqual(report["status"], "inconclusive")
            self.assertIsNone(report["response_test"])
            self.assertIn("cannot be reproduced", report["limitations"][-1])

    def test_unsupported_model_is_not_silently_replayed(self):
        with self.custom(self.experiments["broken_signal"]) as (_, database):
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("UPDATE decisions SET model_version = 'unsupported-model'")
            tools = InvestigationTools(database, self.experiments["broken_signal"].incident_id, as_of=MATURE)
            with self.assertRaisesRegex(ToolError, "model version"):
                tools.call("replay", response="restore_device_signal")

    def test_tools_reject_unknown_names_invalid_parameters_and_budget_overrun(self):
        tools = self.tools(max_calls=2)
        with self.assertRaises(ToolError):
            tools.call("read_private_answers")
        with self.assertRaises(ToolError):
            tools.call("compare_cohorts", group_by="channel; DROP TABLE transactions")
        with self.assertRaises(ToolError):
            tools.call("linked_activity", min_accounts=True)
        self.assertEqual(len(tools.history), 2)
        with self.assertRaisesRegex(ToolError, "budget"):
            tools.call("incident_context")

    def test_queries_and_replay_do_not_change_evidence_files(self):
        before = (self.database.read_bytes(), self.outcomes.read_bytes())
        report = investigate(self.tools())
        after = (self.database.read_bytes(), self.outcomes.read_bytes())
        self.assertEqual(before, after)
        self.assertNotIn("expected_cause", str(report))
        self.assertNotIn("evaluation-private", str(report))

    def test_future_feature_timestamp_is_rejected(self):
        with self.custom(self.experiments["broken_signal"]) as (_, database):
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("UPDATE transactions SET device_observed_at = '2026-01-28T00:00:00+00:00'")
            with self.assertRaisesRegex(ToolError, "later than"):
                InvestigationTools(database, self.experiments["broken_signal"].incident_id, as_of=MATURE)

    def test_decision_trace_is_scoped_and_contains_no_grading_truth(self):
        tools = self.tools(as_of="2026-01-12T10:00:00Z")
        visible = tools.rows[0]["transaction_id"]
        trace = tools.call("decision_trace", transaction_id=visible)["data"]
        self.assertEqual(trace["input_snapshot"]["transaction_id"], visible)
        self.assertNotIn("is_fraud", str(trace))
        future = self.experiments["broken_signal"].transactions[-1].transaction_id
        with self.assertRaisesRegex(ToolError, "not visible"):
            tools.call("decision_trace", transaction_id=future)
        other = self.experiments["emerging_fraud"].transactions[0].transaction_id
        with self.assertRaisesRegex(ToolError, "not visible"):
            tools.call("decision_trace", transaction_id=other)

    def test_outcome_and_replay_results_share_frozen_outcome_digest(self):
        tools = self.tools()
        outcome = tools.call("outcome_metrics")["data"]
        replay = tools.call("replay", response="restore_device_signal")["data"]
        self.assertEqual(outcome["eligible_outcomes_digest"], replay["eligible_outcomes_digest"])

    def test_timestamps_require_timezone_and_normalize_offsets(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            normalize_time("2026-01-27T00:00:00")
        self.assertEqual(normalize_time("2026-01-27T05:30:00+05:30"), "2026-01-27T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
