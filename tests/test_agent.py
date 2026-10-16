from copy import deepcopy
import io
import json
import os
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from risk_investigator.agent import AgentLimits, new_run, run_agent
from risk_investigator.agent_report import accept_report
from risk_investigator.config import load_local_env
from risk_investigator.demo_provider import DemoProvider
from risk_investigator.outcomes import write_outcomes
from risk_investigator.provider import OpenAIProvider, ProviderError, ProviderReply
from risk_investigator.run_store import RunStore
from risk_investigator.schemas import REPORT_SCHEMA, SchemaError, tool_schemas, validate
from risk_investigator.simulation import SCENARIOS, simulate
from risk_investigator.storage import write_experiments
from risk_investigator.tools import InvestigationTools

MATURE = "2026-01-27T00:00:00Z"
EARLY = "2026-01-13T00:00:00Z"


class InterruptDemo(DemoProvider):
    def __init__(self, at=3):
        self.count = 0
        self.at = at

    def respond(self, payload, *, timeout):
        self.count += 1
        if self.count == self.at:
            raise KeyboardInterrupt
        return super().respond(payload, timeout=timeout)


class MockLive(DemoProvider):
    kind = "openai"
    model = "mock-model"

    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    def respond(self, payload, *, timeout):
        self.calls += 1
        if self.failure and self.calls == 1:
            raise self.failure
        result = super().respond(payload, timeout=timeout)
        return ProviderReply(result.output, 100, 50)


class InvalidToolDemo(DemoProvider):
    def respond(self, payload, *, timeout):
        if not any(i.get("type") == "function_call_output" for i in payload["input"]):
            return ProviderReply([{"type": "function_call", "call_id": "call_invalid", "name": "replay",
                                   "arguments": '{"response":"restore_device_signal","min_accounts":true}'}])
        return super().respond(payload, timeout=timeout)


class WrongFinalOnce(DemoProvider):
    def __init__(self):
        self.changed = False

    def respond(self, payload, *, timeout):
        reply = super().respond(payload, timeout=timeout)
        if not self.changed and reply.output[0]["type"] == "message":
            proposal = json.loads(reply.output[0]["content"][0]["text"])
            proposal["findings"][0]["evidence_id"] = "ev_999"
            reply.output[0]["content"][0]["text"] = json.dumps(proposal)
            self.changed = True
        return reply


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.experiments = {scenario: simulate(scenario) for scenario in SCENARIOS}
        cls.database, private = write_experiments(list(cls.experiments.values()), cls.root)
        cls.outcomes = write_outcomes(list(cls.experiments.values()), cls.root / "outcomes.sqlite")
        private.unlink()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.run_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_directory.cleanup)
        self.store = RunStore(Path(self.run_directory.name) / "runs.sqlite")

    def tools(self, scenario="broken_signal", as_of=MATURE):
        return InvestigationTools(self.database, self.experiments[scenario].incident_id,
                                  as_of=as_of, outcomes_database=self.outcomes)

    def start(self, provider=None, limits=None, scenario="broken_signal", as_of=MATURE):
        provider = provider or DemoProvider()
        run_id = new_run(self.store, self.tools(scenario, as_of), provider, limits or AgentLimits(),
                         database=self.database, outcomes_database=self.outcomes)
        return run_id, provider

    def completed_proposal(self, scenario="broken_signal", as_of=MATURE):
        run_id, provider = self.start(scenario=scenario, as_of=as_of)
        report = run_agent(self.store, run_id, provider)
        state = self.store.load(run_id)
        message = next(item for item in reversed(state["conversation"]) if item.get("type") == "message")
        proposal = json.loads(message["content"][0]["text"])
        return proposal, state["tool_history"], report

    def test_demo_runs_all_cases_with_no_model_usage(self):
        for scenario in SCENARIOS:
            run_id, provider = self.start(scenario=scenario)
            report = run_agent(self.store, run_id, provider)
            self.assertEqual(report["run_status"], "completed")
            self.assertEqual(report["diagnosis"], scenario)
            self.assertEqual(report["model_calls"], 0)
            self.assertEqual(report["usage"]["estimated_api_cost_usd"], 0)
            self.assertIn("No LLM", report["mode_notice"])
            self.assertTrue(all("observed_value" in f for f in report["findings"]))
            self.assertEqual(self.store.load(run_id)["report"], report)

    def test_outcomes_do_not_confirm_fraud_early(self):
        _, _, report = self.completed_proposal("emerging_fraud", EARLY)
        self.assertEqual(report["status"], "suspected")
        self.assertEqual(report["diagnosis"], "suspected_emerging_fraud")

    def test_application_rejects_fraud_confirmation_without_eligible_labels(self):
        proposal, history, _ = self.completed_proposal("emerging_fraud", EARLY)
        proposal.update(status="supported", diagnosis="emerging_fraud")
        with self.assertRaisesRegex(SchemaError, "eligible outcomes"):
            accept_report(proposal, history)

    def test_evidence_values_come_from_tools_not_model(self):
        proposal, history, _ = self.completed_proposal()
        proposal["findings"][1]["interpretation"] = "A model interpretation still requires review."
        accepted = accept_report(proposal, history)
        self.assertEqual(accepted["findings"][1]["observed_value"], 31.0)
        self.assertEqual(accepted["validation"]["narrative_faithfulness"], "not_automatically_verified")

    def test_hallucinated_citations_and_paths_are_rejected(self):
        proposal, history, _ = self.completed_proposal()
        wrong = deepcopy(proposal)
        wrong["findings"][0]["evidence_id"] = "ev_999"
        with self.assertRaises(SchemaError):
            accept_report(wrong, history)
        wrong = deepcopy(proposal)
        wrong["findings"][0]["json_pointer"] = "/data/nonexistent"
        with self.assertRaises(SchemaError):
            accept_report(wrong, history)
        wrong = deepcopy(proposal)
        wrong["hypotheses"][0]["evidence_ids"] = ["ev_999"]
        with self.assertRaises(SchemaError):
            accept_report(wrong, history)

    def test_response_must_reference_replay(self):
        proposal, history, _ = self.completed_proposal()
        proposal["response_evidence_id"] = "ev_001"
        with self.assertRaisesRegex(SchemaError, "successful replay"):
            accept_report(proposal, history)

    def test_rejected_fraud_hypothesis_requires_eligible_outcome_evidence(self):
        proposal, history, _ = self.completed_proposal(as_of=EARLY)
        fraud = next(h for h in proposal["hypotheses"] if h["category"] == "emerging_fraud")
        fraud["verdict"] = "rejected"
        with self.assertRaisesRegex(SchemaError, "outcome_metrics"):
            accept_report(proposal, history)
        fraud["evidence_ids"] = [next(e["evidence_id"] for e in history if e["tool"] == "outcome_metrics")]
        with self.assertRaisesRegex(SchemaError, "adequate outcomes"):
            accept_report(proposal, history)

    def test_observed_signal_degradation_cannot_be_rejected(self):
        proposal, history, _ = self.completed_proposal("broken_signal")
        signal = next(h for h in proposal["hypotheses"] if h["category"] == "broken_signal")
        signal["verdict"] = "rejected"
        with self.assertRaisesRegex(SchemaError, "Signal degradation"):
            accept_report(proposal, history)

    def test_observed_lower_threshold_cannot_be_rejected(self):
        proposal, history, _ = self.completed_proposal("restrictive_rule")
        rule = next(h for h in proposal["hypotheses"] if h["category"] == "restrictive_rule")
        rule["verdict"] = "rejected"
        with self.assertRaisesRegex(SchemaError, "lower threshold"):
            accept_report(proposal, history)

    def test_fraud_response_must_test_linkage_mechanism(self):
        proposal, history, _ = self.completed_proposal("emerging_fraud")
        # Even a replay that changes decisions must test the diagnosed mechanism.
        unrelated = self.tools("broken_signal").call("replay", response="restore_device_signal")
        replay = next(e for e in history if e["tool"] == "replay")
        replay["data"] = unrelated["data"]
        with self.assertRaisesRegex(SchemaError, "linked_activity_control"):
            accept_report(proposal, history)

    def test_whole_tool_data_can_be_cited(self):
        proposal, history, _ = self.completed_proposal()
        proposal["findings"][0]["json_pointer"] = "/data"
        accepted = accept_report(proposal, history)
        source = next(e for e in history if e["evidence_id"] == proposal["findings"][0]["evidence_id"])
        self.assertEqual(accepted["findings"][0]["observed_value"], source["data"])

    def test_report_repair_keeps_rejection_and_reuses_evidence(self):
        run_id, provider = self.start(provider=WrongFinalOnce())
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["tool_calls"], 7)
        self.assertEqual(report["provider_turns"], 9)
        self.assertTrue(any(e["type"] == "report_rejection" for e in report["errors"]))

    def test_false_no_material_change_is_rejected(self):
        proposal, history, _ = self.completed_proposal()
        proposal.update(status="no_material_change", diagnosis=None, response_evidence_id=None)
        with self.assertRaisesRegex(SchemaError, "contradict"):
            accept_report(proposal, history)

    def test_unknown_tool_cannot_read_files_or_private_answers(self):
        class UnknownTool(DemoProvider):
            def respond(self, payload, *, timeout):
                if not any(i.get("type") == "function_call_output" for i in payload["input"]):
                    return ProviderReply([{"type": "function_call", "call_id": "call_file",
                                           "name": "read_file", "arguments": '{"path":"evaluation-private.json"}'}])
                return super().respond(payload, timeout=timeout)
        run_id, provider = self.start(provider=UnknownTool(), limits=AgentLimits(max_model_calls=2))
        report = run_agent(self.store, run_id, provider)
        self.assertIsNone(report["diagnosis"])
        self.assertEqual(report["evidence"], [])
        self.assertTrue(any(e["type"] == "tool_rejection" for e in report["errors"]))

    def test_interrupt_and_resume_preserve_progress_and_attempt_counts(self):
        provider = InterruptDemo()
        run_id, _ = self.start(provider=provider)
        with self.assertRaises(KeyboardInterrupt):
            run_agent(self.store, run_id, provider)
        saved = self.store.load(run_id)
        self.assertEqual(saved["run_status"], "interrupted")
        self.assertEqual(len(saved["tool_history"]), 2)
        report = run_agent(self.store, run_id, DemoProvider())
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["tool_calls"], 7)
        self.assertFalse(report["usage"]["billing_uncertain"])
        self.assertEqual(len(report["evidence"]), 7)

    def test_completed_run_returns_saved_report_without_provider_call(self):
        run_id, provider = self.start()
        report = run_agent(self.store, run_id, provider)
        provider.respond = lambda *a, **k: self.fail("Completed run should not call provider")
        self.assertEqual(run_agent(self.store, run_id, provider), report)

    def test_resume_rejects_different_provider(self):
        run_id, _ = self.start()
        with self.assertRaisesRegex(ValueError, "same provider"):
            run_agent(self.store, run_id, MockLive())

    def test_resume_rejects_changed_observation_evidence(self):
        provider = InterruptDemo()
        run_id, _ = self.start(provider=provider)
        with self.assertRaises(KeyboardInterrupt):
            run_agent(self.store, run_id, provider)
        with self.store.claim(run_id) as owner:
            saved = self.store.load(run_id)
            saved["evidence_digest"] = "changed-evidence"
            self.store.save(saved, owner)
        with self.assertRaisesRegex(ValueError, "Evidence changed"):
            run_agent(self.store, run_id, DemoProvider())

    def test_resume_preserves_tool_budget_instead_of_resetting_it(self):
        provider = InterruptDemo(at=2)
        run_id, _ = self.start(provider=provider, limits=AgentLimits(max_tool_calls=2))
        with self.assertRaises(KeyboardInterrupt):
            run_agent(self.store, run_id, provider)
        report = run_agent(self.store, run_id, DemoProvider())
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(report["tool_calls"], 2)

    def test_time_budget_stops_slow_provider_after_return(self):
        elapsed = [0.0]
        class SlowDemo(DemoProvider):
            def respond(self, payload, *, timeout):
                elapsed[0] += 2
                return super().respond(payload, timeout=timeout)
        run_id, provider = self.start(provider=SlowDemo(), limits=AgentLimits(time_limit_seconds=1))
        report = run_agent(self.store, run_id, provider, clock=lambda: elapsed[0])
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(report["tool_calls"], 0)

    def test_active_worker_lease_blocks_second_worker(self):
        run_id, _ = self.start()
        with self.store.claim(run_id):
            with self.assertRaisesRegex(ValueError, "active worker"):
                with self.store.claim(run_id):
                    self.fail("Another worker acquired the run")

    def test_tool_budget_stops_with_inconclusive_report(self):
        run_id, provider = self.start(limits=AgentLimits(max_tool_calls=2))
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(report["tool_calls"], 2)
        self.assertIsNone(report["diagnosis"])

    def test_provider_request_budget_stops_loop(self):
        run_id, provider = self.start(limits=AgentLimits(max_model_calls=1))
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(self.store.load(run_id)["provider_attempts"], 1)

    def test_input_budget_stops_before_provider_call(self):
        run_id, provider = self.start(limits=AgentLimits(max_input_bytes=1000))
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(report["provider_turns"], 0)

    def test_cost_guard_stops_before_live_request(self):
        limits = AgentLimits(max_cost_usd=0.000001, input_usd_per_million=1, output_usd_per_million=1)
        run_id, provider = self.start(provider=MockLive(), limits=limits)
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "budget_exhausted")
        self.assertEqual(provider.calls, 0)

    def test_mocked_live_usage_is_accounted(self):
        limits = AgentLimits(input_usd_per_million=1, output_usd_per_million=2, max_cost_usd=1)
        run_id, provider = self.start(provider=MockLive(), limits=limits)
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["model_calls"], 8)
        self.assertEqual(report["usage"]["input_tokens"], 800)
        self.assertEqual(report["usage"]["output_tokens"], 400)
        self.assertAlmostEqual(report["usage"]["estimated_api_cost_usd"], 0.0016)

    def test_explicit_rate_limit_retry_is_bounded(self):
        provider = MockLive(ProviderError("rate limit", retryable=True))
        run_id, _ = self.start(provider=provider)
        report = run_agent(self.store, run_id, provider, sleep=lambda _: None)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["model_calls"], 9)
        self.assertEqual(len(report["errors"]), 1)

    def test_ambiguous_transport_failure_is_not_automatically_retried(self):
        provider = MockLive(ProviderError("connection lost", ambiguous=True))
        limits = AgentLimits(input_usd_per_million=1, output_usd_per_million=2, max_cost_usd=1)
        run_id, _ = self.start(provider=provider, limits=limits)
        report = run_agent(self.store, run_id, provider)
        self.assertEqual(report["run_status"], "failed")
        self.assertEqual(provider.calls, 1)
        self.assertTrue(report["usage"]["billing_uncertain"])
        self.assertGreater(report["usage"]["unknown_cost_reserve_usd"], 0)

    def test_invalid_tool_arguments_are_rejected_before_execution(self):
        run_id, provider = self.start(provider=InvalidToolDemo())
        report = run_agent(self.store, run_id, provider)
        self.assertTrue(any(e["type"] == "tool_rejection" for e in report["errors"]))
        self.assertFalse(any(e["tool"] == "replay" for e in report["evidence"]))

    def test_all_tools_have_strict_schemas_and_boolean_is_not_integer(self):
        schemas = tool_schemas()
        self.assertEqual({s["name"] for s in schemas}, set(InvestigationTools.NAMES))
        for schema in schemas:
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["parameters"]["additionalProperties"])
        count_schema = next(s for s in schemas if s["name"] == "linked_activity")["parameters"]
        with self.assertRaises(SchemaError):
            validate({"min_accounts": True}, count_schema)

    def test_invalid_limits_rejected(self):
        for args in ({"max_model_calls": True}, {"max_retries": 5}, {"time_limit_seconds": float("nan")},
                     {"max_cost_usd": 1}, {"input_usd_per_million": 1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                AgentLimits(**args)


class ProviderAndConfigTests(unittest.TestCase):
    def test_request_uses_stateless_responses_strict_tools_and_matching_outputs(self):
        class Opener:
            def open(self, request, timeout):
                self.request, self.timeout = request, timeout
                return io.BytesIO(json.dumps({"status": "completed", "output": [
                    {"type": "function_call", "call_id": "call_1", "name": "incident_context", "arguments": "{}"}],
                    "usage": {"input_tokens": 100, "output_tokens": 20}}).encode())
        opener = Opener()
        provider = OpenAIProvider("mock-model", "test-credential-not-real", opener=opener)
        items = [{"type": "reasoning", "encrypted_content": "opaque"},
                 {"type": "function_call", "call_id": "call_prior", "name": "incident_context", "arguments": "{}"},
                 {"type": "function_call_output", "call_id": "call_prior", "output": "{}"}]
        payload = provider.prepare(items, "instructions", tool_schemas(), REPORT_SCHEMA, 1000)
        reply = provider.respond(payload, timeout=5)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["input"], items)
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(opener.request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(reply.input_tokens, 100)
        self.assertNotIn("test-credential", json.dumps(payload))

    def test_http_errors_do_not_echo_secret_or_raw_response_body(self):
        credential = "test-credential-not-real"
        class Opener:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 401, "bad credentials", {}, io.BytesIO(credential.encode()))
        provider = OpenAIProvider("mock-model", credential, opener=Opener())
        with self.assertRaises(ProviderError) as error:
            provider.respond({}, timeout=5)
        self.assertNotIn(credential, str(error.exception))
        self.assertFalse(error.exception.retryable)

    def test_dns_failure_does_not_claim_unknown_billing_or_retry(self):
        class Opener:
            def open(self, request, timeout):
                raise urllib.error.URLError(socket.gaierror(8, "host lookup failed"))
        provider = OpenAIProvider("mock-model", "test-credential-not-real", opener=Opener())
        with self.assertRaises(ProviderError) as error:
            provider.respond({}, timeout=5)
        self.assertFalse(error.exception.ambiguous)
        self.assertFalse(error.exception.retryable)

    def test_env_loader_preserves_shell_values_and_does_not_execute_or_expand(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_MODEL": "shell-model"}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('OPENAI_MODEL=file-model\nOPENAI_API_KEY="literal-$(not-executed)"\nUNRELATED_VALUE=ignored\n')
            load_local_env(path)
            self.assertEqual(os.environ["OPENAI_MODEL"], "shell-model")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "literal-$(not-executed)")
            self.assertNotIn("UNRELATED_VALUE", os.environ)

    def test_malformed_env_error_does_not_include_assignment_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY='test-credential-not-real\n")
            with self.assertRaises(ValueError) as error:
                load_local_env(path)
            self.assertNotIn("test-credential", str(error.exception))


if __name__ == "__main__":
    unittest.main()
