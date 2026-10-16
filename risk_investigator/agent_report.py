"""Structural and evidence checks; these are not a general natural-language fact checker."""

from .schemas import REPORT_SCHEMA, SchemaError, resolve_pointer, validate


def accept_report(proposal: dict, history: list[dict]) -> dict:
    validate(proposal, REPORT_SCHEMA)
    records = {r["evidence_id"]: r for r in history if "data" in r and "error" not in r}
    by_tool = {}
    for record in records.values():
        by_tool[record["tool"]] = record
    for hypothesis in proposal["hypotheses"]:
        if any(e not in records for e in hypothesis["evidence_ids"]):
            raise SchemaError("Hypothesis cites unavailable or failed evidence")
        category, verdict = hypothesis["category"], hypothesis["verdict"]
        cited_tools = {records[e]["tool"] for e in hypothesis["evidence_ids"]}
        if category == "other" and verdict != "unresolved":
            raise SchemaError("Unimplemented hypothesis categories must remain unresolved")
        if verdict == "rejected":
            required_tool = {"broken_signal": "feature_health", "restrictive_rule": "deployment_history",
                             "emerging_fraud": "outcome_metrics"}[category]
            if required_tool not in cited_tools:
                raise SchemaError(f"Rejected {category} hypothesis must cite {required_tool}")
            if category == "broken_signal":
                periods = by_tool["feature_health"]["data"]["periods"]
                if any(periods["after"][key] > periods["before"][key] for key in
                       ("missing", "source_delivery_mismatches", "stale")):
                    raise SchemaError("Signal degradation is observed; do not reject the signal hypothesis")
            if category == "restrictive_rule":
                periods = by_tool["deployment_history"]["data"]["policies"]
                before = {p["decline_threshold"] for p in periods["before"]}
                after = {p["decline_threshold"] for p in periods["after"]}
                if not before or not after or min(after) < min(before):
                    raise SchemaError("A lower threshold is observed or comparison is incomplete; do not reject it")
            if category == "emerging_fraud":
                metrics = by_tool["outcome_metrics"]["data"]["periods"]
                before, after = metrics["before"], metrics["after"]
                if (min(before["label_coverage"], after["label_coverage"]) < 0.9
                        or after["fraud_rate_pct_among_labelled"] > before["fraud_rate_pct_among_labelled"]):
                    raise SchemaError("Fraud hypothesis cannot be rejected without adequate outcomes showing no growth")
    findings = []
    for finding in proposal["findings"]:
        record = records.get(finding["evidence_id"])
        if record is None:
            raise SchemaError("Finding cites unavailable or failed evidence")
        observed = resolve_pointer(record, finding["json_pointer"])
        findings.append({
            "claim": finding["interpretation"], "evidence_ids": [finding["evidence_id"]],
            "json_pointer": finding["json_pointer"], "observed_value": observed,
            "interpretation_is_model_generated": True,
        })
    context = by_tool.get("incident_context", {}).get("data", {})
    diagnosis, status = proposal["diagnosis"], proposal["status"]
    replay_id = proposal["response_evidence_id"]
    replay = None
    if replay_id is not None:
        record = records.get(replay_id)
        if record is None or record["tool"] != "replay":
            raise SchemaError("Response must cite a successful replay")
        replay = {"evidence_id": replay_id, **record["data"]}
    if status in ("supported", "suspected"):
        if diagnosis is None or not context.get("analysis_window_complete") or replay is None:
            raise SchemaError("A diagnosis needs a complete observed window and a successful response replay")
        if not replay["changed_decisions"]:
            raise SchemaError("Proposed response changed no decisions; use inconclusive")
    elif diagnosis is not None:
        raise SchemaError("Inconclusive/no-material-change reports must have a null diagnosis")
    if status == "no_material_change":
        required = {"incident_context", "compare_cohorts", "feature_health", "deployment_history", "linked_activity"}
        if not required.issubset(by_tool) or not context.get("analysis_window_complete"):
            raise SchemaError("No-material-change requires explicit checks across observed data and controls")
        totals = by_tool["compare_cohorts"]["data"]["totals"]
        health_periods = by_tool["feature_health"]["data"]["periods"]
        if (not totals["before"]["transactions"] or not totals["after"]["transactions"]
                or abs(totals["after"]["decline_rate_pct"] - totals["before"]["decline_rate_pct"]) >= 2
                or health_periods["after"]["missing"] > health_periods["before"]["missing"]
                or health_periods["after"]["stale"] > health_periods["before"]["stale"]):
            raise SchemaError("Observed decision or signal changes contradict no-material-change")
    health = by_tool.get("feature_health", {}).get("data", {}).get("periods", {})
    deployments = by_tool.get("deployment_history", {}).get("data", {})
    if diagnosis == "broken_signal":
        before, after = health.get("before", {}), health.get("after", {})
        if (after.get("missing", 0) <= before.get("missing", 0)
                or after.get("source_delivery_mismatches", 0) <= before.get("source_delivery_mismatches", 0)
                or not after.get("missing_signal_declines", 0)
                or replay["response"] != "restore_device_signal"):
            raise SchemaError("Broken-signal diagnosis lacks missing-signal, source, rule, or repair evidence")
    if diagnosis == "restrictive_rule":
        policies = deployments.get("policies", {})
        before, after = policies.get("before", []), policies.get("after", [])
        if (len(before) != 1 or len(after) != 1
                or after[0]["decline_threshold"] >= before[0]["decline_threshold"]
                or replay["response"] != "previous_threshold"):
            raise SchemaError("Restrictive-rule diagnosis lacks a lower recorded threshold and matching replay")
    if diagnosis in ("emerging_fraud", "suspected_emerging_fraud"):
        linkage = by_tool.get("linked_activity", {}).get("data", {}).get("periods", {})
        before = linkage.get("before", {}).get("young_new_device_low_velocity", 0)
        after = linkage.get("after", {}).get("young_new_device_low_velocity", 0)
        if after <= before or replay["response"] != "linked_activity_control":
                raise SchemaError("Emerging-fraud hypothesis needs increasing linked activity and "
                                  "replay(response=linked_activity_control), not a different response replay")
        if diagnosis == "suspected_emerging_fraud" and status != "suspected":
            raise SchemaError("Suspected fraud must remain a suspected diagnosis")
        if diagnosis == "emerging_fraud":
            metrics = by_tool.get("outcome_metrics", {}).get("data", {})
            periods = metrics.get("periods", {})
            before, after = periods.get("before", {}), periods.get("after", {})
            if (status != "supported" or min(before.get("label_coverage", 0), after.get("label_coverage", 0)) < 0.9
                    or not metrics.get("confirmed_fraud_in_linked_after_groups", 0)
                    or after.get("fraud_rate_pct_among_labelled", 0) <= before.get("fraud_rate_pct_among_labelled", 0)):
                raise SchemaError("Fraud confirmation requires sufficient eligible outcomes and observed fraud growth")
    return {
        "status": status, "diagnosis": diagnosis, "summary": proposal["summary"],
        "findings": findings, "hypotheses": proposal["hypotheses"],
        "candidate_causes": [h["description"] for h in proposal["hypotheses"] if h["verdict"] != "rejected"],
        "response_test": replay,
        "limitations": proposal["limitations"] + [
            "Evidence references and selected diagnosis prerequisites were checked in code.",
            "Natural-language interpretations and hypothesis verdicts still require human verification.",
            "Rejected hypotheses concern the checked indicators and observed sample, not every possible fraud mechanism.",
            "Replay is synthetic and does not estimate realized production loss savings.",
        ],
        "human_review_required": True,
        "validation": {"schema": "passed", "citations": "passed", "evidence_paths": "passed",
                       "diagnosis_prerequisites": "passed", "narrative_faithfulness": "not_automatically_verified"},
    }
