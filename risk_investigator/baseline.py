"""An explicit diagnostic runbook; no LLM and no access to private grading truth."""

from time import perf_counter

from .engine import MODEL_VERSION
from .tools import InvestigationTools, ToolError

RESPONSE_BY_CAUSE = {
    "broken_signal": "restore_device_signal",
    "restrictive_rule": "previous_threshold",
    "emerging_fraud": "linked_activity_control",
}


def investigate(tools: InvestigationTools) -> dict:
    started = perf_counter()
    report = {
        "investigator": "scripted-baseline-v1", "synthetic_data": True,
        "incident_id": tools.incident_id, "as_of": tools.as_of,
        "status": "inconclusive", "diagnosis": None, "candidate_causes": [],
        "findings": [], "response_test": None, "human_review_required": True,
        "limitations": [
            "This runbook covers a small set of explicit diagnostic patterns.",
            "Evidence of a deployment or shared instrument alone does not prove causation or fraud.",
            "Synthetic replay does not establish production impact or agent superiority.",
        ],
    }

    def finish(summary: str) -> dict:
        report["summary"] = summary
        report["tool_calls"] = len(tools.history)
        report["elapsed_seconds"] = round(perf_counter() - started, 6)
        report["evidence"] = tools.history.copy()
        report["evidence_digest"] = tools.evidence_digest
        return report

    def finding(claim: str, *records: dict) -> None:
        report["findings"].append({
            "claim": claim, "evidence_ids": [r["evidence_id"] for r in records]
        })

    context = tools.call("incident_context")
    if not context["data"]["analysis_window_complete"]:
        finding("The configured analysis window is incomplete at this investigation timestamp.", context)
        return finish("Wait for the analysis window or select a completed window; no diagnosis asserted.")

    cohorts = tools.call("compare_cohorts", group_by="channel")
    health = tools.call("feature_health")
    deployments = tools.call("deployment_history")
    links = tools.call("linked_activity", min_accounts=5)
    outcomes = tools.call("outcome_metrics")
    before = cohorts["data"]["totals"]["before"]
    after = cohorts["data"]["totals"]["after"]
    if not before["transactions"] or not after["transactions"]:
        finding("Both comparison periods need visible transactions.", cohorts)
        return finish("Insufficient comparison data; no diagnosis asserted.")
    decline_delta = after["decline_rate_pct"] - before["decline_rate_pct"]
    finding(
        f"Observed decline rate changed from {before['decline_rate_pct']}% to "
        f"{after['decline_rate_pct']}% ({decline_delta:+.4f} percentage points).", cohorts
    )
    hp = health["data"]["periods"]
    policy_periods = deployments["data"]["policies"]
    previous = policy_periods["before"]
    current = policy_periods["after"]
    bp, ap = links["data"]["periods"]["before"], links["data"]["periods"]["after"]
    outcome_periods = outcomes["data"]["periods"]
    mature = all(outcome_periods[p]["label_coverage"] >= 0.9 for p in ("before", "after"))
    fraud_delta = (
        outcome_periods["after"]["fraud_rate_pct_among_labelled"]
        - outcome_periods["before"]["fraud_rate_pct_among_labelled"]
    ) if mature else None
    finding(
        f"Available outcome coverage: before {outcome_periods['before']['label_coverage']:.1%}, "
        f"after {outcome_periods['after']['label_coverage']:.1%}. "
        "Unlabelled payments are not classified as legitimate.", outcomes
    )

    candidates = []
    changed_components = {c["component"] for c in deployments["data"]["changes"]}
    if (hp["after"]["missing_rate"] - hp["before"]["missing_rate"] >= 0.05
            and hp["after"]["source_delivery_mismatches"] > hp["before"]["source_delivery_mismatches"]
            and hp["after"]["missing_signal_declines"] > hp["before"]["missing_signal_declines"]
            and "feature_integration" in changed_components):
        candidates.append("broken_signal")
        finding(
            "Missing delivered device signals increased despite retained upstream values; "
            "recorded missing-signal rules declined affected payments, alongside an integration deployment.",
            health, deployments,
        )
    if (len(previous) == 1 and len(current) == 1
            and current[0]["decline_threshold"] < previous[0]["decline_threshold"]
            and "decision_policy" in changed_components and decline_delta >= 2):
        candidates.append("restrictive_rule")
        finding(
            f"The effective decline threshold fell from {previous[0]['decline_threshold']} to "
            f"{current[0]['decline_threshold']}, with a recorded policy deployment and increased declines. "
            "Whether this change is appropriate requires outcome assessment.", deployments, cohorts,
        )
    linked_increase = ap["young_new_device_low_velocity"] - bp["young_new_device_low_velocity"]
    if linked_increase >= 5:
        candidates.append("emerging_fraud")
        finding(
            f"Young accounts using new devices at low individual velocity increased within shared-instrument "
            f"groups by {linked_increase} transactions. Linkage alone is not fraud confirmation.", links,
        )
    report["candidate_causes"] = candidates
    if len(candidates) > 1:
        return finish("Multiple mechanisms are supported; investigate their interaction before choosing a response.")

    known_models = {version for period in deployments["data"]["models"].values() for version in period}
    if known_models != {MODEL_VERSION}:
        finding("Recorded model versions require an adapter beyond the supported heuristic fixture.", deployments)
        return finish("Model behavior cannot be safely reproduced with the current tools; no response tested.")
    if not candidates:
        unexplained_health = (
            hp["after"]["stale"] > hp["before"]["stale"]
            or hp["after"]["missing"] > hp["before"]["missing"]
            or hp["after"]["source_delivery_mismatches"] > hp["before"]["source_delivery_mismatches"]
        )
        if (abs(decline_delta) < 2 and (fraud_delta is None or abs(fraud_delta) < 2)
                and not changed_components and not unexplained_health):
            report["status"] = "no_material_change"
            return finish(
                "No material change found in the monitored indicators under this runbook's thresholds; "
                "this is not a guarantee that no fraud exists."
            )
        return finish("The observed change is not explained by this runbook; further investigation is required.")

    cause = candidates[0]
    if cause == "emerging_fraud":
        confirmed = mature and fraud_delta >= 2 and outcomes["data"]["confirmed_fraud_in_linked_after_groups"] >= 5
        report["status"] = "supported" if confirmed else "suspected"
        report["diagnosis"] = "emerging_fraud" if confirmed else "suspected_emerging_fraud"
        if confirmed:
            finding("Eligible outcomes show increased fraud and confirmed fraud within linked groups.", outcomes, links)
        else:
            report["limitations"].append("The linked pattern is suspicious; available outcomes do not confirm emerging fraud.")
    else:
        report["diagnosis"] = cause
        report["status"] = "supported"
    try:
        replay = tools.call("replay", response=RESPONSE_BY_CAUSE[cause])
    except ToolError as error:
        report["status"] = "inconclusive"
        report["diagnosis"] = None
        report["limitations"].append(f"Replay failed: {error}")
        return finish("A candidate explanation exists, but reproducibility checks failed; no response endorsed.")
    report["response_test"] = {"evidence_id": replay["evidence_id"], **replay["data"]}
    finding(
        f"Offline replay changes {replay['data']['changed_decisions']} decisions: "
        f"{replay['data']['newly_approved']} additional approvals and "
        f"{replay['data']['newly_declined']} additional declines.", replay,
    )
    if not replay["data"]["changed_decisions"]:
        report["status"] = "inconclusive"
        report["diagnosis"] = None
        return finish("The candidate response did not change decisions; the explanation needs further verification.")
    if not mature:
        report["limitations"].append("Replay decision changes are visible, but full fraud/customer trade-offs await labels.")
    return finish(
        f"{report['diagnosis']}: evidence and a candidate response are available for human review. "
        "No production policy was changed."
    )


def render_markdown(report: dict) -> str:
    notice = report.get("mode_notice", "Scripted baseline; no LLM.")
    lines = [f"# Investigation: {report['incident_id']}", "",
             f"**Mode:** {report['investigator']}", f"**Mode notice:** {notice}",
             f"**As of:** {report['as_of']}", f"**Status:** {report['status']}",
             f"**Diagnosis:** {report['diagnosis'] or 'Not established'}", "", report["summary"], "",
             "## Findings", ""]
    lines.extend(f"- {f['claim']} [{', '.join(f['evidence_ids'])}]" for f in report["findings"])
    bound_values = [finding for finding in report["findings"] if "observed_value" in finding]
    if bound_values:
        lines.extend(["", "## Values attached by the evidence validator", ""])
        for finding in bound_values:
            lines.extend([f"**{finding['evidence_ids'][0]} {finding['json_pointer']}**", "", "```json",
                          json_text(finding["observed_value"]), "```", ""])
        lines.append("Interpretations are model-generated text and remain subject to human verification.")
    if report["response_test"]:
        replay = report["response_test"]
        lines.extend(["", "## Candidate response replay", "",
                      f"**Response:** `{replay['response']}` [{replay['evidence_id']}]",
                      f"Additional approvals: {replay['newly_approved']}; additional declines: {replay['newly_declined']}.",
                      f"Outcome coverage: {replay['baseline_outcomes']['label_coverage']:.1%}.", ""])
        if replay["baseline_outcomes"]["labelled_transactions"]:
            lines.extend(["| Metric (eligible labels only) | Original | Proposed |",
                          "|---|---:|---:|"])
            for key in ("fraud_declined", "legitimate_declined", "approved_fraud_amount_cents"):
                lines.append(f"| {key} | {replay['baseline_outcomes'][key]} | {replay['proposal_outcomes'][key]} |")
        else:
            lines.append("Fraud and legitimacy trade-offs are unknown: no eligible labels are available.")
        lines.extend(["", "### Replay assumptions", ""])
        lines.extend(f"- {note}" for note in replay["assumptions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {note}" for note in report["limitations"])
    lines.extend(["", "## Evidence", ""])
    for record in report["evidence"]:
        lines.extend([f"### {record['evidence_id']}: {record['tool']}", "", "```json",
                      json_text(record), "```", ""])
    lines.extend([f"Tool calls: {report['tool_calls']}; elapsed seconds: {report['elapsed_seconds']}.",
                  f"Evidence digest: `{report['evidence_digest']}`.", "",
                  "Synthetic experiment. Human review required. No production change performed.", ""])
    if "usage" in report:
        lines.extend(["## Run metadata", "", "```json", json_text({
            key: report[key] for key in ("run_id", "run_status", "provider", "model", "model_calls", "usage")
        }), "```", ""])
    return "\n".join(lines)


def json_text(value: dict) -> str:
    import json
    return json.dumps(value, indent=2, sort_keys=True)
