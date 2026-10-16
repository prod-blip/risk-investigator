"""Bounded, read-only investigation tools shared by baseline and future agent."""

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict, replace
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
import sqlite3

from .engine import MODEL_VERSION, decide
from .models import Policy, Transaction
from .outcomes import OutcomeReader
from .time_utils import normalize_time, parse_time


class ToolError(ValueError):
    pass


class InvestigationTools:
    NAMES = (
        "incident_context", "compare_cohorts", "feature_health", "deployment_history",
        "linked_activity", "outcome_metrics", "decision_trace", "replay",
    )
    REPLAYS = ("restore_device_signal", "previous_threshold", "linked_activity_control")

    def __init__(self, database: Path, incident_id: str, *, as_of: str,
                 outcomes_database: Path | None = None, max_calls: int = 12):
        if type(max_calls) is not int or not 1 <= max_calls <= 100:
            raise ToolError("max_calls must be an integer from 1 to 100")
        self.as_of = normalize_time(as_of)
        self.incident_id = incident_id
        self.max_calls = max_calls
        self.history: list[dict] = []
        self._eligible_outcomes: dict | None = None
        self.outcomes = OutcomeReader(outcomes_database) if outcomes_database else None
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            incident = connection.execute(
                "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if incident is None:
                raise ToolError(f"Unknown incident: {incident_id}")
            self.incident = dict(incident)
            self.rows = [dict(row) for row in connection.execute(
                "SELECT * FROM decision_evidence WHERE incident_id = ? AND occurred_at <= ? "
                "ORDER BY occurred_at, transaction_id LIMIT 20001", (incident_id, self.as_of)
            )]
            if len(self.rows) > 20_000:
                raise ToolError("Investigation exceeds the 20000-transaction tool limit")
            self.changes = [dict(row) for row in connection.execute(
                "SELECT * FROM changes WHERE incident_id = ? AND occurred_at <= ? "
                "ORDER BY occurred_at, change_id LIMIT 101", (incident_id, self.as_of)
            )]
        if len(self.changes) > 100:
            raise ToolError("Investigation exceeds the 100-change-event tool limit")
        for row in self.rows:
            if row["device_observed_at"] and parse_time(row["device_observed_at"]) > parse_time(row["occurred_at"]):
                raise ToolError("A device feature timestamp is later than its decision time")
        self.evidence_digest = sha256(json.dumps(
            {"rows": self.rows, "changes": self.changes}, sort_keys=True
        ).encode()).hexdigest()

    def call(self, name: str, **arguments) -> dict:
        if name not in self.NAMES:
            raise ToolError(f"Unknown tool: {name}")
        if len(self.history) >= self.max_calls:
            raise ToolError("Investigation tool-call budget exhausted")
        method = getattr(self, name)
        try:
            signature(method).bind(**arguments)
        except TypeError as error:
            raise ToolError(str(error)) from error
        record = {
            "evidence_id": f"ev_{len(self.history) + 1:03d}", "tool": name,
            "arguments": arguments, "as_of": self.as_of,
            "evidence_digest": self.evidence_digest,
        }
        # Failed calls consume budget too, preventing unlimited retries.
        self.history.append(record)
        try:
            record["data"] = method(**arguments)
        except (ValueError, sqlite3.Error) as error:
            record["error"] = str(error)
            raise ToolError(str(error)) from error
        return record

    @staticmethod
    def _refs(rows: list[dict], limit: int = 3) -> list[str]:
        return [f"decision:{row['transaction_id']}" for row in rows[:limit]]

    def _periods(self) -> dict[str, list[dict]]:
        return {p: [r for r in self.rows if r["period"] == p] for p in ("before", "after")}

    @staticmethod
    def _counts(rows: list[dict]) -> dict:
        n = len(rows)
        declines = sum(r["action"] == "decline" for r in rows)
        return {
            "transactions": n, "declines": declines,
            "decline_rate_pct": round(100 * declines / n, 4) if n else None,
            "missing_device_signals": sum(r["device_status"] is None for r in rows),
            "mean_score": round(sum(r["score"] for r in rows) / n, 4) if n else None,
            "source_refs": InvestigationTools._refs(rows),
        }

    def incident_context(self) -> dict:
        return {
            **self.incident, "as_of": self.as_of,
            "analysis_window_complete": self.as_of >= self.incident["end_at"],
            "visible_transactions": len(self.rows), "synthetic_data": True,
            "outcomes_configured": self.outcomes is not None,
        }

    def compare_cohorts(self, group_by: str = "channel") -> dict:
        if group_by not in ("channel", "country", "merchant_id"):
            raise ToolError("group_by must be channel, country, or merchant_id")
        periods = self._periods()
        totals = {p: self._counts(rows) for p, rows in periods.items()}
        groups = sorted({r[group_by] for r in self.rows})
        return {
            "group_by": group_by, "totals": totals,
            "groups": [{"value": group, **{
                p: self._counts([r for r in rows if r[group_by] == group])
                for p, rows in periods.items()
            }} for group in groups[:20]],
            "groups_truncated": len(groups) > 20,
        }

    def feature_health(self) -> dict:
        result = {}
        for period, rows in self._periods().items():
            missing = [r for r in rows if r["device_status"] is None]
            mismatch = [r for r in rows if r["device_status"] != r["source_device_status"]]
            stale = [r for r in rows if r["device_observed_at"] and (
                parse_time(r["occurred_at"]) - parse_time(r["device_observed_at"])
            ).total_seconds() > 300]
            n = len(rows)
            result[period] = {
                "transactions": n, "missing": len(missing),
                "missing_rate": len(missing) / n if n else None,
                "source_delivery_mismatches": len(mismatch), "stale": len(stale),
                "missing_signal_declines": sum(
                    "DEVICE_SIGNAL_MISSING" in json.loads(r["triggered_rules"]) for r in rows
                ),
                "feature_versions": dict(Counter(r["feature_version"] for r in rows)),
                "source_refs": self._refs(mismatch or stale or rows),
            }
        return {"periods": result, "freshness_limit_seconds": 300}

    def deployment_history(self) -> dict:
        policies = {}
        models = {}
        for period, rows in self._periods().items():
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row["policy_version"], row["decline_threshold"])].append(row)
            policies[period] = [
                {"version": version, "decline_threshold": threshold,
                 "transactions": len(group), "source_refs": self._refs(group)}
                for (version, threshold), group in sorted(grouped.items())
            ]
            models[period] = dict(Counter(r["model_version"] for r in rows))
        return {
            "policies": policies, "models": models,
            "changes": [{**r, "details": json.loads(r["details"]),
                         "source_ref": f"change:{r['change_id']}"} for r in self.changes],
        }

    def _linked(self, rows: list[dict], min_accounts: int = 5) -> list[dict]:
        groups = defaultdict(list)
        for row in rows:
            groups[row["payment_instrument_id"]].append(row)
        result = []
        for instrument, records in groups.items():
            accounts = len({r["customer_id"] for r in records})
            if accounts >= min_accounts:
                suspicious = [r for r in records if r["account_age_days"] < 30
                              and r["device_status"] == "new" and r["velocity_1h"] < 4]
                result.append({
                    "payment_instrument_id": instrument, "distinct_accounts": accounts,
                    "transactions": len(records), "young_new_device_low_velocity": len(suspicious),
                    "approved": sum(r["action"] == "approve" for r in records),
                    "source_refs": self._refs(suspicious or records),
                })
        return sorted(result, key=lambda r: (-r["distinct_accounts"], r["payment_instrument_id"]))

    def linked_activity(self, min_accounts: int = 5) -> dict:
        if type(min_accounts) is not int or not 2 <= min_accounts <= 100:
            raise ToolError("min_accounts must be an integer from 2 to 100")
        periods = {}
        for period, rows in self._periods().items():
            groups = self._linked(rows, min_accounts)
            periods[period] = {
                "qualifying_groups": len(groups),
                "young_new_device_low_velocity": sum(g["young_new_device_low_velocity"] for g in groups),
                "groups": groups[:20], "groups_truncated": len(groups) > 20,
            }
        return {"min_accounts": min_accounts, "periods": periods,
                "interpretation": "Shared instruments are evidence of linkage, not proof of fraud."}

    def _labels(self) -> dict:
        if self._eligible_outcomes is None:
            self._eligible_outcomes = self.outcomes.available(
                [r["transaction_id"] for r in self.rows], self.as_of
            ) if self.outcomes else {}
        return self._eligible_outcomes

    def _outcomes_digest(self) -> str:
        return sha256(json.dumps(self._labels(), sort_keys=True).encode()).hexdigest()

    def decision_trace(self, transaction_id: str) -> dict:
        if not isinstance(transaction_id, str) or not 1 <= len(transaction_id) <= 128:
            raise ToolError("transaction_id must be a string of 1 to 128 characters")
        row = next((r for r in self.rows if r["transaction_id"] == transaction_id), None)
        if row is None:
            raise ToolError("Transaction is not visible in this incident at the investigation timestamp")
        return {
            "transaction_id": transaction_id, "input_snapshot": json.loads(row["input_snapshot"]),
            "score": row["score"], "action": row["action"],
            "model_version": row["model_version"], "policy_version": row["policy_version"],
            "decline_threshold": row["decline_threshold"],
            "triggered_rules": json.loads(row["triggered_rules"]),
            "contributions": json.loads(row["contributions"]),
            "source_refs": [f"decision:{transaction_id}"],
        }

    @staticmethod
    def _outcome_counts(rows: list[dict], labels: dict, actions: dict | None = None) -> dict:
        known = [r for r in rows if r["transaction_id"] in labels]
        def action(row):
            return actions[row["transaction_id"]] if actions is not None else row["action"]
        fraud = [r for r in known if labels[r["transaction_id"]]["is_fraud"]]
        legitimate = [r for r in known if not labels[r["transaction_id"]]["is_fraud"]]
        return {
            "total_transactions": len(rows), "labelled_transactions": len(known),
            "label_coverage": len(known) / len(rows) if rows else 0,
            "unlabelled_transactions": len(rows) - len(known),
            "labelled_fraud": len(fraud), "labelled_legitimate": len(legitimate),
            "fraud_rate_pct_among_labelled": 100 * len(fraud) / len(known) if known else None,
            "fraud_declined": sum(action(r) == "decline" for r in fraud),
            "legitimate_declined": sum(action(r) == "decline" for r in legitimate),
            "approved_fraud_amount_cents": sum(r["amount_cents"] for r in fraud if action(r) == "approve"),
        }

    def outcome_metrics(self) -> dict:
        labels = self._labels()
        periods = self._periods()
        after_groups = self._linked(periods["after"])
        linked_instruments = {g["payment_instrument_id"] for g in after_groups}
        linked_fraud = [r for r in periods["after"] if r["payment_instrument_id"] in linked_instruments
                        and r["transaction_id"] in labels and labels[r["transaction_id"]]["is_fraud"]]
        return {
            "outcomes_configured": self.outcomes is not None,
            "eligible_outcomes_digest": self._outcomes_digest(),
            "periods": {p: self._outcome_counts(rows, labels) for p, rows in periods.items()},
            "confirmed_fraud_in_linked_after_groups": len(linked_fraud),
            "source_refs": [f"outcome:{r['transaction_id']}" for r in linked_fraud[:3]],
            "limitations": [
                "Only outcomes available at the investigation timestamp are included.",
                "Unlabelled payments have unknown legitimacy; partial labels may be unrepresentative.",
                "Synthetic labels include declined payments; equivalent real-world outcomes may be unavailable.",
            ],
        }

    def replay(self, response: str, min_accounts: int = 5) -> dict:
        if response not in self.REPLAYS:
            raise ToolError(f"response must be one of {', '.join(self.REPLAYS)}")
        if type(min_accounts) is not int or not 2 <= min_accounts <= 100:
            raise ToolError("min_accounts must be an integer from 2 to 100")
        periods = self._periods()
        if not periods["before"] or not periods["after"]:
            raise ToolError("Replay requires visible transactions in both periods")
        old_thresholds = {r["decline_threshold"] for r in periods["before"]}
        if response == "previous_threshold" and len(old_thresholds) != 1:
            raise ToolError("Previous threshold is ambiguous")
        old_threshold = next(iter(old_thresholds))
        proposal_actions, seen_accounts = {}, defaultdict(set)
        changed_refs = []
        newly_approved = newly_declined = 0
        for row in self.rows:
            if row["model_version"] != MODEL_VERSION:
                raise ToolError("Replay does not support this recorded model version")
            if row["policy_version"] not in ("rules-v1", "rules-v2"):
                raise ToolError("Replay does not support this recorded policy version")
            transaction = Transaction(**json.loads(row["input_snapshot"]))
            if asdict(transaction) != {key: row[key] for key in asdict(transaction)}:
                raise ToolError("Stored input snapshot differs from observed transaction")
            policy = Policy(version=row["policy_version"], decline_threshold=row["decline_threshold"])
            reproduced = decide(transaction, policy)
            if (reproduced.action != row["action"] or reproduced.score != row["score"]
                    or list(reproduced.triggered_rules) != json.loads(row["triggered_rules"])
                    or reproduced.contributions != json.loads(row["contributions"])):
                raise ToolError("Recorded decision cannot be reproduced; response evaluation stopped")
            instrument = transaction.payment_instrument_id
            seen_accounts[instrument].add(transaction.customer_id)
            if row["period"] == "before":
                continue
            if response == "restore_device_signal" and transaction.device_status is None:
                transaction = replace(transaction, device_status=transaction.source_device_status)
            if response == "previous_threshold":
                policy = replace(policy, decline_threshold=old_threshold)
            proposed = decide(transaction, policy).action
            if (response == "linked_activity_control" and len(seen_accounts[instrument]) >= min_accounts
                    and transaction.account_age_days < 30 and transaction.device_status == "new"):
                proposed = "decline"
            proposal_actions[row["transaction_id"]] = proposed
            if proposed != row["action"]:
                changed_refs.append(f"decision:{row['transaction_id']}")
                newly_approved += proposed == "approve"
                newly_declined += proposed == "decline"
        labels = self._labels()
        baseline = self._outcome_counts(periods["after"], labels)
        proposal = self._outcome_counts(periods["after"], labels, proposal_actions)
        metric_keys = ("fraud_declined", "legitimate_declined", "approved_fraud_amount_cents")
        measurable = baseline["labelled_transactions"] > 0
        return {
            "response": response, "scope": "visible after-period transactions",
            "eligible_outcomes_digest": self._outcomes_digest(),
            "transactions": len(periods["after"]), "newly_approved": newly_approved,
            "newly_declined": newly_declined,
            "baseline_outcomes": baseline, "proposal_outcomes": proposal,
            "labelled_outcome_delta": {
                key: proposal[key] - baseline[key] if measurable else None for key in metric_keys
            },
            "source_refs": changed_refs[:5], "changed_decisions": len(changed_refs),
            "parameters": {"previous_threshold": old_threshold} if response == "previous_threshold" else (
                {"min_accounts": min_accounts, "maximum_account_age_days": 29, "device_status": "new"}
                if response == "linked_activity_control" else {}
            ),
            "assumptions": [
                "This is offline replay, not a production change or a causal estimate of future losses.",
                "Recorded inputs and the supported heuristic must reproduce original decisions.",
                "Customers and attackers are assumed not to change behavior in response to a new policy.",
                "A signal repair assumes the retained upstream value could have been delivered at decision time.",
                "Linkage counts use only accounts seen up to each replayed decision, never later transactions.",
                "Outcome trade-offs cover only timestamp-eligible labels and may be incomplete.",
            ],
        }
