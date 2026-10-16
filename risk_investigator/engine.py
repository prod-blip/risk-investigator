"""Transparent heuristic scoring and explicit policy enforcement, not trained ML."""

from .models import Decision, Policy, Transaction

MODEL_VERSION = "heuristic-v1"


def decide(transaction: Transaction, policy: Policy = Policy()) -> Decision:
    contributions = {
        "large_amount": 20 if transaction.amount_cents >= 30_000 else 0,
        "young_account": 15 if transaction.account_age_days < 30 else 0,
        "new_device": 25 if transaction.device_status == "new" else 0,
        "high_velocity": 30 if transaction.velocity_1h >= 4 else 0,
    }
    score = min(100, sum(contributions.values()))
    triggered = []
    if transaction.device_status is None and policy.decline_missing_device:
        triggered.append("DEVICE_SIGNAL_MISSING")
    if score >= policy.decline_threshold:
        triggered.append("SCORE_AT_OR_ABOVE_THRESHOLD")
    return Decision(
        transaction_id=transaction.transaction_id,
        score=score,
        action="decline" if triggered else "approve",
        model_version=MODEL_VERSION,
        policy_version=policy.version,
        decline_threshold=policy.decline_threshold,
        triggered_rules=tuple(triggered),
        contributions=contributions,
    )
