"""Controlled experiments with private grading truth and timestamped observations."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import random
from typing import Literal

from .engine import decide
from .models import Decision, Policy, Transaction

Scenario = Literal["emerging_fraud", "broken_signal", "restrictive_rule"]
SCENARIOS: tuple[Scenario, ...] = (
    "emerging_fraud", "broken_signal", "restrictive_rule"
)
START = datetime(2026, 1, 12, 9, tzinfo=timezone.utc)


def timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def opaque_id(prefix: str, value: str) -> str:
    return f"{prefix}_{sha256(value.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class EvaluationLabel:
    transaction_id: str
    is_fraud: bool
    available_at: str


@dataclass(frozen=True)
class Change:
    change_id: str
    incident_id: str
    occurred_at: str
    component: str
    previous_version: str
    current_version: str
    details: dict[str, object]


@dataclass(frozen=True)
class Experiment:
    incident_id: str
    scenario: Scenario
    seed: int
    transactions: tuple[Transaction, ...]
    decisions: tuple[Decision, ...]
    labels: tuple[EvaluationLabel, ...]
    changes: tuple[Change, ...]
    boundary_at: str
    evaluation_at: str


def simulate(
    scenario: Scenario, *, seed: int = 42, transactions_per_period: int = 500
) -> Experiment:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not 50 <= transactions_per_period <= 100_000:
        raise ValueError("transactions_per_period must be between 50 and 100000")

    # No scenario-specific words appear in identifiers exposed to investigators.
    incident_id = opaque_id("inc", f"{seed}:{scenario}")
    transactions: list[Transaction] = []
    decisions: list[Decision] = []
    labels: list[EvaluationLabel] = []
    boundary = START + timedelta(seconds=transactions_per_period * 10)
    for period in ("before", "after"):
        # Background traffic is identical across scenarios with the same seed.
        rng = random.Random(f"{seed}:{period}")
        attack_rng = random.Random(f"{seed}:attack:{period}")
        for index in range(transactions_per_period):
            occurred_at = START + timedelta(
                seconds=(index + (transactions_per_period if period == "after" else 0)) * 10
            )
            channel = rng.choice(("mobile_web", "desktop", "app"))
            country = rng.choice(("US", "GB", "IN"))
            amount_cents = rng.randint(500, 50_000)
            account_age_days = rng.randint(1, 730)
            velocity_1h = rng.choice((1, 1, 1, 2, 2, 3))
            device = "new" if rng.random() < 0.15 else "known"
            merchant = opaque_id("mer", str(rng.randrange(20)))
            instrument = opaque_id("pi", f"ordinary:{rng.randrange(1000)}")
            is_fraud = rng.random() < 0.02
            if is_fraud:
                # Existing, individually conspicuous fraud is mostly caught.
                account_age_days, velocity_1h, device = 3, 6, "new"
                amount_cents = max(amount_cents, 30_000)
            if scenario == "emerging_fraud" and period == "after":
                if attack_rng.random() < 0.18:
                    # Attackers share instruments but keep most account velocities low.
                    is_fraud = True
                    instrument = opaque_id("pi", f"shared:{attack_rng.randrange(3)}")
                    account_age_days, device = 5, "new"
                    amount_cents = attack_rng.randint(5_000, 45_000)
                    velocity_1h = 4 if attack_rng.random() < 0.3 else 1

            feature_version = "device-parser-v1"
            delivered_device = device
            observed_at = timestamp(occurred_at - timedelta(seconds=2))
            if scenario == "broken_signal" and period == "after":
                feature_version = "device-parser-v2"
                if channel == "mobile_web":
                    delivered_device, observed_at = None, None

            policy = Policy()
            if scenario == "restrictive_rule" and period == "after":
                policy = Policy(version="rules-v2", decline_threshold=25)
            transaction = Transaction(
                transaction_id=opaque_id("txn", f"{incident_id}:{period}:{index}"),
                incident_id=incident_id,
                period=period,
                occurred_at=timestamp(occurred_at),
                customer_id=opaque_id("cus", f"{seed}:{period}:{index}"),
                merchant_id=merchant,
                payment_instrument_id=instrument,
                country=country,
                channel=channel,
                amount_cents=amount_cents,
                account_age_days=account_age_days,
                velocity_1h=velocity_1h,
                device_status=delivered_device,
                device_observed_at=observed_at,
                source_device_status=device,
                feature_version=feature_version,
            )
            transactions.append(transaction)
            decisions.append(decide(transaction, policy))
            labels.append(EvaluationLabel(
                transaction_id=transaction.transaction_id,
                is_fraud=is_fraud,
                available_at=timestamp(occurred_at + timedelta(days=14)),
            ))

    changes: list[Change] = []
    if scenario == "broken_signal":
        changes.append(Change(
            change_id=opaque_id("chg", incident_id), incident_id=incident_id,
            occurred_at=timestamp(boundary), component="feature_integration",
            previous_version="device-parser-v1", current_version="device-parser-v2",
            details={"deployment": "Device field parser update"},
        ))
    elif scenario == "restrictive_rule":
        changes.append(Change(
            change_id=opaque_id("chg", incident_id), incident_id=incident_id,
            occurred_at=timestamp(boundary), component="decision_policy",
            previous_version="rules-v1", current_version="rules-v2",
            details={"decline_threshold": {"before": 70, "after": 25}},
        ))
    end = START + timedelta(seconds=(transactions_per_period * 2 - 1) * 10)
    return Experiment(
        incident_id=incident_id, scenario=scenario, seed=seed,
        transactions=tuple(transactions), decisions=tuple(decisions),
        labels=tuple(labels), changes=tuple(changes),
        boundary_at=timestamp(boundary),
        evaluation_at=timestamp(end + timedelta(days=14)),
    )
