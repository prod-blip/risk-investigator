"""Decision-time records. These types deliberately contain no fraud labels."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    incident_id: str
    period: Literal["before", "after"]
    occurred_at: str
    customer_id: str
    merchant_id: str
    payment_instrument_id: str
    country: str
    channel: str
    amount_cents: int
    account_age_days: int
    velocity_1h: int
    device_status: Literal["known", "new"] | None
    device_observed_at: str | None
    source_device_status: Literal["known", "new"]
    feature_version: str

    def __post_init__(self) -> None:
        if self.amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        if self.account_age_days < 0 or self.velocity_1h < 0:
            raise ValueError("account age and velocity cannot be negative")


@dataclass(frozen=True)
class Policy:
    version: str = "rules-v1"
    decline_threshold: int = 70
    decline_missing_device: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.decline_threshold <= 100:
            raise ValueError("decline_threshold must be between 1 and 100")


@dataclass(frozen=True)
class Decision:
    transaction_id: str
    score: int
    action: Literal["approve", "decline"]
    model_version: str
    policy_version: str
    decline_threshold: int
    triggered_rules: tuple[str, ...]
    contributions: dict[str, int]
