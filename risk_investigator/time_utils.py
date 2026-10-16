from datetime import datetime, timezone


def parse_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid ISO 8601 timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone, for example 2026-01-27T00:00:00Z")
    return parsed.astimezone(timezone.utc)


def normalize_time(value: str) -> str:
    return parse_time(value).isoformat(timespec="seconds")
