"""Minimal local configuration loading; no expansion, execution, or secret logging."""

import os
from pathlib import Path
import re

ALLOWED_ENV = {
    "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_INPUT_USD_PER_MILLION",
    "OPENAI_OUTPUT_USD_PER_MILLION", "OPENAI_PRICING_MODEL",
}


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    if path.stat().st_size > 16_384:
        raise ValueError("Local .env exceeds the configuration size limit")
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"Invalid .env assignment on line {number}")
        if key not in ALLOWED_ENV:
            continue
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote on line {number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        # Existing shell values take precedence, including deliberately empty values.
        os.environ.setdefault(key, value)
