"""The finite JSON schema subset used by this application's tools and report."""

import math


class SchemaError(ValueError):
    pass


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def validate(value, schema: dict, path: str = "$", depth: int = 0) -> None:
    if depth > 20:
        raise SchemaError("JSON nesting limit exceeded")
    types = schema.get("type")
    types = [types] if isinstance(types, str) else types
    checks = {
        "object": lambda v: type(v) is dict,
        "array": lambda v: type(v) is list,
        "string": lambda v: type(v) is str,
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float) and math.isfinite(v),
        "boolean": lambda v: type(v) is bool,
        "null": lambda v: v is None,
    }
    if types and not any(checks[t](value) for t in types):
        raise SchemaError(f"{path}: expected {types}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: value is outside the allowed enum")
    if value is None:
        return
    if type(value) is dict:
        properties = schema.get("properties", {})
        if set(schema.get("required", ())) - set(value):
            raise SchemaError(f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise SchemaError(f"{path}: unexpected fields")
        for key, child in value.items():
            if key in properties:
                validate(child, properties[key], f"{path}.{key}", depth + 1)
    elif type(value) is list:
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise SchemaError(f"{path}: array length outside limits")
        for index, child in enumerate(value):
            validate(child, schema["items"], f"{path}[{index}]", depth + 1)
    elif type(value) is str:
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10000):
            raise SchemaError(f"{path}: string length outside limits")
    elif type(value) in (int, float):
        if not math.isfinite(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise SchemaError(f"{path}: number outside limits")


def text(limit: int = 500) -> dict:
    return {"type": "string", "minLength": 1, "maxLength": limit}


EVIDENCE_IDS = {"type": "array", "items": text(16), "minItems": 1, "maxItems": 8}
REPORT_SCHEMA = object_schema({
    "status": {"type": "string", "enum": ["supported", "suspected", "inconclusive", "no_material_change"]},
    "diagnosis": {"type": ["string", "null"], "enum": [None, "broken_signal", "restrictive_rule",
                   "emerging_fraud", "suspected_emerging_fraud"]},
    "summary": text(800),
    "findings": {"type": "array", "minItems": 1, "maxItems": 12, "items": object_schema({
        "interpretation": text(500), "evidence_id": text(16), "json_pointer": text(200),
    })},
    "hypotheses": {"type": "array", "minItems": 1, "maxItems": 8, "items": object_schema({
        "category": {"type": "string", "enum": ["broken_signal", "restrictive_rule", "emerging_fraud", "other"]},
        "description": text(250),
        "verdict": {"type": "string", "enum": ["supported", "rejected", "unresolved"]},
        "evidence_ids": EVIDENCE_IDS,
    })},
    "response_evidence_id": {"type": ["string", "null"], "maxLength": 16},
    "limitations": {"type": "array", "minItems": 1, "maxItems": 8, "items": text(500)},
})


def tool_schemas() -> list[dict]:
    count = {"type": "integer", "minimum": 2, "maximum": 100}
    definitions = [
        ("incident_context", "Inspect the analysis window, completeness, and visible evidence scope.", {}),
        ("compare_cohorts", "Compare before/after decisions across a bounded cohort dimension.", {
            "group_by": {"type": "string", "enum": ["channel", "country", "merchant_id"]}}),
        ("feature_health", "Check missing/stale device features and retained source mismatches.", {}),
        ("deployment_history", "Inspect eligible deployments, model versions, policies, and actual thresholds.", {}),
        ("linked_activity", "Inspect shared instruments and account patterns; linkage is not fraud proof.", {
            "min_accounts": count}),
        ("outcome_metrics", "Inspect only timestamp-eligible labels, coverage, and observed fraud outcomes.", {}),
        ("decision_trace", "Inspect a cited decision within this incident/time scope.", {
            "transaction_id": text(128)}),
        ("replay", "Test a candidate response on recorded transactions. restore_device_signal repairs missing "
         "delivered device values from retained upstream inputs; previous_threshold restores the before-period "
         "score cutoff; linked_activity_control adds a chronological rule for young accounts using new devices "
         "when an instrument reaches min_accounts distinct accounts. Use the response that addresses the "
         "supported mechanism. A negative-control replay is not a test of a different proposed response. "
         "This tool never changes stored/production decisions.", {
            "response": {"type": "string", "enum": ["restore_device_signal", "previous_threshold", "linked_activity_control"]},
            "min_accounts": count}),
    ]
    return [{"type": "function", "name": name, "description": description,
             "strict": True, "parameters": object_schema(properties)}
            for name, description, properties in definitions]


def resolve_pointer(record: dict, pointer: str):
    if pointer != "/data" and not pointer.startswith("/data/"):
        raise SchemaError("Finding pointers must identify /data or a value under /data/")
    current = record
    for segment in pointer[1:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        try:
            if type(current) is list:
                if not segment.isdigit():
                    raise ValueError
                current = current[int(segment)]
            else:
                current = current[segment]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise SchemaError("Finding refers to an unavailable evidence value") from error
    return current
