"""Optional control-step budget. Hosts stop only after committing the last step."""


def _positive_steps(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 9007199254740991:
        raise ValueError("end_after_steps must use positive safe integer step counts")
    return value


def encode_end_after_steps(value):
    if not isinstance(value, dict):
        return {"steps": _positive_steps(value)}
    if set(value) != {"parameter", "steps"} or not isinstance(value["parameter"], str) or not value["parameter"].strip():
        raise ValueError("end_after_steps requires parameter and steps")
    choices = value["steps"]
    if not isinstance(choices, dict) or not choices:
        raise ValueError("end_after_steps steps must map option values to positive step counts")
    entries = []
    for option, steps in choices.items():
        if isinstance(option, bool) or not isinstance(option, int) or abs(option) > 9007199254740991:
            raise ValueError("end_after_steps option values must be safe integers")
        entries.append({"value": option, "steps": _positive_steps(steps)})
    return {"parameter": value["parameter"], "steps": entries}


def parse_end_after_steps(raw, parameters):
    """Validate compiled metadata once and return a resolver for a new run."""
    def fail():
        raise ValueError("Invalid end_after_steps: map every option to a positive safe integer step count")

    if not isinstance(raw, dict):
        fail()
    if set(raw) == {"steps"}:
        steps = _positive_steps(raw["steps"])
        return lambda read: steps
    if set(raw) != {"parameter", "steps"} or not isinstance(raw["parameter"], str):
        fail()
    options = parameters.get(raw["parameter"], {}).get("options")
    entries = raw["steps"]
    if not options or not isinstance(entries, list) or len(entries) != len(options):
        fail()
    counts = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"value", "steps"}:
            fail()
        value = entry["value"]
        if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 9007199254740991:
            fail()
        if value in counts or not any(option["value"] == value for option in options):
            fail()
        counts[value] = _positive_steps(entry["steps"])

    def resolve(read):
        selected = read(raw["parameter"])
        if selected not in counts:
            fail()
        return counts[selected]
    return resolve
