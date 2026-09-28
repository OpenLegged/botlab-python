"""Motion reference signals read in the engine hot path."""
from ._resources import _motion_source_data
from ._signal import Signal, _require_signal
from ._vectors import _non_negative_integer, _positive_integer


def motion_reference(
    motion,
    *,
    mode=None,
    command=None,
    command_codes=None,
    active_mode=None,
    start=0,
    end=None,
    pause=0,
    frame_interval=1,
    anchor_link=0,
    loop=False,
):
    """Read the current frame from a motion resource in the engine hot path."""
    if not isinstance(loop, bool):
        raise TypeError("motion_reference loop must be a boolean")
    if command is not None and mode is None:
        raise ValueError("motion_reference command requires mode")
    if command is None and command_codes is not None:
        raise ValueError("motion_reference command_codes requires command")
    inputs = []
    if mode is not None:
        inputs.append(_require_signal(mode, "motion_reference mode"))
    if command is not None:
        inputs.append(_require_signal(command, "motion_reference command"))
        required_codes = {"pauseConfigured", "pauseCurrent", "resume"}
        if not isinstance(command_codes, dict) or set(command_codes) != required_codes:
            raise ValueError(
                "motion_reference command_codes must contain exactly "
                '"pauseConfigured", "pauseCurrent", and "resume"'
            )
        normalized_codes = {}
        for name in ("pauseConfigured", "pauseCurrent", "resume"):
            value = command_codes[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"motion_reference command_codes.{name} must be an integer")
            normalized_codes[name] = value
        if len(set(normalized_codes.values())) != len(normalized_codes):
            raise ValueError("motion_reference command_codes values must be distinct")
    data = _motion_source_data(motion)
    data.update({
        "startIndex": _non_negative_integer(start, "motion_reference start"),
        "pauseIndex": _non_negative_integer(pause, "motion_reference pause"),
        "frameInterval": _positive_integer(
            frame_interval,
            "motion_reference frame_interval",
        ),
        "anchorLinkIndex": _non_negative_integer(
            anchor_link,
            "motion_reference anchor_link",
        ),
        "loop": loop,
    })
    if end is not None:
        data["endIndex"] = _non_negative_integer(end, "motion_reference end")
    if active_mode is not None:
        if isinstance(active_mode, bool) or not isinstance(active_mode, int):
            raise ValueError("motion_reference active_mode must be an integer")
        data["activeMode"] = active_mode
    if command is not None:
        data["commandCodes"] = normalized_codes
    return Signal("proc-motion-reference", "motion-reference", data, inputs)
