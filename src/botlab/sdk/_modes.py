"""Mode switches, state machines, phase clocks, mode select and delay."""
import math

from ._signal import Signal, _require_signal
from ._vectors import _finite_values, _non_negative_integer, _positive_integer


def mode_switch(command, *, initial, transitions):
    """Turn discrete command pulses into a persistent controller mode."""
    command = _require_signal(command, "mode_switch command")
    if isinstance(initial, bool) or not isinstance(initial, int):
        raise ValueError("mode_switch initial must be an integer")
    if isinstance(transitions, dict):
        items = [
            {"from": "any", "command": command_code, "to": target_mode}
            for command_code, target_mode in transitions.items()
        ]
    else:
        items = list(transitions)
    if not items:
        raise ValueError("mode_switch transitions must not be empty")
    normalized = []
    for index, transition in enumerate(items):
        if not isinstance(transition, dict):
            raise TypeError(f"mode_switch transition {index} must be a mapping")
        unknown = set(transition) - {"from", "command", "to"}
        if unknown:
            raise ValueError(
                f"mode_switch transition {index} has unknown fields {sorted(unknown)}"
            )
        if "command" not in transition or "to" not in transition:
            raise ValueError(
                f"mode_switch transition {index} requires command and to"
            )
        command_code = transition["command"]
        to_mode = transition["to"]
        if (
            isinstance(command_code, bool)
            or not isinstance(command_code, int)
            or isinstance(to_mode, bool)
            or not isinstance(to_mode, int)
        ):
            raise ValueError(
                f"mode_switch transition {index} command and to must be integers"
            )
        if command_code == 0:
            raise ValueError(
                f"mode_switch transition {index} command must be non-zero"
            )
        from_modes = transition.get("from", "any")
        if from_modes != "any":
            if isinstance(from_modes, int) and not isinstance(from_modes, bool):
                from_modes = [from_modes]
            else:
                try:
                    from_modes = list(from_modes)
                except TypeError as error:
                    raise ValueError(
                        f"mode_switch transition {index} from must be an integer, "
                        'integer sequence, or "any"'
                    ) from error
                if not from_modes or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in from_modes
                ):
                    raise ValueError(
                        f"mode_switch transition {index} from must contain integers"
                    )
        normalized.append({"from": from_modes, "command": command_code, "to": to_mode})
    return Signal(
        "proc-mode-switch",
        "mode-switch",
        {"initialMode": initial, "transitions": normalized},
        (command,),
    )


class _StateMachine:
    __slots__ = ("state", "steps")

    def __init__(self, signal):
        self.state = signal
        self.steps = signal._output(1, "state-steps")


def state_machine(command, *, initial, transitions):
    """Build a persistent finite-state machine in the native control path.

    Each transition has from and to plus at least one trigger: command for an
    edge-triggered command pulse, after_steps for time in the current state, or
    when for a scalar runtime condition. Triggers on one transition are
    conjunctive. reset_steps=False preserves elapsed steps across a sub-state
    transition.
    """
    command = _require_signal(command, "state_machine command")
    if isinstance(initial, bool) or not isinstance(initial, int):
        raise ValueError("state_machine initial must be an integer")
    items = list(transitions)
    if not items:
        raise ValueError("state_machine transitions must not be empty")
    normalized = []
    conditions = []
    for index, transition in enumerate(items):
        if not isinstance(transition, dict):
            raise TypeError(f"state_machine transition {index} must be a mapping")
        unknown = set(transition) - {
            "from", "to", "command", "after_steps", "when", "reset_steps"
        }
        if unknown:
            raise ValueError(
                f"state_machine transition {index} has unknown fields {sorted(unknown)}"
            )
        if "to" not in transition:
            raise ValueError(f"state_machine transition {index} requires to")
        if not any(name in transition for name in ("command", "after_steps", "when")):
            raise ValueError(
                f"state_machine transition {index} requires command, after_steps, or when"
            )
        to_state = transition["to"]
        if isinstance(to_state, bool) or not isinstance(to_state, int):
            raise ValueError(f"state_machine transition {index} to must be an integer")
        from_states = transition.get("from", "any")
        if from_states != "any":
            if isinstance(from_states, int) and not isinstance(from_states, bool):
                from_states = [from_states]
            else:
                try:
                    from_states = list(from_states)
                except TypeError as error:
                    raise ValueError(
                        f"state_machine transition {index} from must be an integer, "
                        'integer sequence, or "any"'
                    ) from error
                if not from_states or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in from_states
                ):
                    raise ValueError(
                        f"state_machine transition {index} from must contain integers"
                    )
        item = {"from": from_states, "to": to_state}
        if "command" in transition:
            command_code = transition["command"]
            if (
                isinstance(command_code, bool)
                or not isinstance(command_code, int)
                or command_code == 0
            ):
                raise ValueError(
                    f"state_machine transition {index} command must be a non-zero integer"
                )
            item["command"] = command_code
        if "after_steps" in transition:
            item["afterSteps"] = _non_negative_integer(
                transition["after_steps"],
                f"state_machine transition {index} after_steps",
            )
        if "when" in transition:
            condition = _require_signal(
                transition["when"], f"state_machine transition {index} when"
            )
            conditions.append(condition)
            item["conditionSlot"] = len(conditions)
        reset_steps = transition.get("reset_steps", True)
        if not isinstance(reset_steps, bool):
            raise TypeError(
                f"state_machine transition {index} reset_steps must be boolean"
            )
        if not reset_steps:
            item["resetSteps"] = False
        normalized.append(item)
    signal = Signal(
        "proc-state-machine",
        "state-machine",
        {"initialState": initial, "transitions": normalized},
        (command, *conditions),
    )
    return _StateMachine(signal)


def phase_clock(steps, *, period_steps):
    """Return cos/sin phase encoding for a scalar step signal."""
    steps = _require_signal(steps, "phase_clock steps")
    return Signal(
        "proc-phase-clock",
        "phase-clock",
        {"periodSteps": _positive_integer(period_steps, "phase_clock period_steps")},
        (steps,),
    )


def mode_select(mode, hold, choices, *, passive=0):
    """Choose a same-sized signal by mode, retaining ``hold`` for unknown modes."""
    mode = _require_signal(mode, "mode_select mode")
    hold = _require_signal(hold, "mode_select hold")
    if isinstance(passive, bool) or not isinstance(passive, int):
        raise ValueError("mode_select passive must be an integer")
    if not isinstance(choices, dict) or not choices:
        raise ValueError("mode_select choices must be a non-empty mapping")
    slot_modes = []
    inputs = [mode, hold]
    for selected, source in choices.items():
        if isinstance(selected, int) and not isinstance(selected, bool):
            selected_modes = [selected]
        elif isinstance(selected, (tuple, frozenset)):
            selected_modes = list(selected)
        else:
            raise ValueError(
                "mode_select choice keys must be integers or tuples/frozensets of integers"
            )
        if not selected_modes or any(
            isinstance(selected_mode, bool) or not isinstance(selected_mode, int)
            for selected_mode in selected_modes
        ):
            raise ValueError("mode_select choice modes must be integers")
        signal = _require_signal(source, f"mode_select choice {selected}")
        for selected_mode in selected_modes:
            if selected_mode in slot_modes:
                raise ValueError(f"mode_select mode {selected_mode} is duplicated")
            slot_modes.append(selected_mode)
            inputs.append(signal)
    return Signal(
        "proc-mode-select",
        "mode-select",
        {"passiveMode": passive, "slotModes": slot_modes},
        inputs,
    )


def one_hot_index(source, *, threshold=0.0, default=0):
    """Return the index of the largest value above ``threshold`` as a scalar signal."""
    source = _require_signal(source, "one_hot_index source")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("one_hot_index threshold must be finite")
    if isinstance(default, bool) or not isinstance(default, int):
        raise ValueError("one_hot_index default must be an integer")
    return Signal(
        "proc-onehot-index",
        "one-hot-index",
        {"threshold": threshold, "defaultIndex": default},
        (source,),
    )


def delay(source, total_steps, *, age=None, fill=0.0, prime=False):
    """Delay a signal, optionally compensating age and priming from its first sample."""
    source = _require_signal(source, "delay source")
    total_steps = _non_negative_integer(total_steps, "delay total_steps")
    if not isinstance(prime, bool):
        raise TypeError("delay prime must be a bool")
    if isinstance(fill, (int, float)) and not isinstance(fill, bool):
        fill = float(fill)
        if not math.isfinite(fill):
            raise ValueError("delay fill must be finite")
    else:
        fill = _finite_values(fill, "delay fill")
        if not fill:
            raise ValueError("delay fill must not be empty")
    inputs = [source]
    if age is not None:
        inputs.append(_require_signal(age, "delay age"))
    data = {"totalSteps": total_steps, "fill": fill}
    if prime:
        data["primeWithFirstSample"] = True
    return Signal(
        "proc-delay",
        f"{source._label}-delay",
        data,
        inputs,
    )
