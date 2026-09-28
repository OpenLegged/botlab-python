"""Elementwise math, reductions, clock, noise and filter processors."""
from ._signal import Signal, _numeric_signal, _require_signal
from ._vectors import _finite_number, _non_negative_integer, _positive_integer


def _signal_math(operation, source, other=None, *, width=None):
    data = {"operation": operation}
    if width is not None:
        data["width"] = _positive_integer(width, f"{operation} width")
    inputs = [_numeric_signal(source)]
    if other is not None:
        inputs.append(_numeric_signal(other))
    return Signal("proc-signal-math", operation, data, inputs)


def absolute(source):
    return _signal_math("abs", source)


def sqrt(source):
    return _signal_math("sqrt", source)


def sin(source):
    return _signal_math("sin", source)


def cos(source):
    return _signal_math("cos", source)


def exp(source):
    return _signal_math("exp", source)


def log(source):
    return _signal_math("log", source)


def reduce_sum(source, *, width=None):
    return _signal_math("sum", source, width=width)


def reduce_min(source, *, width=None):
    return _signal_math("min", source, width=width)


def reduce_max(source, *, width=None):
    return _signal_math("max", source, width=width)


def norm(source, *, width=None):
    return _signal_math("norm", source, width=width)


def normalize(source, *, width=None):
    return _signal_math("normalize", source, width=width)


def dot(left, right, *, width=None):
    return _signal_math("dot", left, right, width=width)


def cross(left, right):
    return _signal_math("cross", left, right, width=3)


def clock(*, unit="seconds"):
    """Control frame time or step index, supplied by the simulation (never wall time)."""
    if unit not in ("seconds", "steps"):
        raise ValueError('clock unit must be "seconds" or "steps"')
    return Signal("proc-clock", "clock", {"unit": unit})


def noise(source, *, distribution="uniform", scale=1.0, bias=0.0, seed):
    """Add repeatable noise; reset replays the explicitly seeded sequence."""
    if distribution not in ("uniform", "gaussian"):
        raise ValueError('noise distribution must be "uniform" or "gaussian"')
    seed = _non_negative_integer(seed, "noise seed")
    if seed > 0xffffffff:
        raise ValueError("noise seed must fit uint32")
    return Signal("proc-signal-noise", "noise", {
        "distribution": distribution, "scale": scale, "bias": bias, "seed": seed,
    }, (_require_signal(source, "noise source"),))


def low_pass(source, *, alpha):
    """Exponential smoothing, primed by the first real sample."""
    alpha = _finite_number(alpha, "low_pass alpha")
    if not 0 < alpha <= 1:
        raise ValueError("low_pass alpha must be in (0, 1]")
    return Signal("proc-signal-filter", "low-pass", {"kind": "ema", "alpha": alpha},
                  (_require_signal(source, "low_pass source"),))


def moving_average(source, *, window):
    return Signal("proc-signal-filter", "moving-average", {
        "kind": "moving_average", "window": _positive_integer(window, "moving_average window"),
    }, (_require_signal(source, "moving_average source"),))


def derivative(source, *, dt):
    dt = _finite_number(dt, "derivative dt")
    if dt <= 0:
        raise ValueError("derivative dt must be positive")
    return Signal("proc-signal-filter", "derivative", {"kind": "difference", "dt": dt},
                  (_require_signal(source, "derivative source"),))
