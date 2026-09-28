"""History stacking and image resampling processors."""
import math

from ._signal import Signal, _as_signal, _require_signal
from ._vectors import _finite_values, _positive_integer


_HISTORY_LAYOUTS = {"by-frame": "isaacgym", "by-term": "isaaclab"}
_HISTORY_ORDERS = ("oldest-first", "newest-first")
_RESIZE_LAYOUTS = ("hwc", "chw")


def history(
    source,
    size,
    *,
    stride=1,
    order="oldest-first",
    layout="by-frame",
    fill=None,
):
    """Stack the last `size` samples of `source`, taken every `stride` control steps.

    `source` is a Signal, or a list of Signals to concatenate first.

    `layout` says how the stacked values are laid out, and must match how the
    policy was trained: `"by-frame"` keeps each frame's values together, while
    `"by-term"` runs each concatenated term through time before the next term.
    """
    signal = _as_signal(source, "history source")
    size = _positive_integer(size, "history size")
    stride = _positive_integer(stride, "history stride")
    if order not in _HISTORY_ORDERS:
        raise ValueError(f"history order must be one of {_HISTORY_ORDERS}")
    if layout not in _HISTORY_LAYOUTS:
        raise ValueError(f"history layout must be one of {tuple(_HISTORY_LAYOUTS)}")

    data = {
        "stackSize": size,
        "order": order,
        "stackMode": _HISTORY_LAYOUTS[layout],
    }
    if fill is not None:
        if isinstance(fill, (int, float)) and not isinstance(fill, bool):
            fill = float(fill)
            if not math.isfinite(fill):
                raise ValueError("history fill must be finite")
        else:
            fill = _finite_values(fill, "history fill")
            if not fill:
                raise ValueError("history fill must not be empty")
        data["initialFillValue"] = fill
    if stride > 1:
        offsets = [-1 - stride * slot for slot in range(size - 1, -1, -1)]
        if order == "newest-first":
            offsets.reverse()
        data["sourceBufferSize"] = stride * (size - 1) + 2
        data["sampleOffsets"] = offsets
    return Signal("proc-history", f"{signal._label}-history", data, (signal,))


def _size_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a (width, height) pair")
    return (
        _positive_integer(value[0], f"{label} width"),
        _positive_integer(value[1], f"{label} height"),
    )


def resize(source, *, input_size, output_size, channels=None, layout="hwc", flip_x=False, flip_y=False):
    """Resample an image signal, with `input_size` and `output_size` given as `(width, height)`.

    `channels` is inferred from the input width when omitted.
    """
    signal = _require_signal(source, "resize source")
    if layout not in _RESIZE_LAYOUTS:
        raise ValueError(f"resize layout must be one of {_RESIZE_LAYOUTS}")
    if not isinstance(flip_x, bool):
        raise TypeError("resize flip_x must be a boolean")
    if not isinstance(flip_y, bool):
        raise TypeError("resize flip_y must be a boolean")
    input_width, input_height = _size_pair(input_size, "resize input_size")
    output_width, output_height = _size_pair(output_size, "resize output_size")
    data = {
        "inputWidth": input_width,
        "inputHeight": input_height,
        "outputWidth": output_width,
        "outputHeight": output_height,
        "layout": layout,
        "flipX": flip_x,
        "flipY": flip_y,
    }
    if channels is not None:
        data["channels"] = _positive_integer(channels, "resize channels")
    return Signal("proc-resize", f"{signal._label}-resize", data, (signal,))
