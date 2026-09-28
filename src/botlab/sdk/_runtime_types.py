"""Runtime data structures for Python controllers: history, frames, contacts."""
from ._vectors import _positive_integer, require_vector


class HistoryBuffer:
    def __init__(self, capacity, frame_size):
        self.capacity = _positive_integer(capacity, "HistoryBuffer capacity")
        self.frame_size = _positive_integer(frame_size, "HistoryBuffer frame_size")
        self._frames = []

    def clear(self):
        self._frames = []

    def append(self, values):
        frame = require_vector(values, self.frame_size, "HistoryBuffer frame")
        if not self._frames:
            self._frames = [list(frame) for _ in range(self.capacity)]
        else:
            self._frames.append(frame)
            del self._frames[:-self.capacity]

    def _require_ready(self):
        if not self._frames:
            raise RuntimeError("HistoryBuffer has no frames")

    def stack_segments(self, segments):
        self._require_ready()
        output = []
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValueError("HistoryBuffer segments must be (start, size) pairs")
            start, size = segment
            if (
                isinstance(start, bool)
                or isinstance(size, bool)
                or not isinstance(start, int)
                or not isinstance(size, int)
                or start < 0
                or size <= 0
                or start + size > self.frame_size
            ):
                raise ValueError(
                    f"HistoryBuffer segment {(start, size)} is outside frame size {self.frame_size}"
                )
            for frame in self._frames:
                output.extend(frame[start:start + size])
        return output

    def sample_ages(self, ages):
        self._require_ready()
        output = []
        newest = len(self._frames) - 1
        for age in ages:
            if (
                isinstance(age, bool)
                or not isinstance(age, int)
                or age < 0
                or age >= self.capacity
            ):
                raise ValueError(
                    f"HistoryBuffer age {age} must be between 0 and {self.capacity - 1}"
                )
            output.extend(self._frames[newest - age])
        return output


class Frame:
    def __init__(self, dt, time, index):
        self.dt = float(dt)
        self.time = float(time)
        self.index = int(index)


def _entity_targets(value):
    values = [value] if isinstance(value, (str, int)) else list(value)
    if not values:
        raise ValueError("entity selection must not be empty")
    for target in values:
        if isinstance(target, bool) or not isinstance(target, (str, int)):
            raise TypeError("entity targets must be names or non-negative indices")
        if isinstance(target, int) and target < 0 or isinstance(target, str) and not target.strip():
            raise ValueError("entity targets must be names or non-negative indices")
    return values


class _ContactSignals:
    """Only consumed fields enter the compiled plan and worker snapshot request."""

    def __init__(self, robot, targets, kind):
        self.force = robot._entity_signal(f"{kind}.contactforce", targets, 3)
        self.touching = robot._entity_signal(f"{kind}.contact", targets, 1)
        for name in ("air_time", "contact_time", "last_air_time", "last_contact_time", "first_contact"):
            setattr(self, name, robot._entity_signal(f"{kind}.{name}", targets, 1))
