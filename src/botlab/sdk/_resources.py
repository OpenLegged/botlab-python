"""Model, Motion, Sensor and Scene resources plus their shared node builders."""
import botlab_host

from ._declaration import (
    _ResourceValues,
    _declare_resource,
    _host_snapshot,
    _host_value,
    _non_empty_string,
)
from ._runtime_types import _entity_targets
from ._signal import Signal, _as_signal, _numeric_signal
from ._vectors import _finite_values, _non_negative_integer, _positive_integer


def _model_constant_inputs(constants, input_name, label):
    if not isinstance(constants, dict) or not constants:
        raise ValueError(f"{label} constants must be a non-empty mapping")
    extra_inputs = []
    for name, values in constants.items():
        name = _non_empty_string(name, f"{label} constant name")
        if input_name is not None and name == input_name:
            raise ValueError(f'{label} constant "{name}" conflicts with input_name')
        vector = _finite_values(values, f'{label} constant "{name}"')
        if not vector:
            raise ValueError(f'{label} constant "{name}" must not be empty')
        extra_inputs.append({"name": name, "values": vector})
    return extra_inputs


def _model_call_data(raw, label):
    data = {}
    input_name = raw.get("input_name")
    if input_name is not None:
        data["inputName"] = _non_empty_string(input_name, f"{label} input_name")
    output_name = raw.get("output_name")
    if output_name is not None:
        data["outputName"] = _non_empty_string(output_name, f"{label} output_name")
    has_constants = raw.get("constants") is not None
    has_extra_inputs = raw.get("extra_inputs") is not None
    if has_constants and has_extra_inputs:
        raise ValueError(f"{label} accepts only one of constants or extra_inputs")
    if has_constants:
        data["extraInputs"] = _model_constant_inputs(
            raw["constants"],
            data.get("inputName"),
            label,
        )
    elif has_extra_inputs:
        data["extraInputs"] = raw["extra_inputs"]
    if "output_dim" in raw:
        data["outputDim"] = _positive_integer(raw["output_dim"], f"{label} output_dim")
    return data


def _motion_source_data(motion):
    if not isinstance(motion, Motion):
        raise TypeError("motion_reference motion must be a BotLab Motion")
    source = botlab_host.resourceSource(motion.id)
    source = _non_empty_string(str(source), "motion resource source")
    if source.startswith("registry://"):
        prefix = source[len("registry://"):]
        return {"motionSource": {"kind": "registry", "registryPrefix": prefix}}
    return {"motionPath": source}


class Model:
    def __init__(self, source, *, provider=None, config=None, name=None):
        self.id = _declare_resource(
            "model",
            source,
            provider=provider,
            config=config,
            name=name,
        )

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    async def predict(self, values):
        result = await botlab_host.predictModel(self.id, values)
        return _host_value(result)

    def __call__(
        self,
        *inputs,
        input_name=None,
        output_name=None,
        constants=None,
        control_decimation=1,
        call_id=None,
    ):
        """Run this ONNX model on the given signals, concatenated in order."""
        if not inputs:
            raise ValueError(f'model "{self.id}" requires at least one input signal')
        source = _as_signal(
            inputs[0] if len(inputs) == 1 else list(inputs),
            f'model "{self.id}" input',
        )
        # The resource owns the loaded ONNX session; each call is a separate
        # plan node so caches, decimation, statistics, and recurrent state stay isolated.
        data = {
            "resourceId": self.id,
            "controlDecimation": _positive_integer(
                control_decimation,
                "model control_decimation",
            ),
        }
        data.update(_model_call_data({
            "input_name": input_name,
            "output_name": output_name,
            "constants": constants,
        }, "model"))
        if data["controlDecimation"] == 1:
            del data["controlDecimation"]
        if call_id is not None:
            call_id = _non_empty_string(call_id, "model call_id")
        return Signal(
            "model-onnx",
            self.id,
            data,
            (source,),
            node_id=call_id,
        )


class Motion:
    def __init__(self, source, *, config=None, name=None):
        self.id = _declare_resource("motion", source, config=config, name=name)

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    @property
    def source(self):
        return str(botlab_host.resourceSource(self.id))

    def sample(self, time, *, channel, indices, interpolation="linear", loop=False, preview=False):
        """Sample absolute motion channels at a Signal time in seconds.

        Selections preserve explicit order; names require motion metadata. Linear
        quaternion sampling uses shortest-path SLERP. Output width follows selection.
        """
        if channel not in ("joint_position", "joint_velocity", "body_position", "body_quaternion",
                           "body_linear_velocity", "body_angular_velocity"):
            raise ValueError("unsupported motion sample channel")
        if interpolation not in ("linear", "nearest"):
            raise ValueError('motion interpolation must be "linear" or "nearest"')
        if not isinstance(loop, bool):
            raise TypeError("motion sample loop must be a boolean")
        if not isinstance(preview, bool):
            raise TypeError("motion sample preview must be a boolean")
        if not isinstance(indices, _ResourceValues):
            if isinstance(indices, (str, int)):
                raise TypeError("motion indices must be a sequence of names or indices")
            indices = _entity_targets(indices)
        data = _motion_source_data(self)
        data.update({"channel": channel, "indices": indices, "interpolation": interpolation,
                     "loop": loop, "preview": preview})
        return Signal("proc-motion-sample", f"motion-{channel}", data, (_numeric_signal(time),))

    async def load(self):
        """Load this motion's frames for Python control code.

        The signal path reads motion frames in the engine with `motion_reference(...)`;
        this is the same data for a Python controller that plans its own playback. Load
        it once at setup -- the arrays cover every frame, not just the current one.

        Returns `frame_count`, `body_count`, `dof_count`, `fps` and flat per-frame
        buffers: `body_position`, `body_quaternion`, `joint_position`, `joint_velocity`,
        `body_linear_velocity`, `body_angular_velocity`.
        """
        return _host_value(await botlab_host.loadMotion(self.id))


_CAMERA_CHANNEL_SLOTS = {
    "rgb": {"rgb": 0},
    "depth": {"depth": 0},
    "rgbd": {"rgb": 0, "depth": 1},
}


class _DepthFrame:
    __slots__ = ("data", "age_steps")

    def __init__(self, signal, age_slot):
        self.data = signal
        self.age_steps = signal._output(age_slot, "depth-age-steps")


class Sensor:
    def __init__(self, source, *, preset=None, config=None, name=None):
        self.id = _declare_resource(
            "sensor",
            source,
            preset=preset,
            config=config,
            name=name,
        )
        self._camera_nodes = {}

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        handle._camera_nodes = {}
        return handle

    def observe(self, *, channel=None):
        """Snapshot of one camera channel for a Python `Controller.step`.

        A camera with a single channel needs no argument; an `rgbd` camera carries two,
        so it has to be asked for one by name rather than have the SDK pick.
        """
        snapshot = _host_snapshot(botlab_host.observeSensor(self.id))
        if channel is None:
            available = sorted(snapshot)
            if len(available) != 1:
                raise ValueError(
                    f'sensor "{self.id}" publishes {available}; '
                    "observe(channel=...) must name one"
                )
            channel = available[0]
        if channel not in snapshot:
            raise ValueError(
                f'sensor "{self.id}" has no "{channel}" channel; '
                f"published channels are {sorted(snapshot)}"
            )
        return snapshot[channel]

    def _output_mode(self):
        """The camera's channel layout, as its preset calibrated it."""
        mode = _host_value(botlab_host.resourceConfig(self.id, "outputMode"))
        if mode is None:
            return "rgbd"
        if mode not in ("rgb", "depth", "rgbd"):
            raise ValueError(
                f'sensor "{self.id}" outputMode must be "rgb", "depth" or "rgbd", got "{mode}"'
            )
        return mode

    def _channel(self, channel, max_age):
        """One camera channel. `rgbd` emits RGB first, so the slot follows the mode."""
        mode = self._output_mode()
        slots = _CAMERA_CHANNEL_SLOTS[mode]
        if channel not in slots:
            raise ValueError(
                f'sensor "{self.id}" has outputMode "{mode}" '
                f"and produces no {channel} channel"
            )
        max_age = _non_negative_integer(max_age, f"{channel} max_age")
        # One camera node per age budget: an rgbd camera asked for both channels is one
        # render and one plan node with two outputs, not two nodes over the same frame.
        signal = self._camera_nodes.get(max_age)
        if signal is None:
            signal = Signal("obs-camera", "rgb" if mode == "rgb" else "depth", {
                "resourceId": self.id,
                "maxAgeSteps": max_age,
            })
            self._camera_nodes[max_age] = signal
        return signal._output(slots[channel], channel)

    def rgb(self, *, max_age=0):
        """Latest RGB frame, rejected once it is older than `max_age` steps."""
        return self._channel("rgb", max_age)

    def depth(self, *, max_age=0):
        """Latest depth frame, rejected once it is older than `max_age` steps."""
        return self._channel("depth", max_age)

    def depth_frame(self, *, max_age=0):
        """Depth data and its frame age, exposed as outputs of one camera node."""
        mode = self._output_mode()
        if mode != "depth":
            raise ValueError(
                f'sensor "{self.id}" needs outputMode "depth" for depth_frame(), got "{mode}"'
            )
        signal = Signal("obs-camera", "depth-frame", {
            "resourceId": self.id,
            "maxAgeSteps": _non_negative_integer(max_age, "depth_frame max_age"),
            "includeAgeSteps": True,
        })
        return _DepthFrame(signal, 1)


class Scene:
    def __init__(self, source, *, preset=None, config=None, name=None):
        self.id = _declare_resource(
            "scene",
            source,
            preset=preset,
            config=config,
            name=name,
        )

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    async def update_heightfield(self, values):
        """Replace a procedural heightfield's normalized elevations at runtime.

        ``values`` may be a flat row-major sequence or a rectangular list of rows.
        The scene's configured grid fixes its dimensions; values must be finite and
        normalized to the MuJoCo ``[0, 1]`` heightfield range.
        """
        await botlab_host.updateHeightfield(self.id, values)
