"""ONNX inference channel for the local plan executor.

Python mirror of the TypeScript ModelProcessor main path
(src/domain/runtime/engine/processors/ModelProcessor.ts): one onnxruntime
session per model resource (``OnnxSessionPool``, keyed by resource id — the
same "model nodes reuse the project's loaded session" contract as the browser
registry), float32 buffers, constant inputs fed every step, recurrent state
rolled through matched input/output name pairs, policy banks/pipelines,
activeModes gating, control decimation and reset. Deliberate differences,
documented here:

- synchronous strict compute: inference errors propagate (the executor wraps
  them); the browser's fire-and-forget compute() that disables a node after a
  failed inference is UI resilience, not engine semantics;
- feeds are numpy views over the plan's own buffers — the browser's ORT-Web
  feed caches, output-fetch caches, WebGPU IO binding and perf-phase telemetry
  have no native-ORT equivalent;
- outputDim resolution consolidates the two TS paths: a configured outputDim
  wins (registration semantics), a fully static output shape is cross-checked
  against it, and dynamic shapes fall back to the product of the static dims
  ignoring symbolic ones (registry ``tensorDim`` semantics).
"""

import math
import os
import re

import numpy as np

from ._model_session import _io_metadata, _normalized_provider, shared_session_pool
from ._model_spec import (
    _coerce_number,
    _configured_io_name,
    _configured_output_dim,
    _is_integer_number,
    _move_first,
    _number_list,
    _numeric_dim_product,
    _parse_constant_specs,
    _static_element_count,
    input_tensor_dims_from_metadata,
    numeric_tensor_data,
    tensor_dims_from_metadata,
    tensor_element_count,
)

_FLOAT32_DTYPE = np.dtype(np.float32)


class _SessionRuntime:
    """Selected I/O view over one session (the TS ModelRuntime mirror)."""

    __slots__ = (
        "session",
        "input_name",
        "input_names",
        "input_metadata",
        "output_name",
        "output_names",
        "output_metadata",
        "output_dim",
    )

    def __init__(self, session, input_names, input_metadata, output_names, output_metadata,
                 output_dim):
        self.session = session
        self.input_names = input_names
        self.input_metadata = input_metadata
        self.output_names = output_names
        self.output_metadata = output_metadata
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.output_dim = output_dim


class RecurrentStateTensor:
    __slots__ = ("input_name", "output_name", "dims", "buffer")

    def __init__(self, input_name, output_name, dims, buffer):
        self.input_name = input_name
        self.output_name = output_name
        self.dims = dims
        self.buffer = buffer


class ConstantInputTensor:
    __slots__ = ("input_name", "dims", "buffer")

    def __init__(self, input_name, dims, buffer):
        self.input_name = input_name
        self.dims = dims
        self.buffer = buffer


class _RuntimeState:
    """Mutable tensors owned by one model call inside a policy pipeline."""

    __slots__ = ("runtime", "recurrent_states", "constant_inputs", "output_buffer")

    def __init__(self, runtime, recurrent_states, constant_inputs):
        self.runtime = runtime
        self.recurrent_states = recurrent_states
        self.constant_inputs = constant_inputs
        self.output_buffer = np.zeros(runtime.output_dim, dtype=_FLOAT32_DTYPE)


class _PolicyRuntime:
    """A policy normalized to one or more stages (single models have one)."""

    __slots__ = ("key", "mode", "input_slot", "stages")

    def __init__(self, key, mode, input_slot, stages):
        self.key = key
        self.mode = mode
        self.input_slot = input_slot
        self.stages = stages

    @property
    def output_dim(self):
        return self.stages[-1].runtime.output_dim


class ModelProcessor:
    """``model-onnx`` node processor: one ONNX inference per active control step.

    Contract matches the executor's duck-typed processors (``output_buffers``
    of float32 ndarrays, ``init``/``compute``/``reset``); sessions come from
    the shared :class:`OnnxSessionPool` keyed by the node's resource id.
    """

    def __init__(self, session_pool=None):
        self.output_buffers = [np.zeros(0, dtype=_FLOAT32_DTYPE)]
        self._pool = session_pool if session_pool is not None else shared_session_pool()
        self._node_id = ""
        self._resource_id = ""
        self._model_path = ""
        self._requested_provider = "auto"
        self._control_decimation = 1
        self._step_count = 0
        self._use_mode_gate = False
        self._active_modes = set()
        self._constants_config = None
        self._extra_inputs_config = None
        self._configured_input_name = None
        self._configured_output_name = None
        self._configured_output_dim = None
        self._runtime = None
        self._constant_inputs = []
        self._recurrent_states = []
        self._use_policy_bank = False
        self._policy_runtimes = []
        self._last_policy_mode = None
        self._pending_policy_reset_keys = set()

    def init(self, config, input_dims):
        node_id = config.get("nodeId")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("[ModelProcessor] nodeId is required")
        self._node_id = node_id

        resource_id = config.get("resourceId")
        if resource_id is None:
            resource_id = node_id
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError(f"[ModelProcessor] resourceId is required for node {node_id}")
        self._resource_id = resource_id.strip()
        self._configured_input_name = None
        self._configured_output_name = None
        self._configured_output_dim = None
        self._runtime = None
        self._constant_inputs = []
        self._recurrent_states = []

        policies = config.get("policies")
        self._step_count = 0
        self._use_policy_bank = isinstance(policies, (list, tuple)) and bool(policies)
        self._policy_runtimes = []
        self._last_policy_mode = None
        self._pending_policy_reset_keys.clear()
        self._control_decimation = self._parse_control_decimation(config)
        self._constants_config = config.get("constants")
        self._extra_inputs_config = config.get("extraInputs")
        if self._use_policy_bank:
            self._init_policy_bank(config, input_dims, policies)
            return

        if not input_dims or not input_dims[0]:
            raise ValueError(f"[ModelProcessor] inputDims[0] is required for node {node_id}")
        mode_dim = input_dims[1] if len(input_dims) > 1 else 0
        if mode_dim and mode_dim != 1:
            raise ValueError(
                f"[ModelProcessor] mode gate input must have dimension 1 for node {node_id}"
            )
        self._use_mode_gate = bool(mode_dim)
        self._active_modes = self._parse_active_modes(config)

        self._configured_input_name = _configured_io_name(config.get("inputName"), "inputName")
        self._configured_output_name = _configured_io_name(config.get("outputName"), "outputName")
        configured_output_dim = config.get("outputDim")
        if configured_output_dim is None:
            configured_output_dim = self._resource_config(
                self._resource_id, "outputDim"
            )
        self._configured_output_dim = _configured_output_dim(configured_output_dim)
        self._requested_provider = self._provider_for_resource(
            self._resource_id, config.get("executionProvider")
        )
        self._model_path = self._resolve_model_path(config, self._resource_id)

        session = self._pool.get(
            self._resource_id, self._model_path, self._requested_provider
        )
        self._configure_runtime(self._build_runtime(session, self._resource_id))
        self.output_buffers[0] = np.zeros(self._runtime.output_dim, dtype=_FLOAT32_DTYPE)

    def compute(self, inputs):
        if self._use_policy_bank:
            self._compute_policy_bank(inputs)
            return
        input_values = inputs[0] if inputs else None
        if input_values is None:
            return
        input_array = self._prepare_input(input_values)
        if input_array.size == 0:
            return
        if not self._is_active(inputs):
            return
        if not self._should_run_inference():
            return

        runtime = self._current_runtime()
        self.output_buffers[0] = self._run_runtime_inference(
            runtime,
            input_array,
            self._recurrent_states,
            self._constant_inputs,
            self.output_buffers[0],
        )

    def reset(self):
        self._step_count = 0
        self.output_buffers[0][:] = 0.0
        for state in self._recurrent_states:
            state.buffer[:] = 0.0
        for policy in self._policy_runtimes:
            self._reset_policy_runtime(policy)
        self._last_policy_mode = None
        self._pending_policy_reset_keys.clear()

    def dispose(self):
        """Release processor state; shared sessions remain keyed by resource id."""
        self.reset()
        self._runtime = None
        self._constant_inputs = []
        self._recurrent_states = []
        self._policy_runtimes = []
        self._pending_policy_reset_keys.clear()

    def _prepare_input(self, values):
        if isinstance(values, np.ndarray):
            array = values if values.dtype == _FLOAT32_DTYPE else values.astype(_FLOAT32_DTYPE)
            return np.ravel(array)
        return np.asarray(values, dtype=_FLOAT32_DTYPE).ravel()

    def _is_active(self, inputs):
        if not self._use_mode_gate:
            return True
        mode_input = inputs[1] if len(inputs) > 1 else None
        if mode_input is None or np.size(mode_input) != 1:
            raise ValueError(
                f"[ModelProcessor] mode gate input is missing for node {self._node_id}"
            )
        if not self._active_modes:
            return True
        mode = float(np.asarray(mode_input).ravel()[0])
        if math.isnan(mode):
            return False
        return math.trunc(mode) in self._active_modes

    def _should_run_inference(self):
        should_run = self._step_count % self._control_decimation == 0
        self._step_count += 1
        return should_run

    def _parse_active_modes(self, config):
        raw = config.get("activeModes")
        if raw is None or raw == "":
            raw = config.get("activeMode")
        if raw is None or raw == "":
            return set()
        if isinstance(raw, str):
            values = [_coerce_number(part) for part in re.split(r"[\s,]+", raw) if part]
        elif isinstance(raw, (list, tuple, np.ndarray)):
            values = [_coerce_number(item) for item in raw]
        else:
            values = [_coerce_number(raw)]
        if not values or any(not _is_integer_number(value) for value in values):
            raise ValueError(
                f"[ModelProcessor] activeModes must contain integer mode codes for "
                f"node {self._node_id}"
            )
        return {int(value) for value in values}

    def _parse_control_decimation(self, config):
        raw = config.get("controlDecimation")
        if raw is None:
            raw = config.get("control_decimation")
        value = 1.0 if raw is None else _coerce_number(raw)
        if not _is_integer_number(value) or value < 1:
            raise ValueError(
                f"[ModelProcessor] controlDecimation must be a positive integer for "
                f"node {self._node_id}"
            )
        return int(value)

    def _init_policy_bank(self, config, input_dims, policies):
        if not input_dims or input_dims[0] != 1:
            raise ValueError(
                f"[ModelProcessor] policy bank mode input must have dimension 1 "
                f"for node {self._node_id}"
            )

        runtimes = []
        keys = set()
        for policy_index, raw_policy in enumerate(policies):
            if not isinstance(raw_policy, dict):
                raise ValueError(
                    f"[ModelProcessor] policies[{policy_index}] must be an object"
                )
            raw_key = raw_policy.get("key")
            key = str(raw_key if raw_key is not None else f"policy-{policy_index}")
            if key in keys:
                raise ValueError(
                    f'[ModelProcessor] policy key "{key}" is duplicated for node '
                    f"{self._node_id}"
                )
            keys.add(key)

            mode = _coerce_number(raw_policy.get("mode"))
            if not _is_integer_number(mode):
                raise ValueError(
                    f"[ModelProcessor] policy {policy_index} has invalid mode for node "
                    f"{self._node_id}"
                )
            raw_slot = raw_policy.get("inputSlot")
            input_slot = policy_index + 1 if raw_slot is None else _coerce_number(raw_slot)
            if not _is_integer_number(input_slot) or input_slot < 1:
                raise ValueError(
                    f"[ModelProcessor] policy {policy_index} has invalid inputSlot "
                    f"for node {self._node_id}"
                )
            input_slot = int(input_slot)
            if input_slot >= len(input_dims) or not input_dims[input_slot]:
                raise ValueError(
                    f'[ModelProcessor] policy "{key}" input slot {input_slot} is '
                    f"not connected for node {self._node_id}"
                )

            raw_pipeline = raw_policy.get("pipeline")
            if raw_pipeline is not None:
                if not isinstance(raw_pipeline, (list, tuple)) or not raw_pipeline:
                    raise ValueError(
                        f'[ModelProcessor] pipeline policy "{key}" has no stages'
                    )
                stages = [
                    self._create_policy_stage(
                        stage,
                        f'policy "{key}" stage {stage_index}',
                        raw_policy,
                        config,
                    )
                    for stage_index, stage in enumerate(raw_pipeline)
                ]
            else:
                stages = [
                    self._create_policy_stage(
                        raw_policy,
                        f'policy "{key}"',
                        raw_policy,
                        config,
                    )
                ]
            runtimes.append(_PolicyRuntime(key, int(mode), input_slot, stages))

        output_dim = runtimes[0].output_dim
        if any(runtime.output_dim != output_dim for runtime in runtimes):
            raise ValueError(
                f"[ModelProcessor] all policy outputs must have dimension {output_dim} "
                f"for node {self._node_id}"
            )
        self._use_mode_gate = False
        self._active_modes = set()
        self._policy_runtimes = runtimes
        self.output_buffers[0] = np.zeros(output_dim, dtype=_FLOAT32_DTYPE)

    def _create_policy_stage(self, stage, label, policy, node_config):
        if not isinstance(stage, dict):
            raise ValueError(f"[ModelProcessor] {label} must be an object")
        resource_id = stage.get("resourceId")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError(
                f"[ModelProcessor] {label} has no ONNX model resource for node "
                f"{self._node_id}"
            )
        resource_id = resource_id.strip()
        explicit_provider = node_config.get("executionProvider")
        requested_provider = self._provider_for_resource(
            resource_id, explicit_provider
        )
        model_path = self._resolve_model_path(
            {"modelPath": stage.get("modelPath")}
            if stage.get("modelPath") is not None
            else {},
            resource_id,
        )
        session = self._pool.get(resource_id, model_path, requested_provider)
        stage_output_dim = stage.get("outputDim")
        if stage_output_dim is None:
            stage_output_dim = self._resource_config(resource_id, "outputDim")
        runtime = self._build_runtime(
            session,
            resource_id,
            input_name=stage.get("inputName"),
            output_name=stage.get("outputName"),
            output_dim=stage_output_dim,
            cross_check_output_dim=False,
        )
        constants_config = self._inherited_config(
            stage, policy, node_config, field="constants"
        )
        extra_inputs_config = self._inherited_config(
            stage, policy, node_config, field="extraInputs"
        )
        constant_inputs = self._create_constant_inputs(
            runtime, constants_config, extra_inputs_config
        )
        return _RuntimeState(
            runtime,
            self._create_state_tensors(runtime, constant_inputs),
            constant_inputs,
        )

    @staticmethod
    def _inherited_config(*configs, field):
        for config in configs:
            if config.get(field) is not None:
                return config[field]
        return None

    def _resolve_model_path(self, config, resource_id):
        explicit = config.get("modelPath")
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip():
                raise ValueError(
                    "[ModelProcessor] modelPath must be a non-empty string when provided"
                )
            return explicit
        # Unnamed resources use their source path as id; named ones resolve
        # through the host declaration table. Policy pipeline stages do not
        # carry a top-level modelPath, so also resolve the same project/shared
        # asset candidates as the local runner.
        try:
            import botlab_host

            source = botlab_host.resourceSource(resource_id)
            directory = botlab_host.project_state().directory
        except Exception:
            source = None
            directory = None
        if isinstance(source, str) and source:
            if not os.path.isabs(source) and directory:
                candidates = (
                    os.path.join(directory, source),
                    os.path.join(os.path.dirname(os.path.dirname(directory)), source),
                )
                for candidate in candidates:
                    if os.path.isfile(candidate):
                        return candidate
            return source
        return resource_id

    def _provider_for_resource(self, resource_id, explicit=None):
        if explicit is not None and explicit != "":
            return _normalized_provider(explicit)
        try:
            import botlab_host

            provider = botlab_host.project_state().resource(resource_id, "model").get(
                "provider"
            )
        except Exception:
            provider = None
        return _normalized_provider(provider)

    @staticmethod
    def _resource_config(resource_id, field):
        try:
            import botlab_host

            return botlab_host.project_state().resource(resource_id, "model")[
                "config"
            ].get(field)
        except Exception:
            return None

    def _build_runtime(
        self,
        session,
        resource_id,
        *,
        input_name=None,
        output_name=None,
        output_dim=None,
        cross_check_output_dim=True,
    ):
        configured_input_name = (
            self._configured_input_name
            if input_name is None and resource_id == self._resource_id
            else _configured_io_name(input_name, "inputName")
        )
        configured_output_name = (
            self._configured_output_name
            if output_name is None and resource_id == self._resource_id
            else _configured_io_name(output_name, "outputName")
        )
        configured_output_dim = (
            self._configured_output_dim
            if output_dim is None and resource_id == self._resource_id
            else _configured_output_dim(output_dim)
        )
        input_metadata = _io_metadata(session.get_inputs(), "input", resource_id)
        output_metadata = _io_metadata(session.get_outputs(), "output", resource_id)
        input_names = [entry["name"] for entry in input_metadata]
        output_names = [entry["name"] for entry in output_metadata]

        if configured_input_name is not None or configured_output_name is not None:
            if configured_input_name is not None:
                input_index = _name_index(
                    input_names, configured_input_name, "inputName", self._node_id
                )
                entry = input_metadata[input_index]
                if not entry["is_tensor"]:
                    raise ValueError(
                        f'[ModelProcessor] ONNX input "{configured_input_name}" '
                        f"metadata is required for node {self._node_id}"
                    )
                if entry["type"] and entry["type"] != "float32":
                    raise ValueError(
                        f'[ModelProcessor] ONNX input "{configured_input_name}" '
                        f"must be float32 for node {self._node_id}"
                    )
                input_metadata = _move_first(input_metadata, input_index)
                input_names = _move_first(input_names, input_index)
            if configured_output_name is not None:
                output_index = _name_index(
                    output_names, configured_output_name, "outputName", self._node_id
                )
                entry = output_metadata[output_index]
                if not entry["is_tensor"]:
                    raise ValueError(
                        f'[ModelProcessor] ONNX output "{configured_output_name}" '
                        f"metadata is required for node {self._node_id}"
                    )
                if entry["type"] and entry["type"] != "float32":
                    raise ValueError(
                        f'[ModelProcessor] ONNX output "{configured_output_name}" '
                        f"must be float32 for node {self._node_id}"
                    )
                output_metadata = _move_first(output_metadata, output_index)
                output_names = _move_first(output_names, output_index)

        resolved_output_dim = self._resolve_output_dim(
            output_metadata[0],
            output_names[0],
            configured_output_dim,
            cross_check_output_dim,
        )
        return _SessionRuntime(
            session,
            input_names,
            input_metadata,
            output_names,
            output_metadata,
            resolved_output_dim,
        )

    def _resolve_output_dim(
        self, output_metadata, output_name, configured, cross_check=True
    ):
        static_dim = _static_element_count(output_metadata)
        if (
            cross_check
            and static_dim is not None
            and configured is not None
            and static_dim != configured
        ):
            raise ValueError(
                f"[ModelProcessor] outputDim {configured} does not match ONNX output "
                f'"{output_name}" dimension {static_dim} for node {self._node_id}'
            )
        if configured is not None:
            return configured
        if static_dim is not None:
            return static_dim
        inferred = _numeric_dim_product(output_metadata)
        if inferred > 1:
            return inferred
        raise ValueError(
            f'[ModelProcessor] Cannot infer outputDim for ONNX output "{output_name}" '
            f"on node {self._node_id}; set outputDim in the node config"
        )

    def _configure_runtime(self, runtime):
        if self._runtime is runtime:
            return
        self._constant_inputs = self._create_constant_inputs(
            runtime, self._constants_config, self._extra_inputs_config
        )
        self._recurrent_states = self._create_state_tensors(runtime, self._constant_inputs)
        self._runtime = runtime
        if runtime.output_dim > 0 and self.output_buffers[0].size != runtime.output_dim:
            self.output_buffers[0] = np.zeros(runtime.output_dim, dtype=_FLOAT32_DTYPE)

    def _current_runtime(self):
        session = self._pool.get(
            self._resource_id, self._model_path, self._requested_provider
        )
        if self._runtime is None or self._runtime.session is not session:
            self._configure_runtime(self._build_runtime(session, self._resource_id))
        return self._runtime

    def _create_constant_inputs(
        self, runtime, constants_config=None, extra_inputs_config=None
    ):
        specs = []
        for field, raw in (
            ("constants", constants_config),
            ("extraInputs", extra_inputs_config),
        ):
            for index, spec in enumerate(_parse_constant_specs(raw, field)):
                specs.append((f"{field}[{index}]", spec))

        names = set()
        constants = []
        for label, spec in specs:
            raw_name = spec.get("name")
            if raw_name is None:
                raw_name = spec.get("inputName")
            input_name = str(raw_name).strip() if raw_name is not None else ""
            if not input_name:
                raise ValueError(f"[ModelProcessor] {label}.name is required")
            if input_name in names:
                raise ValueError(
                    f'[ModelProcessor] constant input "{input_name}" is declared more than once'
                )
            names.add(input_name)
            input_index = (
                runtime.input_names.index(input_name)
                if input_name in runtime.input_names
                else -1
            )
            if input_index < 0:
                raise ValueError(
                    f'[ModelProcessor] {label}.name "{input_name}" is not an ONNX input'
                )
            if input_index == 0:
                raise ValueError(
                    f"[ModelProcessor] {label}.name cannot target the primary observation input"
                )
            dims = self._dims_from_spec(runtime.input_metadata[input_index], spec, label)
            buffer = self._buffer_from_spec(spec, tensor_element_count(dims), label)
            constants.append(ConstantInputTensor(input_name, dims, buffer))
        return constants

    def _dims_from_spec(self, metadata, spec, label):
        raw = spec.get("dims")
        if raw is None:
            raw = spec.get("shape")
        if raw is None or raw == "":
            return tensor_dims_from_metadata(metadata, f"{label} metadata")
        dims = [math.floor(value) for value in _number_list(raw, f"{label}.dims")]
        if not dims or any(value <= 0 for value in dims):
            raise ValueError(f"[ModelProcessor] {label}.dims must contain positive integers")
        return dims

    def _buffer_from_spec(self, spec, element_count, label):
        raw = spec.get("values")
        if raw is None:
            raw = spec.get("value")
        buffer = np.zeros(element_count, dtype=_FLOAT32_DTYPE)
        if raw is None or raw == "":
            return buffer
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if not math.isfinite(raw):
                raise ValueError(f"[ModelProcessor] {label}.value must be finite")
            buffer[:] = raw
            return buffer
        values = _number_list(raw, f"{label}.values")
        if len(values) != element_count:
            raise ValueError(
                f"[ModelProcessor] {label}.values length {len(values)} does not match "
                f"tensor size {element_count}"
            )
        buffer[:] = values
        return buffer

    def _create_state_tensors(self, runtime, constant_inputs):
        if len(runtime.input_names) == 1:
            return []
        constant_names = {constant.input_name for constant in constant_inputs}
        recurrent_entries = [
            (name, index)
            for index, name in enumerate(runtime.input_names)
            if index >= 1 and name not in constant_names
        ]
        if not recurrent_entries:
            return []

        use_aligned_outputs = len(runtime.output_names) == len(runtime.input_names)
        if not use_aligned_outputs and len(runtime.output_names) < len(recurrent_entries) + 1:
            raise ValueError(
                f"[ModelProcessor] Node {self._node_id} has "
                f"{len(recurrent_entries)} recurrent input(s) but only "
                f"{len(runtime.output_names) - 1} recurrent output(s)"
            )

        states = []
        for state_index, (input_name, input_index) in enumerate(recurrent_entries):
            output_name = (
                runtime.output_names[input_index]
                if use_aligned_outputs
                else runtime.output_names[state_index + 1]
            )
            if not output_name:
                raise ValueError(
                    f'[ModelProcessor] Node {self._node_id} cannot map recurrent output '
                    f'for "{input_name}"'
                )
            dims = tensor_dims_from_metadata(
                runtime.input_metadata[input_index],
                f'recurrent state input "{input_name}"',
            )
            states.append(
                RecurrentStateTensor(
                    input_name,
                    output_name,
                    dims,
                    np.zeros(tensor_element_count(dims), dtype=_FLOAT32_DTYPE),
                )
            )
        return states

    def _compute_policy_bank(self, inputs):
        mode_input = inputs[0] if inputs else None
        if mode_input is None or np.size(mode_input) != 1:
            raise ValueError(
                f"[ModelProcessor] policy bank mode input is missing for node "
                f"{self._node_id}"
            )
        raw_mode = float(np.asarray(mode_input).ravel()[0])
        mode = math.trunc(raw_mode) if math.isfinite(raw_mode) else raw_mode
        mode_changed = mode != self._last_policy_mode
        if mode_changed:
            self._last_policy_mode = mode
            self.output_buffers[0][:] = 0.0

        policy = next(
            (candidate for candidate in self._policy_runtimes if candidate.mode == mode),
            None,
        )
        if policy is None:
            self.output_buffers[0][:] = 0.0
            return
        if mode_changed:
            self._pending_policy_reset_keys.add(policy.key)
        if not self._should_run_inference():
            return
        if policy.key in self._pending_policy_reset_keys:
            self._pending_policy_reset_keys.remove(policy.key)
            self._reset_policy_runtime(policy)

        input_values = inputs[policy.input_slot] if len(inputs) > policy.input_slot else None
        if input_values is None:
            raise ValueError(
                f'[ModelProcessor] policy "{policy.key}" input is missing for node '
                f"{self._node_id}"
            )
        stage_input = self._prepare_input(input_values)
        if stage_input.size == 0:
            raise ValueError(
                f'[ModelProcessor] policy "{policy.key}" input is missing for node '
                f"{self._node_id}"
            )
        for stage in policy.stages:
            stage.output_buffer = self._run_runtime_inference(
                stage.runtime,
                stage_input,
                stage.recurrent_states,
                stage.constant_inputs,
                stage.output_buffer,
            )
            stage_input = stage.output_buffer
        if self.output_buffers[0].size != stage_input.size:
            self.output_buffers[0] = np.zeros(stage_input.size, dtype=_FLOAT32_DTYPE)
        self.output_buffers[0][:] = stage_input

    def _reset_policy_runtime(self, policy):
        for stage in policy.stages:
            stage.output_buffer[:] = 0.0
            for state in stage.recurrent_states:
                state.buffer[:] = 0.0

    def _run_runtime_inference(
        self, runtime, input_array, recurrent_states, constant_inputs, output_buffer
    ):
        input_label = f'ONNX input "{runtime.input_name}" for node {self._node_id}'
        input_dims = input_tensor_dims_from_metadata(
            runtime.input_metadata[0], input_label, input_array.size
        )
        feeds = {runtime.input_name: input_array.reshape(input_dims)}
        for state in recurrent_states:
            feeds[state.input_name] = state.buffer.reshape(state.dims)
        for constant in constant_inputs:
            feeds[constant.input_name] = constant.buffer.reshape(constant.dims)

        fetch_names = _ordered_unique(
            [runtime.output_name] + [state.output_name for state in recurrent_states]
        )
        results = dict(zip(fetch_names, runtime.session.run(fetch_names, feeds)))
        return self._copy_outputs(
            runtime, results, recurrent_states, output_buffer
        )

    def _copy_outputs(self, runtime, results, recurrent_states, output_buffer):
        output_label = f'ONNX output "{runtime.output_name}" for node {self._node_id}'
        output_data = numeric_tensor_data(results.get(runtime.output_name), output_label)
        if output_data.size != output_buffer.size:
            output_buffer = np.zeros(output_data.size, dtype=_FLOAT32_DTYPE)
        output_buffer[:] = output_data.ravel()

        for state in recurrent_states:
            state_label = f'ONNX output "{state.output_name}" for node {self._node_id}'
            state_data = numeric_tensor_data(results.get(state.output_name), state_label)
            if state_data.size != state.buffer.size:
                raise ValueError(
                    f'[ModelProcessor] Recurrent state output "{state.output_name}" length '
                    f"{state_data.size} does not match input \"{state.input_name}\" length "
                    f"{state.buffer.size}"
                )
            state.buffer[:] = state_data.ravel()
        return output_buffer


def _ordered_unique(names):
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _name_index(names, target, field, node_id):
    try:
        return names.index(target)
    except ValueError:
        kind = "input" if field == "inputName" else "output"
        raise ValueError(
            f'[ModelProcessor] {field} "{target}" is not an ONNX {kind} for node {node_id}'
        ) from None


REGISTRY = {"model-onnx": ModelProcessor}
