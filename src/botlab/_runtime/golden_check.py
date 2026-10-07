"""Replay processor-level golden traces exported by the TS engine.

Consumes ``tests/fixtures/processors.json`` (exported by BotLab
``processorGoldenExport.test.ts`` with UPDATE_GOLDEN=1): for every case it
builds the ported processor, replays the recorded step inputs and compares
outputs. The frame contract mirrors the exporter exactly — dt=0.02,
time=index*0.02, index=step — so frame-driven processors (proc-clock) see
the same values on both sides. Run directly or via
``python -m botlab._runtime.golden_check <path>``.
"""

import json
import sys

import numpy as np

from ._processors import build_registry


class _Frame:
    def __init__(self, index):
        self.dt = 0.02
        self.time = index * 0.02
        self.index = index


def default_golden_path():
    import os

    runtime_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(runtime_dir, "..", "..", "..", "tests", "fixtures", "processors.json")


def _inject_motion_assets(case, config):
    """Registry-referenced motions travel as base64 files inside the golden."""
    assets = case.get("motionAssets")
    if not assets:
        return
    import base64

    from ._motion import load_motion_folder

    files = {
        name: base64.b64decode(data) for name, data in assets.items()
    }
    config["motionData"] = load_motion_folder(files)


def replay(golden_path, atol=1e-9):
    """Replay every case; returns (failures, total) with per-case details."""
    with open(golden_path, encoding="utf-8") as handle:
        golden = json.load(handle)
    if golden.get("version") != 1:
        raise ValueError(f"unsupported golden version {golden.get('version')!r}")

    registry = build_registry()
    failures = []
    total = 0
    for case in golden["cases"]:
        total += 1
        type_name = case["type"]
        processor_class = registry.get(type_name)
        if processor_class is None:
            failures.append(
                {
                    "type": type_name,
                    "reason": f"no Python processor registered for {type_name!r}",
                }
            )
            continue

        try:
            processor = processor_class()
            config = dict(case["config"])
            config["nodeId"] = f"golden:{type_name}"
            config["nodeType"] = type_name
            _inject_motion_assets(case, config)
            processor.init(config, list(case["inputDims"]))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            failures.append({"type": type_name, "reason": f"init failed: {error}"})
            continue

        uses_frame = getattr(processor, "uses_frame", False)
        for step_index, step in enumerate(case["steps"]):
            inputs = [
                np.asarray(values, dtype=np.float32) for values in step["inputs"]
            ]
            try:
                if uses_frame:
                    processor.compute(inputs, _Frame(step_index))
                else:
                    processor.compute(inputs)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                failures.append(
                    {"type": type_name, "reason": f"step {step_index} failed: {error}"}
                )
                break

            actual = [np.asarray(buffer) for buffer in processor.output_buffers]
            expected = [np.asarray(values) for values in step["outputs"]]
            if len(actual) != len(expected):
                failures.append(
                    {
                        "type": type_name,
                        "reason": (
                            f"step {step_index}: {len(actual)} output slots, "
                            f"expected {len(expected)}"
                        ),
                    }
                )
                break
            for slot, (actual_slot, expected_slot) in enumerate(zip(actual, expected)):
                if not np.allclose(actual_slot, expected_slot, rtol=0, atol=atol):
                    worst = float(
                        np.max(np.abs(actual_slot.astype(np.float64) - expected_slot))
                    )
                    failures.append(
                        {
                            "type": type_name,
                            "reason": (
                                f"step {step_index} slot {slot}: max abs diff "
                                f"{worst:.3e} exceeds atol {atol:g}"
                            ),
                        }
                    )
                    break
    return failures, total


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    golden_path = args[0] if args else default_golden_path()
    failures, total = replay(golden_path)
    if failures:
        for failure in failures:
            print(f"[golden] FAIL {failure['type']}: {failure['reason']}", file=sys.stderr)
        print(f"[golden] {len(failures)}/{total} cases failed", file=sys.stderr)
        return 1
    print(f"[golden] {total} cases replayed within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
