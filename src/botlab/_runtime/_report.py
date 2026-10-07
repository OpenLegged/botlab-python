"""Machine-readable run reports for ``botlab run --report``.

The headless run otherwise prints only the completed control-step count,
which gives an AI agent no way to judge behavior (fell over? tracking on
target?). The recorder captures per-control-step base poses per robot,
reset events with their reasons, and ``sim.metrics`` updates, then writes
one JSON document at the end of the run (including interrupted runs).
Judgment stays with the reader — the platform sets no thresholds.
"""

import json
import os
import time


class RunRecorder:
    def __init__(self, path, project_name, control_hz, physics_hz, robot_ids):
        if not robot_ids:
            raise ValueError("[botlab.run] --report needs at least one declared robot")
        self._path = os.fspath(path)
        self._project = project_name
        self._control_hz = control_hz
        self._physics_hz = physics_hz
        self._robot_ids = list(robot_ids)
        self.base_position = {robot_id: [] for robot_id in self._robot_ids}
        self.base_quaternion = {robot_id: [] for robot_id in self._robot_ids}
        self.resets = []
        self.metrics = []
        self.steps = 0
        self._started_at = time.monotonic()

    def record_step(self, physics):
        for robot_id in self._robot_ids:
            position, quaternion = physics.base_pose(robot_id)
            self.base_position[robot_id].append(position)
            self.base_quaternion[robot_id].append(quaternion)
        self.steps += 1

    def record_reset(self, time_s, reason):
        self.resets.append({"time": time_s, "reason": str(reason)})

    def record_metrics(self, time_s, values):
        self.metrics.append({"time": time_s, "values": values})

    def write(self):
        report = {
            "project": self._project,
            "control_hz": self._control_hz,
            "physics_hz": self._physics_hz,
            "steps": self.steps,
            "wall_time_s": round(time.monotonic() - self._started_at, 6),
            "robots": self._robot_ids,
            "base_position": self.base_position,
            "base_quaternion": self.base_quaternion,
            "resets": self.resets,
            "metrics": self.metrics,
        }
        try:
            with open(self._path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)
        except OSError as error:
            raise RuntimeError(
                f'[botlab.run] cannot write the run report to "{self._path}": {error}'
            ) from error
        return self._path
