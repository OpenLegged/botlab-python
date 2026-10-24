"""Live PD gain signals through the independent native runner and MuJoCo."""

from unittest import mock

import mujoco
import numpy as np
import pytest

from botlab import open_headless
from botlab._runtime._processors._output import PDProcessor


def test_dynamic_gains_update_both_torque_and_owned_substep_views():
    processor = PDProcessor()
    processor.init(dict(pdType="position", kps=[10, 20], kds=[1, 2],
                        effortLimit=[100, 100], kpsInputSlot=3, kdsInputSlot=4),
                   [2, 2, 2, 2, 2])
    values = [np.array(value, dtype=np.float32) for value in
              ([1, 2], [0, 0], [1, 1], [30, 40], [3, 4])]
    initial = processor.get_substep_pd()
    processor.compute(values)
    np.testing.assert_array_equal(processor.output_buffers[0], [27, 76])
    current = processor.get_substep_pd()
    assert current["kps"] is initial["kps"]
    assert current["kds"] is initial["kds"]
    assert current["kps"] is not values[3]
    values[3][:] = 0
    values[4][:] = 1
    np.testing.assert_array_equal(current["kps"], [30, 40])
    processor.compute(values)
    np.testing.assert_array_equal(processor.output_buffers[0], [-1, -1])
    np.testing.assert_array_equal(current["kps"], [0, 0])


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_gain_does_not_partially_update_the_controller(bad):
    processor = PDProcessor()
    processor.init(dict(kps=[10], kds=[1], effortLimit=[100],
                        kpsInputSlot=3, kdsInputSlot=4), [1, 1, 1, 1, 1])
    values = [np.array(value, dtype=np.float32) for value in
              ([1], [0], [0], [30], [bad])]
    with pytest.raises(ValueError, match=r"kds input\[0\] must be finite and >= 0"):
        processor.compute(values)
    np.testing.assert_array_equal(processor.get_substep_pd()["kps"], [10])
    np.testing.assert_array_equal(processor.get_substep_pd()["kds"], [1])


def test_real_headless_runner_applies_changed_gains_at_every_physics_substep(tmp_path):
    (tmp_path / "robot.xml").write_text('''
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="motor" joint="hinge"/></actuator>
</mujoco>
''')
    (tmp_path / "main.py").write_text('''
from botlab import Simulation
from controller import create_controller
sim = Simulation.from_project()
sim.run(create_controller(sim))
''')
    (tmp_path / "controller.py").write_text('''
from botlab import PolicyController, Robot, constant, mode_select, project
project("Dynamic gains", control_hz=50, physics_hz=200,
        parameters={"mode": {"default": 2, "range": [0, 3]}})
robot = Robot("robot.xml", config={
    "actionDimension": 1, "jointQposOffset": 0, "jointQvelOffset": 0,
    "initialQpos": [0], "initialQvel": [0], "defaultJointPositions": [0],
    "kp": [10], "kd": [1], "effortLimit": [100],
})
def create_controller(sim):
    mode = robot.commands("mode")
    return PolicyController(robot, target=constant([0.5]).action(),
        kp=mode_select(mode, constant([10]), {3: constant([40])}),
        kd=mode_select(mode, constant([1]), {3: constant([4])}))
''')
    with open_headless(tmp_path) as run:
        recorded = []
        native_step = mujoco.mj_step

        def observe_step(model, data):
            recorded.append((float(data.ctrl[0]), float(data.qpos[0]), float(data.qvel[0])))
            native_step(model, data)

        with mock.patch.object(mujoco, "mj_step", side_effect=observe_step):
            first = run.step()
            np.testing.assert_array_equal(first.substep_pd["kps"], [10])
            np.testing.assert_array_equal(first.substep_pd["kds"], [1])
            assert len(recorded) == 4
            for ctrl, qpos, qvel in recorded:
                assert ctrl == pytest.approx(10 * (0.5 - qpos) - qvel)
            assert len({ctrl for ctrl, _, _ in recorded}) == 4

            recorded.clear()
            run.set_parameter("mode", 3)
            second = run.step()
            np.testing.assert_array_equal(second.substep_pd["kps"], [40])
            np.testing.assert_array_equal(second.substep_pd["kds"], [4])
            assert len(recorded) == 4
            for ctrl, qpos, qvel in recorded:
                assert ctrl == pytest.approx(40 * (0.5 - qpos) - 4 * qvel)
            assert len({ctrl for ctrl, _, _ in recorded}) == 4
