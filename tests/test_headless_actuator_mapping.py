"""Exercise compiled PolicyController mappings through the public native runner."""
import numpy as np
import pytest

from botlab import open_headless
from tests.conftest import MAIN


def mapped_project(path, *, floating, order, mapping):
    path.mkdir()
    motors = [f'<motor joint="joint{i}"/>' for i in order]
    root = '<freejoint/>' if floating else ''
    (path / 'robot.xml').write_text(f'''<mujoco>
      <option gravity="0 0 0"/>
      <worldbody><body name="root" pos="0 0 1">{root}
        <geom type="sphere" size="0.05" mass="1"/>
        <body pos="0 0 -0.2"><joint name="joint0" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="1"/>
          <body pos="0 0 -0.2"><joint name="joint1" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="1"/>
          </body>
        </body>
      </body></worldbody>
      <actuator>{''.join(motors)}</actuator>
    </mujoco>''')
    config = dict(actionDimension=2, jointQposOffset=7 if floating else 0,
                  jointQvelOffset=6 if floating else 0, policyJointOrder=[0, 1],
                  motorJointOrder=[0, 1], defaultJointPositions=[0, 0], actionScale=1,
                  kp=[10, 20], kd=[0.5, 0.8], effortLimit=[5, 7])
    if mapping is not None:
        config['actuatorJointOrder'] = mapping
    (path / 'controller.py').write_text('''from botlab import Robot, PolicyController, constant, project
project("Actuator mapping", control_hz=50, physics_hz=200)
robot = Robot("robot.xml", config=''' + repr(config) + ''')
def create_controller(sim):
    return PolicyController(robot, target=constant([0.2, -0.3]).action())
''')
    (path / 'main.py').write_text(MAIN)
    return path


def rollout(path):
    states, forces, controls = [], [], []
    with open_headless(path) as run:
        for _ in range(25):
            run.step()
            data = run.physics.data
            states.append(np.concatenate([data.qpos, data.qvel]))
            forces.append(data.qfrc_actuator.copy())
            controls.append(data.ctrl.copy())
    return np.array(states), np.array(forces), np.array(controls)


@pytest.mark.parametrize('floating', [False, True])
def test_single_policy_actuator_mapping_preserves_joint_forces_and_trajectory(tmp_path, floating):
    baseline = rollout(mapped_project(tmp_path / 'plain', floating=floating,
                                      order=[0, 1], mapping=None))
    for name, order, mapping in [('identity', [0, 1], [0, 1]), ('swapped', [1, 0], [1, 0])]:
        actual = rollout(mapped_project(tmp_path / name, floating=floating,
                                        order=order, mapping=mapping))
        np.testing.assert_allclose(actual[0], baseline[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[1], baseline[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[2][:, mapping], baseline[2], rtol=0, atol=1e-12)
    assert np.max(np.abs(baseline[1])) > 0.1


def test_single_policy_rejects_out_of_scope_actuator_mapping(tmp_path):
    path = mapped_project(tmp_path / 'invalid', floating=True, order=[0, 1], mapping=[2, 0])
    with open_headless(path) as run:
        with pytest.raises(ValueError, match='outside robot action dimension'):
            run.step()
