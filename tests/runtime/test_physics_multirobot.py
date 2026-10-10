"""Multi-robot native physics tests (tier B: needs mujoco + numpy)."""

import unittest

import mujoco
import numpy as np

from botlab._runtime import _compose, _physics

DUCK_XML = """
<mujoco>
  <worldbody>
    <body name="duck" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="body_geom" type="sphere" size="0.05" mass="0.5"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_motor" joint="knee" gear="1"/>
  </actuator>
</mujoco>
"""

ROBOT_CONFIG = {
    "actionDimension": 1,
    "jointQposOffset": 7,
    "jointQvelOffset": 6,
    "initialQpos": [0, 0, 0.3, 1, 0, 0, 0, 0.1],
    "initialQvel": [0] * 7,
}


def _two_duck_physics():
    """Two free-Root ducks composed; second one spawned aside."""
    worldbody, assets, roots = _compose.load_extra_robot_text(DUCK_XML, "r1_")
    composed = _compose.compose_entry_xml(DUCK_XML, [(worldbody, assets, roots)])
    model = mujoco.MjModel.from_xml_string(composed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    resources = [
        {"id": "first", "kind": "robot", "config": dict(ROBOT_CONFIG)},
        {"id": "second", "kind": "robot", "config": dict(ROBOT_CONFIG)},
    ]
    robot_states = [
        {**ROBOT_CONFIG, "robotId": "first", "namespace": ""},
        {
            **ROBOT_CONFIG,
            "robotId": "second",
            "namespace": "r1_",
            "spawnPosition": [1.5, 0.0, 0.3],
        },
    ]
    physics = _physics.MujocoPhysics(
        model, data, mujoco, resources=resources, robot_states=robot_states
    )
    return physics


class ComposedPhysicsTest(unittest.TestCase):
    def test_scopes_and_layouts(self):
        physics = _two_duck_physics()
        self.assertEqual(physics.model.nq, 16)
        self.assertEqual(physics.model.nv, 14)
        self.assertEqual(physics.model.nu, 2)
        first = physics.robot_layout("first")
        second = physics.robot_layout("second")
        self.assertEqual(first["qpos_offset"], 7)
        self.assertEqual(first["actuator_start"], 0)
        self.assertEqual(second["qpos_offset"], 8 + 7)
        self.assertEqual(second["qvel_offset"], 7 + 6)
        self.assertEqual(second["actuator_start"], 1)
        self.assertEqual(second["name_prefix"], "r1_")

    def test_initial_state_and_spawn_per_robot(self):
        physics = _two_duck_physics()
        qpos = np.asarray(physics.data.qpos)
        # first：preset 初始值；second：spawn 位置覆盖根、关节取 preset
        np.testing.assert_allclose(qpos[0:8], ROBOT_CONFIG["initialQpos"])
        np.testing.assert_allclose(qpos[8:16], [1.5, 0.0, 0.3, 1, 0, 0, 0, 0.1])

    def test_robot_observation_uses_scope(self):
        physics = _two_duck_physics()
        first = physics.robot_observation("first")
        second = physics.robot_observation("second")
        self.assertEqual(second["base_position"], [1.5, 0.0, 0.3])
        self.assertEqual(first["base_position"], [0.0, 0.0, 0.3])
        self.assertEqual(second["joint_position"], [0.1])
        self.assertEqual(first["joint_position"], [0.1])

    def test_apply_action_raw_ctrl_lands_in_scope(self):
        physics = _two_duck_physics()
        physics.apply_action("second", [0.5])
        physics.step_substeps(1)
        ctrl = np.asarray(physics.data.ctrl)
        self.assertEqual(ctrl[0], 0.0)
        self.assertEqual(ctrl[1], 0.5)

    def test_apply_action_position_pd_uses_absolute_indices(self):
        physics = _two_duck_physics()
        physics.apply_action("second", {
            "kind": "position-pd",
            "target": [0.4],
            "kp": [10.0],
            "kd": [1.0],
            "effort_limit": [50.0],
        })
        kind, view = physics._pending["second"]
        self.assertEqual(kind, "pd")
        # pending 里是机器人局部索引；compose 时平移到绝对下标
        self.assertEqual(list(view["qpos_indices"]), [0])
        self.assertEqual(list(view["ctrl_indices"]), [0])
        ctrl, pd = physics.compose_payloads(physics._pending)
        self.assertIsNone(ctrl)
        self.assertEqual(list(pd["qpos_indices"]), [15])
        self.assertEqual(list(pd["qvel_indices"]), [13])
        self.assertEqual(list(pd["ctrl_indices"]), [1])
        physics.step_substeps(1)
        # second 的膝盖被 PD 拉向 0.4，first 不受影响
        self.assertGreater(physics.data.qpos[15], 0.1)
        physics.step_substeps(3)
        self.assertAlmostEqual(float(physics.data.qpos[7]), 0.1, places=5)

    def test_reset_replays_composed_snapshot(self):
        physics = _two_duck_physics()
        physics.data.qpos[8] = 9.9  # 扰动 second 的根
        physics.reset()
        np.testing.assert_allclose(
            np.asarray(physics.data.qpos)[8:16], [1.5, 0.0, 0.3, 1, 0, 0, 0, 0.1]
        )

    def test_unknown_robot_error_lists_declared(self):
        physics = _two_duck_physics()
        with self.assertRaisesRegex(ValueError, 'unknown robot "third"'):
            physics.robot_observation("third")

    def test_actuator_joint_order_is_robot_local(self):
        # actuatorJointOrder 越出本机器人维度 → 明确报错（此前按全局 nu 校验会漏）
        resources = [
            {"id": "first", "kind": "robot", "config": dict(ROBOT_CONFIG)},
            {
                "id": "second",
                "kind": "robot",
                "config": {**ROBOT_CONFIG, "actuatorJointOrder": [1]},
            },
        ]
        worldbody, assets, roots = _compose.load_extra_robot_text(DUCK_XML, "r1_")
        composed = _compose.compose_entry_xml(DUCK_XML, [(worldbody, assets, roots)])
        model = mujoco.MjModel.from_xml_string(composed)
        data = mujoco.MjData(model)
        physics = _physics.MujocoPhysics(
            model,
            data,
            mujoco,
            resources=resources,
            robot_states=[
                {**ROBOT_CONFIG, "robotId": "first", "namespace": ""},
                {**ROBOT_CONFIG, "robotId": "second", "namespace": "r1_"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "actuatorJointOrder"):
            physics.apply_action("second", {
                "kind": "position-pd",
                "target": [0.0],
                "kp": [1.0],
                "kd": [1.0],
                "effort_limit": [1.0],
            })


if __name__ == "__main__":
    unittest.main()
