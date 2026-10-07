"""Multi-robot MJCF composition tests (tier B: needs mujoco + numpy)."""

import os
import unittest

import mujoco
import numpy as np

from botlab._runtime import _compose


DUCK_XML = """
<mujoco>
  <worldbody>
    <body name="duck" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="body_geom" type="sphere" size="0.05" mass="0.5"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015" mass="0.1"/>
        <site name="mouth" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_motor" joint="knee" gear="1"/>
  </actuator>
</mujoco>
"""

PRIMARY_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="duck" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="body_geom" type="sphere" size="0.05" mass="0.5"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015" mass="0.1"/>
        <site name="mouth" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_motor" joint="knee" gear="1"/>
  </actuator>
</mujoco>
"""


class NamespaceTest(unittest.TestCase):
    def test_prefixes_name_references_but_not_paths_or_enums(self):
        xml = (
            '<default class="leg"><joint damping="1"/></default>'
            '<body name="duck" childclass="leg"><joint name="knee" class="leg"/></body>'
            '<equality><connect body1="duck" body2="anchor"/></equality>'
            '<contact><exclude body1="duck" body2="floor"/></contact>'
            '<sensor><framequat objtype="body" objname="duck"/></sensor>'
            '<mesh name="shell" file="meshes/shell.stl"/>'
        )
        out = _compose.apply_robot_namespace(xml, "r1_")
        self.assertIn('class="r1_leg"', out)
        self.assertIn('childclass="r1_leg"', out)
        self.assertIn('name="r1_duck"', out)
        self.assertIn('body1="r1_duck"', out)
        self.assertIn('body2="r1_anchor"', out)
        self.assertIn('objname="r1_duck"', out)
        self.assertIn('file="meshes/shell.stl"', out)  # 文件路径不加前缀
        self.assertIn('objtype="body"', out)  # 枚举值不加前缀

    def test_single_quoted_attributes(self):
        out = _compose.apply_robot_namespace("<joint name='hip'/>", "r2_")
        self.assertIn("name='r2_hip'", out)


class ExtractTest(unittest.TestCase):
    def test_nested_blocks_and_inner_content(self):
        xml = "<worldbody><body name='a'><body name='b'/></body></worldbody>"
        inner = _compose.extract_tag_inner_content(xml, "worldbody")
        self.assertIn("name='a'", inner)
        self.assertIn("name='b'", inner)

    def test_root_sections_collect_known_tags(self):
        xml = (
            "<mujoco><worldbody/><equality><connect body1='a' body2='b'/></equality>"
            "<contact><exclude body1='a' body2='c'/></contact>"
            "<sensor><framequat name='q' objname='a'/></sensor></mujoco>"
        )
        out = _compose.extra_root_sections(xml)
        self.assertIn("<equality>", out)
        self.assertIn("<contact>", out)
        self.assertIn("<sensor>", out)
        self.assertNotIn("worldbody", out)


class InjectTest(unittest.TestCase):
    def test_inject_worldbody_and_root_sections(self):
        composed = _compose.compose_entry_xml(
            "<mujoco><worldbody><geom name='floor'/></worldbody></mujoco>",
            [("<body name='r1_duck'/>", "", "<equality><connect body1='r1_a' body2='r1_b'/></equality>")],
        )
        self.assertIn("<geom name='floor'/>", composed)
        self.assertIn("<body name='r1_duck'/>", composed)
        self.assertIn("<equality>", composed)
        self.assertLess(composed.index("r1_duck"), composed.index("</worldbody>"))

    def test_missing_worldbody_raises(self):
        with self.assertRaisesRegex(ValueError, "worldbody"):
            _compose.inject_before_closing_tag("<mujoco/>", "worldbody", "<body/>")


class LoadExtraRobotTest(unittest.TestCase):
    def _write_tree(self, root):
        meshes = os.path.join(root, "meshes")
        os.makedirs(meshes)
        with open(os.path.join(meshes, "shell.stl"), "wb") as handle:
            handle.write(b"solid shell\nendsolid shell\n")
        with open(os.path.join(root, "parts.xml"), "w", encoding="utf-8") as handle:
            handle.write(
                "<mujocoinclude>"
                '<body name="duck"><freejoint name="root"/>'
                '<geom name="shell_geom" mesh="shell" type="mesh" mass="0.5"/>'
                '<joint name="knee" type="hinge"/></body>'
                "</mujocoinclude>"
            )
        with open(os.path.join(root, "robot.xml"), "w", encoding="utf-8") as handle:
            handle.write(
                '<mujoco><compiler meshdir="meshes"/>'
                '<asset><mesh name="shell" file="shell.stl"/></asset>'
                '<worldbody><include file="parts.xml"/></worldbody>'
                "</mujoco>"
            )

    def test_include_expansion_asset_rewrite_and_namespace(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            self._write_tree(root)
            worldbody, assets, roots = _compose.load_extra_robot(
                os.path.join(root, "robot.xml"), "r1_", True
            )
            self.assertIn('name="r1_duck"', worldbody)
            self.assertNotIn("<include", worldbody)
            self.assertIn('name="r1_shell"', assets)
            self.assertIn(os.path.join(root, "meshes", "shell.stl"), assets)
            # 无执行器的模型经 auto-actuator 注入：root sections 带 actuator 段
            self.assertIn("<actuator>", roots)
            self.assertIn('joint="r1_knee"', roots)

    def test_include_cycle_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            for name in ("a.xml", "b.xml"):
                other = "b.xml" if name == "a.xml" else "a.xml"
                with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                    handle.write(f'<mujoco><include file="{other}"/></mujoco>')
            with self.assertRaisesRegex(ValueError, "include cycle"):
                _compose.load_extra_robot(os.path.join(root, "a.xml"), "r1_")

    def test_missing_asset_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "robot.xml"), "w", encoding="utf-8") as handle:
                handle.write(
                    '<mujoco><asset><mesh name="x" file="nope.stl"/></asset>'
                    "<worldbody/></mujoco>"
                )
            with self.assertRaisesRegex(ValueError, "nope.stl"):
                _compose.load_extra_robot(os.path.join(root, "robot.xml"), "r1_")


class ComposedModelTest(unittest.TestCase):
    """Compose two real robots with equality + contact and compile natively."""

    def _composed_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            extra_path = os.path.join(root, "extra.xml")
            with open(extra_path, "w", encoding="utf-8") as handle:
                handle.write(DUCK_XML)
            worldbody, assets, roots = _compose.load_extra_robot(extra_path, "r1_")
            primary = PRIMARY_XML.replace(
                "</mujoco>",
                "<contact><exclude body1='duck' body2='r1_duck'/></contact>"
                "<equality><connect body1='duck' body2='r1_duck' anchor='0 0 0'/></equality>"
                "</mujoco>",
            )
            composed = _compose.compose_entry_xml(primary, [(worldbody, assets, roots)])
            return mujoco.MjModel.from_xml_string(composed)

    def test_composed_model_compiles_with_namespaces(self):
        model = self._composed_model()
        self.assertEqual(model.nu, 2)
        self.assertIsNotNone(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_duck")
        )
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_duck"), 0)
        self.assertEqual(int(model.neq), 1)  # equality connect 合并生效

    def test_resolve_scopes(self):
        model = self._composed_model()
        scopes = _compose.resolve_robot_scopes(mujoco, model, ["main", "second"])
        self.assertEqual(
            scopes["main"],
            {"qpos_start": 0, "qvel_start": 0, "actuator_start": 0, "name_prefix": ""},
        )
        second = scopes["second"]
        # 主机器人：freejoint 7 + hinge 1 = qpos 8；附加机器人从 8 开始
        self.assertEqual(second["qpos_start"], 8)
        self.assertEqual(second["qvel_start"], 7)
        self.assertEqual(second["actuator_start"], 1)
        self.assertEqual(second["name_prefix"], "r1_")

    def test_missing_namespace_raises(self):
        model = self._composed_model()
        with self.assertRaisesRegex(ValueError, 'namespace "r2_"'):
            _compose.resolve_robot_scopes(mujoco, model, ["main", "s1", "s2"])

    def test_initial_state_and_spawn_pose(self):
        model = self._composed_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        states = [
            {"robotId": "main", "namespace": "", "initialQpos": [0] * 8},
            {
                "robotId": "second",
                "namespace": "r1_",
                "initialQpos": [0.5, 0, 0.4, 1, 0, 0, 0, 0.2],
                "spawnPosition": [1.0, 2.0, 0.6],
            },
        ]
        _compose.apply_composed_initial_state(mujoco, model, data, states)
        qpos = np.asarray(data.qpos)
        np.testing.assert_allclose(qpos[0:8], [0] * 8)
        # spawnPosition 覆盖初始根位置，关节值保留
        np.testing.assert_allclose(qpos[8:16], [1.0, 2.0, 0.6, 1, 0, 0, 0, 0.2])

    def test_spawn_requires_freejoint(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='r1_arm' pos='0 0 1'>"
            "<joint name='r1_j' type='hinge'/><geom size='0.1' mass='1'/>"
            "</body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        with self.assertRaisesRegex(ValueError, "freejoint"):
            _compose.apply_composed_initial_state(
                mujoco,
                model,
                data,
                [{"robotId": "arm", "namespace": "r1_", "spawnPosition": [0, 0, 1]}],
            )


if __name__ == "__main__":
    unittest.main()
