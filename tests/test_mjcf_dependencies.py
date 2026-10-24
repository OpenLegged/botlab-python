"""Dependency closure must agree with real MuJoCo and remain project-contained."""
import numpy as np
import pytest

from botlab._host import BotLabError
from botlab._mjcf import mjcf_dependencies
from botlab._runtime.run import _load_model_with_transform


@pytest.fixture()
def scene(tmp_path):
    robot = tmp_path / "robot"
    robot.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "tetra.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    (shared / "body.xml").write_text("<mujoco><worldbody><geom type='mesh' mesh='terrain'/></worldbody></mujoco>")
    (shared / "nested.xml").write_text("<mujoco><include file='body.xml'/></mujoco>")
    entry = robot / "scene.xml"
    entry.write_text("""<mujoco><compiler meshdir='../shared'/>
        <asset><mesh name='terrain' file='tetra.obj'/></asset>
        <!-- <include file='nonexistent.xml'/> -->
        <include file='../shared/nested.xml'/></mujoco>""")
    return entry


def test_sibling_nested_dependencies_match_native(scene):
    mujoco = pytest.importorskip("mujoco")
    documents, assets = mjcf_dependencies(scene, root=scene.parent.parent)
    assert len(documents) == 3
    assert {p.name for p in assets} == {"tetra.obj"}
    original = mujoco.MjModel.from_xml_path(str(scene))
    transformed = _load_model_with_transform(scene, lambda xml: xml, mujoco)
    np.testing.assert_array_equal(original.mesh_vert, transformed.mesh_vert)
    np.testing.assert_array_equal(original.geom_pos, transformed.geom_pos)
    np.testing.assert_array_equal(original.body_mass, transformed.body_mass)


def test_missing_dependency_and_cycle_are_diagnostic(scene):
    (scene.parent.parent / 'shared/tetra.obj').unlink()
    with pytest.raises(BotLabError, match='tetra.obj'):
        mjcf_dependencies(scene)
    scene.write_text("<mujoco><include file='scene.xml'/></mujoco>")
    with pytest.raises(BotLabError, match='include cycle'):
        mjcf_dependencies(scene)


def test_dependencies_cannot_escape_project_or_follow_external_symlinks(scene, tmp_path):
    with pytest.raises(BotLabError, match='escapes the project'):
        mjcf_dependencies(scene, root=scene.parent)
    external = tmp_path.parent / (tmp_path.name + '-external.obj')
    external.write_text('external')
    mesh = tmp_path / 'shared/tetra.obj'
    mesh.unlink()
    mesh.symlink_to(external)
    with pytest.raises(BotLabError, match='escapes the project'):
        mjcf_dependencies(scene, root=tmp_path)


def test_compiler_assetdir_strippath_and_texture(tmp_path):
    (tmp_path / 'assets').mkdir()
    for name in ('mesh.obj', 'height.bin', 'color.png'):
        (tmp_path / 'assets' / name).write_bytes(b'fixture')
    entry = tmp_path / 'scene.xml'
    entry.write_text("""<mujoco><compiler assetdir='assets' strippath='true'/><asset>
      <mesh file='ignored/mesh.obj'/><hfield file='ignored/height.bin'/>
      <texture file='ignored/color.png'/></asset></mujoco>""")
    _, assets = mjcf_dependencies(entry, root=tmp_path)
    assert {p.name for p in assets} == {'mesh.obj', 'height.bin', 'color.png'}


def test_transform_applies_to_included_document(scene):
    mujoco = pytest.importorskip('mujoco')
    result = _load_model_with_transform(scene, lambda xml: xml.replace(
        '<worldbody>', "<worldbody><geom type='sphere' size='.1' pos='0 0 3'/>"), mujoco)
    assert result.ngeom == 2
    np.testing.assert_array_equal(result.geom_pos[0], [0, 0, 3])


def test_packaging_rejects_absolute_dependencies(scene):
    scene.write_text(f"<mujoco><include file='{scene.parent.parent}/shared/body.xml'/></mujoco>")
    with pytest.raises(BotLabError, match='must be project-relative'):
        mjcf_dependencies(scene, root=scene.parent.parent)


def test_attached_model_cycles_fail_without_recursion(tmp_path):
    a = tmp_path / 'a.xml'
    b = tmp_path / 'b.xml'
    a.write_text("<mujoco><asset><model name='b' file='b.xml'/></asset></mujoco>")
    b.write_text("<mujoco><asset><model name='a' file='a.xml'/></asset></mujoco>")
    with pytest.raises(BotLabError, match='attached model cycle'):
        mjcf_dependencies(a, root=tmp_path)
