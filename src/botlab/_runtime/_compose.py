"""Multi-robot MJCF composition for the native runner.

Mirrors the browser worker's composition pipeline
(src/platform/workers/mujocoWorkerVfs.ts) with native filesystem reads in
place of the VFS: each extra robot's XML gets its include chain expanded,
asset ``file`` attributes rewritten to absolute paths, auto-actuators
injected, and every name-reference attribute prefixed with its ``r{N}_``
namespace; the composed worldbody/asset/root sections are then injected
into the primary robot's entry XML.

Scope resolution (qpos/qvel/ctrl slices per robot) mirrors
src/domain/mujoco/mujocoRobotScope.ts: the primary robot keeps its original
names and starts at zero; extra robots resolve through their namespace.
"""

import os
import re

from ._physics_xml import _strip_xml_comments, inject_auto_actuators

# Name-reference attributes rewritten by the namespace pass. Must match
# ROBOT_NAME_ATTRIBUTES in src/platform/workers/mujocoWorkerVfs.ts.
ROBOT_NAME_ATTRIBUTES = frozenset({
    "name", "class", "childclass", "body", "body1", "body2",
    "joint", "joint1", "joint2", "geom", "geom1", "geom2",
    "site", "site1", "site2", "tendon", "tendon1", "tendon2",
    "actuator", "mesh", "material", "texture", "hfield", "skin", "flex",
    "target", "objname", "refname", "cranksite", "slidersite",
})

# Root-level sections merged from extra robots into the composed model.
# Must match EXTRA_ROOT_SECTIONS in mujocoWorkerVfs.ts.
EXTRA_ROOT_SECTIONS = (
    "default", "contact", "equality", "tendon", "actuator", "sensor",
    "keyframe", "custom",
)

_TAG_PATTERN = re.compile(r"<\s*[A-Za-z_][\w:.-]*(?:\s[^<>]*?)?\s*/?\s*>")
_ATTR_PATTERN = re.compile(r"\b([A-Za-z_][\w:.-]*)\s*=\s*([\"'])(.*?)\2")
_INCLUDE_PATTERN = re.compile(r"<\s*include\b([^<>]*?)/\s*>", re.IGNORECASE)


def apply_robot_namespace(xml, prefix):
    """Prefix structural declarations and every name-reference attribute."""

    def rewrite_tag(match):
        tag = match.group(0)

        def rewrite_attr(attr_match):
            raw_name, quote, value = attr_match.groups()
            if not value or raw_name.lower() not in ROBOT_NAME_ATTRIBUTES:
                return attr_match.group(0)
            return f"{raw_name}={quote}{prefix}{value}{quote}"

        return _ATTR_PATTERN.sub(rewrite_attr, tag)

    return _TAG_PATTERN.sub(rewrite_tag, xml)


def extract_tag_blocks(xml, tag_name):
    """Top-level blocks of ``tag_name`` as (full, inner) string pairs."""
    pattern = re.compile(r"<\s*(/?)\s*" + tag_name + r"\b[^>]*>", re.IGNORECASE)
    blocks = []
    depth = 0
    start = -1
    inner_start = -1
    for match in pattern.finditer(xml):
        closing = match.group(1) == "/"
        self_closing = not closing and bool(re.search(r"/\s*>$", match.group(0)))
        if not closing:
            if depth == 0:
                start = match.start()
                inner_start = match.end()
            if self_closing:
                if depth == 0:
                    blocks.append((match.group(0), ""))
            else:
                depth += 1
            continue
        if depth == 0:
            continue
        depth -= 1
        if depth == 0 and start >= 0:
            blocks.append((xml[start : match.end()], xml[inner_start : match.start()]))
            start = -1
            inner_start = -1
    return blocks


def extract_tag_inner_content(xml, tag_name):
    return "\n".join(
        inner.strip() for _full, inner in extract_tag_blocks(xml, tag_name) if inner.strip()
    )


def _top_level_section_blocks(xml):
    """(tag, full) blocks that are direct children of the <mujoco> root.

    Depth-tracked over all tags so e.g. an ``<equality>`` inside a
    ``<default>`` block (a defaults-class attribute, not a constraint
    section) is never collected.
    """
    tag_pattern = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)([^<>]*?)(/?)\s*>")
    blocks = []
    depth = 0
    start = -1
    current = None
    for match in tag_pattern.finditer(xml):
        closing = match.group(1) == "/"
        name = match.group(2).lower()
        self_closing = match.group(4) == "/"
        if closing:
            if depth == 2 and current == name:
                blocks.append((name, xml[start : match.end()]))
                current = None
            depth -= 1
            continue
        if self_closing:
            if depth == 1:
                blocks.append((name, match.group(0)))
            continue
        if depth == 1:
            current = name
            start = match.start()
        depth += 1
    return blocks


def extra_root_sections(xml):
    return "\n".join(
        block.strip()
        for name, block in _top_level_section_blocks(xml)
        if name in EXTRA_ROOT_SECTIONS and block.strip()
    )


def inject_before_closing_tag(xml, tag_name, content):
    trimmed = content.strip()
    if not trimmed:
        return xml
    pattern = re.compile(r"</\s*" + tag_name + r"\s*>", re.IGNORECASE)
    if not pattern.search(xml):
        raise ValueError(f"[mujocoWorker] Cannot inject: <{tag_name}> is missing")
    return pattern.sub(f"{trimmed}\n</{tag_name}>", xml, count=1)


def inject_asset_xml(xml, content):
    trimmed = content.strip()
    if not trimmed:
        return xml
    if re.search(r"<\s*/\s*asset\s*>", xml, re.IGNORECASE):
        return inject_before_closing_tag(xml, "asset", trimmed)
    match = re.search(r"<\s*worldbody\b", xml, re.IGNORECASE)
    if not match:
        raise ValueError("[mujocoWorker] Cannot inject asset: <worldbody> is missing")
    return xml[: match.start()] + f"<asset>\n{trimmed}\n</asset>\n" + xml[match.start() :]


def inject_root_sections(xml, content):
    trimmed = content.strip()
    if not trimmed:
        return xml
    if not re.search(r"<\s*/\s*mujoco\s*>", xml, re.IGNORECASE):
        raise ValueError("[mujocoWorker] Cannot compose extra robot: </mujoco> is missing")
    return re.sub(
        r"<\s*/\s*mujoco\s*>", f"{trimmed}\n</mujoco>", xml, count=1, flags=re.IGNORECASE
    )


def _read_xml_attrs(text):
    return {match.group(1): match.group(3) for match in _ATTR_PATTERN.finditer(text)}


def _compiler_asset_dir(xml, tag_name):
    blocks = extract_tag_blocks(xml, "compiler")
    attrs = {}
    for block, _inner in blocks:
        attrs = _read_xml_attrs(block)
        break
    assetdir = (attrs.get("assetdir") or "").strip()
    if tag_name == "mesh":
        return (attrs.get("meshdir") or "").strip() or assetdir
    if tag_name == "texture":
        return (attrs.get("texturedir") or "").strip() or assetdir
    return assetdir


def _asset_file_refs(xml):
    refs = []
    tag_name = re.compile(r"<\s*([A-Za-z_][\w:.-]*)")
    for block, _inner in extract_tag_blocks(xml, "asset"):
        for tag in _TAG_PATTERN.finditer(block):
            name_match = tag_name.match(tag.group(0))
            attrs = _read_xml_attrs(tag.group(0))
            file = (attrs.get("file") or "").strip()
            if name_match and file:
                refs.append((name_match.group(1).lower(), file))
    return refs


def _rewrite_assets_to_absolute(xml, xml_path):
    """Rewrite asset ``file`` attributes to absolute native paths.

    The browser loads extra-robot assets into the VFS and rewrites to
    absolute VFS paths; natively the real filesystem path plays that role.
    """
    base_dir = os.path.dirname(os.path.abspath(xml_path))
    result = xml
    for tag_name, file in _asset_file_refs(xml):
        if re.match(r"^(https?:|data:|blob:)", file, re.IGNORECASE):
            raise ValueError(
                f'[mujocoWorker] Extra robot asset "{file}" uses a URL; '
                "the local runtime requires local files"
            )
        compiler_dir = _compiler_asset_dir(xml, tag_name)
        candidate = os.path.join(base_dir, compiler_dir, file) if compiler_dir else os.path.join(base_dir, file)
        absolute = os.path.normpath(candidate)
        if not os.path.isfile(absolute):
            raise ValueError(
                f'[mujocoWorker] Extra robot asset "{file}" was not found under {base_dir}'
            )
        pattern = re.compile(r"(\bfile\s*=\s*)([\"'])" + re.escape(file) + r"\2")
        result = pattern.sub(
            lambda m, _absolute=absolute: f"{m.group(1)}{m.group(2)}{_absolute}{m.group(2)}", result
        )
    return result


def _unwrap_included_xml(xml):
    without_declaration = re.sub(r"^\s*<\?xml[\s\S]*?\?>\s*", "", xml).strip()
    wrapper = re.match(
        r"^<\s*(mujoco|mujocoinclude)\b[^>]*>([\s\S]*)<\s*/\s*\1\s*>$",
        without_declaration,
        re.IGNORECASE,
    )
    return (wrapper.group(2) if wrapper else without_declaration).strip()


def _load_extra_xml_tree(xml_path, include_stack):
    """Expand the include chain of an extra robot XML into one flat text."""
    absolute = os.path.abspath(xml_path)
    if absolute in include_stack:
        raise ValueError(
            "[mujocoWorker] Extra robot include cycle: "
            + " -> ".join([*include_stack, absolute])
        )
    with open(absolute, encoding="utf-8") as handle:
        xml = handle.read()
    # 注释里的标签文本（如被注释掉的 <contact>）不是结构：扫描前先剥掉，
    # 否则 include 展开与 section 提取会把注释切碎。编译语义不变。
    xml = _strip_xml_comments(xml)
    base_dir = os.path.dirname(absolute)

    output = []
    cursor = 0
    for match in _INCLUDE_PATTERN.finditer(xml):
        output.append(xml[cursor : match.start()])
        file = (_read_xml_attrs(match.group(1)).get("file") or "").strip()
        if not file:
            raise ValueError("[mujocoWorker] Extra robot include is missing file")
        include_path = os.path.normpath(os.path.join(base_dir, file))
        if not os.path.isfile(include_path):
            raise ValueError(
                f'[mujocoWorker] Extra robot include "{file}" was not found under {base_dir}'
            )
        output.append(_load_extra_xml_tree(include_path, [*include_stack, absolute]))
        cursor = match.end()
    output.append(xml[cursor:])
    # Includes land inside the including document: strip the wrapper only for
    # nested files (the entry keeps its own <mujoco> root for section extraction).
    combined = "".join(output)
    return combined if not include_stack else _unwrap_included_xml(combined)


def load_extra_robot(xml_path, prefix, auto_actuators=None, label=None):
    """Load one extra robot: expand includes, resolve assets, namespace it.

    Returns ``(worldbody_xml, asset_xml, root_sections_xml)`` ready for
    injection into the primary robot's entry XML.
    """
    label = label or xml_path
    xml = _load_extra_xml_tree(xml_path, [])
    xml = _rewrite_assets_to_absolute(xml, xml_path)
    xml = inject_auto_actuators(xml, auto_actuators, label)
    namespaced = apply_robot_namespace(xml, prefix)
    namespaced = _name_unnamed_assets(namespaced, prefix)
    return (
        extract_tag_inner_content(namespaced, "worldbody"),
        extract_tag_inner_content(namespaced, "asset"),
        extra_root_sections(namespaced),
    )


def _name_unnamed_assets(xml, prefix):
    """Give file-bearing assets an explicit namespaced name.

    MuJoCo derives unnamed mesh/texture names from the file basename; two
    identical robots composed into one scene would collide. The namespace
    pass can only prefix written names, so derive and write them ourselves.
    """
    def fix_block(match):
        block = match.group(0)

        def fix_tag(tag_match):
            tag = tag_match.group(0)
            attrs = _read_xml_attrs(tag)
            file = (attrs.get("file") or "").strip()
            if not file or (attrs.get("name") or "").strip():
                return tag
            derived = os.path.splitext(os.path.basename(file))[0]
            if not derived:
                return tag
            closing = re.search(r"/?\s*>$", tag).group(0)
            body = tag[: -len(closing)]
            suffix = "/>" if closing.lstrip().startswith("/") else ">"
            return f'{body} name="{prefix}{derived}"{suffix}'

        return _TAG_PATTERN.sub(fix_tag, block)

    return re.sub(r"<\s*asset\b[^>]*>[\s\S]*?<\s*/\s*asset\s*>", fix_block, xml, flags=re.IGNORECASE)


def load_extra_robot_text(xml, prefix, auto_actuators=None, label=None):
    """Text-in variant of load_extra_robot for embedded XML (no includes/assets)."""
    xml = _strip_xml_comments(xml)
    xml = inject_auto_actuators(xml, auto_actuators, label or "embedded extra robot")
    namespaced = apply_robot_namespace(xml, prefix)
    namespaced = _name_unnamed_assets(namespaced, prefix)
    return (
        extract_tag_inner_content(namespaced, "worldbody"),
        extract_tag_inner_content(namespaced, "asset"),
        extra_root_sections(namespaced),
    )


def compose_entry_xml(entry_xml, extras):
    """Inject extra robots' sections into the primary entry XML text."""
    composed = _strip_xml_comments(entry_xml)
    for worldbody_xml, asset_xml, root_sections_xml in extras:
        if worldbody_xml:
            composed = inject_before_closing_tag(composed, "worldbody", worldbody_xml)
        if asset_xml:
            composed = inject_asset_xml(composed, asset_xml)
        if root_sections_xml:
            composed = inject_root_sections(composed, root_sections_xml)
    return composed


def _model_name(mujoco_module, model, obj_type, index):
    name = mujoco_module.mj_id2name(model, obj_type, index)
    return name or ""


def _extra_robot_scope(mujoco_module, model, prefix):
    qpos_start = -1
    qvel_start = -1
    namespace_found = False
    body_type = mujoco_module.mjtObj.mjOBJ_BODY
    for body_index in range(model.nbody):
        if not _model_name(mujoco_module, model, body_type, body_index).startswith(prefix):
            continue
        namespace_found = True
        joint_index = int(model.body_jntadr[body_index])
        if joint_index < 0:
            continue
        candidate_qpos = int(model.jnt_qposadr[joint_index])
        candidate_qvel = int(model.jnt_dofadr[joint_index])
        if candidate_qpos >= 0 and candidate_qvel >= 0:
            qpos_start = candidate_qpos
            qvel_start = candidate_qvel
            break
    if not namespace_found:
        raise ValueError(f'[mujocoRobotScope] body namespace "{prefix}" was not found')
    if qpos_start < 0 or qvel_start < 0:
        raise ValueError(
            f'[mujocoRobotScope] robot namespace "{prefix}" has no addressable root joint'
        )
    actuator_type = mujoco_module.mjtObj.mjOBJ_ACTUATOR
    actuator_start = -1
    for index in range(model.nu):
        if _model_name(mujoco_module, model, actuator_type, index).startswith(prefix):
            actuator_start = index
            break
    if actuator_start < 0:
        raise ValueError(
            f'[mujocoRobotScope] actuator namespace "{prefix}" was not found — '
            "composed extra robots need explicitly named actuators "
            "(MuJoCo does not auto-name actuators)"
        )
    return {
        "qpos_start": qpos_start,
        "qvel_start": qvel_start,
        "actuator_start": actuator_start,
        "name_prefix": prefix,
    }


def resolve_robot_scopes(mujoco_module, model, robot_ids):
    """Resolve state-array slices for composed robots.

    The primary robot keeps its original names and starts at zero. Extra
    robots use the exact ``rN_`` namespace of the composer; missing names are
    configuration errors, not silent aliases of the primary robot.
    """
    if not robot_ids:
        raise ValueError("[mujocoRobotScope] at least one robot resource is required")
    scopes = {
        robot_ids[0]: {
            "qpos_start": 0,
            "qvel_start": 0,
            "actuator_start": 0,
            "name_prefix": "",
        }
    }
    for index in range(1, len(robot_ids)):
        scopes[robot_ids[index]] = _extra_robot_scope(
            mujoco_module, model, f"r{index}_"
        )
    return scopes


def _namespaced_free_joint(mujoco_module, model, namespace):
    free_type = mujoco_module.mjtJoint.mjJNT_FREE
    body_type = mujoco_module.mjtObj.mjOBJ_BODY
    for joint in range(model.njnt):
        if int(model.jnt_type[joint]) != int(free_type):
            continue
        if not namespace:
            return joint
        body_index = int(model.jnt_bodyid[joint])
        if body_index >= 0 and _model_name(
            mujoco_module, model, body_type, body_index
        ).startswith(namespace):
            return joint
    return -1


def _finite_vector(value, field, length=None):
    import math

    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        raise ValueError(
            f"[mujocoWorker] {field} must be a {length}-number array"
            if length is not None
            else f"[mujocoWorker] {field} must be a number array"
        )
    if not value:
        raise ValueError(f"[mujocoWorker] {field} must not be empty")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"[mujocoWorker] {field} contains a non-finite number")
    return result


def _resolve_spawn_position(state):
    import random

    if state.get("randomizePosition") is True:
        low = _finite_vector(state.get("spawnPositionMin"), "spawnPositionMin", 3)
        high = _finite_vector(state.get("spawnPositionMax"), "spawnPositionMax", 3)
        if not low or not high:
            raise ValueError(
                "[mujocoWorker] spawnPositionMin and spawnPositionMax are required "
                "when randomization is enabled"
            )
        for index in range(3):
            if low[index] > high[index]:
                raise ValueError(
                    f"[mujocoWorker] spawnPositionMin[{index}] must be <= spawnPositionMax[{index}]"
                )
        return [low[i] + random.random() * (high[i] - low[i]) for i in range(3)]
    return _finite_vector(state.get("spawnPosition"), "spawnPosition", 3)


def _resolve_spawn_euler_deg(state):
    import random

    if state.get("randomizeEuler") is True:
        low = _finite_vector(state.get("spawnEulerDegMin"), "spawnEulerDegMin", 3)
        high = _finite_vector(state.get("spawnEulerDegMax"), "spawnEulerDegMax", 3)
        if not low or not high:
            raise ValueError(
                "[mujocoWorker] spawnEulerDegMin and spawnEulerDegMax are required "
                "when randomization is enabled"
            )
        for index in range(3):
            if low[index] > high[index]:
                raise ValueError(
                    f"[mujocoWorker] spawnEulerDegMin[{index}] must be <= spawnEulerDegMax[{index}]"
                )
        return [low[i] + random.random() * (high[i] - low[i]) for i in range(3)]
    return _finite_vector(state.get("spawnEulerDeg"), "spawnEulerDeg", 3)


def _quat_from_euler_xyz_degrees(euler_deg):
    import math

    x, y, z = (value * math.pi / 180 for value in euler_deg)
    c1, c2, c3 = math.cos(x / 2), math.cos(y / 2), math.cos(z / 2)
    s1, s2, s3 = math.sin(x / 2), math.sin(y / 2), math.sin(z / 2)
    return (
        c1 * c2 * c3 - s1 * s2 * s3,
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
    )


def apply_composed_initial_state(mujoco_module, model, data, robot_states):
    """Apply per-robot initial qpos/qvel and spawn poses on a composed model.

    ``robot_states`` carries one entry per declared robot in declaration
    order: ``{"robotId", "namespace", "initialQpos"?, "initialQvel"?,
    "spawnPosition"?, ...}`` (namespace "" for the primary). Mirrors
    applyInitialPose's composed branch in mujocoWorkerInitialPose.ts.
    """
    for state in robot_states:
        namespace = state.get("namespace") or ""
        joint = _namespaced_joint_any(mujoco_module, model, namespace)
        initial_qpos = state.get("initialQpos")
        initial_qvel = state.get("initialQvel")
        if initial_qpos is None and initial_qvel is None:
            continue
        if joint < 0:
            raise ValueError(
                f'[mujocoWorker] Robot "{state["robotId"]}" namespace "{namespace}" '
                "has no addressable joint"
            )
        qpos_start = int(model.jnt_qposadr[joint])
        qvel_start = int(model.jnt_dofadr[joint])
        if initial_qpos is not None:
            values = _finite_vector(initial_qpos, f'{state["robotId"]}.initialQpos')
            if qpos_start < 0 or qpos_start + len(values) > len(data.qpos):
                raise ValueError(
                    f'[mujocoWorker] {state["robotId"]}.initialQpos is outside composed qpos'
                )
            data.qpos[qpos_start : qpos_start + len(values)] = values
        if initial_qvel is not None:
            values = _finite_vector(initial_qvel, f'{state["robotId"]}.initialQvel')
            if qvel_start < 0 or qvel_start + len(values) > len(data.qvel):
                raise ValueError(
                    f'[mujocoWorker] {state["robotId"]}.initialQvel is outside composed qvel'
                )
            data.qvel[qvel_start : qvel_start + len(values)] = values

    for state in robot_states:
        position = _resolve_spawn_position(state)
        euler_deg = _resolve_spawn_euler_deg(state)
        if position is None and euler_deg is None:
            continue
        joint = _namespaced_free_joint(mujoco_module, model, state.get("namespace") or "")
        if joint < 0:
            raise ValueError(
                f'[mujocoWorker] Robot "{state["robotId"]}" spawn pose requires a '
                "namespaced freejoint"
            )
        qpos_adr = int(model.jnt_qposadr[joint])
        if qpos_adr < 0 or qpos_adr + 6 >= len(data.qpos):
            raise ValueError(
                f"[mujocoWorker] Freejoint qpos address {qpos_adr} is outside qpos "
                f"length {len(data.qpos)}"
            )
        if position is not None:
            data.qpos[qpos_adr : qpos_adr + 3] = position
        if euler_deg is not None:
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = _quat_from_euler_xyz_degrees(euler_deg)


def _namespaced_joint_any(mujoco_module, model, namespace):
    body_type = mujoco_module.mjtObj.mjOBJ_BODY
    for joint in range(model.njnt):
        if not namespace:
            return joint
        body_index = int(model.jnt_bodyid[joint])
        if body_index >= 0 and _model_name(
            mujoco_module, model, body_type, body_index
        ).startswith(namespace):
            return joint
    return -1
