"""Auto-actuator injection into MJCF text."""

import re

_ACTUATOR_TAG_PATTERN = re.compile(
    r"<\s*(motor|position|velocity|intvelocity|damper|cylinder|muscle|adhesion|general)\b",
    re.IGNORECASE,
)


def _xml_attr(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _strip_xml_comments(xml):
    return re.sub(r"<!--.*?-->", "", xml, flags=re.DOTALL)


def _read_xml_attrs(text):
    attrs = {}
    for match in re.finditer(r"([\w:.-]+)\s*=\s*\"([^\"]*)\"", text):
        attrs[match.group(1)] = match.group(2)
    return attrs


def _auto_actuator_options(raw):
    if raw is True:
        return {
            "enabled": True, "joint_names": None, "exclude": set(),
            "strip_suffix": "_joint", "ctrl_range": None,
            "ctrl_range_from_joint": True, "gear": None, "class_name": None,
        }
    if not raw:
        return {
            "enabled": False, "joint_names": None, "exclude": set(),
            "strip_suffix": "_joint", "ctrl_range": None,
            "ctrl_range_from_joint": True, "gear": None, "class_name": None,
        }
    if not isinstance(raw, dict):
        raise ValueError("[mujocoAutoActuators] autoActuators must be true, false, or an object")

    joint_names = raw.get("jointNames", raw.get("joints"))
    if joint_names is not None:
        if not isinstance(joint_names, list):
            raise ValueError("[mujocoAutoActuators] autoActuators.jointNames must be a string array")
        joint_names = [str(name).strip() for name in joint_names]
        if any(not name for name in joint_names):
            raise ValueError(
                "[mujocoAutoActuators] autoActuators.jointNames must be non-empty strings"
            )
    exclude = raw.get("excludeJointNames", raw.get("excludeJoints", raw.get("exclude"))) or []
    ctrl_range = raw.get("ctrlRange", raw.get("ctrlrange"))
    if ctrl_range is not None and ctrl_range != "":
        if isinstance(ctrl_range, str):
            parts = [float(part) for part in re.split(r"[\s,]+", ctrl_range.strip())]
        else:
            if not isinstance(ctrl_range, list) or len(ctrl_range) != 2:
                raise ValueError("[mujocoAutoActuators] autoActuators.ctrlRange must be a 2-number array")
            parts = [float(part) for part in ctrl_range]
        if len(parts) != 2:
            raise ValueError("[mujocoAutoActuators] autoActuators.ctrlRange must contain two finite numbers")
        ctrl_range = f"{parts[0]} {parts[1]}"
    else:
        ctrl_range = None
    return {
        "enabled": raw.get("enabled", True),
        "joint_names": joint_names,
        "exclude": {str(name).strip() for name in exclude},
        "strip_suffix": str(raw.get("stripJointNameSuffix", raw.get("nameSuffixToStrip")) or "_joint"),
        "ctrl_range": ctrl_range,
        "ctrl_range_from_joint": raw.get("ctrlRangeFromJoint", True),
        "gear": raw.get("gear"),
        "class_name": raw.get("className", raw.get("class")),
    }


def inject_auto_actuators(xml, raw_config, label="MuJoCo XML"):
    """Mirror mujocoAutoActuators.injectAutoActuators: add <motor> lines for
    hinge/slide joints when the XML has no actuators of its own."""
    options = _auto_actuator_options(raw_config)
    if not options["enabled"]:
        return xml
    stripped = _strip_xml_comments(xml)
    if _ACTUATOR_TAG_PATTERN.search(stripped):
        return xml

    joints = []
    for match in re.finditer(r"<\s*joint\b([^<>]*)>", stripped, re.IGNORECASE):
        attrs = _read_xml_attrs(match.group(1))
        name = (attrs.get("name") or "").strip()
        if not name:
            continue
        joint_type = (attrs.get("type") or "hinge").strip().lower()
        if joint_type not in ("hinge", "slide"):
            continue
        joints.append((name, attrs))

    by_name = {}
    for name, attrs in joints:
        if name in by_name:
            raise ValueError(f'[mujocoAutoActuators] {label} has duplicate joint "{name}"')
        by_name[name] = attrs

    if options["joint_names"] is not None:
        selected = []
        for wanted in options["joint_names"]:
            if wanted not in by_name:
                raise ValueError(
                    f'[mujocoAutoActuators] {label} is missing configured joint "{wanted}"'
                )
            selected.append((wanted, by_name[wanted]))
    else:
        selected = joints
    selected = [(name, attrs) for name, attrs in selected if name not in options["exclude"]]
    if not selected:
        return xml

    lines = []
    for name, attrs in selected:
        motor_name = name
        suffix = options["strip_suffix"]
        if suffix and motor_name.endswith(suffix):
            motor_name = motor_name[: -len(suffix)] or motor_name
        motor_attrs = [f'name="{_xml_attr(motor_name)}"', f'joint="{_xml_attr(name)}"']
        ctrl_range = options["ctrl_range"]
        if ctrl_range is None and options["ctrl_range_from_joint"]:
            ctrl_range = (attrs.get("actuatorfrcrange") or "").strip() or None
        if ctrl_range:
            motor_attrs.append(f'ctrlrange="{_xml_attr(ctrl_range)}"')
        if options["gear"]:
            motor_attrs.append(f'gear="{_xml_attr(options["gear"])}"')
        if options["class_name"]:
            motor_attrs.append(f'class="{_xml_attr(options["class_name"])}"')
        lines.append(f"    <motor {' '.join(motor_attrs)} />")
    content = "\n".join(lines)

    if re.search(r"<\s*actuator\b[^>]*/\s*>", xml, re.IGNORECASE):
        return re.sub(
            r"<\s*actuator\b[^>]*/\s*>",
            f"<actuator>\n{content}\n  </actuator>",
            xml,
            count=1,
            flags=re.IGNORECASE,
        )
    if re.search(r"<\s*/\s*actuator\s*>", xml, re.IGNORECASE):
        return re.sub(
            r"<\s*/\s*actuator\s*>", f"{content}\n  </actuator>", xml, count=1, flags=re.IGNORECASE
        )
    block = f"  <actuator>\n{content}\n  </actuator>"
    if re.search(r"<\s*sensor\b", xml, re.IGNORECASE):
        return re.sub(r"<\s*sensor\b", f"{block}\n  <sensor", xml, count=1, flags=re.IGNORECASE)
    if re.search(r"<\s*/\s*mujoco\s*>", xml, re.IGNORECASE):
        return re.sub(r"<\s*/\s*mujoco\s*>", f"{block}\n</mujoco>", xml, count=1, flags=re.IGNORECASE)
    raise ValueError("[mujocoAutoActuators] cannot inject actuators: </mujoco> is missing")
