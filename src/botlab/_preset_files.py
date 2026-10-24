"""Resolve a resource preset id to its (kind, config) from a project file or the built-ins."""

from __future__ import annotations

from pathlib import Path

import yaml

from ._contract import PRESET_FILE_SUFFIX, RESOURCE_KINDS
from ._errors import BotLabError
from ._json import json_value
from ._presets import PRESETS


def _parse_preset_file(preset_path: Path) -> tuple[str, dict]:
    with preset_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise BotLabError(f"A BotLab resource preset must be a mapping: {preset_path}")
    kind = data.pop("kind", None)
    if not isinstance(kind, str) or kind not in RESOURCE_KINDS:
        raise BotLabError(
            f"Resource preset kind must be one of: {', '.join(sorted(RESOURCE_KINDS))} "
            f"in {preset_path}"
        )
    return kind, data


def _preset_lookup(project_dir: Path):
    cache: dict[str, tuple[str, dict, bool]] = {}

    def lookup(preset_id: str):
        if preset_id in cache:
            return cache[preset_id]
        if preset_id.endswith(PRESET_FILE_SUFFIX):
            if "/" in preset_id:
                raise BotLabError(
                    f'BotLab resource preset "{preset_id}" must be a flat '
                    f'"<name>{PRESET_FILE_SUFFIX}" file next to controller.py.'
                )
            preset_path = project_dir / preset_id
            if not preset_path.is_file():
                raise BotLabError(
                    f'BotLab resource preset file "{preset_id}" is not part of this project.'
                )
            kind, config = _parse_preset_file(preset_path)
            entry = (kind, json_value(config, f"preset {preset_id}"), True)
        else:
            built_in = PRESETS.get(preset_id)
            if built_in is None:
                return None
            entry = (
                built_in["kind"],
                json_value(built_in.get("config", {}), f"preset {preset_id}"),
                True,
            )
        cache[preset_id] = entry
        return entry

    return lookup
