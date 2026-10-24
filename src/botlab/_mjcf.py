"""Local MJCF dependency closure shared by validation and native staging."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ._host import BotLabError


def mjcf_dependencies(source: Path, *, root: Path | None = None, transform=None,
                      entry_transform=None, _model_stack=()) -> tuple[dict[Path, ET.Element], set[Path]]:
    """Collect includes and file assets without walking unrelated directories.

    Include lookup follows MuJoCo: main XML directory first, then the including
    directory. Asset paths use the main XML's compiler directories. When packaging,
    ``root`` bounds both lexical paths and symlink targets to the project.
    """
    source = Path(os.path.abspath(source))
    if source.resolve() in _model_stack:
        raise BotLabError(f'MJCF attached model cycle: "{source}"')
    base = source.parent
    documents: dict[Path, ET.Element] = {}
    assets: set[Path] = set()
    compiler: dict[str, str] = {}
    elements: list[ET.Element] = []

    def checked(path: Path) -> Path:
        path = Path(os.path.abspath(path))
        if root is not None and (not path.is_relative_to(root.resolve())
                                 or not path.resolve().is_relative_to(root.resolve())):
            raise BotLabError(f'MJCF dependency escapes the project: "{path}"')
        if not path.is_file():
            raise BotLabError(f'MJCF dependency does not exist: "{path}"')
        return path

    def visit(path: Path, stack: tuple[Path, ...]):
        path = checked(path)
        if path.resolve() in {item.resolve() for item in stack}:
            raise BotLabError(f'MJCF include cycle: {" -> ".join(map(str, (*stack, path)))}')
        if path in documents:
            return
        text = path.read_text(encoding="utf-8")
        if transform is not None:
            text = transform(text)
        if path == source and entry_transform is not None:
            text = entry_transform(text)
        try:
            document = ET.fromstring(text)
        except ET.ParseError as error:
            raise BotLabError(f'Invalid MJCF XML "{path}": {error}') from error
        documents[path] = document
        for element in document.iter():
            if element.tag == "include":
                file = element.get("file")
                if not file:
                    raise BotLabError(f'MJCF include has no file in "{path}"')
                if root is not None and Path(file).is_absolute():
                    raise BotLabError(f'MJCF include must be project-relative: "{file}"')
                candidate = base / file
                if not candidate.is_file():
                    candidate = path.parent / file
                candidate = checked(candidate)
                # Explicit main-relative paths preserve resolution after staging.
                element.set("file", os.path.relpath(candidate, base))
                visit(candidate, (*stack, path))
            elif element.tag == "compiler":
                compiler.update(element.attrib)
            else:
                elements.append(element)

    visit(source, ())
    for element in elements:
        if element.tag not in {"mesh", "texture", "hfield", "skin", "model", "flexcomp"}:
            continue
        keys = ("file", "filename")
        if element.tag == "texture":
            keys += ("fileleft", "fileright", "fileup", "filedown", "filefront", "fileback")
        for key in keys:
            file = element.get(key)
            if not file:
                continue
            if compiler.get("strippath") == "true":
                file = file.replace("\\", "/").rsplit("/", 1)[-1]
            directory = ""
            if element.tag in {"mesh", "hfield"}:
                directory = compiler.get("meshdir", compiler.get("assetdir", ""))
            elif element.tag == "texture":
                directory = compiler.get("texturedir", compiler.get("assetdir", ""))
            if root is not None and (Path(directory).is_absolute() or Path(file).is_absolute()):
                raise BotLabError(f'MJCF asset must be project-relative: "{directory}/{file}"')
            candidate = checked(base / directory / file)
            if element.tag == "model":
                # Attached MJCF models have their own compiler/include scope.
                nested, nested_assets = mjcf_dependencies(
                    candidate, root=root, transform=transform,
                    _model_stack=(*_model_stack, source.resolve()),
                )
                documents.update(nested)
                assets.update(nested_assets)
            else:
                assets.add(candidate)
    return documents, assets
