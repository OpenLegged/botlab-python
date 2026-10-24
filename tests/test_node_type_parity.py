"""Keep SUPPORTED_NODE_TYPES in lockstep with the engine's ProcessorRegistry.

``botlab build`` rejects plans whose node types are missing from the offline
whitelist, but the engine (BotLab's SimulationEngine.ts ProcessorRegistry) is
what actually runs them — the two lists drifted by eight types before anyone
noticed (projects built fine in the editor, then failed to package). These
tests parse the TypeScript registry directly and compare it against
``botlab._host.SUPPORTED_NODE_TYPES`` so the drift cannot silently recur.

The BotLab repository is looked up next to this repository (the monorepo
layout ``<workspace>/botlab-python`` + ``<workspace>/BotLab``) or through
``BOTLAB_ENGINE_REPO``; without it the parity test skips locally — the parser
itself stays covered by the inline-snippet unit tests below. With
``BOTLAB_REQUIRE_ENGINE_REPO=1`` a missing repository is a hard failure.
.github/workflows/ci.yml does not check out BotLab (it is a separate private
repository), so CI currently skips this test and does not set the variable;
set it once the workflow checks the engine repo out.

``test_whitelist_matches_local_processor_registries`` is the in-repo
counterpart: it needs no engine checkout and pins the whitelist to what the
native runtime's registries actually execute.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from botlab import _host

REPO_ROOT = Path(__file__).resolve().parents[1]

# The registry literal: keys are 'quoted strings' or [SYMBOLIC_CONSTANTS] that
# resolve to string constants defined elsewhere in the engine source.
# The type annotation contains "=>" so the header match may not exclude "=".
_REGISTRY_RE = re.compile(
    r"const\s+ProcessorRegistry.*?=\s*\{(.*?)\n\};",
    re.DOTALL,
)
_KEY_RE = re.compile(r"^\s*(?:'([^']+)'|\[([A-Z][A-Z0-9_]*)\])\s*:", re.MULTILINE)


def _engine_repository() -> Path | None:
    override = os.environ.get("BOTLAB_ENGINE_REPO", "").strip()
    candidates = [
        Path(override) if override else None,
        REPO_ROOT.parent / "BotLab",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        engine = candidate / "src" / "domain" / "runtime" / "engine" / "SimulationEngine.ts"
        if engine.is_file():
            return candidate
    return None


def _resolve_symbol(root: Path, symbol: str) -> str:
    """Find ``export const <symbol> = '...'`` under root and return the value."""
    pattern = re.compile(r"export\s+const\s+" + re.escape(symbol) + r"\s*=\s*'([^']+)'")
    for path in sorted(root.rglob("*.ts")):
        match = pattern.search(path.read_text(encoding="utf-8"))
        if match is not None:
            return match.group(1)
    raise AssertionError(f"node type constant {symbol} is not defined under {root}")


def parse_processor_registry(source: str, symbol_resolver) -> set[str]:
    """Extract the registry's node-type keys from SimulationEngine.ts source."""
    block = _REGISTRY_RE.search(source)
    if block is None:
        raise AssertionError("ProcessorRegistry literal not found in SimulationEngine.ts")
    types: set[str] = set()
    for literal, symbol in _KEY_RE.findall(block.group(1)):
        types.add(literal if literal else symbol_resolver(symbol))
    if not types:
        raise AssertionError("ProcessorRegistry literal contains no node type keys")
    return types


def test_registry_keys_match_the_offline_whitelist():
    repository = _engine_repository()
    if repository is None:
        message = "BotLab engine repository not found (set BOTLAB_ENGINE_REPO)"
        if os.environ.get("BOTLAB_REQUIRE_ENGINE_REPO") == "1":
            pytest.fail(message + "; BOTLAB_REQUIRE_ENGINE_REPO=1 forbids skipping")
        pytest.skip(message)
    engine_source = (
        repository / "src" / "domain" / "runtime" / "engine" / "SimulationEngine.ts"
    ).read_text(encoding="utf-8")
    engine_types = parse_processor_registry(
        engine_source, lambda symbol: _resolve_symbol(repository / "src", symbol)
    )
    whitelist = set(_host.SUPPORTED_NODE_TYPES)
    missing = engine_types - whitelist
    extra = whitelist - engine_types
    assert not missing, (
        "engine node types missing from botlab._host.SUPPORTED_NODE_TYPES "
        f"(botlab build would reject them): {sorted(missing)}"
    )
    assert not extra, (
        "SUPPORTED_NODE_TYPES entries the engine registry no longer defines "
        f"(stale whitelist entries): {sorted(extra)}"
    )


def test_parser_reads_literals_and_symbols():
    source = """
const ProcessorRegistry: Record<string, new () => INodeProcessor> = {
  // Observations
  'obs-qpos': ObservationProcessor,
  [CAMERA_OBSERVATION_NODE_TYPE]: ObservationProcessor,
  'proc-clock': ClockProcessor,
};
"""
    resolved = parse_processor_registry(
        source, lambda symbol: {"CAMERA_OBSERVATION_NODE_TYPE": "obs-camera"}[symbol]
    )
    assert resolved == {"obs-qpos", "obs-camera", "proc-clock"}


def test_parser_rejects_unresolved_symbols():
    source = """
const ProcessorRegistry: Record<string, new () => INodeProcessor> = {
  [MYSTERY_NODE_TYPE]: ObservationProcessor,
};
"""
    with pytest.raises(KeyError):
        parse_processor_registry(source, lambda symbol: (_ for _ in ()).throw(KeyError(symbol)))


def test_parser_requires_a_registry_literal():
    with pytest.raises(AssertionError, match="not found"):
        parse_processor_registry("export const nothing = 1;", lambda symbol: symbol)


# Node types the whitelist accepts that no local registry executes. Empty today;
# add an entry (with the reason) only for types validated but run elsewhere.
_ACCEPTED_BUT_NOT_EXECUTED: frozenset[str] = frozenset()


def test_whitelist_matches_local_processor_registries():
    pytest.importorskip("numpy")
    from botlab._runtime._processors import build_registry

    executed = set(build_registry())
    whitelist = set(_host.SUPPORTED_NODE_TYPES)
    assert whitelist - _ACCEPTED_BUT_NOT_EXECUTED == executed
