"""Processor implementations for the local plan executor.

Each group module exposes a ``REGISTRY`` mapping plan node types to processor
classes; ``build_registry`` merges them. A missing group module fails loudly —
the local runtime is only complete when every group is ported.
"""

_GROUP_MODULES = ("_parts", "_math", "_temporal", "_modes", "_output", "_model")


def build_registry():
    from importlib import import_module

    from .._observations import REGISTRY as observation_registry

    registry = dict(observation_registry)
    for name in _GROUP_MODULES:
        module = import_module(f".{name}", __package__)
        registry.update(module.REGISTRY)
    return registry
