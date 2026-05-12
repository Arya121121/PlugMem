"""Process-wide registry of memory pipelines.

Pipelines self-register at import time. Lookup is by name. The first
pipeline registered with the name ``plugmem-default`` is the fallback
for any graph that doesn't have an explicit binding.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from plugmem.pipelines.base import MemoryPipeline

logger = logging.getLogger(__name__)


DEFAULT_PIPELINE_NAME = "plugmem-default"

_REGISTRY: Dict[str, MemoryPipeline] = {}


def register(pipeline: MemoryPipeline) -> None:
    """Add a pipeline to the registry (last write wins on name collision)."""
    if not pipeline.name or pipeline.name == "abstract":
        raise ValueError(f"Pipeline {pipeline!r} has no name set.")
    if pipeline.name in _REGISTRY:
        logger.warning(
            "Pipeline name collision on '%s'; replacing %r with %r",
            pipeline.name, _REGISTRY[pipeline.name], pipeline,
        )
    _REGISTRY[pipeline.name] = pipeline
    logger.info("Registered pipeline: %s", pipeline.name)


def get(name: str) -> MemoryPipeline:
    """Resolve a pipeline by name. Falls back to ``plugmem-default`` if missing."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if DEFAULT_PIPELINE_NAME in _REGISTRY:
        logger.warning(
            "Pipeline '%s' not registered; falling back to '%s'.",
            name, DEFAULT_PIPELINE_NAME,
        )
        return _REGISTRY[DEFAULT_PIPELINE_NAME]
    raise KeyError(f"Pipeline '{name}' not registered (and no default available)")


def list_pipelines() -> List[Dict[str, str]]:
    """Return [{name, description}] for every registered pipeline."""
    return [
        {"name": p.name, "description": p.description}
        for p in sorted(_REGISTRY.values(), key=lambda x: x.name)
    ]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def clear_registry() -> None:
    """Wipe the registry. Only used by hot-reload — never call from user code."""
    _REGISTRY.clear()


def reload_pipeline_modules() -> List[str]:
    """Re-import every loaded ``plugmem.pipelines.*`` module so on-disk
    edits take effect without restarting the server.

    Order: children first (deepest module names), package ``__init__``
    last. The package's ``__init__`` runs ``register(...)`` for each
    built-in pipeline, so by the time it executes the children modules
    already hold the latest class definitions. ``clear_registry()`` is
    called first so stale entries don't linger if a pipeline is renamed
    or removed.

    Returns the list of module names that were successfully reloaded.

    Caution: this is in-process Python reload. Existing references held
    by long-lived objects (e.g., graph instances that captured a
    pipeline at bind time) keep pointing at the old class. Routes that
    resolve via :func:`get` see the new class on the next call.
    """
    import importlib
    import sys

    targets = [
        name for name in list(sys.modules)
        if name == "plugmem.pipelines" or name.startswith("plugmem.pipelines.")
    ]
    # Deepest first, package itself last.
    targets.sort(key=lambda n: (n.count("."), n), reverse=True)

    clear_registry()
    reloaded: List[str] = []
    for name in targets:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
            reloaded.append(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Reload failed for %s: %s", name, e)
    return reloaded
