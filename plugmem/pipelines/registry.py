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
