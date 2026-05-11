"""Memory pipeline implementations + a process-wide registry.

Pipelines self-register at package import. Researchers add a baseline
by writing a class in this package, calling :func:`register` on it, and
adding the module to the imports below.
"""
from __future__ import annotations

from plugmem.pipelines.base import MemoryPipeline
from plugmem.pipelines.default import PlugMemDefaultPipeline
from plugmem.pipelines.naive_rag import NaiveRAGPipeline
from plugmem.pipelines.spec_driven import SpecDrivenPipeline
from plugmem.pipelines.registry import (
    DEFAULT_PIPELINE_NAME,
    get,
    is_registered,
    list_pipelines,
    register,
)

# Register built-ins. Names are distinct so registration order is just docs.
register(PlugMemDefaultPipeline())
register(NaiveRAGPipeline())
register(SpecDrivenPipeline())

__all__ = [
    "MemoryPipeline",
    "DEFAULT_PIPELINE_NAME",
    "register",
    "get",
    "list_pipelines",
    "is_registered",
]
