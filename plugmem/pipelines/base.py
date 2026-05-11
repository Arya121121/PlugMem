"""Memory pipeline protocol.

A *pipeline* is the algorithm that turns trajectory steps into stored
memory, retrieves them, reasons over them, and consolidates them.
Different research baselines (PlugMem's graph-based algorithm, naive
RAG, simple memorize-everything, etc.) implement the same four-method
interface so the routes can dispatch generically and a single graph can
be A/B'd across implementations by changing one binding.

The default ``plugmem-default`` pipeline wraps the original PlugMem
algorithm exactly — same prompts, same control flow, same data model.
Alternative pipelines are free to ignore parts of MemoryGraph (e.g. a
RAG baseline that only uses the semantic collection) or layer their
own state on top.

Pipelines are stateless; per-graph state lives on the
:class:`MemoryGraph` instance passed into every method. Lookup happens
via :mod:`plugmem.pipelines.registry`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugmem.api.schemas import (
        ConsolidateRequest,
        ConsolidateResponse,
        MemoryInsertRequest,
        MemoryInsertResponse,
        ReasonRequest,
        ReasonResponse,
        RetrieveRequest,
        RetrieveResponse,
    )
    from plugmem.core.memory_graph import MemoryGraph


class MemoryPipeline:
    """Base class for pluggable memory algorithms.

    Subclasses set ``name`` (the lookup key) and ``description``, and
    override whichever methods they support. Anything not overridden
    raises :class:`NotImplementedError`; the routes catch that and
    return a 501.
    """

    name: str = "abstract"
    description: str = ""

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (optional)
    # ------------------------------------------------------------------ #

    def on_graph_create(self, graph: "MemoryGraph") -> None:
        """Called once when a fresh graph is bound to this pipeline."""

    def on_graph_delete(self, graph_id: str) -> None:
        """Called when a graph is being deleted (after route already removed it)."""

    # ------------------------------------------------------------------ #
    # The four required operations
    # ------------------------------------------------------------------ #

    def ingest(
        self, graph: "MemoryGraph", body: "MemoryInsertRequest",
    ) -> "MemoryInsertResponse":
        raise NotImplementedError(f"{self.name}: ingest not implemented")

    def retrieve(
        self, graph: "MemoryGraph", body: "RetrieveRequest",
    ) -> "RetrieveResponse":
        raise NotImplementedError(f"{self.name}: retrieve not implemented")

    def reason(
        self, graph: "MemoryGraph", body: "ReasonRequest",
    ) -> "ReasonResponse":
        raise NotImplementedError(f"{self.name}: reason not implemented")

    def consolidate(
        self, graph: "MemoryGraph", body: "ConsolidateRequest",
    ) -> "ConsolidateResponse":
        raise NotImplementedError(f"{self.name}: consolidate not implemented")


__all__ = ["MemoryPipeline"]
