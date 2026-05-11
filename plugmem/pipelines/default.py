"""PlugMem's original algorithm, packaged as a pipeline implementation.

This is a verbatim move of the logic that used to live inline in the
``/memories`` and ``/retrieve`` / ``/reason`` / ``/consolidate`` route
handlers. No behavior change — same prompts, same control flow, same
chroma collections, same audit writes. The spec lint + runtime trace
tests still cover it.

A different pipeline (e.g. NaiveRAG) implements the same four methods
with totally different internals.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Any, Dict

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
from plugmem.core.memory import Memory
from plugmem.core.memory_graph import MemoryGraph
from plugmem.core.pipeline_trace import record_llm_step
from plugmem.pipelines.base import MemoryPipeline


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_audit(
    graph,
    *,
    endpoint: str,
    body,
    audit: Dict[str, Any],
    mode: str,
    n_messages: int,
) -> None:
    """Best-effort audit write — never breaks the recall path."""
    try:
        graph.storage.add_recall(
            graph.graph_id,
            endpoint=endpoint,
            ts=_now_iso(),
            graph_time=graph.semantic_time,
            session_id=getattr(body, "session_id", None),
            observation=body.observation or "",
            goal=body.goal or "",
            subgoal=body.subgoal or "",
            state=body.state or "",
            task_type=body.task_type or "",
            mode=mode,
            next_subgoal=audit.get("next_subgoal", ""),
            query_tags=audit.get("query_tags", []),
            selected_semantic_ids=audit.get("selected_semantic_ids", []),
            selected_procedural_ids=audit.get("selected_procedural_ids", []),
            n_messages=n_messages,
        )
    except Exception:
        pass


class PlugMemDefaultPipeline(MemoryPipeline):
    """Original PlugMem: episodic + semantic + procedural graph with consolidation."""

    name = "plugmem-default"
    description = (
        "PlugMem's structured-memory graph. Ingest builds episodic / "
        "semantic / procedural / subgoal / tag nodes via the structuring "
        "LLM calls (get_subgoal / get_reward / get_state / get_semantic / "
        "get_procedural). Retrieve uses tag-based recall planning + "
        "embedding scoring. Consolidate merges high-similarity semantics."
    )

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def ingest(self, graph: MemoryGraph, body: MemoryInsertRequest) -> MemoryInsertResponse:
        if body.mode == "trajectory":
            return self._ingest_trajectory(graph, body)
        return self._ingest_structured(graph, body)

    def _ingest_trajectory(self, graph: MemoryGraph, body: MemoryInsertRequest) -> MemoryInsertResponse:
        if not body.goal:
            raise ValueError("'goal' is required for trajectory mode")
        if not body.steps:
            raise ValueError("'steps' is required for trajectory mode")

        mem = Memory(
            goal=body.goal,
            observation=body.steps[0].observation,
            llm=graph.llm,
            embedder=graph.embedder,
            time=graph.semantic_time,
            session_id=body.session_id,
        )
        for step in body.steps:
            mem.append(action_t0=step.action, observation_t1=step.observation)
        mem.close()
        graph.insert(mem)

        stats = graph.storage.get_graph_stats(graph.graph_id)
        return MemoryInsertResponse(status="ok", stats=stats)

    def _ingest_structured(self, graph: MemoryGraph, body: MemoryInsertRequest) -> MemoryInsertResponse:
        embedder = graph.embedder

        mem = Memory.__new__(Memory)
        mem.time = graph.semantic_time
        mem.session_id = body.session_id
        mem.llm = graph.llm
        mem.embedder = embedder
        mem.memory = {"goal": "", "episodic": [], "semantic": [], "procedural": []}
        mem.memory_embedding = {"semantic": [], "procedural": []}

        if body.episodic:
            for trajectory in body.episodic:
                mem.memory["episodic"].append([
                    {
                        "observation": step.observation,
                        "action": step.action,
                        "subgoal": step.subgoal,
                        "state": step.state,
                        "reward": step.reward,
                        "time": step.time or graph.semantic_time,
                    }
                    for step in trajectory
                ])

        if body.semantic:
            for sem in body.semantic:
                mem.memory["semantic"].append({
                    "semantic_memory": sem.semantic_memory,
                    "tags": sem.tags,
                })
                mem.memory_embedding["semantic"].append({
                    "semantic_memory": embedder.embed(sem.semantic_memory),
                    "tags": [embedder.embed(tag) for tag in sem.tags],
                })

        if body.procedural:
            for proc in body.procedural:
                mem.memory["procedural"].append({
                    "subgoal": proc.subgoal,
                    "procedural_memory": proc.procedural_memory,
                    "time": graph.semantic_time,
                    "return": proc.return_value,
                })
                mem.memory_embedding["procedural"].append({
                    "subgoal": embedder.embed(proc.subgoal),
                })

        graph.insert(mem)
        stats = graph.storage.get_graph_stats(graph.graph_id)
        return MemoryInsertResponse(status="ok", stats=stats)

    # ------------------------------------------------------------------ #
    # Retrieve
    # ------------------------------------------------------------------ #

    def retrieve(self, graph: MemoryGraph, body: RetrieveRequest) -> RetrieveResponse:
        audit: Dict[str, Any] = {}
        messages, variables, mode = graph.retrieve_memory(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            _audit=audit,
        )
        _write_audit(graph, endpoint="retrieve", body=body, audit=audit, mode=mode, n_messages=len(messages))
        return RetrieveResponse(mode=mode, reasoning_prompt=messages, variables=variables)

    # ------------------------------------------------------------------ #
    # Reason
    # ------------------------------------------------------------------ #

    def reason(self, graph: MemoryGraph, body: ReasonRequest) -> ReasonResponse:
        audit: Dict[str, Any] = {}
        messages, variables, mode = graph.retrieve_memory(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            _audit=audit,
        )

        reasoning_llm = getattr(graph, "reasoning_llm", graph.llm)
        started = _time.monotonic()
        reasoning = reasoning_llm.complete(messages=messages)
        latency_ms = int((_time.monotonic() - started) * 1000)
        record_llm_step(
            name="reason_llm_call",
            llm=reasoning_llm,
            variables={"observation": body.observation or "", "mode": mode},
            response=reasoning,
            parsed={"reasoning": reasoning},
            latency_ms=latency_ms,
        )
        _write_audit(graph, endpoint="reason", body=body, audit=audit, mode=mode, n_messages=len(messages))
        return ReasonResponse(mode=mode, reasoning=reasoning, reasoning_prompt=messages)

    # ------------------------------------------------------------------ #
    # Consolidate
    # ------------------------------------------------------------------ #

    def consolidate(self, graph: MemoryGraph, body: ConsolidateRequest) -> ConsolidateResponse:
        stats = graph.update_semantic_subgraph(
            merge_threshold=body.merge_threshold,
            max_merges_per_node=body.max_merges_per_node,
            max_candidates_per_tag=body.max_candidates_per_tag,
            max_total_candidates=body.max_total_candidates,
            min_credibility_to_keep_active=body.min_credibility_to_keep_active,
            credibility_decay=body.credibility_decay,
            only_update_recent_window=body.only_update_recent_window,
            allow_merge_with_common_episodic_nodes=body.allow_merge_with_common_episodic_nodes,
        )
        return ConsolidateResponse(status="ok", stats=stats)
