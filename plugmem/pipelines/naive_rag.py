"""Naive flat-RAG baseline.

Treats every (observation, action) step as a single flat fact. No
trajectory segmentation, no subgoals, no procedural memories, no
consolidation. Retrieve is plain top-K embedding similarity over the
flat fact set.

Shipped as a baseline so researchers can A/B PlugMem's structured graph
against a no-frills RAG implementation on the same observation stream
by changing one binding.

The fact set is stored under the existing graph's ``semantic`` chroma
collection — no new collections — so existing tooling (Browse tab,
trace recorder, etc.) keeps working. Tags are left empty, episodic /
procedural / subgoal collections stay untouched.
"""
from __future__ import annotations

import time as _time
from typing import Any, Dict, List

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
from plugmem.core.pipeline_trace import record_llm_step
from plugmem.pipelines.base import MemoryPipeline


_DEFAULT_REASONING_TEMPLATE = (
    "You are answering a question with the help of retrieved facts.\n\n"
    "Question / observation:\n{observation}\n\n"
    "Retrieved facts:\n{facts}\n\n"
    "Answer concisely using only the facts above."
)


class NaiveRAGPipeline(MemoryPipeline):
    """Flat embedding-based fact store + top-K retrieval. No graph structure."""

    name = "naive-rag"
    description = (
        "Baseline RAG: stores each step's observation (and structured semantic "
        "memories) as a flat fact in the semantic collection — no episodic, "
        "procedural, or subgoal nodes. Retrieve is top-K embedding similarity "
        "over the fact set. Reason runs a single LLM call. Consolidate is a "
        "no-op. Useful as an A/B baseline against the default pipeline."
    )

    # ------------------------------------------------------------------ #
    # Ingest: flatten everything into the semantic store.
    # ------------------------------------------------------------------ #

    def ingest(self, graph: MemoryGraph, body: MemoryInsertRequest) -> MemoryInsertResponse:
        facts: List[str] = []

        if body.mode == "trajectory":
            if not body.steps:
                raise ValueError("'steps' is required for trajectory mode")
            if body.goal:
                facts.append(f"Goal: {body.goal}")
            for step in body.steps:
                if step.observation:
                    facts.append(step.observation)
                if step.action:
                    facts.append(f"Action: {step.action}")
        else:
            if body.semantic:
                for sem in body.semantic:
                    if sem.semantic_memory:
                        facts.append(sem.semantic_memory)
            if body.episodic:
                for traj in body.episodic:
                    for step in traj:
                        if step.observation:
                            facts.append(step.observation)

        # Persist each fact as a semantic node. No tags, no episodic linkage.
        for text in facts:
            if not text.strip():
                continue
            embedding = graph.embedder.embed(text)
            sem_id = len(graph.semantic_nodes)
            from plugmem.core.graph_node import SemanticNode
            node = SemanticNode(
                semantic_id=sem_id,
                semantic_memory_str=text,
                embedding=embedding,
                time=graph.semantic_time,
                session_id=body.session_id,
            )
            graph.semantic_nodes.append(node)
            graph.semantic_id2node[sem_id] = node
            emb_list = embedding if isinstance(embedding, list) else embedding.tolist()
            graph.storage.add_semantic(
                graph.graph_id,
                semantic_id=sem_id,
                text=text,
                embedding=emb_list,
                tags=[],
                tag_ids=[],
                time=graph.semantic_time,
                session_id=body.session_id,
                episodic_ids=[],
                bro_semantic_ids=[],
            )
            graph.semantic_time += 1

        stats = graph.storage.get_graph_stats(graph.graph_id)
        return MemoryInsertResponse(status="ok", stats=stats)

    # ------------------------------------------------------------------ #
    # Retrieve: top-K embedding similarity. No tag voting, no planning.
    # ------------------------------------------------------------------ #

    def retrieve(self, graph: MemoryGraph, body: RetrieveRequest) -> RetrieveResponse:
        topk = self._topk_facts(graph, body.observation or "", k=5)
        facts_block = "\n".join(f"- {t}" for t in topk) if topk else "(no facts in memory)"
        variables = {"observation": body.observation or "", "facts": facts_block}
        rendered = _DEFAULT_REASONING_TEMPLATE.format(**variables)
        messages = [{"role": "user", "content": rendered}]
        return RetrieveResponse(
            mode="semantic_memory",
            reasoning_prompt=messages,
            variables=variables,
        )

    # ------------------------------------------------------------------ #
    # Reason: retrieve + one LLM call.
    # ------------------------------------------------------------------ #

    def reason(self, graph: MemoryGraph, body: ReasonRequest) -> ReasonResponse:
        retrieve_body = RetrieveRequest(
            observation=body.observation or "",
            goal=body.goal, subgoal=body.subgoal, state=body.state,
            task_type=body.task_type, time=body.time, mode=body.mode,
            session_id=body.session_id,
        )
        r = self.retrieve(graph, retrieve_body)

        reasoning_llm = getattr(graph, "reasoning_llm", graph.llm)
        started = _time.monotonic()
        reasoning = reasoning_llm.complete(messages=r.reasoning_prompt)
        latency_ms = int((_time.monotonic() - started) * 1000)
        record_llm_step(
            name="reason_llm_call",
            llm=reasoning_llm,
            variables={"observation": body.observation or "", "mode": r.mode},
            response=reasoning,
            parsed={"reasoning": reasoning},
            latency_ms=latency_ms,
        )
        return ReasonResponse(mode=r.mode, reasoning=reasoning, reasoning_prompt=r.reasoning_prompt)

    # ------------------------------------------------------------------ #
    # Consolidate: no-op — flat RAG doesn't merge.
    # ------------------------------------------------------------------ #

    def consolidate(self, graph: MemoryGraph, body: ConsolidateRequest) -> ConsolidateResponse:
        return ConsolidateResponse(status="ok", stats={"skipped": 1})

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _topk_facts(self, graph: MemoryGraph, query: str, *, k: int) -> List[str]:
        from plugmem.clients.embedding import get_similarity
        if not query or not graph.semantic_nodes:
            return [n.get_semantic_memory() for n in graph.semantic_nodes[:k]]
        qv = graph.embedder.embed(query)
        scored: List[tuple] = []
        for n in graph.semantic_nodes:
            if not getattr(n, "is_active", True):
                continue
            try:
                score = get_similarity(qv, n.embedding)
            except Exception:
                score = 0.0
            scored.append((score, n))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [n.get_semantic_memory() for _, n in scored[:k]]
