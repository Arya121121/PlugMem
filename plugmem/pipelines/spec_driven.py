"""User-defined memory pipeline driven by a per-graph YAML spec.

Phase 6.2 MVP: implements ``retrieve`` only. Other phases delegate to
:class:`PlugMemDefaultPipeline` so researchers can swap retrieve without
re-implementing ingest / reason / consolidate.

Spec file path: ``{PROMPTS_DIR}/{graph_id}.pipeline.yaml`` (reuses the
existing ``PROMPTS_DIR`` env var). Missing file → 422 from the route.

Node types in the MVP:

- ``Input``       — phase entry. Outputs = the request body fields.
- ``Output``      — phase exit. Inputs = the response fields
                    (mode, reasoning_prompt, variables).
- ``LLMCall``     — calls a registered prompt via PromptRegistry,
                    records to the trace recorder, returns {raw, parsed}.
                    ``parsed`` is best-effort: a small set of known
                    prompts (get_plan, get_mode) have parsers; others
                    return {text: <raw>}.
- ``PromptRender`` — renders a template into messages without an LLM call.
- ``Constant``    — emits a fixed value from config.value.

Inputs reference other nodes via ``"node_id.port"`` strings. A one-off
literal can be written inline as ``{ const: <value> }``.
"""
from __future__ import annotations

import ast as _ast
import json
import os
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from plugmem.api.schemas import RetrieveRequest, RetrieveResponse
from plugmem.core.memory_graph import MemoryGraph
from plugmem.core.pipeline_trace import record_llm_step
from plugmem.pipelines.base import MemoryPipeline
from plugmem.pipelines.default import PlugMemDefaultPipeline
from plugmem.prompts.registry import PromptRegistry

LLM_CALL_CAP = 20

NODE_TYPES = {"Input", "Output", "LLMCall", "PromptRender", "Constant"}


# ----------------------------------------------------------------------- #
# Data model
# ----------------------------------------------------------------------- #


@dataclass
class NodeSpec:
    id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineGraph:
    phase: str
    nodes: List[NodeSpec]


def _is_const_ref(ref: Any) -> bool:
    return isinstance(ref, dict) and "const" in ref


# ----------------------------------------------------------------------- #
# YAML loader (with validation)
# ----------------------------------------------------------------------- #


def load_yaml(path: Path) -> PipelineGraph:
    """Parse + validate a pipeline YAML. Raises ValueError on bad spec."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a top-level dict: {path}")
    phase = raw.get("phase", "retrieve")
    if phase != "retrieve":
        raise ValueError(
            f"MVP only supports phase=retrieve, got {phase!r}. "
            f"Other phases delegate to plugmem-default."
        )
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("'nodes' must be a list")

    nodes: List[NodeSpec] = []
    seen: set = set()
    for entry in raw_nodes:
        if not isinstance(entry, dict):
            raise ValueError(f"Bad node entry: {entry!r}")
        nid = entry.get("id")
        if not isinstance(nid, str) or not nid:
            raise ValueError(f"Node missing 'id' or 'id' not a string: {entry!r}")
        if nid in seen:
            raise ValueError(f"Duplicate node id: {nid!r}")
        seen.add(nid)
        ntype = entry.get("type")
        if ntype not in NODE_TYPES:
            raise ValueError(
                f"Node {nid!r}: unknown type {ntype!r} "
                f"(allowed: {sorted(NODE_TYPES)})"
            )
        cfg = entry.get("config") or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Node {nid!r}: 'config' must be a dict")
        ins = entry.get("inputs") or {}
        if not isinstance(ins, dict):
            raise ValueError(f"Node {nid!r}: 'inputs' must be a dict")
        nodes.append(NodeSpec(id=nid, type=ntype, config=cfg, inputs=ins))

    type_counts: Dict[str, int] = {}
    for n in nodes:
        type_counts[n.type] = type_counts.get(n.type, 0) + 1
    if type_counts.get("Input", 0) != 1:
        raise ValueError("Exactly one Input node is required")
    if type_counts.get("Output", 0) != 1:
        raise ValueError("Exactly one Output node is required")

    # Validate all input references resolve.
    for n in nodes:
        for port, ref in n.inputs.items():
            if _is_const_ref(ref):
                continue
            if not isinstance(ref, str) or "." not in ref:
                raise ValueError(
                    f"Node {n.id!r} input {port!r}: bad reference {ref!r} "
                    f"(expected 'node_id.port' or {{const: value}})"
                )
            src_id = ref.split(".", 1)[0]
            if src_id not in seen:
                raise ValueError(
                    f"Node {n.id!r} input {port!r}: references unknown "
                    f"node {src_id!r}"
                )

    _topo_sort(nodes)  # raises ValueError on cycle / self-ref
    return PipelineGraph(phase=phase, nodes=nodes)


def _topo_sort(nodes: List[NodeSpec]) -> List[NodeSpec]:
    """Kahn's algorithm. Returns execution order. Raises on cycles."""
    by_id = {n.id: n for n in nodes}
    incoming: Dict[str, set] = {n.id: set() for n in nodes}
    for n in nodes:
        for ref in n.inputs.values():
            if _is_const_ref(ref):
                continue
            src_id = ref.split(".", 1)[0]
            if src_id == n.id:
                raise ValueError(f"Node {n.id!r} references itself")
            incoming[n.id].add(src_id)

    queue = [nid for nid, deps in incoming.items() if not deps]
    order: List[NodeSpec] = []
    while queue:
        nid = queue.pop(0)
        order.append(by_id[nid])
        for other_nid, deps in incoming.items():
            if nid in deps:
                deps.remove(nid)
                if not deps:
                    queue.append(other_nid)
    if len(order) != len(nodes):
        remaining = sorted(set(by_id) - {n.id for n in order})
        raise ValueError(f"Pipeline has a cycle (unresolved nodes: {remaining})")
    return order


# ----------------------------------------------------------------------- #
# Parsers for known prompts (so downstream nodes can use structured outputs)
# ----------------------------------------------------------------------- #


def _parse_get_plan(response: str) -> Dict[str, Any]:
    tags_match = re.search(r"\*\*Tags:\*\*\s*(.*)\n", response or "")
    tags: List[str] = []
    if tags_match:
        raw = tags_match.group(1).strip()
        try:
            tags = json.loads(raw)
        except json.JSONDecodeError:
            try:
                tags = _ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                tags = []
    sub_match = re.search(r"### Next Subgoal\n(.*)", response or "", re.S)
    next_subgoal = sub_match.group(1).strip() if sub_match else ""
    return {"next_subgoal": next_subgoal, "query_tags": tags}


def _parse_get_mode(response: str) -> Dict[str, Any]:
    m = re.search(r"### Memory Type\n(.*)", response or "")
    return {"mode": m.group(1).strip() if m else "semantic_memory"}


PARSERS = {
    "get_plan": _parse_get_plan,
    "get_mode": _parse_get_mode,
}


# ----------------------------------------------------------------------- #
# Executor
# ----------------------------------------------------------------------- #


class PipelineExecutor:
    """Runs one pipeline graph against a MemoryGraph + LLM router."""

    def __init__(self, graph: PipelineGraph, memory_graph: MemoryGraph):
        self.graph = graph
        self.memory_graph = memory_graph
        self.llm_call_count = 0
        self._order = _topo_sort(self.graph.nodes)

    def run(self, phase_inputs: Dict[str, Any]) -> Dict[str, Any]:
        env: Dict[str, Dict[str, Any]] = {}
        for node in self._order:
            if node.type == "Input":
                env[node.id] = dict(phase_inputs)
                continue
            resolved = self._resolve_inputs(node, env)
            if node.type == "Output":
                env[node.id] = resolved
            elif node.type == "Constant":
                env[node.id] = {"value": node.config.get("value")}
            elif node.type == "PromptRender":
                env[node.id] = self._render_prompt(node, resolved)
            elif node.type == "LLMCall":
                env[node.id] = self._call_llm(node, resolved)
            else:
                raise ValueError(f"Unknown node type {node.type!r}")

        out_node = next(n for n in self.graph.nodes if n.type == "Output")
        return env[out_node.id]

    def _resolve_inputs(self, node: NodeSpec, env: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for port, ref in node.inputs.items():
            if _is_const_ref(ref):
                out[port] = ref["const"]
                continue
            parts = ref.split(".")
            src_id, path = parts[0], parts[1:]
            src_env = env.get(src_id)
            if src_env is None:
                raise RuntimeError(
                    f"Node {node.id}.{port}: source {src_id!r} not yet evaluated"
                )
            val: Any = src_env
            for part in path:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    raise RuntimeError(
                        f"Node {node.id}.{port}: cannot resolve {ref!r} "
                        f"(missing key {part!r})"
                    )
            out[port] = val
        return out

    def _registry(self) -> PromptRegistry:
        return self.memory_graph.prompts or PromptRegistry()

    def _render_prompt(self, node: NodeSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt_name = node.config.get("prompt")
        if not prompt_name:
            raise ValueError(f"PromptRender {node.id!r}: config.prompt required")
        messages = self._registry().render_messages(
            prompt_name, inputs, graph_id=self.memory_graph.graph_id,
        )
        return {"messages": messages}

    def _call_llm(self, node: NodeSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_call_count >= LLM_CALL_CAP:
            raise RuntimeError(
                f"LLM call cap ({LLM_CALL_CAP}) exceeded for pipeline "
                f"{self.graph.phase!r}"
            )
        self.llm_call_count += 1
        prompt_name = node.config.get("prompt")
        role = node.config.get("role", "default")
        if not prompt_name:
            raise ValueError(f"LLMCall {node.id!r}: config.prompt required")
        llm = self.memory_graph._router.for_role(role)
        messages = self._registry().render_messages(
            prompt_name, inputs, graph_id=self.memory_graph.graph_id,
        )
        started = _time.monotonic()
        response = llm.complete(messages=messages)
        latency_ms = int((_time.monotonic() - started) * 1000)
        parser = PARSERS.get(prompt_name)
        parsed = parser(response) if parser else {"text": response}
        record_llm_step(
            name=node.id,
            llm=llm,
            variables=inputs,
            response=response,
            parsed=parsed,
            latency_ms=latency_ms,
        )
        return {"raw": response, "parsed": parsed}


# ----------------------------------------------------------------------- #
# Pipeline class
# ----------------------------------------------------------------------- #


class SpecDrivenPipeline(MemoryPipeline):
    """Interprets a per-graph YAML pipeline at runtime (retrieve only, MVP)."""

    name = "spec-driven"
    description = (
        "Runs a user-authored YAML pipeline for the retrieve phase. Looks for "
        "{PROMPTS_DIR}/{graph_id}.pipeline.yaml; missing file → 422. Other "
        "phases (ingest / reason / consolidate) delegate to plugmem-default."
    )

    def __init__(self) -> None:
        self._default = PlugMemDefaultPipeline()

    # ---- retrieve runs the executor ----
    def retrieve(self, graph: MemoryGraph, body: RetrieveRequest) -> RetrieveResponse:
        spec = self._load_for_graph(graph.graph_id)
        executor = PipelineExecutor(spec, graph)
        out = executor.run(self._body_to_inputs(body))
        return RetrieveResponse(
            mode=out.get("mode") or "semantic_memory",
            reasoning_prompt=out.get("reasoning_prompt") or [],
            variables=out.get("variables") or {},
        )

    # ---- other phases delegate ----
    def ingest(self, graph, body):
        return self._default.ingest(graph, body)

    def reason(self, graph, body):
        return self._default.reason(graph, body)

    def consolidate(self, graph, body):
        return self._default.consolidate(graph, body)

    # ---- helpers ----
    @staticmethod
    def yaml_path_for(graph_id: str) -> Path:
        prompts_dir = os.getenv("PROMPTS_DIR", "./data/prompts")
        return Path(prompts_dir) / f"{graph_id}.pipeline.yaml"

    def _load_for_graph(self, graph_id: str) -> PipelineGraph:
        path = self.yaml_path_for(graph_id)
        if not path.exists():
            raise ValueError(
                f"No pipeline YAML for graph {graph_id!r} at {path}. "
                f"Write one or switch the pipeline binding back to plugmem-default."
            )
        return load_yaml(path)

    @staticmethod
    def _body_to_inputs(body: RetrieveRequest) -> Dict[str, Any]:
        return {
            "observation": body.observation or "",
            "goal": body.goal or "",
            "state": body.state or "",
            "subgoal": body.subgoal or "",
            "task_type": body.task_type or "",
            "time": body.time or "",
            "mode": body.mode,
        }
