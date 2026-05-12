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
MAX_LOOP_ITERATIONS = 1000

NODE_TYPES = {
    "Input", "Output", "LLMCall", "PromptRender", "Constant",
    "ForEach", "Branch", "Compute", "StorageRead", "Embed",
}
STORAGE_COLLECTIONS = {"semantic", "procedural", "episodic"}
BODY_FORBIDDEN_TYPES = {"Input", "Output"}


# ----------------------------------------------------------------------- #
# Compute ops (whitelist)
# ----------------------------------------------------------------------- #


def _op_eq(i):       return {"value": i["a"] == i["b"]}
def _op_ne(i):       return {"value": i["a"] != i["b"]}
def _op_lt(i):       return {"value": i["a"] <  i["b"]}
def _op_gt(i):       return {"value": i["a"] >  i["b"]}
def _op_lte(i):      return {"value": i["a"] <= i["b"]}
def _op_gte(i):      return {"value": i["a"] >= i["b"]}
def _op_and(i):      return {"value": bool(i["a"]) and bool(i["b"])}
def _op_or(i):       return {"value": bool(i["a"]) or  bool(i["b"])}
def _op_not(i):      return {"value": not bool(i["a"])}
def _op_length(i):   return {"value": len(i["list"])}
def _op_contains(i): return {"value": i["item"] in i["list"]}
def _op_concat(i):   return {"value": i["a"] + i["b"]}


COMPUTE_OPS: Dict[str, Dict[str, Any]] = {
    "eq":       {"fn": _op_eq,       "inputs": ["a", "b"]},
    "ne":       {"fn": _op_ne,       "inputs": ["a", "b"]},
    "lt":       {"fn": _op_lt,       "inputs": ["a", "b"]},
    "gt":       {"fn": _op_gt,       "inputs": ["a", "b"]},
    "lte":      {"fn": _op_lte,      "inputs": ["a", "b"]},
    "gte":      {"fn": _op_gte,      "inputs": ["a", "b"]},
    "and":      {"fn": _op_and,      "inputs": ["a", "b"]},
    "or":       {"fn": _op_or,       "inputs": ["a", "b"]},
    "not":      {"fn": _op_not,      "inputs": ["a"]},
    "length":   {"fn": _op_length,   "inputs": ["list"]},
    "contains": {"fn": _op_contains, "inputs": ["list", "item"]},
    "concat":   {"fn": _op_concat,   "inputs": ["a", "b"]},
}


# ----------------------------------------------------------------------- #
# Data model
# ----------------------------------------------------------------------- #


@dataclass
class NodeSpec:
    id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    body: Optional[List["NodeSpec"]] = None  # only set for ForEach nodes


@dataclass
class PipelineGraph:
    phase: str
    nodes: List[NodeSpec]


def _is_const_ref(ref: Any) -> bool:
    return isinstance(ref, dict) and "const" in ref


def _resolve_ref(
    ref: Any,
    env: Dict[str, Dict[str, Any]],
    item_bindings: Dict[str, Any],
    *,
    hint: str = "",
) -> Any:
    """Resolve a single ref (string or const-dict) against env + item bindings.

    Lookup order:
      1. ``{const: <value>}``                → literal.
      2. ``<item_var>[.path]``               → item_bindings.
      3. ``<node_id>[.path]``                → env.

    ``hint`` is included in error messages for traceability.
    """
    if _is_const_ref(ref):
        return ref["const"]
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{hint}: bad reference {ref!r}")
    parts = ref.split(".")
    head, path = parts[0], parts[1:]
    if head in item_bindings:
        val: Any = item_bindings[head]
    elif head in env:
        val = env[head]
    else:
        raise ValueError(
            f"{hint}: source {head!r} not in env or item bindings (ref={ref!r})"
        )
    for part in path:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            raise ValueError(
                f"{hint}: cannot resolve {ref!r} — missing key {part!r}"
            )
    return val


# ----------------------------------------------------------------------- #
# YAML loader (with validation)
# ----------------------------------------------------------------------- #


def load_yaml(path: Path) -> PipelineGraph:
    """Parse + validate a pipeline YAML. Raises ValueError on bad spec."""
    return load_yaml_str(path.read_text(), source=str(path))


def load_yaml_str(text: str, *, source: str = "<inline>") -> PipelineGraph:
    """Same as :func:`load_yaml` but takes the YAML source as a string."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {source}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a top-level dict: {source}")
    phase = raw.get("phase", "retrieve")
    if phase != "retrieve":
        raise ValueError(
            f"MVP only supports phase=retrieve, got {phase!r}. "
            f"Other phases delegate to plugmem-default."
        )
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("'nodes' must be a list")

    nodes = [_parse_node(entry, context="top-level") for entry in raw_nodes]

    # Top-level structural constraints.
    seen: set = set()
    for n in nodes:
        if n.id in seen:
            raise ValueError(f"Duplicate node id: {n.id!r}")
        seen.add(n.id)
    type_counts: Dict[str, int] = {}
    for n in nodes:
        type_counts[n.type] = type_counts.get(n.type, 0) + 1
    if type_counts.get("Input", 0) != 1:
        raise ValueError("Exactly one Input node is required")
    if type_counts.get("Output", 0) != 1:
        raise ValueError("Exactly one Output node is required")

    # Validate references (and recursively for ForEach bodies). Cycle
    # detection happens via the topo sort below.
    _validate_refs(nodes, allowed_ids=seen, item_vars=set(), context="top-level")
    _topo_sort(nodes)

    # Recurse into ForEach / Branch bodies to validate refs + cycles.
    for n in nodes:
        if n.type == "ForEach":
            _validate_foreach(n, outer_ids=seen)
        elif n.type == "Branch":
            _validate_branch(n, outer_ids=seen)

    return PipelineGraph(phase=phase, nodes=nodes)


def _parse_node(entry: Any, *, context: str) -> NodeSpec:
    """Convert one YAML entry into a NodeSpec. Recurses for ForEach.body."""
    if not isinstance(entry, dict):
        raise ValueError(f"{context}: bad node entry: {entry!r}")
    nid = entry.get("id")
    if not isinstance(nid, str) or not nid:
        raise ValueError(f"{context}: node missing 'id' (or not a string): {entry!r}")
    ntype = entry.get("type")
    if ntype not in NODE_TYPES:
        raise ValueError(
            f"{context}: node {nid!r} has unknown type {ntype!r} "
            f"(allowed: {sorted(NODE_TYPES)})"
        )
    cfg = entry.get("config") or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{context}: node {nid!r} 'config' must be a dict")
    ins = entry.get("inputs") or {}
    if not isinstance(ins, dict):
        raise ValueError(f"{context}: node {nid!r} 'inputs' must be a dict")

    body: Optional[List[NodeSpec]] = None
    if ntype == "ForEach":
        # ForEach: validate config + body shape; recurse into body parse.
        if "items" not in ins:
            raise ValueError(f"{context}: ForEach {nid!r}: inputs.items is required")
        item_var = cfg.get("item_var")
        if not isinstance(item_var, str) or not item_var:
            raise ValueError(f"{context}: ForEach {nid!r}: config.item_var must be a non-empty string")
        _validate_output_decls(cfg.get("outputs"), label=f"ForEach {nid!r}", context=context)
        body = _parse_subgraph_body(entry, label=f"ForEach {nid!r}", context=context)

    elif ntype == "Branch":
        if "condition" not in ins:
            raise ValueError(f"{context}: Branch {nid!r}: inputs.condition is required")
        _validate_output_decls(
            cfg.get("outputs"), label=f"Branch {nid!r}", context=context, require_non_empty=True,
        )
        if "else_value" in cfg:
            ev = cfg["else_value"]
            if not (_is_const_ref(ev) or (isinstance(ev, str) and ev)):
                raise ValueError(
                    f"{context}: Branch {nid!r}: config.else_value must be a ref "
                    f"string or a {{const: value}} dict, got {ev!r}"
                )
        body = _parse_subgraph_body(entry, label=f"Branch {nid!r}", context=context)

    elif ntype == "Compute":
        op = cfg.get("op")
        if op not in COMPUTE_OPS:
            raise ValueError(
                f"{context}: Compute {nid!r}: op must be one of "
                f"{sorted(COMPUTE_OPS)}, got {op!r}"
            )
        required = set(COMPUTE_OPS[op]["inputs"])
        provided = set(ins.keys())
        missing = required - provided
        if missing:
            raise ValueError(
                f"{context}: Compute {nid!r} op={op!r}: requires inputs "
                f"{sorted(required)}, missing {sorted(missing)}"
            )

    elif ntype == "StorageRead":
        coll = cfg.get("collection")
        if coll not in STORAGE_COLLECTIONS:
            raise ValueError(
                f"{context}: StorageRead {nid!r}: config.collection must be "
                f"one of {sorted(STORAGE_COLLECTIONS)}, got {coll!r}"
            )
        # Per-collection input shape — same as retrieve_memory's call sites.
        required_inputs = {
            "semantic": {"query", "tags"},
            "procedural": {"subgoal"},
            "episodic": {"query"},
        }[coll]
        provided = set(ins.keys())
        missing = required_inputs - provided
        if missing:
            raise ValueError(
                f"{context}: StorageRead {nid!r} collection={coll!r}: "
                f"requires inputs {sorted(required_inputs)}, missing "
                f"{sorted(missing)}"
            )
        top_k = cfg.get("top_k", 5)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"{context}: StorageRead {nid!r}: config.top_k must be a "
                f"positive integer, got {top_k!r}"
            )

    elif ntype == "Embed":
        if "text" not in ins:
            raise ValueError(
                f"{context}: Embed {nid!r}: inputs.text is required"
            )

    return NodeSpec(id=nid, type=ntype, config=cfg, inputs=ins, body=body)


def _validate_output_decls(
    out_decls: Any,
    *,
    label: str,
    context: str,
    require_non_empty: bool = False,
) -> None:
    """Shared validation for ForEach/Branch ``config.outputs`` declarations."""
    if out_decls is None:
        out_decls = {}
    if not isinstance(out_decls, dict):
        raise ValueError(f"{context}: {label}: config.outputs must be a dict")
    if require_non_empty and not out_decls:
        raise ValueError(f"{context}: {label}: config.outputs must be a non-empty dict")
    for out_name, ref in out_decls.items():
        if not isinstance(out_name, str) or not out_name:
            raise ValueError(f"{context}: {label}: bad output name {out_name!r}")
        if not (isinstance(ref, str) and "." in ref):
            raise ValueError(
                f"{context}: {label}: output {out_name!r} must reference "
                f"a body node port like 'node.port', got {ref!r}"
            )


def _parse_subgraph_body(entry: dict, *, label: str, context: str) -> List[NodeSpec]:
    """Shared body parsing for ForEach + Branch: type filter, dup id check, recursion."""
    raw_body = entry.get("body")
    if not isinstance(raw_body, list) or not raw_body:
        raise ValueError(f"{context}: {label}: 'body' must be a non-empty list")
    body = [_parse_node(b, context=f"{label}.body") for b in raw_body]
    for b in body:
        if b.type in BODY_FORBIDDEN_TYPES:
            raise ValueError(
                f"{context}: {label}: body cannot contain {b.type!r} nodes"
            )
    body_ids: set = set()
    for b in body:
        if b.id in body_ids:
            raise ValueError(f"{label}.body: duplicate node id {b.id!r}")
        body_ids.add(b.id)
    return body


def _validate_refs(
    nodes: List[NodeSpec],
    *,
    allowed_ids: set,
    item_vars: set,
    context: str,
) -> None:
    """Each input ref's head must resolve to an allowed id or item var.

    Bare references (no ``.``) are valid only if they name a known
    ``item_var`` — they bind to the whole per-iteration value.
    """
    for n in nodes:
        for port, ref in n.inputs.items():
            if _is_const_ref(ref):
                continue
            if not isinstance(ref, str) or not ref:
                raise ValueError(
                    f"{context}: node {n.id!r} input {port!r}: bad reference {ref!r} "
                    f"(expected 'node_id.port', a bare item_var/node, or {{const: value}})"
                )
            head = ref.split(".", 1)[0]
            has_dot = "." in ref
            # Order: item_var → known node id → unknown.
            # Bare refs to known nodes are allowed and mean "the node's
            # whole output dict" (useful for piping into Output.variables
            # or any consumer that wants the full record).
            if head in item_vars:
                continue
            if head in allowed_ids:
                continue
            # Head is unknown. Surface the more helpful error message
            # depending on whether a dot was present.
            if not has_dot:
                raise ValueError(
                    f"{context}: node {n.id!r} input {port!r}: bare reference "
                    f"{ref!r} must be a known item_var or top-level node id "
                    f"(item_vars: {sorted(item_vars)})"
                )
            raise ValueError(
                f"{context}: node {n.id!r} input {port!r}: references unknown "
                f"node {head!r}"
            )


def _validate_foreach(node: NodeSpec, *, outer_ids: set) -> None:
    """Validate one ForEach: body refs + cycle check + nested ForEach recursion."""
    assert node.type == "ForEach" and node.body is not None
    item_var = node.config["item_var"]
    body_ids = {b.id for b in node.body}

    # An item_var must not collide with any visible id at the body level.
    if item_var in outer_ids or item_var in body_ids:
        raise ValueError(
            f"ForEach {node.id!r}: item_var {item_var!r} collides with an existing node id"
        )

    # Output declarations must reference body nodes.
    for out_name, ref in node.config.get("outputs", {}).items():
        head = ref.split(".", 1)[0]
        if head not in body_ids:
            raise ValueError(
                f"ForEach {node.id!r}: output {out_name!r} references {head!r} "
                f"which is not a body node"
            )

    allowed = outer_ids | body_ids
    _validate_refs(node.body, allowed_ids=allowed, item_vars={item_var},
                   context=f"ForEach {node.id!r}.body")

    # Topo-sort body. Externals (outer + item_var) are skipped as deps.
    _topo_sort(node.body, external_ids=outer_ids | {item_var})

    # Recurse for nested ForEach / Branch.
    nested_outer = outer_ids | body_ids | {item_var}
    for b in node.body:
        if b.type == "ForEach":
            _validate_foreach(b, outer_ids=nested_outer)
        elif b.type == "Branch":
            _validate_branch(b, outer_ids=nested_outer)


def _validate_branch(node: NodeSpec, *, outer_ids: set) -> None:
    """Validate one Branch: body refs + cycle check + nested recursion."""
    assert node.type == "Branch" and node.body is not None
    body_ids = {b.id for b in node.body}

    # Output declarations must reference body nodes.
    for out_name, ref in node.config.get("outputs", {}).items():
        head = ref.split(".", 1)[0]
        if head not in body_ids:
            raise ValueError(
                f"Branch {node.id!r}: output {out_name!r} references {head!r} "
                f"which is not a body node"
            )

    allowed = outer_ids | body_ids
    _validate_refs(node.body, allowed_ids=allowed, item_vars=set(),
                   context=f"Branch {node.id!r}.body")

    # Topo-sort body. Externals (outer) are skipped as deps.
    _topo_sort(node.body, external_ids=outer_ids)

    # Recurse for nested ForEach / Branch inside the body.
    nested_outer = outer_ids | body_ids
    for b in node.body:
        if b.type == "ForEach":
            _validate_foreach(b, outer_ids=nested_outer)
        elif b.type == "Branch":
            _validate_branch(b, outer_ids=nested_outer)


def _topo_sort(
    nodes: List[NodeSpec],
    *,
    external_ids: Optional[set] = None,
) -> List[NodeSpec]:
    """Kahn's algorithm. Returns execution order. Raises on cycles.

    ``external_ids`` (when given) is a set of identifiers that appear in
    refs but live outside this scope — they're already evaluated, so they
    don't count as dependencies. Used for ForEach body sorts where outer
    nodes + the item_var are visible.
    """
    external_ids = external_ids or set()
    by_id = {n.id: n for n in nodes}
    incoming: Dict[str, set] = {n.id: set() for n in nodes}
    for n in nodes:
        for ref in n.inputs.values():
            if _is_const_ref(ref):
                continue
            src_id = ref.split(".", 1)[0]
            if src_id == n.id:
                raise ValueError(f"Node {n.id!r} references itself")
            if src_id in external_ids:
                continue
            if src_id not in by_id:
                continue
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
        # Stack of iteration indices for nested ForEach; used to disambiguate
        # trace step names across iterations.
        self._iter_stack: List[int] = []

    def run(self, phase_inputs: Dict[str, Any]) -> Dict[str, Any]:
        env: Dict[str, Dict[str, Any]] = {}
        for node in self._order:
            if node.type == "Input":
                env[node.id] = dict(phase_inputs)
                continue
            env[node.id] = self._eval_node(node, env, item_bindings={})

        out_node = next(n for n in self.graph.nodes if n.type == "Output")
        return env[out_node.id]

    # ----- per-node dispatcher (shared with body subgraph execution) -----

    def _eval_node(
        self,
        node: NodeSpec,
        env: Dict[str, Dict[str, Any]],
        *,
        item_bindings: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved = self._resolve_inputs(node, env, item_bindings=item_bindings)
        if node.type == "Output":
            return resolved
        if node.type == "Constant":
            return {"value": node.config.get("value")}
        if node.type == "PromptRender":
            return self._render_prompt(node, resolved)
        if node.type == "LLMCall":
            return self._call_llm(node, resolved)
        if node.type == "ForEach":
            return self._run_foreach(node, resolved, env, item_bindings=item_bindings)
        if node.type == "Branch":
            return self._run_branch(node, resolved, env, item_bindings=item_bindings)
        if node.type == "Compute":
            return self._run_compute(node, resolved)
        if node.type == "StorageRead":
            return self._run_storage_read(node, resolved)
        if node.type == "Embed":
            return self._run_embed(node, resolved)
        raise ValueError(f"Unknown node type {node.type!r}")

    def _resolve_inputs(
        self,
        node: NodeSpec,
        env: Dict[str, Dict[str, Any]],
        *,
        item_bindings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        item_bindings = item_bindings or {}
        out: Dict[str, Any] = {}
        for port, ref in node.inputs.items():
            out[port] = _resolve_ref(ref, env, item_bindings, hint=f"{node.id}.{port}")
        return out

    # ----- ForEach -----

    def _run_foreach(
        self,
        node: NodeSpec,
        resolved_inputs: Dict[str, Any],
        env: Dict[str, Dict[str, Any]],
        *,
        item_bindings: Dict[str, Any],
    ) -> Dict[str, Any]:
        items_value = resolved_inputs.get("items")
        if not isinstance(items_value, list):
            raise ValueError(
                f"ForEach {node.id!r}: 'items' must resolve to a list, "
                f"got {type(items_value).__name__}"
            )
        if len(items_value) > MAX_LOOP_ITERATIONS:
            raise ValueError(
                f"ForEach {node.id!r}: {len(items_value)} items exceeds "
                f"MAX_LOOP_ITERATIONS={MAX_LOOP_ITERATIONS}"
            )
        item_var: str = node.config["item_var"]
        declared_outputs: Dict[str, str] = node.config.get("outputs") or {}
        collected: Dict[str, List[Any]] = {name: [] for name in declared_outputs}

        body_nodes = node.body or []
        external_ids = set(env.keys()) | set(item_bindings.keys()) | {item_var}
        body_order = _topo_sort(body_nodes, external_ids=external_ids)

        for i, item in enumerate(items_value):
            self._iter_stack.append(i)
            try:
                # Sub-env shadows over outer env so body nodes can read both.
                sub_env: Dict[str, Dict[str, Any]] = dict(env)
                sub_bindings: Dict[str, Any] = dict(item_bindings)
                sub_bindings[item_var] = item
                for body_node in body_order:
                    sub_env[body_node.id] = self._eval_node(
                        body_node, sub_env, item_bindings=sub_bindings,
                    )
                for out_name, ref in declared_outputs.items():
                    val = _resolve_ref(
                        ref, sub_env, sub_bindings,
                        hint=f"{node.id}.outputs.{out_name}",
                    )
                    collected[out_name].append(val)
            finally:
                self._iter_stack.pop()

        return collected

    # ----- Branch -----

    def _run_branch(
        self,
        node: NodeSpec,
        resolved_inputs: Dict[str, Any],
        env: Dict[str, Dict[str, Any]],
        *,
        item_bindings: Dict[str, Any],
    ) -> Dict[str, Any]:
        cond_truthy = bool(resolved_inputs.get("condition", False))
        declared_outputs: Dict[str, str] = node.config.get("outputs") or {}
        else_ref: Any = node.config.get("else_value", {"const": None})

        if not cond_truthy:
            falsy_val = _resolve_ref(
                else_ref, env, item_bindings,
                hint=f"{node.id}.else_value",
            )
            return {name: falsy_val for name in declared_outputs}

        # Truthy: run the body once and collect declared outputs.
        body_nodes = node.body or []
        external_ids = set(env.keys()) | set(item_bindings.keys())
        body_order = _topo_sort(body_nodes, external_ids=external_ids)

        sub_env: Dict[str, Dict[str, Any]] = dict(env)
        for body_node in body_order:
            sub_env[body_node.id] = self._eval_node(
                body_node, sub_env, item_bindings=item_bindings,
            )
        return {
            out_name: _resolve_ref(
                ref, sub_env, item_bindings,
                hint=f"{node.id}.outputs.{out_name}",
            )
            for out_name, ref in declared_outputs.items()
        }

    # ----- Compute -----

    def _run_compute(self, node: NodeSpec, resolved_inputs: Dict[str, Any]) -> Dict[str, Any]:
        op_name = node.config["op"]
        spec = COMPUTE_OPS[op_name]
        try:
            return spec["fn"](resolved_inputs)
        except (TypeError, KeyError, ValueError, ZeroDivisionError) as e:
            raise ValueError(
                f"Compute {node.id!r} op={op_name!r}: {type(e).__name__}: {e}"
            ) from e

    # ----- StorageRead -----

    def _run_storage_read(
        self, node: NodeSpec, resolved_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read memory nodes from the bound MemoryGraph, formatted for prompts.

        Wraps ``MemoryGraph.retrieve_{semantic,procedural,episodic}_nodes``
        using the graph's default value functions (``tag_relevant``,
        ``semantic_relevant``, ``subgoal_relevant``, ``procedural_relevant``).
        Returns the same formatted strings that ``MemoryGraph.retrieve_memory``
        builds for the reasoning templates plus the list of selected ids
        (useful for the trace recorder).
        """
        collection = node.config["collection"]
        graph = self.memory_graph

        if collection == "semantic":
            query = resolved_inputs.get("query") or ""
            tags = resolved_inputs.get("tags") or []
            if not isinstance(tags, list):
                raise ValueError(
                    f"StorageRead {node.id!r}: inputs.tags must be a list, "
                    f"got {type(tags).__name__}"
                )
            nodes = graph.retrieve_semantic_nodes(
                semantic_memory={"semantic_memory": query, "tags": list(tags)},
                value_func_tag=graph.tag_relevant,
                value_func=graph.semantic_relevant,
            )
            if not nodes:
                return {"value": "No relevant fact", "ids": []}
            text = "".join(
                f"Fact {i}: {n.get_semantic_memory()}\n"
                for i, n in enumerate(nodes)
            )
            return {"value": text, "ids": [n.semantic_id for n in nodes]}

        if collection == "procedural":
            subgoal = resolved_inputs.get("subgoal") or ""
            nodes = graph.retrieve_procedural_nodes(
                subgoal=subgoal,
                value_func_subgoal=graph.subgoal_relevant,
                value_func=graph.procedural_relevant,
            )
            if not nodes:
                return {"value": "No relevant experiences", "ids": []}
            text = "".join(
                f"Experience {i}: {n.get_procedural_memory()}\n"
                for i, n in enumerate(nodes)
            )
            return {"value": text, "ids": [n.procedural_id for n in nodes]}

        if collection == "episodic":
            query = resolved_inputs.get("query") or ""
            text = graph.retrieve_episodic_nodes(observation=query) or ""
            return {"value": text, "ids": []}

        # Unreachable — load-time check rejects unknown collections.
        raise ValueError(
            f"StorageRead {node.id!r}: unknown collection {collection!r}"
        )

    # ----- Embed -----

    def _run_embed(
        self, node: NodeSpec, resolved_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = resolved_inputs.get("text", "")
        if not isinstance(text, str):
            raise ValueError(
                f"Embed {node.id!r}: inputs.text must be a string, "
                f"got {type(text).__name__}"
            )
        emb = self.memory_graph.embedder.embed(text)
        # Normalize to a Python list so concat/length ops behave predictably.
        vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        return {"embedding": vec}

    # ----- trace step name (suffixed by iter indices when inside a loop) -----

    def _trace_step_name(self, node_id: str) -> str:
        if not self._iter_stack:
            return node_id
        return node_id + "#" + "#".join(str(i) for i in self._iter_stack)

    def _registry(self) -> PromptRegistry:
        return self.memory_graph.prompts or PromptRegistry()

    def _render_prompt(self, node: NodeSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Render a registered prompt into both a string and a messages list.

        Outputs:
        - ``value``    — the finished template as a single string. Each
                         message's content is concatenated with a blank
                         line between them. This is the "filled-in
                         template" view: a pure ``vars → str`` function.
        - ``messages`` — the same content broken into a list of
                         ``{role, content}`` dicts, mirroring the prompt
                         registry's structured output. Kept for callers
                         that need role information (e.g. an LLMCall in
                         messages-mode).
        """
        prompt_name = node.config.get("prompt")
        if not prompt_name:
            raise ValueError(f"PromptRender {node.id!r}: config.prompt required")
        messages = self._registry().render_messages(
            prompt_name, inputs, graph_id=self.memory_graph.graph_id,
        )
        value = "\n\n".join(
            (m.get("content") if isinstance(m, dict) else "") or "" for m in messages
        )
        return {"messages": messages, "value": value}

    def _call_llm(self, node: NodeSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call the LLM. Three input modes, exactly one must be configured.

        - **Prompt-mode** (``config.prompt`` set): renders the named
          registered prompt using ``inputs`` as template variables, then
          calls the LLM.
        - **Messages-mode** (``inputs.messages`` set): pre-rendered list
          of ``{role, content}`` dicts is passed to the LLM verbatim.
        - **Text-mode** (``inputs.text`` set): a single string is wrapped
          as ``[{role: "user", content: text}]`` and passed to the LLM.
          This is the natural pairing for ``PromptRender.value`` (the
          finished template as a string).
        """
        if self.llm_call_count >= LLM_CALL_CAP:
            raise ValueError(
                f"LLM call cap ({LLM_CALL_CAP}) exceeded for pipeline "
                f"{self.graph.phase!r}"
            )
        self.llm_call_count += 1
        prompt_name = node.config.get("prompt")
        role = node.config.get("role", "default")
        has_messages_input = "messages" in node.inputs
        has_text_input = "text" in node.inputs
        modes_set = sum([bool(prompt_name), has_messages_input, has_text_input])
        if modes_set > 1:
            raise ValueError(
                f"LLMCall {node.id!r}: set exactly one of "
                f"config.prompt, inputs.messages, inputs.text"
            )
        if modes_set == 0:
            raise ValueError(
                f"LLMCall {node.id!r}: must set one of "
                f"config.prompt (prompt-mode), inputs.messages "
                f"(messages-mode), or inputs.text (text-mode)"
            )

        llm = self.memory_graph._router.for_role(role)
        if has_messages_input:
            explicit_messages = inputs["messages"]
            if not isinstance(explicit_messages, list):
                raise ValueError(
                    f"LLMCall {node.id!r}: inputs.messages must be a list, "
                    f"got {type(explicit_messages).__name__}"
                )
            messages = explicit_messages
        elif has_text_input:
            text = inputs["text"]
            if not isinstance(text, str):
                raise ValueError(
                    f"LLMCall {node.id!r}: inputs.text must be a string, "
                    f"got {type(text).__name__}"
                )
            messages = [{"role": "user", "content": text}]
        else:
            messages = self._registry().render_messages(
                prompt_name, inputs, graph_id=self.memory_graph.graph_id,
            )

        started = _time.monotonic()
        response = llm.complete(messages=messages)
        latency_ms = int((_time.monotonic() - started) * 1000)
        parser = PARSERS.get(prompt_name) if prompt_name else None
        parsed = parser(response) if parser else {"text": response}
        record_llm_step(
            name=self._trace_step_name(node.id),
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
