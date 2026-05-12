"""Per-pipeline visualization specs for the Pipeline tab.

Each registered pipeline exposes a ``PipelineSpec``-shaped dict (phases /
steps / edges / roles / kinds) so the inspector canvas can render the
correct graph for whichever pipeline a graph is currently bound to.

- ``plugmem-default``  → hand-curated spec from ``pipeline_spec.py``.
- ``naive-rag``        → hand-coded small spec (4 steps) — flat RAG has
                         no structured phases.
- ``spec-driven``      → derived from the user's saved YAML; falls back
                         to a single-node placeholder if no YAML exists.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from plugmem.core.pipeline_spec import to_dict as default_spec_dict
from plugmem.pipelines import spec_storage
from plugmem.pipelines.spec_driven import (
    PipelineGraph,
    NodeSpec,
    load_yaml_str,
)


# --------------------------------------------------------------------- #
# naive-rag
# --------------------------------------------------------------------- #


def naive_rag_spec() -> Dict[str, Any]:
    """Hand-coded spec for the flat-RAG baseline.

    Mirrors what ``NaiveRAGPipeline`` actually does: embed/store on ingest,
    top-K similarity on retrieve, one LLM call on reason.
    """
    return {
        "phases": [
            {
                "id": "ingest",
                "label": "Ingest",
                "description": (
                    "Flatten incoming trajectory/structured payload into "
                    "individual facts and write each one to the semantic "
                    "collection. No segmentation, no procedural memory."
                ),
                "trigger": "Once per /memories call",
                "triggered_by": ["POST /memories"],
            },
            {
                "id": "retrieve",
                "label": "Retrieve",
                "description": (
                    "Top-K embedding similarity over the flat fact set. "
                    "Renders a single reasoning prompt with the retrieved "
                    "facts pasted in."
                ),
                "trigger": "Once per /retrieve call",
                "triggered_by": ["POST /retrieve", "POST /reason"],
            },
            {
                "id": "reason",
                "label": "Reason",
                "description": "One LLM call against the rendered prompt.",
                "trigger": "Once per /reason call",
                "triggered_by": ["POST /reason"],
            },
        ],
        "steps": [
            _step("flatten_facts", "Flatten payload to facts",
                  "Extract observation/action strings + semantic_memory "
                  "fields from the request into a flat list of strings.",
                  kind="compute", phase="ingest",
                  inputs=["body.steps", "body.semantic"], outputs=["facts[]"]),
            _step("embed_facts", "Embed each fact",
                  "graph.embedder.embed(text) per fact.",
                  kind="embed", phase="ingest",
                  inputs=["facts[]"], outputs=["fact_embeddings[]"]),
            _step("persist_semantic_flat", "Persist flat semantics",
                  "storage.add_semantic with empty tags and no episodic "
                  "linkage.",
                  kind="storage", phase="ingest",
                  inputs=["fact_embeddings[]"]),

            _step("embed_query", "Embed observation",
                  "graph.embedder.embed(body.observation).",
                  kind="embed", phase="retrieve",
                  inputs=["body.observation"], outputs=["query_emb"]),
            _step("topk_similarity", "Top-K cosine similarity",
                  "Score every active semantic node against query_emb and "
                  "keep the top 5.",
                  kind="compute", phase="retrieve",
                  inputs=["query_emb", "semantic_nodes"],
                  outputs=["topk_facts"]),
            _step("render_flat_prompt", "Render flat-RAG prompt",
                  "Plug the topk facts into a fixed template (no "
                  "PromptRegistry lookup).",
                  kind="template_render", phase="retrieve",
                  prompt_name="(inline naive-rag template)",
                  inputs=["body.observation", "topk_facts"],
                  outputs=["messages"]),

            _step("reason_inherit_messages", "Use messages from Retrieve",
                  "Re-uses the messages produced in the retrieve phase.",
                  kind="compute", phase="reason",
                  inputs=["messages"], outputs=["messages"]),
            _step("reason_llm_call", "Reasoning LLM call",
                  "graph.reasoning_llm.complete(messages=messages).",
                  kind="llm", phase="reason",
                  prompt_name="", role="reasoning",
                  inputs=["messages"], outputs=["reasoning"]),
        ],
        "edges": [
            _edge("flatten_facts", "embed_facts"),
            _edge("embed_facts", "persist_semantic_flat"),
            _edge("embed_query", "topk_similarity"),
            _edge("topk_similarity", "render_flat_prompt"),
            _edge("reason_inherit_messages", "reason_llm_call"),
        ],
        "cross_phase_edges": [
            _edge("render_flat_prompt", "reason_inherit_messages",
                  kind="alt", label="if /reason"),
        ],
        "roles": ["reasoning"],
        "kinds": ["llm", "template_render", "embed", "compute", "storage"],
    }


# --------------------------------------------------------------------- #
# spec-driven  — YAML → PipelineSpec converter
# --------------------------------------------------------------------- #


_TYPE_TO_KIND = {
    "Input": "compute",
    "Output": "compute",
    "LLMCall": "llm",
    "PromptRender": "template_render",
    "Constant": "compute",
    "Compute": "compute",
    "ForEach": "loop_marker",
    "Branch": "branch",
}


def spec_driven_spec(graph_id: str) -> Dict[str, Any]:
    """Convert the saved YAML for *graph_id* into the inspector spec shape.

    Falls back to a one-node placeholder when no YAML exists yet so the
    canvas still renders something useful.
    """
    yaml_text = spec_storage.read_current_content(graph_id) or ""
    if not yaml_text.strip():
        return _empty_spec_driven_placeholder()

    try:
        graph = load_yaml_str(yaml_text)
    except ValueError as e:
        return _invalid_spec_driven_placeholder(str(e))

    return _yaml_to_spec(graph)


def _empty_spec_driven_placeholder() -> Dict[str, Any]:
    return {
        "phases": [_retrieve_phase("No YAML saved — use the Spec editor below.")],
        "steps": [
            _step("__empty", "(no YAML)",
                  "Drop a YAML into the Spec editor and Save. "
                  "Or click the 'Insert plugmem-default template' button to "
                  "start from a ready-to-run sample.",
                  kind="compute", phase="retrieve"),
        ],
        "edges": [],
        "cross_phase_edges": [],
        "roles": ["retrieval"],
        "kinds": ["llm", "template_render", "compute", "branch", "loop_marker"],
    }


def _invalid_spec_driven_placeholder(error: str) -> Dict[str, Any]:
    return {
        "phases": [_retrieve_phase(f"YAML failed to load: {error}")],
        "steps": [
            _step("__invalid", "Invalid YAML",
                  f"The saved spec did not load. Validator: {error}",
                  kind="compute", phase="retrieve"),
        ],
        "edges": [],
        "cross_phase_edges": [],
        "roles": ["retrieval"],
        "kinds": ["llm", "template_render", "compute", "branch", "loop_marker"],
    }


def _yaml_to_spec(graph: PipelineGraph) -> Dict[str, Any]:
    """Convert a parsed YAML pipeline graph into the inspector spec shape.

    Body node ids are namespaced by their parent container's id (e.g. a
    PromptRender ``r`` inside Branch ``render_reasoning_semantic``
    becomes step id ``render_reasoning_semantic.r``). Without this,
    multiple branches sharing a body node id (a perfectly valid YAML
    pattern — body scopes are independent) collapse into a single
    canvas node.

    References from inside a body are resolved against three scopes,
    in order: (a) local body siblings, (b) outer top-level ids,
    (c) item_var bindings (skipped — they aren't first-class nodes).
    """
    steps: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_kinds: set = set()
    seen_roles: set = set()

    top_level_ids = {n.id for n in graph.nodes}

    # Build a rewire table so consumers of ``<container>.<declared_output>``
    # see the body node (the actual data source) as the edge source. Without
    # this, Branch/ForEach bodies look like dead-end leaves on the canvas
    # — the user can't see where the rendered template/loop output goes.
    # Mapping shape: {(container_id, output_name): body_display_id}.
    container_output_map: Dict[tuple, str] = {}
    for tn in graph.nodes:
        if tn.body and tn.type in ("Branch", "ForEach"):
            decls = (tn.config or {}).get("outputs") or {}
            body_local_ids = {b.id for b in tn.body}
            for out_name, ref in decls.items():
                if isinstance(ref, str) and "." in ref:
                    body_head = ref.split(".", 1)[0]
                    if body_head in body_local_ids:
                        container_output_map[(tn.id, out_name)] = f"{tn.id}.{body_head}"

    def display_id(local_id: str, parent_id: Optional[str]) -> str:
        return f"{parent_id}.{local_id}" if parent_id else local_id

    def resolve_ref_src(
        ref_head: str, *,
        local_ids: set, parent_id: Optional[str], item_vars: set,
    ) -> Optional[str]:
        """Map a ref's head id to a display id, or None if it doesn't refer
        to a first-class node (item_var, unknown)."""
        if ref_head in item_vars:
            return None
        if ref_head in local_ids:
            return display_id(ref_head, parent_id)
        if ref_head in top_level_ids:
            return ref_head
        return None

    def walk(
        nodes: List[NodeSpec], *,
        parent_id: Optional[str], item_vars: set,
    ) -> None:
        local_ids = {n.id for n in nodes}
        for n in nodes:
            kind = _TYPE_TO_KIND.get(n.type, "compute")
            seen_kinds.add(kind)
            prompt = ""
            role = ""
            if n.type in ("LLMCall", "PromptRender"):
                prompt = str(n.config.get("prompt") or "")
            if n.type == "LLMCall":
                role = str(n.config.get("role") or "default")
                seen_roles.add(role)

            inputs = list(n.inputs.keys())
            outputs = _outputs_for(n)
            description = _desc_for(n)
            this_display = display_id(n.id, parent_id)

            steps.append(_step(
                this_display, _label_for(n, parent_id=parent_id), description,
                kind=kind, phase="retrieve",
                prompt_name=prompt, role=role,
                inputs=inputs, outputs=outputs,
                branch_condition=_branch_condition_label(n),
                branch_outcomes=(
                    ["yes → run body", "no  → emit else_value"]
                    if n.type == "Branch" else []
                ),
                loop_scope=(
                    f"for each item in {list(n.inputs.values())[0] if n.inputs else 'items'}"
                    if n.type == "ForEach" else ""
                ),
            ))

            # Edges from each input's referenced source.
            for port, ref in n.inputs.items():
                if isinstance(ref, dict) and "const" in ref:
                    continue
                if not isinstance(ref, str):
                    continue
                parts = ref.split(".")
                src_head = parts[0]
                src_port = parts[1] if len(parts) > 1 else None
                # Rewire: when a ref reads a container's declared output
                # (e.g. ``render_reasoning_semantic.rendered``), source
                # the edge from the body node that actually produces it
                # (``render_reasoning_semantic.render``) so the canvas
                # shows the data flowing OUT of the body, not from the
                # container wrapper.
                if src_port and (src_head, src_port) in container_output_map:
                    src_display = container_output_map[(src_head, src_port)]
                else:
                    src_display = resolve_ref_src(
                        src_head,
                        local_ids=local_ids,
                        parent_id=parent_id,
                        item_vars=item_vars,
                    )
                if src_display is None:
                    continue
                edges.append({
                    "source": src_display, "target": this_display,
                    "kind": "seq", "label": port,
                })

            # Container → body edges (control flow into the body).
            if n.body:
                edge_kind = "branch" if n.type == "Branch" else "seq"
                for b in n.body:
                    edges.append({
                        "source": this_display,
                        "target": display_id(b.id, n.id),
                        "kind": edge_kind,
                        "label": "if true" if n.type == "Branch" else "iter",
                    })

                new_item_vars = set(item_vars)
                if n.type == "ForEach":
                    iv = n.config.get("item_var")
                    if iv:
                        new_item_vars.add(iv)
                walk(n.body, parent_id=n.id, item_vars=new_item_vars)

    walk(graph.nodes, parent_id=None, item_vars=set())

    # Drop edges whose endpoints aren't known steps (defensive).
    known = {s["id"] for s in steps}
    edges = [e for e in edges if e["source"] in known and e["target"] in known]

    return {
        "phases": [_retrieve_phase(
            "User-authored spec-driven pipeline (retrieve phase only)."
        )],
        "steps": steps,
        "edges": edges,
        "cross_phase_edges": [],
        "roles": sorted(seen_roles) or ["retrieval"],
        "kinds": sorted(seen_kinds) or ["compute"],
    }


def _branch_condition_label(n: NodeSpec) -> str:
    """Human-readable condition text for Branch nodes."""
    if n.type != "Branch":
        return ""
    cond = n.inputs.get("condition")
    if isinstance(cond, str):
        return f"{cond} truthy?"
    if isinstance(cond, dict) and "const" in cond:
        return f"{cond['const']!r} truthy?"
    return "condition truthy?"


# --------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------- #


def view_for_pipeline(pipeline_name: str, *, graph_id: str) -> Dict[str, Any]:
    """Return the visualization spec for the given bound pipeline."""
    if pipeline_name == "plugmem-default":
        return default_spec_dict()
    if pipeline_name == "naive-rag":
        return naive_rag_spec()
    if pipeline_name == "spec-driven":
        return spec_driven_spec(graph_id)
    # Unknown pipeline — return a stub so the UI doesn't break.
    return {
        "phases": [_retrieve_phase(
            f"No visualization registered for pipeline '{pipeline_name}'."
        )],
        "steps": [],
        "edges": [],
        "cross_phase_edges": [],
        "roles": ["retrieval"],
        "kinds": ["compute"],
    }


# --------------------------------------------------------------------- #
# Small builders
# --------------------------------------------------------------------- #


def _step(
    sid: str, label: str, description: str, *,
    kind: str, phase: str,
    prompt_name: str = "",
    role: str = "",
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    branch_condition: str = "",
    branch_outcomes: Optional[List[str]] = None,
    loop_scope: str = "",
) -> Dict[str, Any]:
    return {
        "id": sid,
        "label": label,
        "description": description,
        "kind": kind,
        "phase": phase,
        "prompt_name": prompt_name,
        "role": role,
        "inputs": list(inputs or []),
        "outputs": list(outputs or []),
        "per": "per_call",
        "optional": False,
        "branch_condition": branch_condition,
        "branch_outcomes": list(branch_outcomes or []),
        "loop_scope": loop_scope,
    }


def _edge(source: str, target: str, *, kind: str = "seq", label: str = ""):
    return {"source": source, "target": target, "kind": kind, "label": label}


def _retrieve_phase(description: str) -> Dict[str, Any]:
    return {
        "id": "retrieve",
        "label": "Retrieve",
        "description": description,
        "trigger": "POST /retrieve",
        "triggered_by": ["POST /retrieve"],
    }


def _label_for(n: NodeSpec, *, parent_id: Optional[str] = None) -> str:
    # Body-local names like ``r`` are unhelpful — qualify them with the
    # parent's id so the user sees ``render_reasoning_semantic.r`` etc.
    qualified_id = f"{parent_id}.{n.id}" if parent_id else n.id
    if n.type == "LLMCall":
        return f"LLM · {n.config.get('prompt', n.id)}"
    if n.type == "PromptRender":
        return f"Template · {n.config.get('prompt', n.id)}"
    if n.type == "Compute":
        return f"Compute · {n.config.get('op', '?')}"
    if n.type == "Constant":
        return f"Constant · {qualified_id}"
    if n.type == "ForEach":
        iv = n.config.get("item_var", "item")
        return f"ForEach ({iv})"
    if n.type == "Branch":
        return f"Branch · {qualified_id}"
    if n.type == "Input":
        return "Input"
    if n.type == "Output":
        return "Output"
    return n.type


def _desc_for(n: NodeSpec) -> str:
    if n.type == "Input":
        return "Phase entry. Outputs are the request body fields (observation, goal, …)."
    if n.type == "Output":
        return "Phase exit. Inputs become the response (mode, reasoning_prompt, variables)."
    if n.type == "Constant":
        return f"Emits a fixed value: {n.config.get('value')!r}"
    if n.type == "Compute":
        return f"Single-op compute: {n.config.get('op')}"
    if n.type == "LLMCall":
        return (
            f"Renders prompt '{n.config.get('prompt')}' (role="
            f"{n.config.get('role', 'default')}) and calls the LLM."
        )
    if n.type == "PromptRender":
        return f"Renders prompt '{n.config.get('prompt')}' into messages (no LLM call)."
    if n.type == "ForEach":
        return (
            f"Iterates over items; body runs once per element; declared "
            f"outputs are collected into lists. item_var="
            f"{n.config.get('item_var')!r}."
        )
    if n.type == "Branch":
        return (
            "Runs body if condition is truthy; otherwise emits else_value "
            "for each declared output."
        )
    return n.type


def _outputs_for(n: NodeSpec) -> List[str]:
    if n.type == "Input":
        return ["observation", "goal", "subgoal", "state", "task_type", "time", "mode"]
    if n.type == "Output":
        return []
    if n.type == "LLMCall":
        return ["raw", "parsed.*"]
    if n.type == "PromptRender":
        # ``value`` is the rendered template as a single string;
        # ``messages`` is the same content as a list[{role, content}].
        return ["value", "messages"]
    if n.type == "Constant":
        return ["value"]
    if n.type == "Compute":
        return ["value"]
    if n.type in ("ForEach", "Branch"):
        decls = (n.config or {}).get("outputs") or {}
        return list(decls.keys())
    return []
