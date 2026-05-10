"""Declarative description of PlugMem's processing pipeline.

Single source of truth for the inspector's Pipeline tab. Encodes the actual
call graph: every LLM call, every embedding call, every compute / storage
step, every branch, and every loop boundary that the production code path
executes — mapped phase by phase.

Step kinds
----------
- ``llm``             — paid LLM call. ``prompt_name`` + ``role`` populated.
- ``template_render`` — renders a prompt template (no LLM call). The
                        rendered messages flow into a downstream ``llm``
                        step. ``prompt_name`` populated so the user can
                        find it for editing; ``role`` left blank.
- ``embed``           — embedding call (sentence-transformer / fastembed).
- ``compute``         — pure CPU work (no I/O), e.g. similarity, list ops.
- ``storage``         — ChromaDB persist.
- ``branch``          — control-flow split. ``branch_condition`` is the
                        question; ``branch_outcomes`` lists what each
                        outcome does in plain text.
- ``loop_marker``     — marks an iteration boundary. ``loop_scope`` is the
                        per-iteration label, e.g. "for each trajectory".

The frontend renders each phase as a vertical column; steps stack in the
order given in :data:`STEPS`. Edges are explicit (see :data:`EDGES`) so
branching alternatives can be drawn distinctly from straight-line flow.

When the source code changes, update this file *and* re-run the inspector
to verify the diagram still matches reality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


# ----------------------------------------------------------------------- #
# Data model
# ----------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineStep:
    id: str
    label: str
    description: str
    kind: str
    phase: str
    prompt_name: str = ""
    role: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    per: str = "per_call"
    optional: bool = False
    branch_condition: str = ""
    branch_outcomes: List[str] = field(default_factory=list)
    loop_scope: str = ""


@dataclass(frozen=True)
class PipelineEdge:
    source: str
    target: str
    kind: str = "seq"
    label: str = ""


@dataclass(frozen=True)
class PipelinePhase:
    id: str
    label: str
    description: str
    trigger: str
    triggered_by: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------- #
# Phases
# ----------------------------------------------------------------------- #


PHASES: List[PipelinePhase] = [
    PipelinePhase(
        id="append",
        label="Append",
        description=(
            "Per-step structuring: runs once for each (action, observation) "
            "pair as a trajectory is being appended in Memory.append()."
        ),
        trigger="Once per trajectory step",
        triggered_by=["POST /memories (mode=trajectory)"],
    ),
    PipelinePhase(
        id="close",
        label="Close",
        description=(
            "Trajectory finalization in Memory.close(): distills semantic "
            "facts per step and a procedural memory per closed trajectory."
        ),
        trigger="Once on trajectory close (after all steps appended)",
        triggered_by=["POST /memories (mode=trajectory)"],
    ),
    PipelinePhase(
        id="insert",
        label="Insert",
        description=(
            "MemoryGraph.insert(): persists structured memory into ChromaDB "
            "and merges duplicate subgoals."
        ),
        trigger="Once per memory after Close (or directly for mode=structured)",
        triggered_by=["POST /memories"],
    ),
    PipelinePhase(
        id="retrieve",
        label="Retrieve",
        description=(
            "Recall planning + node retrieval in retrieve_memory(): plans "
            "tags / next-subgoal, picks a memory mode, fetches matching "
            "nodes, and renders the reasoning prompt template."
        ),
        trigger="Once per recall call",
        triggered_by=["POST /retrieve", "POST /reason", "POST /recall_trace"],
    ),
    PipelinePhase(
        id="reason",
        label="Reason",
        description=(
            "Final reasoning LLM call in retrieve_and_reason(): consumes "
            "the messages built during Retrieve."
        ),
        trigger="Once per /reason call (after Retrieve)",
        triggered_by=["POST /reason"],
    ),
    PipelinePhase(
        id="consolidate",
        label="Consolidate",
        description=(
            "update_semantic_subgraph(): walks recently-added semantic "
            "nodes, finds candidate merges via tag overlap, and merges "
            "high-similarity pairs via merge_semantic()."
        ),
        trigger="Once per consolidation pass",
        triggered_by=["POST /consolidate"],
    ),
]


# ----------------------------------------------------------------------- #
# Builders
# ----------------------------------------------------------------------- #


def _llm(id, label, desc, *, prompt, role, phase, inputs, outputs, per="per_call", optional=False):
    return PipelineStep(
        id=id, label=label, description=desc, kind="llm",
        phase=phase, prompt_name=prompt, role=role,
        inputs=list(inputs), outputs=list(outputs), per=per, optional=optional,
    )


def _tmpl(id, label, desc, *, prompt, phase, inputs, outputs, optional=False):
    return PipelineStep(
        id=id, label=label, description=desc, kind="template_render",
        phase=phase, prompt_name=prompt,
        inputs=list(inputs), outputs=list(outputs), optional=optional,
    )


def _emb(id, label, desc, *, phase, inputs, outputs, optional=False):
    return PipelineStep(
        id=id, label=label, description=desc, kind="embed",
        phase=phase, inputs=list(inputs), outputs=list(outputs), optional=optional,
    )


def _comp(id, label, desc, *, phase, inputs=(), outputs=(), optional=False):
    return PipelineStep(
        id=id, label=label, description=desc, kind="compute",
        phase=phase, inputs=list(inputs), outputs=list(outputs), optional=optional,
    )


def _store(id, label, desc, *, phase, inputs=()):
    return PipelineStep(
        id=id, label=label, description=desc, kind="storage",
        phase=phase, inputs=list(inputs),
    )


def _branch(id, label, *, phase, condition, outcomes):
    return PipelineStep(
        id=id, label=label, description=condition, kind="branch",
        phase=phase, branch_condition=condition, branch_outcomes=list(outcomes),
    )


def _loop(id, label, *, phase, scope):
    return PipelineStep(
        id=id, label=label, description=scope, kind="loop_marker",
        phase=phase, loop_scope=scope,
    )


# ----------------------------------------------------------------------- #
# Steps — encoded directly from the call sites (see references in comments)
# ----------------------------------------------------------------------- #


STEPS: List[PipelineStep] = [
    # ============ Append (Memory.append, core/memory.py:59) ============
    _llm("get_subgoal", "Subgoal",
         "Infer the immediate intent behind the action. Template variables "
         "are sourced from (goal, state_t0, observation_t0, action_t0).",
         prompt="get_subgoal", role="structuring", phase="append",
         inputs=["goal", "state", "observation", "action"],
         outputs=["subgoal"], per="per_step"),
    _llm("get_reward", "Reward",
         "Evaluate how well the action moved toward the (sub)goal. The "
         "template's {goal} receives the just-inferred subgoal (not the "
         "original goal); {observation} receives observation_t1.",
         prompt="get_reward", role="structuring", phase="append",
         inputs=["goal", "state", "action", "observation"],
         outputs=["reward"], per="per_step"),
    _branch("br_traj_empty", "Trajectory empty?", phase="append",
            condition="Is self.trajectory empty (first step in a new segment)?",
            outcomes=[
                "yes → skip embed + similarity; jump to append-step",
                "no  → embed previous + current subgoal and check similarity",
            ]),
    _emb("embed_prev_subgoal", "Embed previous subgoal",
         "self.embedder.embed(self.trajectory[-1]['subgoal']).",
         phase="append", inputs=["trajectory[-1].subgoal"],
         outputs=["emb_prev"], optional=True),
    _emb("embed_curr_subgoal", "Embed current subgoal",
         "self.embedder.embed(subgoal).",
         phase="append", inputs=["subgoal"],
         outputs=["emb_curr"], optional=True),
    _comp("subgoal_similarity", "Cosine similarity",
          "get_similarity(emb_prev, emb_curr) → similarity_subgoal.",
          phase="append",
          inputs=["emb_prev", "emb_curr"], outputs=["similarity_subgoal"],
          optional=True),
    _branch("br_segment", "Segment trajectory?", phase="append",
            condition="Is similarity_subgoal < 0.75?",
            outcomes=[
                "yes → close current trajectory segment, start a new one",
                "no  → continue same trajectory",
            ]),
    _comp("close_segment", "Close trajectory segment",
          "memory['episodic'].append(self.trajectory); self.trajectory = [].",
          phase="append", optional=True),
    _comp("append_step", "Append step",
          "Append (subgoal, state_t0, observation_t0, action_t0, reward, "
          "similarity_subgoal, time) to self.trajectory.",
          phase="append",
          inputs=["subgoal", "state_t0", "observation_t0", "action_t0", "reward"],
          outputs=["trajectory[+]"]),
    _llm("get_state", "State (next)",
         "Update the running state summary after observation_t1. The "
         "output ``state`` becomes state_t0 in the next iteration.",
         prompt="get_state", role="structuring", phase="append",
         inputs=["goal", "state", "action", "observation"],
         outputs=["state"], per="per_step"),

    # ============ Close (Memory.close, core/memory.py:104) ============
    _comp("finalize_traj", "Finalize trajectories",
          "memory['episodic'].append(self.trajectory); self.trajectory = [].",
          phase="close"),
    _loop("loop_traj", "for each trajectory in episodic", phase="close",
          scope="Iterate over closed trajectories (j, trajectory)."),
    _loop("loop_step_in_traj", "for each step in trajectory", phase="close",
          scope="Iterate over (i, step) within the current trajectory."),
    _comp("build_traj_str", "Build trajectory_str",
          "Append `Step i:\\n-State: …\\n-Action: …\\n-Reward: …\\n` to "
          "trajectory_str.",
          phase="close", inputs=["step.state", "step.action", "step.reward"],
          outputs=["trajectory_str"]),
    _llm("get_semantic", "Semantic facts",
         "Extract durable facts (with tags) from one observation. "
         "Each parsed fact has {statement, tags}; the list lives under "
         "``facts``.",
         prompt="get_semantic", role="structuring", phase="close",
         inputs=["observation"],
         outputs=["facts"],
         per="per_step"),
    _emb("embed_facts_and_tags", "Embed facts + tags",
         "For each fact: embed semantic_memory and embed every tag.",
         phase="close",
         inputs=["facts[].semantic_memory", "facts[].tags"],
         outputs=["semantic_emb[]"]),
    _llm("get_procedural", "Procedural memory",
         "Distill an experiential insight + goal from the trajectory. The "
         "template's {trajectory} variable is the concatenated trajectory_str "
         "built in Memory.close().",
         prompt="get_procedural", role="structuring", phase="close",
         inputs=["trajectory"],
         outputs=["goal", "experience", "return"],
         per="per_trajectory"),
    _emb("embed_proc_subgoal", "Embed subgoal",
         "Embed the goal returned by get_procedural for procedural lookup.",
         phase="close", inputs=["goal"], outputs=["procedural_emb.subgoal"]),
    _llm("get_return", "Return score (unused)",
         "Numeric score for a procedural memory. Defined but NOT called by "
         "Memory.close — included for completeness.",
         prompt="get_return", role="structuring", phase="close",
         inputs=["subgoal", "procedural_memory"], outputs=["score"],
         optional=True),

    # ============ Insert (MemoryGraph.insert, core/memory_graph.py:308) ============
    _store("persist_episodic", "Persist episodic nodes",
           "Create EpisodicNode per step and self.storage.add_episodic(...).",
           phase="insert", inputs=["memory.episodic"]),
    _store("persist_semantic", "Persist semantic + tag links",
           "Create SemanticNode + TagNode (or link existing tags), embed "
           "new tags, then self.storage.add_semantic / add_tag.",
           phase="insert", inputs=["memory.semantic"]),
    _loop("loop_proc", "for each procedural memory", phase="insert",
          scope="Iterate over (proc_item, proc_emb_item) pairs."),
    _comp("lookup_subgoal", "Lookup subgoal node",
          "subgoal_node = self.subgoal2node.get(subgoal_str).",
          phase="insert", inputs=["proc_item.subgoal"],
          outputs=["subgoal_node?"]),
    _branch("br_subgoal_exists", "Subgoal exists?", phase="insert",
            condition="Is subgoal_node already in the graph?",
            outcomes=[
                "yes → call get_new_subgoal to merge old + new",
                "no  → create a fresh SubgoalNode",
            ]),
    _llm("get_new_subgoal", "Merge subgoals",
         "Combine an existing subgoal with the incoming one. The template "
         "uses {goal_1}/{goal_2} (existing / new respectively).",
         prompt="get_new_subgoal", role="consolidation", phase="insert",
         inputs=["goal_1", "goal_2"],
         outputs=["merged"], optional=True),
    _emb("embed_merged_subgoal", "Embed merged subgoal",
         "Re-embed the merged subgoal text.",
         phase="insert", inputs=["merged_subgoal"],
         outputs=["subgoal_node.embedding"], optional=True),
    _comp("create_subgoal", "Create new subgoal node",
          "SubgoalNode(subgoal=…, embedding=…) and add to self.subgoal_nodes.",
          phase="insert", optional=True),
    _store("persist_proc_subgoal", "Persist procedural + subgoal",
           "self.storage.add_procedural / add_subgoal / update_subgoal.",
           phase="insert"),

    # ============ Retrieve (retrieve_memory, core/memory_graph.py:899) ============
    _llm("get_plan", "Recall plan",
         "Decide which tags and which next_subgoal to query against.",
         prompt="get_plan", role="retrieval", phase="retrieve",
         inputs=["goal", "subgoal", "state", "observation"],
         outputs=["next_subgoal", "query_tags"]),
    _branch("br_mode_set", "Mode supplied?", phase="retrieve",
            condition="Did the caller pass `mode`?",
            outcomes=[
                "no  → call get_mode (LLM) to classify",
                "yes → use the supplied mode",
            ]),
    _llm("get_mode", "Mode classifier",
         "Classify which memory mode to use (semantic / procedural / "
         "episodic).",
         prompt="get_mode", role="retrieval", phase="retrieve",
         inputs=["observation", "task_type"], outputs=["mode"],
         optional=True),
    _branch("br_node_retrieval", "Node retrieval by mode", phase="retrieve",
            condition="Which retriever(s) to run, by mode?",
            outcomes=[
                "mode ∈ {semantic, episodic} → retrieve_semantic_nodes",
                "mode ∈ {procedural, episodic} → retrieve_procedural_nodes",
                "mode = episodic → also retrieve_episodic_nodes",
            ]),
    _comp("retrieve_semantic_nodes", "Retrieve semantic nodes",
          "Tag voting + similarity scoring against semantic embeddings.",
          phase="retrieve",
          inputs=["observation", "query_tags"],
          outputs=["semantic_nodes"], optional=True),
    _comp("retrieve_procedural_nodes", "Retrieve procedural nodes",
          "Subgoal lookup + similarity scoring against procedural memories.",
          phase="retrieve",
          inputs=["next_subgoal"],
          outputs=["procedural_nodes"], optional=True),
    _comp("retrieve_episodic_nodes", "Retrieve episodic nodes",
          "Pulls episodic context strings (uses retrieve_semantic_nodes_wo_tag "
          "internally).",
          phase="retrieve",
          inputs=["observation"],
          outputs=["episodic_memory_str"], optional=True),
    _branch("br_template_by_mode", "Pick reasoning template", phase="retrieve",
            condition="Which reasoning_* template to render, by mode?",
            outcomes=[
                "episodic_memory  → reasoning_episodic",
                "semantic_memory  → reasoning_semantic",
                "procedural_memory → reasoning_procedural",
            ]),
    _tmpl("render_reasoning_episodic", "Render reasoning_episodic",
          "Render the reasoning_episodic template into chat messages. "
          "Runs only when mode=episodic_memory.",
          prompt="reasoning_episodic", phase="retrieve",
          inputs=["observation", "episodic_memory"],
          outputs=["messages"], optional=True),
    _tmpl("render_reasoning_semantic", "Render reasoning_semantic",
          "Render the reasoning_semantic template. Runs only when "
          "mode=semantic_memory.",
          prompt="reasoning_semantic", phase="retrieve",
          inputs=["observation", "semantic_memory"],
          outputs=["messages"], optional=True),
    _tmpl("render_reasoning_procedural", "Render reasoning_procedural",
          "Render the reasoning_procedural template. Runs only when "
          "mode=procedural_memory.",
          prompt="reasoning_procedural", phase="retrieve",
          inputs=["observation", "procedural_memory"],
          outputs=["messages"], optional=True),

    # ============ Reason (retrieve_and_reason, core/memory_graph.py:1138) ============
    _comp("reason_inherit_messages", "Use messages from Retrieve",
          "messages produced by the rendered reasoning_* template.",
          phase="reason", inputs=["messages"], outputs=["messages"]),
    _llm("reason_llm_call", "Reasoning LLM call",
         "self.reasoning_llm.complete(messages=messages). Uses whichever "
         "reasoning_* template was rendered upstream — there is no separate "
         "prompt at this step.",
         prompt="", role="reasoning", phase="reason",
         inputs=["messages"], outputs=["reasoning"]),

    # ============ Consolidate (update_semantic_subgraph, core/memory_graph.py:1209) ============
    _comp("scope_semantic", "Scope candidate semantics",
          "Pick semantic nodes within `only_update_recent_window` (or all "
          "older than time_st) and apply credibility decay.",
          phase="consolidate", outputs=["scope[]"]),
    _loop("loop_semantic", "for each semantic node in scope",
          phase="consolidate",
          scope="Walks scope[]; skips inactive / already-updated nodes."),
    _branch("br_keep_active", "Below credibility floor?",
            phase="consolidate",
            condition="Is sem_node.Credibility < min_credibility_to_keep_active?",
            outcomes=[
                "yes → soft-deactivate and skip",
                "no  → continue to candidate gathering",
            ]),
    _comp("collect_candidates", "Collect candidates by tag",
          "Gather sibling semantics from each tag_node (capped by "
          "max_candidates_per_tag / max_total_candidates).",
          phase="consolidate", outputs=["cand_ids"]),
    _comp("filter_candidates", "Filter candidates",
          "Drop inactive, future-time, already-updated, and self-referential "
          "candidates.",
          phase="consolidate", inputs=["cand_ids"], outputs=["filtered"]),
    _comp("score_candidates", "Score candidates",
          "Compute embedding similarity → semantic_equal.evaluate → top-k.",
          phase="consolidate", inputs=["filtered"], outputs=["topk"]),
    _loop("loop_topk", "for each top-k candidate", phase="consolidate",
          scope="Walk top-k by score, descending."),
    _branch("br_merge_threshold", "Score ≥ merge_threshold?",
            phase="consolidate",
            condition="Is the candidate's score above merge_threshold?",
            outcomes=[
                "yes → call merge_semantic (which calls get_new_semantic)",
                "no  → skip",
            ]),
    _llm("get_new_semantic", "Merge decision",
         "Merge two overlapping semantic memories — produces "
         "merged_statement + relationship + dedup decisions.",
         prompt="get_new_semantic", role="consolidation", phase="consolidate",
         inputs=["memory_earlier", "memory_later"],
         outputs=["merged_statement", "relationship",
                  "deactivate_earlier", "deactivate_later"],
         optional=True),
    _emb("embed_merged_semantic", "Embed merged statement",
         "self.embedder.embed(merged_str).",
         phase="consolidate",
         inputs=["merged_statement"],
         outputs=["merged_emb"], optional=True),
    _store("persist_merged", "Persist merged semantic",
           "Create new SemanticNode (with son links) and storage.add_semantic; "
           "deactivate originals as flagged.",
           phase="consolidate"),
]


# ----------------------------------------------------------------------- #
# Edges — explicit so branches and loops are not muddled with sequence.
# ----------------------------------------------------------------------- #


EDGES: List[PipelineEdge] = [
    # ---- Append ----
    PipelineEdge("get_subgoal", "get_reward", "seq"),
    PipelineEdge("get_reward", "br_traj_empty", "seq"),
    PipelineEdge("br_traj_empty", "embed_prev_subgoal", "branch", "non-empty"),
    PipelineEdge("br_traj_empty", "append_step", "branch", "empty"),
    PipelineEdge("embed_prev_subgoal", "embed_curr_subgoal", "seq"),
    PipelineEdge("embed_curr_subgoal", "subgoal_similarity", "seq"),
    PipelineEdge("subgoal_similarity", "br_segment", "seq"),
    PipelineEdge("br_segment", "close_segment", "branch", "sim < 0.75"),
    PipelineEdge("br_segment", "append_step", "branch", "sim ≥ 0.75"),
    PipelineEdge("close_segment", "append_step", "seq"),
    PipelineEdge("append_step", "get_state", "seq"),

    # ---- Close ----
    PipelineEdge("finalize_traj", "loop_traj", "seq"),
    PipelineEdge("loop_traj", "loop_step_in_traj", "seq"),
    PipelineEdge("loop_step_in_traj", "build_traj_str", "seq"),
    PipelineEdge("build_traj_str", "get_semantic", "seq"),
    PipelineEdge("get_semantic", "embed_facts_and_tags", "seq"),
    PipelineEdge("embed_facts_and_tags", "loop_step_in_traj", "loop_back",
                 "next step"),
    PipelineEdge("embed_facts_and_tags", "get_procedural", "seq",
                 "after step loop"),
    PipelineEdge("get_procedural", "embed_proc_subgoal", "seq"),
    PipelineEdge("embed_proc_subgoal", "loop_traj", "loop_back",
                 "next trajectory"),
    PipelineEdge("embed_proc_subgoal", "get_return", "seq", "(unused path)"),

    # ---- Insert ----
    PipelineEdge("persist_episodic", "persist_semantic", "seq"),
    PipelineEdge("persist_semantic", "loop_proc", "seq"),
    PipelineEdge("loop_proc", "lookup_subgoal", "seq"),
    PipelineEdge("lookup_subgoal", "br_subgoal_exists", "seq"),
    PipelineEdge("br_subgoal_exists", "get_new_subgoal", "branch", "exists"),
    PipelineEdge("br_subgoal_exists", "create_subgoal", "branch", "new"),
    PipelineEdge("get_new_subgoal", "embed_merged_subgoal", "seq"),
    PipelineEdge("embed_merged_subgoal", "persist_proc_subgoal", "seq"),
    PipelineEdge("create_subgoal", "persist_proc_subgoal", "seq"),
    PipelineEdge("persist_proc_subgoal", "loop_proc", "loop_back",
                 "next procedural"),

    # ---- Retrieve ----
    PipelineEdge("get_plan", "br_mode_set", "seq"),
    PipelineEdge("br_mode_set", "get_mode", "branch", "mode is None"),
    PipelineEdge("br_mode_set", "br_node_retrieval", "branch", "mode supplied"),
    PipelineEdge("get_mode", "br_node_retrieval", "seq"),
    PipelineEdge("br_node_retrieval", "retrieve_semantic_nodes", "branch",
                 "semantic | episodic"),
    PipelineEdge("br_node_retrieval", "retrieve_procedural_nodes", "branch",
                 "procedural | episodic"),
    PipelineEdge("br_node_retrieval", "retrieve_episodic_nodes", "branch",
                 "episodic"),
    PipelineEdge("retrieve_semantic_nodes", "br_template_by_mode", "seq"),
    PipelineEdge("retrieve_procedural_nodes", "br_template_by_mode", "seq"),
    PipelineEdge("retrieve_episodic_nodes", "br_template_by_mode", "seq"),
    PipelineEdge("br_template_by_mode", "render_reasoning_episodic", "branch",
                 "episodic_memory"),
    PipelineEdge("br_template_by_mode", "render_reasoning_semantic", "branch",
                 "semantic_memory"),
    PipelineEdge("br_template_by_mode", "render_reasoning_procedural", "branch",
                 "procedural_memory"),

    # ---- Reason ----
    PipelineEdge("reason_inherit_messages", "reason_llm_call", "seq"),

    # Cross-phase: messages flow from Retrieve into Reason
    PipelineEdge("render_reasoning_episodic", "reason_inherit_messages",
                 "alt", "if /reason"),
    PipelineEdge("render_reasoning_semantic", "reason_inherit_messages",
                 "alt", "if /reason"),
    PipelineEdge("render_reasoning_procedural", "reason_inherit_messages",
                 "alt", "if /reason"),

    # ---- Consolidate ----
    PipelineEdge("scope_semantic", "loop_semantic", "seq"),
    PipelineEdge("loop_semantic", "br_keep_active", "seq"),
    PipelineEdge("br_keep_active", "loop_semantic", "branch",
                 "yes (skip)"),
    PipelineEdge("br_keep_active", "collect_candidates", "branch",
                 "no (continue)"),
    PipelineEdge("collect_candidates", "filter_candidates", "seq"),
    PipelineEdge("filter_candidates", "score_candidates", "seq"),
    PipelineEdge("score_candidates", "loop_topk", "seq"),
    PipelineEdge("loop_topk", "br_merge_threshold", "seq"),
    PipelineEdge("br_merge_threshold", "loop_topk", "branch", "no (skip)"),
    PipelineEdge("br_merge_threshold", "get_new_semantic", "branch", "yes"),
    PipelineEdge("get_new_semantic", "embed_merged_semantic", "seq"),
    PipelineEdge("embed_merged_semantic", "persist_merged", "seq"),
    PipelineEdge("persist_merged", "loop_topk", "loop_back", "next candidate"),
    PipelineEdge("persist_merged", "loop_semantic", "loop_back",
                 "next semantic"),
]


# Cross-phase ghost edges (rendered subtly, hint at data flow between phases)
CROSS_PHASE_EDGES: List[PipelineEdge] = [
    PipelineEdge("get_state", "get_subgoal", "loop_back",
                 "state_t1 → state_t0 (next iteration)"),
    PipelineEdge("close_segment", "finalize_traj", "alt",
                 "feeds episodic[]"),
    PipelineEdge("get_procedural", "persist_episodic", "alt",
                 "memory → insert"),
]


# ----------------------------------------------------------------------- #
# Serialization
# ----------------------------------------------------------------------- #


KINDS: List[str] = [
    "llm", "template_render", "embed", "compute", "storage",
    "branch", "loop_marker",
]


def to_dict() -> Dict[str, Any]:
    """Return a JSON-serializable representation of the pipeline spec."""
    return {
        "phases": [asdict(p) for p in PHASES],
        "steps": [asdict(s) for s in STEPS],
        "edges": [asdict(e) for e in EDGES],
        "cross_phase_edges": [asdict(e) for e in CROSS_PHASE_EDGES],
        "roles": ["default", "structuring", "retrieval", "reasoning", "consolidation"],
        "kinds": list(KINDS),
    }
