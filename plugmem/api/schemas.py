"""Pydantic request/response models for the PlugMem API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Graphs
# ------------------------------------------------------------------ #

class GraphCreateRequest(BaseModel):
    graph_id: Optional[str] = Field(
        None,
        description="Optional custom graph ID. Auto-generated if omitted.",
    )


class GraphResponse(BaseModel):
    graph_id: str
    stats: Dict[str, int] = Field(default_factory=dict)


class GraphListResponse(BaseModel):
    graphs: List[str]


# ------------------------------------------------------------------ #
# Memory insertion
# ------------------------------------------------------------------ #

class TrajectoryStep(BaseModel):
    observation: str
    action: str


class SemanticMemoryInput(BaseModel):
    semantic_memory: str
    tags: List[str] = Field(default_factory=list)


class ProceduralMemoryInput(BaseModel):
    subgoal: str
    procedural_memory: str
    return_value: float = Field(0.0, alias="return")

    model_config = {"populate_by_name": True}


class EpisodicStep(BaseModel):
    observation: str = ""
    action: str = ""
    subgoal: str = ""
    state: str = ""
    reward: str = ""
    time: Any = ""


class MemoryInsertRequest(BaseModel):
    mode: str = Field(
        ...,
        description='"trajectory" or "structured"',
        pattern="^(trajectory|structured)$",
    )
    session_id: Optional[str] = Field(
        None,
        description=(
            "Stamps every node created by this insert with the given session id. "
            "Used by the Sessions view + recall audit log to group nodes by run."
        ),
    )

    # trajectory mode
    goal: Optional[str] = None
    steps: Optional[List[TrajectoryStep]] = None

    # structured mode
    episodic: Optional[List[List[EpisodicStep]]] = None
    semantic: Optional[List[SemanticMemoryInput]] = None
    procedural: Optional[List[ProceduralMemoryInput]] = None


class MemoryInsertResponse(BaseModel):
    status: str = "ok"
    stats: Dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Retrieval
# ------------------------------------------------------------------ #

class RetrieveRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = Field(
        None,
        description=(
            'null (auto-detect), "semantic_memory", '
            '"episodic_memory", or "procedural_memory"'
        ),
    )
    session_id: Optional[str] = Field(
        None,
        description="If set, the recall is logged against this session id.",
    )


class RetrieveResponse(BaseModel):
    mode: str
    reasoning_prompt: List[Dict[str, str]]
    variables: Dict[str, Any] = Field(default_factory=dict)


class ReasonRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = None
    session_id: Optional[str] = Field(
        None,
        description="If set, the reasoning recall is logged against this session id.",
    )


class ReasonResponse(BaseModel):
    mode: str
    reasoning: str
    reasoning_prompt: List[Dict[str, str]]


# ------------------------------------------------------------------ #
# Consolidation
# ------------------------------------------------------------------ #

class ConsolidateRequest(BaseModel):
    merge_threshold: float = 0.5
    max_merges_per_node: int = 1
    max_candidates_per_tag: int = 200
    max_total_candidates: int = 800
    min_credibility_to_keep_active: int = -10
    credibility_decay: int = 0
    only_update_recent_window: Optional[int] = None
    allow_merge_with_common_episodic_nodes: bool = False


class ConsolidateResponse(BaseModel):
    status: str = "ok"
    stats: Dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Stats / Nodes
# ------------------------------------------------------------------ #

class StatsResponse(BaseModel):
    graph_id: str
    stats: Dict[str, int]


class NodeListResponse(BaseModel):
    graph_id: str
    node_type: str
    count: int
    nodes: List[Dict[str, Any]]


# ------------------------------------------------------------------ #
# Inspector
# ------------------------------------------------------------------ #

class SearchResponse(BaseModel):
    graph_id: str
    node_type: str
    query: str
    count: int
    nodes: List[Dict[str, Any]]


class NodeDetailResponse(BaseModel):
    graph_id: str
    node_type: str
    node: Dict[str, Any]
    edges: Dict[str, List[Dict[str, Any]]]


class SemanticUpdateRequest(BaseModel):
    is_active: Optional[bool] = None
    text: Optional[str] = Field(
        None,
        description="New semantic text. If supplied, the embedding is recomputed.",
    )
    tags: Optional[List[str]] = Field(
        None,
        description=(
            "Replacement tag list. Tags are reconciled — existing tag nodes "
            "are reused by exact-match string, new tags are created and "
            "embedded, removed tags are detached (the tag node is kept)."
        ),
    )
    credibility: Optional[int] = None


class ProceduralUpdateRequest(BaseModel):
    text: Optional[str] = Field(
        None,
        description="New procedural text. If supplied, the embedding is recomputed.",
    )
    return_value: Optional[float] = Field(None, alias="return")

    model_config = {"populate_by_name": True}


class TagUpdateRequest(BaseModel):
    tag: Optional[str] = Field(
        None,
        description="New tag string. If supplied, the embedding is recomputed.",
    )
    importance: Optional[int] = None


class SubgoalUpdateRequest(BaseModel):
    subgoal: Optional[str] = Field(
        None,
        description="New subgoal string. If supplied, the embedding is recomputed.",
    )


class EpisodicUpdateRequest(BaseModel):
    observation: Optional[str] = None
    action: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    reward: Optional[str] = None


class RecallTraceRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = Field(
        None,
        description=(
            'null/omit (default = semantic_memory unless auto_plan), '
            '"semantic_memory", "episodic_memory", or "procedural_memory"'
        ),
    )
    query_tags: Optional[List[str]] = Field(
        None,
        description="Manual tags to skip the LLM planner. Empty list disables tag voting.",
    )
    next_subgoal: Optional[str] = None
    auto_plan: bool = Field(
        False,
        description="If True, fill missing mode/tags/subgoal via the LLM planner (paid).",
    )
    session_id: Optional[str] = Field(
        None,
        description="If set, the trace is logged to the recall audit under this session id.",
    )


class RecallTraceResponse(BaseModel):
    mode: str
    plan: Dict[str, Any]
    trace: Dict[str, Any]
    selected: Dict[str, List[int]]
    rendered_prompt: List[Dict[str, str]]


class TopologyResponse(BaseModel):
    graph_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    counts: Dict[str, int]
    truncated: bool
    node_limit: int


class RecallAuditEntry(BaseModel):
    recall_id: int
    endpoint: str
    ts: str
    graph_time: int = 0
    session_id: Optional[str] = None
    observation: str = ""
    goal: str = ""
    subgoal: str = ""
    state: str = ""
    task_type: str = ""
    mode: str = ""
    next_subgoal: str = ""
    query_tags: List[str] = Field(default_factory=list)
    selected_semantic_ids: List[int] = Field(default_factory=list)
    selected_procedural_ids: List[int] = Field(default_factory=list)
    n_messages: int = 0


class RecallListResponse(BaseModel):
    graph_id: str
    count: int
    session_id: Optional[str] = None
    recalls: List[RecallAuditEntry]


class SessionListResponse(BaseModel):
    graph_id: str
    sessions: List[str]


class SessionEvent(BaseModel):
    """One row in the chronological session view.

    Two flavours: ``kind="insert"`` (a node was created) and
    ``kind="recall"`` (a /retrieve, /reason, or /recall_trace fired).
    Both carry ``time`` so the frontend sorts on a unified axis; nodes
    use the graph's monotonic time counter, recalls use ``graph_time``
    captured when the recall was logged.
    """
    kind: str
    time: int
    # insert fields
    node_type: Optional[str] = None
    node_id: Optional[int] = None
    label: Optional[str] = None
    text: Optional[str] = None
    is_active: Optional[bool] = None
    credibility: Optional[int] = None
    return_value: Optional[float] = None
    subgoal: Optional[str] = None
    # recall fields
    endpoint: Optional[str] = None
    recall_id: Optional[int] = None
    ts: Optional[str] = None
    observation: Optional[str] = None
    mode: Optional[str] = None
    next_subgoal: Optional[str] = None
    query_tags: List[str] = Field(default_factory=list)
    selected_semantic_ids: List[int] = Field(default_factory=list)
    selected_procedural_ids: List[int] = Field(default_factory=list)
    n_messages: Optional[int] = None


class SessionTimelineResponse(BaseModel):
    graph_id: str
    session_id: str
    count: int
    events: List[SessionEvent]


# ------------------------------------------------------------------ #
# Pipeline (inspector)
# ------------------------------------------------------------------ #

class PipelinePhase(BaseModel):
    id: str
    label: str
    description: str
    trigger: str = ""
    triggered_by: List[str] = Field(default_factory=list)


class PipelineStep(BaseModel):
    id: str
    label: str
    description: str
    kind: str = "llm"
    phase: str
    prompt_name: str = ""
    role: str = ""
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    per: str = "per_call"
    optional: bool = False
    branch_condition: str = ""
    branch_outcomes: List[str] = Field(default_factory=list)
    loop_scope: str = ""


class PipelineEdge(BaseModel):
    source: str
    target: str
    kind: str = "seq"
    label: str = ""


class PipelineSpecResponse(BaseModel):
    phases: List[PipelinePhase]
    steps: List[PipelineStep]
    edges: List[PipelineEdge]
    cross_phase_edges: List[PipelineEdge] = Field(default_factory=list)
    roles: List[str]
    kinds: List[str] = Field(default_factory=list)


class PromptLayer(BaseModel):
    """system + user template for one resolution layer of a prompt."""
    system: str
    user: str


class PromptInfo(BaseModel):
    """Per-layer view of a single prompt + which layer is currently winning."""
    name: str
    builtin: PromptLayer
    service: Optional[PromptLayer] = None
    graph: Optional[PromptLayer] = None
    effective: PromptLayer
    has_graph_override: bool = False


class PromptListResponse(BaseModel):
    graph_id: str
    prompts: List[PromptInfo]


class PromptUpdateRequest(BaseModel):
    system: str = Field(..., description="System message template (Python {var} placeholders).")
    user: str = Field(..., description="User message template (Python {var} placeholders).")


class PromptUpdateResponse(BaseModel):
    name: str
    graph_id: str
    persisted_to: Optional[str] = Field(
        None,
        description="Filesystem path the per-graph YAML was written to.",
    )
    info: PromptInfo


class PromptResetResponse(BaseModel):
    name: str
    graph_id: str
    cleared: bool
    persisted_to: Optional[str] = None
    info: PromptInfo


class PromptPreviewRequest(BaseModel):
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Substitution variables for the {placeholder} fields.",
    )
    system: Optional[str] = Field(
        None,
        description=(
            "Optional unsaved system template. If both system and user are "
            "provided, preview renders these instead of the registered "
            "template — useful for previewing in-progress edits."
        ),
    )
    user: Optional[str] = Field(
        None,
        description="Optional unsaved user template (paired with system).",
    )


class PromptPreviewMessage(BaseModel):
    role: str
    content: str


class PromptPreviewResponse(BaseModel):
    name: str
    graph_id: str
    messages: List[PromptPreviewMessage]


# ------------------------------------------------------------------ #
# Model bindings (Phase 3)
# ------------------------------------------------------------------ #


class ModelBinding(BaseModel):
    role: str
    base_url: str
    model: str
    has_api_key: bool
    is_azure: bool
    azure_api_version: str = ""
    falls_back_to_default: bool = False


class ModelListResponse(BaseModel):
    bindings: List[ModelBinding]
    roles: List[str]


class ModelUpdateRequest(BaseModel):
    base_url: str
    model: str
    api_key: Optional[str] = Field(
        None,
        description=(
            "Set to a non-empty string to update the API key. None or omitted "
            "keeps the existing key. An empty string clears it."
        ),
    )
    is_azure: bool = False
    azure_api_version: str = "2024-05-01-preview"


class ModelUpdateResponse(BaseModel):
    role: str
    binding: ModelBinding


class ModelTestRequest(BaseModel):
    base_url: str
    model: str
    api_key: Optional[str] = None
    is_azure: bool = False
    azure_api_version: str = "2024-05-01-preview"
    prompt: str = Field(
        "ping",
        description="Probe message — sent as a single user message via complete().",
    )
    max_tokens: int = 32
    role: Optional[str] = Field(
        None,
        description=(
            "If set and api_key is None, the existing api_key for that role "
            "is used (so the user can test without re-typing the key)."
        ),
    )


class ModelTestResponse(BaseModel):
    ok: bool
    latency_ms: int = 0
    sample: str = ""
    error: Optional[str] = None


# ------------------------------------------------------------------ #
# Pipeline traces (Phase 4)
# ------------------------------------------------------------------ #


class TraceSummary(BaseModel):
    trace_id: str
    ts: str
    endpoint: str
    duration_ms: int
    ok: bool
    num_steps: int
    session_id: Optional[str] = None
    error: Optional[str] = None


class TraceListResponse(BaseModel):
    graph_id: str
    traces: List[TraceSummary]
    cap: int = 100


class TraceStep(BaseModel):
    name: str
    model: str = ""
    variables: Dict[str, Any] = Field(default_factory=dict)
    response: str = ""
    parsed: Optional[Any] = None
    latency_ms: int = 0
    ts_offset_ms: int = 0
    error: Optional[str] = None


class TraceDetailResponse(BaseModel):
    graph_id: str
    trace_id: str
    ts: str
    endpoint: str
    duration_ms: int
    ok: bool
    num_steps: int
    session_id: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    steps: List[TraceStep] = Field(default_factory=list)


class TraceCapResponse(BaseModel):
    graph_id: str
    cap: int


class TraceCapRequest(BaseModel):
    cap: int = Field(
        ...,
        ge=0,
        description="Retention cap. 0 means unlimited.",
    )


class PipelineStepStats(BaseModel):
    name: str
    count: int = 0
    mean_latency_ms: int = 0
    errors: int = 0
    last_ts: str = ""
    last_latency_ms: int = 0
    recent_latencies: List[int] = Field(default_factory=list)


class PipelineStatsResponse(BaseModel):
    graph_id: str
    stats: Dict[str, PipelineStepStats] = Field(default_factory=dict)


class StepTraceSummary(BaseModel):
    trace_id: str
    ts: str
    endpoint: str
    duration_ms: int
    ok: bool
    step_latency_ms: int
    step_error: Optional[str] = None


class StepTracesResponse(BaseModel):
    graph_id: str
    step_name: str
    traces: List[StepTraceSummary]


# ------------------------------------------------------------------ #
# Pipeline binding (Phase 5.5)
# ------------------------------------------------------------------ #


class PipelineInfo(BaseModel):
    name: str
    description: str = ""


class PipelineListResponse(BaseModel):
    pipelines: List[PipelineInfo]
    default: str


class PipelineBindingResponse(BaseModel):
    graph_id: str
    pipeline: str


class PipelineBindingRequest(BaseModel):
    pipeline: str


class PipelineReloadResponse(BaseModel):
    reloaded_modules: List[str]
    registered: List[str]


# ------------------------------------------------------------------ #
# Spec-driven YAML editor (Phase 6.5a)
# ------------------------------------------------------------------ #


class SpecVersionInfo(BaseModel):
    version_id: str
    ts: str
    parent_version_id: Optional[str] = None
    note: str = ""
    active: bool = False


class SpecDocumentResponse(BaseModel):
    graph_id: str
    content: str = ""
    active_version_id: Optional[str] = None
    exists: bool = False
    live_path: str


class SpecSaveRequest(BaseModel):
    content: str = Field(..., description="Full YAML source — saved verbatim.")
    note: str = Field("", description="Optional human-readable note for the audit log.")


class SpecSaveResponse(BaseModel):
    graph_id: str
    version: SpecVersionInfo


class SpecValidateRequest(BaseModel):
    content: str


class SpecValidateResponse(BaseModel):
    ok: bool
    error: Optional[str] = None


class SpecVersionsResponse(BaseModel):
    graph_id: str
    versions: List[SpecVersionInfo] = Field(default_factory=list)
    active_version_id: Optional[str] = None


class SpecVersionContentResponse(BaseModel):
    graph_id: str
    version: SpecVersionInfo
    content: str


class PipelineSpecSample(BaseModel):
    key: str
    label: str
    description: str
    content: str


class PipelineSpecSamplesResponse(BaseModel):
    samples: List[PipelineSpecSample]


class PipelineNodeSnippet(BaseModel):
    key: str
    label: str
    description: str
    snippet: str


class PipelineNodeSnippetsResponse(BaseModel):
    snippets: List[PipelineNodeSnippet]


# ------------------------------------------------------------------ #
# Canvas layout (Phase 6.5b) — per-graph sidecar
# ------------------------------------------------------------------ #


class CanvasNodePosition(BaseModel):
    x: float
    y: float


class CanvasViewport(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


class CanvasLayoutResponse(BaseModel):
    graph_id: str
    positions: Dict[str, CanvasNodePosition] = Field(default_factory=dict)
    viewport: Optional[CanvasViewport] = None


class CanvasLayoutSaveRequest(BaseModel):
    positions: Dict[str, CanvasNodePosition]
    viewport: Optional[CanvasViewport] = None


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    embedding_available: bool
    chroma_available: bool
