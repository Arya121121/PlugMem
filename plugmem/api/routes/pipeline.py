"""Pipeline inspector endpoints.

Phase 1: read-only spec for the inspector UI.
Phase 2: per-graph prompt CRUD + preview. Builtins and the service-wide
``_defaults.yaml`` are read-only from the API; edits land only in the
per-graph layer and are persisted to ``{prompts_dir}/{graph_id}.yaml``.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

import logging
import time

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager, get_llm, get_prompt_registry
from plugmem.api.schemas import (
    ModelBinding,
    ModelListResponse,
    ModelTestRequest,
    ModelTestResponse,
    ModelUpdateRequest,
    ModelUpdateResponse,
    PipelineSpecResponse,
    PromptInfo,
    PromptLayer,
    PromptListResponse,
    PromptPreviewMessage,
    PromptPreviewRequest,
    PromptPreviewResponse,
    PromptResetResponse,
    PromptUpdateRequest,
    PromptUpdateResponse,
    PipelineBindingRequest,
    PipelineBindingResponse,
    PipelineInfo,
    CanvasLayoutResponse,
    CanvasLayoutSaveRequest,
    CanvasNodePosition,
    CanvasViewport,
    PipelineListResponse,
    PipelineNodeSnippet,
    PipelineNodeSnippetsResponse,
    PipelineSpecSample,
    PipelineSpecSamplesResponse,
    PipelineStatsResponse,
    PipelineStepStats,
    SpecDocumentResponse,
    SpecSaveRequest,
    SpecSaveResponse,
    SpecValidateRequest,
    SpecValidateResponse,
    SpecVersionContentResponse,
    SpecVersionInfo,
    SpecVersionsResponse,
    StepTraceSummary,
    StepTracesResponse,
    TraceCapRequest,
    TraceCapResponse,
    TraceDetailResponse,
    TraceListResponse,
    TraceStep,
    TraceSummary,
)
from openai import AzureOpenAI, OpenAI

from plugmem.clients.llm import OpenAICompatibleLLMClient
from plugmem.clients.llm_router import ROLES as ROUTER_ROLES, LLMRouter, expand_env_vars
from plugmem.core.pipeline_spec import to_dict as pipeline_spec_dict
from plugmem.graph_manager import GraphManager
from plugmem.pipelines import (
    DEFAULT_PIPELINE_NAME,
    is_registered as pipeline_is_registered,
    list_pipelines as list_registered_pipelines,
)
from plugmem.pipelines import layout_storage, pipeline_views, spec_storage
from plugmem.pipelines.spec_driven import load_yaml_str
from plugmem.pipelines.sample_specs import PLUGMEM_DEFAULT_RETRIEVE_YAML
from plugmem.prompts.registry import PromptRegistry, TemplatePrompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"], dependencies=[Depends(require_api_key)])
graph_router = APIRouter(prefix="/graphs", tags=["pipeline"], dependencies=[Depends(require_api_key)])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _check_graph_exists(graph_id: str) -> GraphManager:
    gm = get_graph_manager()
    try:
        gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    return gm


def _registry() -> PromptRegistry:
    return get_prompt_registry()


def _layer(d) -> PromptLayer | None:
    if d is None:
        return None
    return PromptLayer(system=d.get("system", ""), user=d.get("user", ""))


def _to_info(name: str, registry: PromptRegistry, graph_id: str) -> PromptInfo:
    try:
        info = registry.inspect(name, graph_id=graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt: '{name}'")
    return PromptInfo(
        name=info["name"],
        builtin=_layer(info["builtin"]),
        service=_layer(info["service"]),
        graph=_layer(info["graph"]),
        effective=_layer(info["effective"]),
        has_graph_override=info["has_graph_override"],
    )


# ------------------------------------------------------------------ #
# Spec (Phase 1)
# ------------------------------------------------------------------ #


@router.get("/spec", response_model=PipelineSpecResponse)
def get_pipeline_spec() -> PipelineSpecResponse:
    return PipelineSpecResponse(**pipeline_spec_dict())


# ------------------------------------------------------------------ #
# Pipeline binding (Phase 5.5) — global registry + per-graph selection
# ------------------------------------------------------------------ #


@router.get("/pipelines", response_model=PipelineListResponse)
def list_pipelines_endpoint() -> PipelineListResponse:
    """Available memory-pipeline implementations registered in this process."""
    rows = list_registered_pipelines()
    return PipelineListResponse(
        pipelines=[PipelineInfo(**r) for r in rows],
        default=DEFAULT_PIPELINE_NAME,
    )


@graph_router.get(
    "/{graph_id}/pipeline",
    response_model=PipelineBindingResponse,
)
def get_graph_pipeline(graph_id: str) -> PipelineBindingResponse:
    gm = _check_graph_exists(graph_id)
    name = gm.storage.get_pipeline_name(graph_id)
    return PipelineBindingResponse(graph_id=graph_id, pipeline=name)


@graph_router.put(
    "/{graph_id}/pipeline",
    response_model=PipelineBindingResponse,
)
def set_graph_pipeline(graph_id: str, body: PipelineBindingRequest) -> PipelineBindingResponse:
    gm = _check_graph_exists(graph_id)
    if not pipeline_is_registered(body.pipeline):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown pipeline '{body.pipeline}'. "
                f"Registered: {[p['name'] for p in list_registered_pipelines()]}"
            ),
        )
    name = gm.storage.set_pipeline_name(graph_id, body.pipeline)
    return PipelineBindingResponse(graph_id=graph_id, pipeline=name)


# ------------------------------------------------------------------ #
# Prompt CRUD (Phase 2) — scoped to one graph
# ------------------------------------------------------------------ #


@graph_router.get(
    "/{graph_id}/pipeline/prompts",
    response_model=PromptListResponse,
)
def list_prompts(graph_id: str) -> PromptListResponse:
    _check_graph_exists(graph_id)
    registry = _registry()
    items: List[PromptInfo] = [
        _to_info(name, registry, graph_id) for name in registry.list_prompts()
    ]
    return PromptListResponse(graph_id=graph_id, prompts=items)


@graph_router.put(
    "/{graph_id}/pipeline/prompts/{name}",
    response_model=PromptUpdateResponse,
)
def update_prompt(
    graph_id: str, name: str, body: PromptUpdateRequest,
) -> PromptUpdateResponse:
    _check_graph_exists(graph_id)
    registry = _registry()
    if name not in registry.list_prompts():
        raise HTTPException(status_code=404, detail=f"Unknown prompt: '{name}'")

    template = TemplatePrompt(
        name=name,
        system_template=body.system,
        user_template=body.user,
    )
    registry.set(name, template, graph_id=graph_id)
    persisted = registry.save_graph_yaml(graph_id)
    return PromptUpdateResponse(
        name=name, graph_id=graph_id,
        persisted_to=persisted,
        info=_to_info(name, registry, graph_id),
    )


@graph_router.post(
    "/{graph_id}/pipeline/prompts/{name}/reset",
    response_model=PromptResetResponse,
)
def reset_prompt(graph_id: str, name: str) -> PromptResetResponse:
    _check_graph_exists(graph_id)
    registry = _registry()
    if name not in registry.list_prompts():
        raise HTTPException(status_code=404, detail=f"Unknown prompt: '{name}'")

    cleared = registry.clear_graph_override(name, graph_id)
    persisted = registry.save_graph_yaml(graph_id) if cleared else None
    return PromptResetResponse(
        name=name, graph_id=graph_id,
        cleared=cleared,
        persisted_to=persisted,
        info=_to_info(name, registry, graph_id),
    )


# ------------------------------------------------------------------ #
# Model bindings (Phase 3) — global, applies to all graphs
# ------------------------------------------------------------------ #


def _router() -> LLMRouter:
    llm = get_llm()
    if not isinstance(llm, LLMRouter):
        # Defensive: the dependency now always returns a router, but if
        # someone bypassed it we surface a clear error.
        raise HTTPException(
            status_code=503,
            detail="LLM is not a router; per-role swap not available.",
        )
    return llm


def _binding_from_summary(summary: dict) -> ModelBinding:
    return ModelBinding(
        role=summary["role"],
        base_url=summary["base_url"],
        model=summary["model"],
        has_api_key=summary["has_api_key"],
        is_azure=summary["is_azure"],
        azure_api_version=summary.get("azure_api_version", "") or "",
        falls_back_to_default=summary["falls_back_to_default"],
    )


@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Snapshot of every router role's current binding (api_keys redacted)."""
    summary = _router().role_summary()
    bindings = [_binding_from_summary(summary[r]) for r in ROUTER_ROLES]
    return ModelListResponse(bindings=bindings, roles=list(ROUTER_ROLES))


@router.put("/models/{role}", response_model=ModelUpdateResponse)
def update_model(role: str, body: ModelUpdateRequest) -> ModelUpdateResponse:
    """Atomically swap the LLM bound to *role*.

    Applies process-wide — all graphs that use this role on subsequent
    calls will hit the new endpoint.
    """
    if role not in ROUTER_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role '{role}'. Allowed: {list(ROUTER_ROLES)}",
        )
    try:
        _router().set_role(
            role,
            base_url=body.base_url,
            model=body.model,
            api_key=body.api_key,
            is_azure=body.is_azure,
            azure_api_version=body.azure_api_version,
        )
    except Exception as e:  # noqa: BLE001 — surface client-construction errors verbatim
        raise HTTPException(status_code=400, detail=f"set_role failed: {e}")
    summary = _router().role_summary()[role]
    return ModelUpdateResponse(role=role, binding=_binding_from_summary(summary))


@router.post("/models/test", response_model=ModelTestResponse)
def test_model(body: ModelTestRequest) -> ModelTestResponse:
    """Probe a candidate {base_url, api_key, model} without mutating router state.

    Builds a one-off client, runs a short ``complete([{user: prompt}])``, and
    returns the first chunk + latency. Useful as a "Test connection" check
    before committing a swap.
    """
    api_key = body.api_key
    if api_key is None and body.role and body.role in ROUTER_ROLES:
        # Reuse the role's existing key if the caller didn't re-type it.
        existing = _router()._clients.get(body.role) or _router()._clients.get("default")
        api_key = getattr(existing, "api_key", "") if existing else ""
    elif api_key is not None:
        # ``${VAR}`` references resolve against the server's environment.
        api_key = expand_env_vars(api_key)
    if api_key is None:
        api_key = ""
    base_url = expand_env_vars(body.base_url)

    started = time.monotonic()
    try:
        # Bypass OpenAICompatibleLLMClient.complete — its retry loop swallows
        # exceptions and returns "" on failure, which would mask connection
        # errors as "ok with empty response". Call the SDK directly instead.
        if body.is_azure:
            sdk = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=body.azure_api_version,
            )
        else:
            sdk = OpenAI(base_url=base_url, api_key=api_key)
        resp = sdk.chat.completions.create(
            model=body.model,
            messages=[{"role": "user", "content": body.prompt}],
            max_tokens=body.max_tokens,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return ModelTestResponse(ok=False, latency_ms=elapsed_ms, error=str(e))

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return ModelTestResponse(
        ok=True,
        latency_ms=elapsed_ms,
        sample=content[:240],
    )


# ------------------------------------------------------------------ #
# Pipeline traces (Phase 4)
# ------------------------------------------------------------------ #


@graph_router.get(
    "/{graph_id}/pipeline/traces",
    response_model=TraceListResponse,
)
def list_traces(graph_id: str, limit: int = 20) -> TraceListResponse:
    gm = _check_graph_exists(graph_id)
    rows = gm.storage.list_pipeline_traces(graph_id, limit=limit)
    cap = gm.storage.get_pipeline_trace_cap(graph_id)
    return TraceListResponse(
        graph_id=graph_id,
        traces=[TraceSummary(**r) for r in rows],
        cap=cap,
    )


@graph_router.get(
    "/{graph_id}/pipeline/traces/cap",
    response_model=TraceCapResponse,
)
def get_trace_cap(graph_id: str) -> TraceCapResponse:
    gm = _check_graph_exists(graph_id)
    cap = gm.storage.get_pipeline_trace_cap(graph_id)
    return TraceCapResponse(graph_id=graph_id, cap=cap)


@graph_router.put(
    "/{graph_id}/pipeline/traces/cap",
    response_model=TraceCapResponse,
)
def set_trace_cap(graph_id: str, body: TraceCapRequest) -> TraceCapResponse:
    gm = _check_graph_exists(graph_id)
    new_cap = gm.storage.set_pipeline_trace_cap(graph_id, body.cap)
    return TraceCapResponse(graph_id=graph_id, cap=new_cap)


@graph_router.get(
    "/{graph_id}/pipeline/stats",
    response_model=PipelineStatsResponse,
)
def get_pipeline_stats(graph_id: str) -> PipelineStatsResponse:
    """Per-step-name aggregates across all stored traces (count, mean latency, …)."""
    gm = _check_graph_exists(graph_id)
    raw = gm.storage.aggregate_step_stats(graph_id)
    stats = {name: PipelineStepStats(**v) for name, v in raw.items()}
    return PipelineStatsResponse(graph_id=graph_id, stats=stats)


@graph_router.get(
    "/{graph_id}/pipeline/steps/{step_name}/traces",
    response_model=StepTracesResponse,
)
def list_step_traces(
    graph_id: str, step_name: str, limit: int = 10,
) -> StepTracesResponse:
    """Recent trace summaries that include the named step."""
    gm = _check_graph_exists(graph_id)
    rows = gm.storage.list_traces_for_step(graph_id, step_name, limit=limit)
    return StepTracesResponse(
        graph_id=graph_id, step_name=step_name,
        traces=[StepTraceSummary(**r) for r in rows],
    )


@graph_router.get(
    "/{graph_id}/pipeline/traces/{trace_id}",
    response_model=TraceDetailResponse,
)
def get_trace(graph_id: str, trace_id: str) -> TraceDetailResponse:
    gm = _check_graph_exists(graph_id)
    row = gm.storage.get_pipeline_trace(graph_id, trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")
    return TraceDetailResponse(
        graph_id=graph_id,
        trace_id=row["trace_id"],
        ts=row["ts"],
        endpoint=row["endpoint"],
        duration_ms=row["duration_ms"],
        ok=row["ok"],
        num_steps=row["num_steps"],
        session_id=row.get("session_id"),
        error=row.get("error"),
        meta=row.get("meta") or {},
        steps=[TraceStep(**s) for s in (row.get("steps") or [])],
    )


# ------------------------------------------------------------------ #
# Per-pipeline visualization view (Phase 6.5a follow-up)
# ------------------------------------------------------------------ #


@graph_router.get(
    "/{graph_id}/pipeline/spec_view",
    response_model=PipelineSpecResponse,
)
def get_pipeline_spec_view(graph_id: str) -> PipelineSpecResponse:
    """Return the visualization spec for whichever pipeline is bound to *graph_id*.

    - ``plugmem-default``  → static spec from ``pipeline_spec.py``.
    - ``naive-rag``        → hand-coded baseline spec.
    - ``spec-driven``      → derived from the saved YAML; empty placeholder
                             if no YAML has been saved yet.
    """
    gm = _check_graph_exists(graph_id)
    name = gm.storage.get_pipeline_name(graph_id)
    return PipelineSpecResponse(**pipeline_views.view_for_pipeline(name, graph_id=graph_id))


@router.get("/samples", response_model=PipelineSpecSamplesResponse)
def list_pipeline_spec_samples() -> PipelineSpecSamplesResponse:
    """Starter YAMLs the editor can paste as a starting point."""
    from plugmem.pipelines.sample_specs import SAMPLES
    rows = [
        PipelineSpecSample(key=k, label=v["label"], description=v["description"],
                           content=v["content"])
        for k, v in SAMPLES.items()
    ]
    return PipelineSpecSamplesResponse(samples=rows)


@router.get("/node_snippets", response_model=PipelineNodeSnippetsResponse)
def list_pipeline_node_snippets() -> PipelineNodeSnippetsResponse:
    """Per-node-type YAML stubs the editor palette inserts at the cursor."""
    from plugmem.pipelines.sample_specs import NODE_SNIPPETS
    rows = [
        PipelineNodeSnippet(key=k, label=v["label"], description=v["description"],
                            snippet=v["snippet"])
        for k, v in NODE_SNIPPETS.items()
    ]
    return PipelineNodeSnippetsResponse(snippets=rows)


# ------------------------------------------------------------------ #
# Spec-driven YAML editor (Phase 6.5a) — non-destructive, versioned
# ------------------------------------------------------------------ #


def _validate_spec_text(text: str) -> None:
    """Raises ValueError if YAML is invalid; mapped to 422 by the route."""
    load_yaml_str(text)


def _version_info(v: spec_storage.SpecVersion) -> SpecVersionInfo:
    return SpecVersionInfo(
        version_id=v.version_id, ts=v.ts,
        parent_version_id=v.parent_version_id,
        note=v.note, active=v.active,
    )


@graph_router.get(
    "/{graph_id}/pipeline/spec",
    response_model=SpecDocumentResponse,
)
def get_pipeline_spec_yaml(graph_id: str) -> SpecDocumentResponse:
    """Return the current spec-driven YAML source for *graph_id*."""
    _check_graph_exists(graph_id)
    content = spec_storage.read_current_content(graph_id) or ""
    active = spec_storage.get_active_version_id(graph_id)
    return SpecDocumentResponse(
        graph_id=graph_id,
        content=content,
        active_version_id=active,
        exists=bool(content),
        live_path=str(spec_storage.live_path(graph_id)),
    )


@graph_router.put(
    "/{graph_id}/pipeline/spec",
    response_model=SpecSaveResponse,
)
def save_pipeline_spec_yaml(graph_id: str, body: SpecSaveRequest) -> SpecSaveResponse:
    """Validate + save a new version. Prior content is preserved in history."""
    _check_graph_exists(graph_id)
    try:
        version = spec_storage.save_new_version(
            graph_id, body.content,
            note=body.note,
            validator=_validate_spec_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return SpecSaveResponse(graph_id=graph_id, version=_version_info(version))


@graph_router.post(
    "/{graph_id}/pipeline/spec/validate",
    response_model=SpecValidateResponse,
)
def validate_pipeline_spec_yaml(
    graph_id: str, body: SpecValidateRequest,
) -> SpecValidateResponse:
    """Run the loader against *content* without writing anything to disk."""
    _check_graph_exists(graph_id)
    try:
        _validate_spec_text(body.content)
    except ValueError as e:
        return SpecValidateResponse(ok=False, error=str(e))
    return SpecValidateResponse(ok=True)


@graph_router.get(
    "/{graph_id}/pipeline/spec/versions",
    response_model=SpecVersionsResponse,
)
def list_pipeline_spec_versions(graph_id: str) -> SpecVersionsResponse:
    _check_graph_exists(graph_id)
    versions = spec_storage.list_versions(graph_id)
    return SpecVersionsResponse(
        graph_id=graph_id,
        versions=[_version_info(v) for v in versions],
        active_version_id=spec_storage.get_active_version_id(graph_id),
    )


@graph_router.get(
    "/{graph_id}/pipeline/spec/versions/{version_id}",
    response_model=SpecVersionContentResponse,
)
def get_pipeline_spec_version(
    graph_id: str, version_id: str,
) -> SpecVersionContentResponse:
    _check_graph_exists(graph_id)
    content = spec_storage.read_version(graph_id, version_id)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version_id}' not found for graph '{graph_id}'",
        )
    # Build a SpecVersionInfo from the list (cheapest accurate way to set `active`).
    info = next(
        (_version_info(v) for v in spec_storage.list_versions(graph_id)
         if v.version_id == version_id),
        SpecVersionInfo(version_id=version_id, ts="", note="", active=False),
    )
    return SpecVersionContentResponse(
        graph_id=graph_id, version=info, content=content,
    )


# ------------------------------------------------------------------ #
# Canvas layout (Phase 6.5b) — per-graph hand-positioned node coords
# ------------------------------------------------------------------ #


@graph_router.get(
    "/{graph_id}/pipeline/layout",
    response_model=CanvasLayoutResponse,
)
def get_pipeline_layout(graph_id: str) -> CanvasLayoutResponse:
    """Return the saved node-position sidecar (empty if none persisted)."""
    _check_graph_exists(graph_id)
    layout = layout_storage.load_layout(graph_id)
    return CanvasLayoutResponse(
        graph_id=graph_id,
        positions={
            nid: CanvasNodePosition(**pos) for nid, pos in layout.positions.items()
        },
        viewport=CanvasViewport(**layout.viewport) if layout.viewport else None,
    )


@graph_router.put(
    "/{graph_id}/pipeline/layout",
    response_model=CanvasLayoutResponse,
)
def save_pipeline_layout(
    graph_id: str, body: CanvasLayoutSaveRequest,
) -> CanvasLayoutResponse:
    """Persist hand-positioned node coordinates."""
    _check_graph_exists(graph_id)
    try:
        positions = {nid: {"x": p.x, "y": p.y} for nid, p in body.positions.items()}
        viewport = None
        if body.viewport is not None:
            viewport = {"x": body.viewport.x, "y": body.viewport.y, "zoom": body.viewport.zoom}
        layout = layout_storage.save_layout(
            graph_id, positions, viewport=viewport,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return CanvasLayoutResponse(
        graph_id=graph_id,
        positions={
            nid: CanvasNodePosition(**pos) for nid, pos in layout.positions.items()
        },
        viewport=CanvasViewport(**layout.viewport) if layout.viewport else None,
    )


@graph_router.delete(
    "/{graph_id}/pipeline/layout",
    response_model=CanvasLayoutResponse,
)
def reset_pipeline_layout(graph_id: str) -> CanvasLayoutResponse:
    """Forget hand-positions; canvas falls back to dagre auto-layout."""
    _check_graph_exists(graph_id)
    layout_storage.clear_layout(graph_id)
    return CanvasLayoutResponse(graph_id=graph_id, positions={}, viewport=None)


@graph_router.post(
    "/{graph_id}/pipeline/spec/versions/{version_id}/rollback",
    response_model=SpecSaveResponse,
)
def rollback_pipeline_spec_version(
    graph_id: str, version_id: str,
) -> SpecSaveResponse:
    """Make *version_id* active. Live file is rewritten; no new version row."""
    _check_graph_exists(graph_id)
    try:
        version = spec_storage.rollback(graph_id, version_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SpecSaveResponse(graph_id=graph_id, version=_version_info(version))


@graph_router.post(
    "/{graph_id}/pipeline/prompts/{name}/preview",
    response_model=PromptPreviewResponse,
)
def preview_prompt(
    graph_id: str, name: str, body: PromptPreviewRequest,
) -> PromptPreviewResponse:
    _check_graph_exists(graph_id)
    registry = _registry()
    if name not in registry.list_prompts():
        raise HTTPException(status_code=404, detail=f"Unknown prompt: '{name}'")

    try:
        if body.system is not None and body.user is not None:
            adhoc = TemplatePrompt(
                name=name,
                system_template=body.system,
                user_template=body.user,
            )
            rendered = adhoc.render(body.variables)
            messages = [{"role": m.role, "content": m.content} for m in rendered]
        else:
            messages = registry.render_messages(name, body.variables, graph_id=graph_id)
    except (KeyError, ValueError) as e:
        # PromptBase.format_text wraps KeyError as ValueError; either signals a
        # missing template variable in the variables payload.
        raise HTTPException(
            status_code=400,
            detail=f"Missing variable in variables payload: {e}",
        )
    return PromptPreviewResponse(
        name=name, graph_id=graph_id,
        messages=[PromptPreviewMessage(**m) for m in messages],
    )
