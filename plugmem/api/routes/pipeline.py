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
    TraceCapRequest,
    TraceCapResponse,
    TraceDetailResponse,
    TraceListResponse,
    TraceStep,
    TraceSummary,
)
from openai import AzureOpenAI, OpenAI

from plugmem.clients.llm import OpenAICompatibleLLMClient
from plugmem.clients.llm_router import ROLES as ROUTER_ROLES, LLMRouter
from plugmem.core.pipeline_spec import to_dict as pipeline_spec_dict
from plugmem.graph_manager import GraphManager
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
        summary = _router().role_summary()[body.role]
        # role_summary doesn't expose the key; reach into the router directly.
        existing = _router()._clients.get(body.role) or _router()._clients.get("default")
        api_key = getattr(existing, "api_key", "") if existing else ""
    if api_key is None:
        api_key = ""

    started = time.monotonic()
    try:
        # Bypass OpenAICompatibleLLMClient.complete — its retry loop swallows
        # exceptions and returns "" on failure, which would mask connection
        # errors as "ok with empty response". Call the SDK directly instead.
        if body.is_azure:
            sdk = AzureOpenAI(
                azure_endpoint=body.base_url,
                api_key=api_key,
                api_version=body.azure_api_version,
            )
        else:
            sdk = OpenAI(base_url=body.base_url, api_key=api_key)
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
