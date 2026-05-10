"""Pipeline inspector endpoints.

Phase 1: read-only spec for the inspector UI.
Phase 2: per-graph prompt CRUD + preview. Builtins and the service-wide
``_defaults.yaml`` are read-only from the API; edits land only in the
per-graph layer and are persisted to ``{prompts_dir}/{graph_id}.yaml``.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager, get_prompt_registry
from plugmem.api.schemas import (
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
)
from plugmem.core.pipeline_spec import to_dict as pipeline_spec_dict
from plugmem.graph_manager import GraphManager
from plugmem.prompts.registry import PromptRegistry, TemplatePrompt

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
