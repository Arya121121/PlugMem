"""Retrieve / Reason / Consolidate endpoints — dispatch to the bound pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import (
    ConsolidateRequest,
    ConsolidateResponse,
    ReasonRequest,
    ReasonResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from plugmem.core.pipeline_trace import trace_run
from plugmem.graph_manager import GraphManager
from plugmem.pipelines import get as get_pipeline

router = APIRouter(prefix="/graphs", tags=["retrieval"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


def _get_graph(graph_id: str):
    gm = _manager()
    try:
        return gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")


def _pipeline_for(graph_id: str):
    name = _manager().storage.get_pipeline_name(graph_id)
    return get_pipeline(name)


@router.post("/{graph_id}/retrieve", response_model=RetrieveResponse)
async def retrieve(graph_id: str, body: RetrieveRequest) -> RetrieveResponse:
    graph = _get_graph(graph_id)
    pipeline = _pipeline_for(graph_id)
    with trace_run(
        graph_id, f"POST /retrieve (pipeline={pipeline.name})",
        storage=graph.storage,
        session_id=getattr(body, "session_id", None),
        meta={"pipeline": pipeline.name},
    ):
        try:
            return pipeline.retrieve(graph, body)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))


@router.post("/{graph_id}/reason", response_model=ReasonResponse)
async def reason(graph_id: str, body: ReasonRequest) -> ReasonResponse:
    graph = _get_graph(graph_id)
    pipeline = _pipeline_for(graph_id)
    with trace_run(
        graph_id, f"POST /reason (pipeline={pipeline.name})",
        storage=graph.storage,
        session_id=getattr(body, "session_id", None),
        meta={"pipeline": pipeline.name},
    ):
        try:
            return pipeline.reason(graph, body)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))


@router.post("/{graph_id}/consolidate", response_model=ConsolidateResponse)
async def consolidate(graph_id: str, body: ConsolidateRequest) -> ConsolidateResponse:
    graph = _get_graph(graph_id)
    pipeline = _pipeline_for(graph_id)
    with trace_run(
        graph_id, f"POST /consolidate (pipeline={pipeline.name})",
        storage=graph.storage,
        meta={"pipeline": pipeline.name},
    ):
        try:
            return pipeline.consolidate(graph, body)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))
