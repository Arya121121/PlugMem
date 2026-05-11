"""Memory insertion endpoint — dispatches to the bound pipeline."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import MemoryInsertRequest, MemoryInsertResponse
from plugmem.core.pipeline_trace import trace_run
from plugmem.graph_manager import GraphManager
from plugmem.pipelines import get as get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graphs", tags=["memories"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


@router.post("/{graph_id}/memories", response_model=MemoryInsertResponse)
async def insert_memories(graph_id: str, body: MemoryInsertRequest) -> MemoryInsertResponse:
    gm = _manager()

    try:
        graph = gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    pipeline_name = gm.storage.get_pipeline_name(graph_id)
    pipeline = get_pipeline(pipeline_name)

    endpoint = f"POST /memories (mode={body.mode}, pipeline={pipeline.name})"
    with trace_run(
        graph_id, endpoint,
        storage=graph.storage,
        session_id=body.session_id,
        meta={"mode": body.mode, "pipeline": pipeline.name},
    ):
        try:
            return pipeline.ingest(graph, body)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
