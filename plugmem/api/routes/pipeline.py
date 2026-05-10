"""Pipeline inspector endpoints.

Phase 1: read-only spec for the inspector UI. Subsequent phases will add
prompt and model edit endpoints, plus pipeline-run trace browsing.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from plugmem.api.auth import require_api_key
from plugmem.api.schemas import PipelineSpecResponse
from plugmem.core.pipeline_spec import to_dict as pipeline_spec_dict

router = APIRouter(prefix="/pipeline", tags=["pipeline"], dependencies=[Depends(require_api_key)])


@router.get("/spec", response_model=PipelineSpecResponse)
def get_pipeline_spec() -> PipelineSpecResponse:
    """Return the static pipeline shape used by the inspector frontend."""
    return PipelineSpecResponse(**pipeline_spec_dict())
