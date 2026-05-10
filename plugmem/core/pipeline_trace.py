"""Pipeline trace capture.

A "pipeline run" is one outer API call (e.g. POST /memories, /retrieve,
/reason) and the LLM calls it triggers internally. The recorder is set on
a ContextVar so the inference layer can record per-LLM-call telemetry
without every function signature growing a recorder parameter.

Usage from a route::

    from plugmem.core.pipeline_trace import trace_run

    with trace_run(graph_id, "/memories", session_id=...):
        graph.insert(memory)

Inside the inference layer::

    record_llm_step(
        name="get_subgoal", llm=llm, variables=vars,
        response=response, parsed={"subgoal": result},
        latency_ms=..., error=None,
    )
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
# Data model
# ----------------------------------------------------------------------- #


@dataclass
class StepRecord:
    name: str
    model: str
    variables: Dict[str, Any]
    response: str
    parsed: Optional[Any]
    latency_ms: int
    ts_offset_ms: int
    error: Optional[str] = None


@dataclass
class TraceRecord:
    trace_id: str
    graph_id: str
    endpoint: str
    ts: str
    duration_ms: int
    ok: bool
    num_steps: int
    error: Optional[str] = None
    session_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepRecord] = field(default_factory=list)


# ----------------------------------------------------------------------- #
# Recorder
# ----------------------------------------------------------------------- #


class PipelineTraceRecorder:
    """Collects step records for one pipeline run.

    Designed to be cheap when no storage is configured — finalize/persist is
    a no-op if the storage backend wasn't supplied.
    """

    def __init__(
        self,
        graph_id: str,
        endpoint: str,
        *,
        storage: Optional[Any] = None,
        session_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph_id = graph_id
        self.endpoint = endpoint
        self.session_id = session_id
        self.meta = dict(meta or {})
        self._storage = storage
        self._steps: List[StepRecord] = []
        self._started_monotonic: Optional[float] = None
        self._started_iso: Optional[str] = None
        self._trace_id = uuid.uuid4().hex[:16]

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def start(self) -> None:
        self._started_monotonic = time.monotonic()
        self._started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def record_step(
        self,
        *,
        name: str,
        model: str,
        variables: Dict[str, Any],
        response: str,
        parsed: Any = None,
        latency_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        if self._started_monotonic is None:
            self.start()
        ts_off = int((time.monotonic() - (self._started_monotonic or 0)) * 1000)
        self._steps.append(StepRecord(
            name=name,
            model=model or "",
            variables=_safe_json(variables),
            response=response or "",
            parsed=_safe_json(parsed) if parsed is not None else None,
            latency_ms=int(latency_ms),
            ts_offset_ms=ts_off,
            error=error,
        ))

    def finish(self, *, ok: bool = True, error: Optional[str] = None) -> Optional[TraceRecord]:
        if self._started_monotonic is None:
            return None
        duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        record = TraceRecord(
            trace_id=self._trace_id,
            graph_id=self.graph_id,
            endpoint=self.endpoint,
            ts=self._started_iso or "",
            duration_ms=duration_ms,
            ok=ok and error is None,
            num_steps=len(self._steps),
            error=error,
            session_id=self.session_id,
            meta=self.meta,
            steps=list(self._steps),
        )
        if self._storage is not None and self._steps:
            try:
                self._storage.add_pipeline_trace(self.graph_id, record)
            except Exception as e:  # noqa: BLE001 — never fail the request because of trace IO
                logger.warning("Failed to persist pipeline trace: %s", e)
        return record


# ----------------------------------------------------------------------- #
# ContextVar plumbing
# ----------------------------------------------------------------------- #


_recorder_var: ContextVar[Optional[PipelineTraceRecorder]] = ContextVar(
    "_plugmem_recorder", default=None,
)


def get_recorder() -> Optional[PipelineTraceRecorder]:
    return _recorder_var.get()


def record_llm_step(
    *,
    name: str,
    llm: Any,
    variables: Dict[str, Any],
    response: str,
    parsed: Any = None,
    latency_ms: int = 0,
    error: Optional[str] = None,
) -> None:
    """Helper for the inference layer. No-op when no recorder is active."""
    rec = _recorder_var.get()
    if rec is None:
        return
    rec.record_step(
        name=name,
        model=getattr(llm, "model", "") or "",
        variables=variables,
        response=response,
        parsed=parsed,
        latency_ms=latency_ms,
        error=error,
    )


@contextmanager
def trace_run(
    graph_id: str,
    endpoint: str,
    *,
    storage: Optional[Any] = None,
    session_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    """Set a recorder on the current context for the duration of the block.

    The recorder is finalized (and persisted if storage is supplied) when
    the block exits — even on exception. Caller never sees a trace IO
    error: failures during persistence are logged but swallowed.
    """
    rec = PipelineTraceRecorder(
        graph_id=graph_id,
        endpoint=endpoint,
        storage=storage,
        session_id=session_id,
        meta=meta,
    )
    rec.start()
    token = _recorder_var.set(rec)
    error_msg: Optional[str] = None
    try:
        yield rec
    except Exception as e:  # noqa: BLE001
        error_msg = str(e)
        raise
    finally:
        try:
            rec.finish(ok=error_msg is None, error=error_msg)
        finally:
            _recorder_var.reset(token)


def _safe_json(obj: Any) -> Any:
    """Best-effort coerce *obj* into JSON-serializable form (for chroma metadata)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return repr(obj)
