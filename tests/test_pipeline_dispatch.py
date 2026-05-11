"""Verify the pipeline registry + per-graph binding.

- Default binding is ``plugmem-default``.
- Registry lists both built-ins.
- Switching to ``naive-rag`` changes the runtime behavior:
  ingest skips the structuring LLM calls; retrieve goes through the
  flat-RAG path.
- Binding survives across requests (persisted to chroma).
- Unknown pipeline names get a 400.
"""
from __future__ import annotations

import pytest


def _seed_graph(client, name: str) -> str:
    """Create a graph with a test-unique id.

    EphemeralClient shares in-memory backend state across instances in the
    same process — using a stable graph_id leaks bindings between tests.
    """
    r = client.post("/api/v1/graphs", json={"graph_id": name})
    assert r.status_code in (200, 201), r.text
    return r.json()["graph_id"]


def test_lists_registered_pipelines(client):
    r = client.get("/api/v1/pipeline/pipelines")
    assert r.status_code == 200, r.text
    names = {p["name"] for p in r.json()["pipelines"]}
    assert {"plugmem-default", "naive-rag"} <= names
    assert r.json()["default"] == "plugmem-default"


def test_default_binding_is_plugmem_default(client):
    gid = _seed_graph(client, "g-default")
    r = client.get(f"/api/v1/graphs/{gid}/pipeline")
    assert r.status_code == 200
    assert r.json()["pipeline"] == "plugmem-default"


def test_binding_persists_after_set(client):
    gid = _seed_graph(client, "g-persist")
    r = client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "naive-rag"})
    assert r.status_code == 200
    assert r.json()["pipeline"] == "naive-rag"
    r2 = client.get(f"/api/v1/graphs/{gid}/pipeline")
    assert r2.json()["pipeline"] == "naive-rag"


def test_unknown_pipeline_rejected(client):
    gid = _seed_graph(client, "g-unknown")
    r = client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "made-up"})
    assert r.status_code == 400
    assert "made-up" in r.json()["detail"]


def test_default_pipeline_records_structuring_calls(client, fake_llm):
    """plugmem-default ingest fires the structuring LLM (subgoal/reward/state/…)."""
    gid = _seed_graph(client, "g-default-ing")
    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/memories", json={
        "mode": "trajectory",
        "goal": "test",
        "steps": [{"observation": "o1", "action": "a1"}],
    })
    assert r.status_code == 200, r.text
    # default fires at least: get_subgoal + get_reward + get_state + get_semantic + get_procedural
    assert len(fake_llm.calls) - n_before >= 5


def test_naive_rag_skips_structuring(client, fake_llm):
    """naive-rag ingest is a pure storage write — no LLM calls."""
    gid = _seed_graph(client, "g-naive-ing")
    client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "naive-rag"})
    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/memories", json={
        "mode": "trajectory",
        "goal": "test",
        "steps": [{"observation": "o1", "action": "a1"}],
    })
    assert r.status_code == 200, r.text
    # No LLM calls during ingest under naive-rag.
    assert len(fake_llm.calls) == n_before


def test_naive_rag_retrieve_uses_default_template(client, fake_llm):
    """naive-rag retrieve returns its own messages (no get_plan / get_mode fired)."""
    gid = _seed_graph(client, "g-naive-ret")
    client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "naive-rag"})
    client.post(f"/api/v1/graphs/{gid}/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "Paris is the capital of France.", "tags": []}],
    })
    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={
        "observation": "What is the capital of France?",
        "mode": None,
    })
    assert r.status_code == 200, r.text
    # naive-rag's retrieve is non-LLM — no get_plan / get_mode calls.
    assert len(fake_llm.calls) == n_before
    body = r.json()
    assert body["mode"] == "semantic_memory"
    content = body["reasoning_prompt"][0]["content"]
    assert "Paris" in content


def test_naive_rag_consolidate_is_noop(client):
    gid = _seed_graph(client, "g-naive-cons")
    client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "naive-rag"})
    r = client.post(f"/api/v1/graphs/{gid}/consolidate", json={})
    assert r.status_code == 200
    assert r.json()["stats"] == {"skipped": 1}
