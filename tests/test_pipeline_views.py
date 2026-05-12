"""Tests for per-pipeline visualization views + the spec_view route."""
from __future__ import annotations

import textwrap

import pytest

from plugmem.pipelines import pipeline_views
from plugmem.pipelines.sample_specs import PLUGMEM_DEFAULT_RETRIEVE_YAML
from plugmem.pipelines.spec_driven import load_yaml_str


def _create_graph(client) -> str:
    r = client.post("/api/v1/graphs", json={})
    assert r.status_code == 201, r.text
    return r.json()["graph_id"]


# --------------------------------------------------------------------- #
# View module (pure)
# --------------------------------------------------------------------- #


def test_default_view_returns_full_pipeline():
    v = pipeline_views.view_for_pipeline("plugmem-default", graph_id="anything")
    # All six production phases should be present.
    phase_ids = {p["id"] for p in v["phases"]}
    assert {"append", "close", "insert", "retrieve", "reason", "consolidate"} <= phase_ids
    # And a known step.
    assert any(s["id"] == "get_plan" for s in v["steps"])


def test_naive_rag_view_has_ingest_retrieve_reason():
    v = pipeline_views.view_for_pipeline("naive-rag", graph_id="anything")
    phase_ids = [p["id"] for p in v["phases"]]
    assert phase_ids == ["ingest", "retrieve", "reason"]
    # No structuring/consolidation steps.
    assert not any(s["id"] == "get_plan" for s in v["steps"])
    # Embed + storage steps are present.
    assert any(s["kind"] == "embed" for s in v["steps"])
    assert any(s["kind"] == "storage" for s in v["steps"])


def test_spec_driven_view_empty_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    v = pipeline_views.view_for_pipeline("spec-driven", graph_id="ghost")
    # One placeholder step is fine.
    assert len(v["steps"]) == 1
    assert v["steps"][0]["id"] == "__empty"


def test_spec_driven_view_renders_saved_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    # Pre-seed a saved version via the storage module directly.
    from plugmem.pipelines import spec_storage
    spec_storage.save_new_version(
        "g1", PLUGMEM_DEFAULT_RETRIEVE_YAML,
        validator=load_yaml_str,
    )
    v = pipeline_views.view_for_pipeline("spec-driven", graph_id="g1")
    step_ids = {s["id"] for s in v["steps"]}
    # Top-level ids mirror plugmem.core.pipeline_spec for the retrieve phase.
    assert {"in", "get_plan", "get_mode", "out"} <= step_ids
    assert {"retrieve_semantic_nodes", "retrieve_procedural_nodes",
            "retrieve_episodic_nodes"} <= step_ids
    # Three reasoning Branches, one per mode.
    assert {"render_reasoning_semantic", "render_reasoning_procedural",
            "render_reasoning_episodic"} <= step_ids
    # Their body PromptRenders are namespaced — no collision.
    assert {"render_reasoning_semantic.render",
            "render_reasoning_procedural.render",
            "render_reasoning_episodic.render"} <= step_ids

    plan = next(s for s in v["steps"] if s["id"] == "get_plan")
    assert plan["kind"] == "llm"
    assert plan["prompt_name"] == "get_plan"
    assert plan["role"] == "retrieval"


def test_spec_driven_view_invalid_yaml_shows_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    # Write an invalid YAML directly into the live file (bypass validator).
    (tmp_path / "g1.pipeline.yaml").write_text(
        "phase: retrieve\nnodes: not-a-list\n"
    )
    v = pipeline_views.view_for_pipeline("spec-driven", graph_id="g1")
    assert v["steps"][0]["id"] == "__invalid"
    assert "nodes" in v["steps"][0]["description"].lower()


# --------------------------------------------------------------------- #
# Route layer
# --------------------------------------------------------------------- #


def test_spec_view_default_binding(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/spec_view")
    assert r.status_code == 200
    body = r.json()
    assert any(p["id"] == "append" for p in body["phases"])  # plugmem-default phase


def test_spec_view_naive_rag_binding(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline",
        json={"pipeline": "naive-rag"},
    )
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/spec_view")
    assert r.status_code == 200
    phases = [p["id"] for p in r.json()["phases"]]
    assert phases == ["ingest", "retrieve", "reason"]


def test_spec_view_spec_driven_with_yaml(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline",
        json={"pipeline": "spec-driven"},
    )
    client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": PLUGMEM_DEFAULT_RETRIEVE_YAML, "note": "init"},
    )
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/spec_view")
    assert r.status_code == 200
    step_ids = {s["id"] for s in r.json()["steps"]}
    assert {"in", "get_plan", "get_mode",
            "render_reasoning_semantic", "out"} <= step_ids


def test_samples_endpoint_lists_default(client):
    r = client.get("/api/v1/pipeline/samples")
    assert r.status_code == 200
    keys = {s["key"] for s in r.json()["samples"]}
    assert "plugmem-default" in keys


def test_sample_yaml_validates(client):
    """The shipped starter YAML must parse cleanly through the loader."""
    g = load_yaml_str(PLUGMEM_DEFAULT_RETRIEVE_YAML)
    assert g.phase == "retrieve"
    top_level = {n.id for n in g.nodes}
    assert {"in", "get_plan", "get_mode",
            "retrieve_semantic_nodes", "retrieve_procedural_nodes",
            "retrieve_episodic_nodes",
            "render_reasoning_semantic", "render_reasoning_procedural",
            "render_reasoning_episodic", "out"} <= top_level


def test_sample_yaml_runs_end_to_end(client, monkeypatch, tmp_path):
    """The starter YAML must run via /retrieve when bound."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline",
        json={"pipeline": "spec-driven"},
    )
    client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": PLUGMEM_DEFAULT_RETRIEVE_YAML},
    )
    r = client.post(
        f"/api/v1/graphs/{gid}/retrieve",
        json={"observation": "what is the weather?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reasoning_prompt"]
