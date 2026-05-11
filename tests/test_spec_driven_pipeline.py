"""Tests for the SpecDrivenPipeline (Phase 6.2 MVP).

Covers the YAML loader's validation, the executor's behaviour on simple
pipelines, and the route-level integration with the per-graph pipeline
binding.
"""
from __future__ import annotations

import textwrap

import pytest

from plugmem.pipelines.spec_driven import (
    PARSERS,
    PipelineExecutor,
    SpecDrivenPipeline,
    load_yaml,
)


# ------------------------------------------------------------------ #
# Loader / validator
# ------------------------------------------------------------------ #


def _write_yaml(path, text: str):
    path.write_text(textwrap.dedent(text))
    return path


def test_load_minimal_yaml(tmp_path):
    p = _write_yaml(tmp_path / "x.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    g = load_yaml(p)
    assert g.phase == "retrieve"
    assert {n.id for n in g.nodes} == {"in", "out"}


def test_cycle_rejected(tmp_path):
    # Two PromptRender nodes wire into each other → cycle.
    p = _write_yaml(tmp_path / "cyc.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: a
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs: { observation: b.messages, semantic_memory: { const: "" } }
          - id: b
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs: { observation: a.messages, semantic_memory: { const: "" } }
          - id: out
            type: Output
            inputs: { mode: { const: semantic_memory }, reasoning_prompt: a.messages, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="cycle"):
        load_yaml(p)


def test_unknown_reference_rejected(tmp_path):
    p = _write_yaml(tmp_path / "bad.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: out
            type: Output
            inputs: { mode: ghost.value, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="unknown node"):
        load_yaml(p)


def test_self_reference_rejected(tmp_path):
    p = _write_yaml(tmp_path / "self.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: c
            type: Constant
            config: { value: 1 }
            inputs: { x: c.value }
          - id: out
            type: Output
            inputs: { mode: { const: semantic_memory }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="itself"):
        load_yaml(p)


def test_duplicate_id_rejected(tmp_path):
    p = _write_yaml(tmp_path / "dup.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: x, type: Input }
          - id: x
            type: Output
            inputs: { mode: { const: a }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="Duplicate"):
        load_yaml(p)


def test_phase_must_be_retrieve(tmp_path):
    p = _write_yaml(tmp_path / "wrongphase.pipeline.yaml", """
        phase: ingest
        nodes:
          - { id: in, type: Input }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="phase=retrieve"):
        load_yaml(p)


def test_parsers_cover_known_prompts():
    """Sanity: get_plan + get_mode parsers exist (used by the round-trip path)."""
    assert "get_plan" in PARSERS
    assert "get_mode" in PARSERS


def test_parse_get_plan_extracts_subgoal_and_tags():
    out = PARSERS["get_plan"](
        '**Tags:** ["a", "b"]\n### Next Subgoal\nfoo\n'
    )
    assert out["query_tags"] == ["a", "b"]
    assert out["next_subgoal"] == "foo"


# ------------------------------------------------------------------ #
# Route-level integration (uses the `client` fixture from conftest.py)
# ------------------------------------------------------------------ #


def _bind(client, gid, name="spec-driven"):
    r = client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": name})
    assert r.status_code == 200, r.text


def test_route_returns_422_without_yaml(client, monkeypatch, tmp_path):
    """Binding to spec-driven but providing no YAML returns 422, not 500."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    r = client.post("/api/v1/graphs", json={"graph_id": "g-spec-noyaml"})
    assert r.status_code in (200, 201), r.text
    _bind(client, "g-spec-noyaml")
    r = client.post("/api/v1/graphs/g-spec-noyaml/retrieve", json={
        "observation": "?", "mode": None,
    })
    assert r.status_code == 422, r.text
    assert "No pipeline YAML" in r.json()["detail"]


def test_route_runs_user_pipeline_for_retrieve(client, fake_llm, monkeypatch, tmp_path):
    """A YAML that calls one LLMCall + one PromptRender renders into the response."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = "g-spec-run"
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text
    _bind(client, gid)

    _write_yaml(tmp_path / f"{gid}.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: plan
            type: LLMCall
            config: { prompt: get_plan, role: retrieval }
            inputs: { goal: in.goal, subgoal: in.subgoal, state: in.state, observation: in.observation }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              goal: in.goal
              subgoal: in.subgoal
              state: in.state
              observation: in.observation
              semantic_memory: { const: "(none)" }
              procedural_memory: { const: "" }
              episodic_memory: { const: "" }
              time: { const: "" }
              information: { const: "" }
              question: in.observation
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: msgs.messages
              variables: { const: {} }
    """)

    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={
        "observation": "what's next",
        "goal": "ship",
        "subgoal": "test",
        "state": "ok",
        "mode": None,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "semantic_memory"
    assert isinstance(body["reasoning_prompt"], list) and body["reasoning_prompt"], body
    # The LLMCall (plan) hit the fake LLM exactly once. PromptRender doesn't call.
    assert len(fake_llm.calls) - n_before == 1


def test_executor_call_is_traced(client, monkeypatch, tmp_path):
    """The LLMCall node's record_llm_step entry shows up in the persisted trace."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = "g-spec-trace"
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text
    _bind(client, gid)
    _write_yaml(tmp_path / f"{gid}.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: my_plan_step
            type: LLMCall
            config: { prompt: get_plan, role: retrieval }
            inputs: { goal: in.goal, subgoal: in.subgoal, state: in.state, observation: in.observation }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={
        "observation": "?",
        "goal": "g", "subgoal": "s", "state": "x",
        "mode": None,
    })
    assert r.status_code == 200, r.text

    r = client.get(f"/api/v1/graphs/{gid}/pipeline/traces")
    traces = r.json()["traces"]
    assert traces, "no trace recorded"
    detail = client.get(
        f"/api/v1/graphs/{gid}/pipeline/traces/{traces[0]['trace_id']}"
    ).json()
    step_names = [s["name"] for s in detail["steps"]]
    assert "my_plan_step" in step_names, step_names


def test_fallback_ingest_delegates_to_default(client, fake_llm, monkeypatch, tmp_path):
    """With spec-driven bound, /memories still records the structuring LLM calls."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = "g-spec-fallback"
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text
    _bind(client, gid)
    # No YAML required — ingest delegates to plugmem-default.
    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/memories", json={
        "mode": "trajectory",
        "goal": "test",
        "steps": [{"observation": "o", "action": "a"}],
    })
    assert r.status_code == 200, r.text
    # default fires get_subgoal + get_reward + get_state + get_semantic + get_procedural.
    assert len(fake_llm.calls) - n_before >= 5


def test_listed_in_pipeline_registry(client):
    r = client.get("/api/v1/pipeline/pipelines")
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["pipelines"]}
    assert "spec-driven" in names
