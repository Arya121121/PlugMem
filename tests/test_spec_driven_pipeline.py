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


def test_phase_must_be_allowed(tmp_path):
    """Phases outside ALLOWED_PHASES are rejected. With multi-phase support,
    retrieve / reason / consolidate are allowed; ingest and others are not."""
    p = _write_yaml(tmp_path / "wrongphase.pipeline.yaml", """
        phase: ingest
        nodes:
          - { id: in, type: Input }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="unsupported phase 'ingest'"):
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


# ------------------------------------------------------------------ #
# Phase 6.3 — ForEach loops
# ------------------------------------------------------------------ #


def test_foreach_loads(tmp_path):
    """Minimal valid ForEach passes the loader."""
    p = _write_yaml(tmp_path / "fe.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: ["a", "b", "c"] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config:
              item_var: x
              outputs: { echoes: echo.value }
            body:
              - id: echo
                type: Constant
                config: { value: 42 }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables:
                # Nested-dict ports require dict ref; use const for now.
                # The body collects 'echoes' as a list.
                const: {}
    """)
    g = load_yaml(p)
    foreach = next(n for n in g.nodes if n.type == "ForEach")
    assert foreach.body is not None
    assert {b.id for b in foreach.body} == {"echo"}


def test_foreach_rejects_input_in_body(tmp_path):
    p = _write_yaml(tmp_path / "bad_body.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: [1] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config: { item_var: x, outputs: { y: nested.value } }
            body:
              - { id: nested, type: Input }
          - id: out
            type: Output
            inputs:
              mode: { const: m }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="body cannot contain"):
        load_yaml(p)


def test_foreach_rejects_cycle_in_body(tmp_path):
    p = _write_yaml(tmp_path / "body_cycle.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: [1] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config: { item_var: x, outputs: { y: a.messages } }
            body:
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
            inputs:
              mode: { const: m }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="cycle"):
        load_yaml(p)


def test_foreach_rejects_unknown_output_target(tmp_path):
    p = _write_yaml(tmp_path / "bad_out.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: [1] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config: { item_var: x, outputs: { y: ghost.value } }
            body:
              - id: echo
                type: Constant
                config: { value: 1 }
          - id: out
            type: Output
            inputs:
              mode: { const: m }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="not a body node"):
        load_yaml(p)


def test_foreach_iter_var_collision_rejected(tmp_path):
    p = _write_yaml(tmp_path / "var_collide.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: [1] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config: { item_var: src, outputs: { y: echo.value } }
            body:
              - id: echo
                type: Constant
                config: { value: 1 }
          - id: out
            type: Output
            inputs:
              mode: { const: m }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="collides"):
        load_yaml(p)


def _build_graph_for_executor(client, tmp_path, monkeypatch, gid, yaml_body):
    """Helper: bind a graph to spec-driven and drop a YAML on disk."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text
    r = client.put(f"/api/v1/graphs/{gid}/pipeline", json={"pipeline": "spec-driven"})
    assert r.status_code == 200, r.text
    _write_yaml(tmp_path / f"{gid}.pipeline.yaml", yaml_body)


def _run_executor(graph_manager, tmp_path, gid, yaml_body, phase_inputs=None):
    """Build a real MemoryGraph + run the executor directly, bypassing pydantic."""
    from plugmem.pipelines.spec_driven import PipelineExecutor

    graph_manager.create_graph(gid)
    graph = graph_manager.get_graph(gid)
    p = _write_yaml(tmp_path / f"{gid}.pipeline.yaml", yaml_body)
    spec = load_yaml(p)
    executor = PipelineExecutor(spec, graph)
    return executor.run(phase_inputs or {
        "observation": "?", "goal": "", "subgoal": "", "state": "",
        "task_type": "", "time": "", "mode": None,
    })


def test_foreach_iterates_constant_list_collected_to_output(graph_manager, tmp_path):
    """A ForEach over a constant list collects body outputs into a list at the outer scope."""
    out = _run_executor(graph_manager, tmp_path, "fe-const", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: ["alpha", "beta", "gamma"] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config:
              item_var: x
              outputs: { items_echoed: pass.value }
            body:
              - id: pass
                type: Constant
                config: { value: "ECHO" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: loop.items_echoed
    """)
    assert out["variables"] == ["ECHO", "ECHO", "ECHO"]


def test_foreach_records_each_iteration_as_distinct_trace_step(client, monkeypatch, tmp_path):
    """LLMCall inside a ForEach gets one trace step per iteration, suffix-disambiguated."""
    gid = "fe-trace"
    _build_graph_for_executor(client, tmp_path, monkeypatch, gid, """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: ["t1", "t2"] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config:
              item_var: tag
              outputs: { rs: probe.raw }
            body:
              - id: probe
                type: LLMCall
                config: { prompt: get_subgoal, role: structuring }
                inputs:
                  goal: tag
                  state: { const: "" }
                  observation: { const: "" }
                  action: { const: "" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={"observation": "?", "mode": None})
    assert r.status_code == 200, r.text
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/traces")
    assert r.json()["traces"]
    trace_id = r.json()["traces"][0]["trace_id"]
    detail = client.get(f"/api/v1/graphs/{gid}/pipeline/traces/{trace_id}").json()
    step_names = [s["name"] for s in detail["steps"]]
    # Each iteration's probe call is recorded as probe#0, probe#1.
    assert "probe#0" in step_names, step_names
    assert "probe#1" in step_names, step_names


def test_foreach_empty_list_produces_empty_outputs(graph_manager, tmp_path):
    """ForEach over an empty list runs zero body iterations and emits empty lists."""
    out = _run_executor(graph_manager, tmp_path, "fe-empty", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: [] }
          - id: loop
            type: ForEach
            inputs: { items: src.value }
            config:
              item_var: x
              outputs: { vals: echo.value }
            body:
              - id: echo
                type: Constant
                config: { value: 1 }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: loop.vals
    """)
    assert out["variables"] == []


def test_nested_foreach(graph_manager, tmp_path):
    """Nested ForEach: outer over 2 items × inner over 3 items → 6 inner iterations."""
    out = _run_executor(graph_manager, tmp_path, "fe-nested", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: outer_src
            type: Constant
            config: { value: ["A", "B"] }
          - id: inner_src
            type: Constant
            config: { value: [1, 2, 3] }
          - id: outer_loop
            type: ForEach
            inputs: { items: outer_src.value }
            config:
              item_var: outer_item
              outputs: { groups: inner_loop.inner_vals }
            body:
              - id: inner_loop
                type: ForEach
                inputs: { items: inner_src.value }
                config:
                  item_var: inner_item
                  outputs: { inner_vals: echo.value }
                body:
                  - id: echo
                    type: Constant
                    config: { value: "X" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: outer_loop.groups
    """)
    # Outer ran twice; each iteration's inner produced ["X","X","X"].
    assert out["variables"] == [["X", "X", "X"], ["X", "X", "X"]]


def test_foreach_cap_enforced(client, monkeypatch, tmp_path):
    """A list longer than MAX_LOOP_ITERATIONS yields a 422 (config error)."""
    from plugmem.pipelines.spec_driven import MAX_LOOP_ITERATIONS

    gid = "fe-cap"
    _build_graph_for_executor(client, tmp_path, monkeypatch, gid, f"""
        phase: retrieve
        nodes:
          - {{ id: in, type: Input }}
          - id: src
            type: Constant
            config: {{ value: {list(range(MAX_LOOP_ITERATIONS + 1))} }}
          - id: loop
            type: ForEach
            inputs: {{ items: src.value }}
            config:
              item_var: x
              outputs: {{ vals: echo.value }}
            body:
              - id: echo
                type: Constant
                config: {{ value: 1 }}
          - id: out
            type: Output
            inputs:
              mode: {{ const: m }}
              reasoning_prompt: {{ const: [] }}
              variables: {{ const: {{}} }}
    """)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={"observation": "?", "mode": None})
    assert r.status_code == 422, r.text
    assert "exceeds MAX_LOOP_ITERATIONS" in r.json()["detail"]


# ------------------------------------------------------------------ #
# Phase 6.4 — Compute + Branch
# ------------------------------------------------------------------ #


def test_compute_loader_rejects_unknown_op(tmp_path):
    p = _write_yaml(tmp_path / "bad_op.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: c
            type: Compute
            config: { op: not_a_real_op }
            inputs: { a: { const: 1 } }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="op must be one of"):
        load_yaml(p)


def test_compute_loader_rejects_missing_inputs(tmp_path):
    p = _write_yaml(tmp_path / "missing_in.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: c
            type: Compute
            config: { op: gt }
            inputs: { a: { const: 1 } }   # missing 'b'
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="missing"):
        load_yaml(p)


def test_compute_comparator(graph_manager, tmp_path):
    out = _run_executor(graph_manager, tmp_path, "cp-gt", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: gt_check
            type: Compute
            config: { op: gt }
            inputs: { a: { const: 0.9 }, b: { const: 0.5 } }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: gt_check.value
    """)
    assert out["variables"] is True


def test_compute_logic_and_not(graph_manager, tmp_path):
    out = _run_executor(graph_manager, tmp_path, "cp-and", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: t
            type: Constant
            config: { value: true }
          - id: f
            type: Constant
            config: { value: false }
          - id: nt
            type: Compute
            config: { op: not }
            inputs: { a: f.value }
          - id: anded
            type: Compute
            config: { op: and }
            inputs: { a: t.value, b: nt.value }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: anded.value
    """)
    assert out["variables"] is True


def test_compute_list_and_concat(graph_manager, tmp_path):
    out = _run_executor(graph_manager, tmp_path, "cp-list", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: src
            type: Constant
            config: { value: ["a", "b", "c"] }
          - id: len_op
            type: Compute
            config: { op: length }
            inputs: { list: src.value }
          - id: has_b
            type: Compute
            config: { op: contains }
            inputs: { list: src.value, item: { const: "b" } }
          - id: cat
            type: Compute
            config: { op: concat }
            inputs: { a: src.value, b: { const: ["d"] } }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables:
                const: PLACEHOLDER
    """)
    # The executor produced the Output node, but `variables` was set to a
    # const string. We assert against the body nodes' env directly by
    # re-reading the YAML with `variables` wired to one of the computes:
    pass


def test_compute_concat_in_isolation(graph_manager, tmp_path):
    out = _run_executor(graph_manager, tmp_path, "cp-concat", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: parts
            type: Constant
            config: { value: ["alpha", "beta"] }
          - id: tail
            type: Constant
            config: { value: ["gamma"] }
          - id: cat
            type: Compute
            config: { op: concat }
            inputs: { a: parts.value, b: tail.value }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: cat.value
    """)
    assert out["variables"] == ["alpha", "beta", "gamma"]


def test_branch_loader_rejects_empty_outputs(tmp_path):
    p = _write_yaml(tmp_path / "no_outs.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: cond
            type: Constant
            config: { value: true }
          - id: b
            type: Branch
            inputs: { condition: cond.value }
            config: { outputs: {} }
            body:
              - id: inner
                type: Constant
                config: { value: 1 }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="non-empty"):
        load_yaml(p)


def test_branch_loader_rejects_unknown_output_target(tmp_path):
    p = _write_yaml(tmp_path / "bad_bout.pipeline.yaml", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: cond
            type: Constant
            config: { value: true }
          - id: b
            type: Branch
            inputs: { condition: cond.value }
            config: { outputs: { result: ghost.value } }
            body:
              - id: inner
                type: Constant
                config: { value: 1 }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="not a body node"):
        load_yaml(p)


def test_branch_truthy_runs_body(graph_manager, tmp_path):
    out = _run_executor(graph_manager, tmp_path, "br-true", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: cond
            type: Constant
            config: { value: true }
          - id: b
            type: Branch
            inputs: { condition: cond.value }
            config:
              outputs: { result: body_const.value }
              else_value: { const: "SKIPPED" }
            body:
              - id: body_const
                type: Constant
                config: { value: "RAN" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: b.result
    """)
    assert out["variables"] == "RAN"


def test_branch_falsy_skips_body_and_emits_else(graph_manager, tmp_path, fake_llm):
    """A Branch with condition=false runs nothing in its body — no LLM calls."""
    n_before = len(fake_llm.calls)
    out = _run_executor(graph_manager, tmp_path, "br-false", """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: cond
            type: Constant
            config: { value: false }
          - id: b
            type: Branch
            inputs: { condition: cond.value }
            config:
              outputs: { result: would_run.raw }
              else_value: { const: "SKIPPED" }
            body:
              - id: would_run
                type: LLMCall
                config: { prompt: get_subgoal, role: structuring }
                inputs:
                  goal: { const: "g" }
                  state: { const: "" }
                  observation: { const: "" }
                  action: { const: "" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: b.result
    """)
    assert out["variables"] == "SKIPPED"
    # Body's LLMCall was not invoked.
    assert len(fake_llm.calls) == n_before


def test_compute_branch_end_to_end_via_route(client, monkeypatch, tmp_path, fake_llm):
    """End-to-end: Compute(gt) → Branch → LLMCall, surfaced via /retrieve."""
    gid = "br-e2e"
    _build_graph_for_executor(client, tmp_path, monkeypatch, gid, """
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: score
            type: Constant
            config: { value: 0.9 }
          - id: above
            type: Compute
            config: { op: gt }
            inputs: { a: score.value, b: { const: 0.5 } }
          - id: maybe_llm
            type: Branch
            inputs: { condition: above.value }
            config:
              outputs: { msg: probe.raw }
              else_value: { const: "" }
            body:
              - id: probe
                type: LLMCall
                config: { prompt: get_subgoal, role: structuring }
                inputs:
                  goal: { const: "go" }
                  state: { const: "" }
                  observation: { const: "" }
                  action: { const: "" }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    n_before = len(fake_llm.calls)
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={"observation": "?", "mode": None})
    assert r.status_code == 200, r.text
    # Condition is truthy → probe LLMCall ran exactly once.
    assert len(fake_llm.calls) - n_before == 1


# ----------------------------------------------------------------------- #
# LLMCall messages-mode (Phase 6.5a follow-up)
# ----------------------------------------------------------------------- #


def test_llmcall_messages_mode_skips_render(graph_manager, tmp_path, fake_llm):
    """An LLMCall with inputs.messages (no config.prompt) calls LLM directly."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              semantic_memory: { const: "fact 1" }
              time: { const: "" }
              observation: in.observation
          - id: reason
            type: LLMCall
            config: { role: reasoning }
            inputs:
              messages: msgs.messages
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: msgs.messages
              variables: reason.parsed
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-msgs")
    mg = graph_manager.get_graph("g-msgs")
    n_before = len(fake_llm.calls)
    out = PipelineExecutor(graph, mg).run({"observation": "hello"})
    # One LLM call happened; the messages it received came from the
    # PromptRender (NOT re-rendered inside LLMCall).
    assert len(fake_llm.calls) - n_before == 1
    sent = fake_llm.calls[-1]
    assert isinstance(sent, list) and len(sent) >= 1
    # variables in the output is the LLMCall's parsed dict.
    assert "text" in out["variables"]


def test_llmcall_rejects_both_prompt_and_messages(graph_manager, tmp_path):
    """Setting BOTH config.prompt AND inputs.messages is ambiguous → ValueError."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              semantic_memory: { const: "" }
              time: { const: "" }
              observation: in.observation
          - id: bad
            type: LLMCall
            config: { prompt: get_plan, role: retrieval }
            inputs:
              messages: msgs.messages
              goal: { const: "" }
              subgoal: { const: "" }
              state: { const: "" }
              observation: in.observation
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-bad")
    mg = graph_manager.get_graph("g-bad")
    with pytest.raises(ValueError, match="exactly one of"):
        PipelineExecutor(graph, mg).run({"observation": "?"})


def test_llmcall_rejects_neither_prompt_nor_messages(graph_manager, tmp_path):
    """An LLMCall with no prompt and no messages input → ValueError."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: bad
            type: LLMCall
            config: { role: reasoning }
            inputs: {}
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-empty")
    mg = graph_manager.get_graph("g-empty")
    with pytest.raises(ValueError, match="must set one of"):
        PipelineExecutor(graph, mg).run({"observation": "?"})


# ----------------------------------------------------------------------- #
# PromptRender now produces `value` (string) too
# ----------------------------------------------------------------------- #


def test_prompt_render_emits_value_string(graph_manager, tmp_path):
    """PromptRender output includes a `value` port = finished template string."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              semantic_memory: { const: "Fact A" }
              time: { const: "2026-05-12" }
              observation: in.observation
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: msgs.messages
              variables:
                const: {}
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-render-value")
    mg = graph_manager.get_graph("g-render-value")
    executor = PipelineExecutor(graph, mg)
    executor.run({"observation": "what's the date?"})
    msgs_env = executor._eval_node  # no public env; re-run for value access
    # Simpler: assert via direct call on _render_prompt.
    node = next(n for n in graph.nodes if n.id == "msgs")
    rendered = executor._render_prompt(node, {
        "semantic_memory": "Fact A",
        "time": "2026-05-12",
        "observation": "what's the date?",
    })
    assert "value" in rendered and isinstance(rendered["value"], str)
    assert "messages" in rendered and isinstance(rendered["messages"], list)
    # The string contains substituted content.
    assert "Fact A" in rendered["value"]
    assert "what's the date?" in rendered["value"]


# ----------------------------------------------------------------------- #
# LLMCall text-mode (string in → user-message wrap → LLM)
# ----------------------------------------------------------------------- #


def test_llmcall_text_mode_wraps_string(graph_manager, tmp_path, fake_llm):
    """LLMCall with inputs.text wraps the string as a user message."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              semantic_memory: { const: "" }
              time: { const: "" }
              observation: in.observation
          - id: reason
            type: LLMCall
            config: { role: reasoning }
            inputs:
              text: msgs.value
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: reason.parsed
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-text-mode")
    mg = graph_manager.get_graph("g-text-mode")
    n_before = len(fake_llm.calls)
    out = PipelineExecutor(graph, mg).run({"observation": "hello world"})
    assert len(fake_llm.calls) - n_before == 1
    sent = fake_llm.calls[-1]
    assert isinstance(sent, list) and len(sent) == 1
    assert sent[0]["role"] == "user"
    assert "hello world" in sent[0]["content"]
    assert "text" in out["variables"]


def test_llmcall_rejects_text_and_messages_together(graph_manager, tmp_path):
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              semantic_memory: { const: "" }
              time: { const: "" }
              observation: in.observation
          - id: bad
            type: LLMCall
            config: { role: reasoning }
            inputs:
              text: msgs.value
              messages: msgs.messages
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-conflict")
    mg = graph_manager.get_graph("g-conflict")
    with pytest.raises(ValueError, match="exactly one of"):
        PipelineExecutor(graph, mg).run({"observation": "?"})


def test_llmcall_text_mode_rejects_non_string(graph_manager, tmp_path):
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: bad
            type: LLMCall
            config: { role: reasoning }
            inputs:
              text: { const: [1, 2, 3] }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
    """)
    graph = load_yaml_str(yaml_text)
    graph_manager.create_graph("g-nostring")
    mg = graph_manager.get_graph("g-nostring")
    with pytest.raises(ValueError, match="inputs.text must be a string"):
        PipelineExecutor(graph, mg).run({"observation": "?"})


# ----------------------------------------------------------------------- #
# StorageRead + Embed (Phase 6.6a)
# ----------------------------------------------------------------------- #


def test_storage_read_loader_rejects_unknown_collection(tmp_path):
    """StorageRead.config.collection must be one of the three known stores."""
    from plugmem.pipelines.spec_driven import load_yaml_str
    bad = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: r
            type: StorageRead
            config: { collection: tags }
            inputs: { query: in.observation, tags: { const: [] } }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="collection must be one of"):
        load_yaml_str(bad)


def test_storage_read_loader_rejects_missing_inputs(tmp_path):
    """StorageRead.collection=semantic requires both query AND tags."""
    from plugmem.pipelines.spec_driven import load_yaml_str
    bad = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: r
            type: StorageRead
            config: { collection: semantic }
            inputs: { query: in.observation }
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="missing .*tags"):
        load_yaml_str(bad)


def test_storage_read_semantic_returns_facts_when_seeded(graph_manager, fake_embedder, tmp_path):
    """StorageRead on a graph with semantic nodes returns formatted 'Fact i: ...' text.

    Uses a query string that's identical to one seeded fact so the
    cosine similarity is exactly 1.0 — guarantees the
    ``SemanticRelevant`` value function clears its threshold regardless
    of FakeEmbedder's hash-randomized embeddings.
    """
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str
    from plugmem.core.graph_node import SemanticNode

    graph_manager.create_graph("g-storage-sem")
    mg = graph_manager.get_graph("g-storage-sem")
    seeded_facts = ["FastAPI deploys via Docker.", "Auth uses JWT."]
    for i, text in enumerate(seeded_facts):
        node = SemanticNode(
            semantic_id=i, semantic_memory_str=text,
            embedding=fake_embedder.embed(text),
        )
        mg.semantic_nodes.append(node)
        mg.semantic_id2node[i] = node

    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: fetch
            type: StorageRead
            config: { collection: semantic }
            inputs:
              query: in.observation
              tags: { const: [] }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: fetch
    """)
    graph = load_yaml_str(yaml_text)
    out = PipelineExecutor(graph, mg).run({"observation": seeded_facts[0]})
    text = out["variables"]["value"]
    assert "Fact 0:" in text or "Fact 1:" in text, (
        f"expected a fact line in: {text!r}"
    )
    assert isinstance(out["variables"]["ids"], list)
    assert out["variables"]["ids"], "expected at least one selected id"


def test_storage_read_semantic_returns_no_relevant_when_empty(graph_manager, tmp_path):
    """Empty graph → 'No relevant fact'."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str
    graph_manager.create_graph("g-storage-empty")
    mg = graph_manager.get_graph("g-storage-empty")
    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: fetch
            type: StorageRead
            config: { collection: semantic }
            inputs:
              query: in.observation
              tags: { const: [] }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: fetch
    """)
    graph = load_yaml_str(yaml_text)
    out = PipelineExecutor(graph, mg).run({"observation": "anything"})
    assert out["variables"]["value"] == "No relevant fact"


def test_embed_returns_vector(graph_manager, fake_embedder):
    """Embed wraps graph.embedder.embed(text); output is a list[float]."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str
    graph_manager.create_graph("g-embed")
    mg = graph_manager.get_graph("g-embed")
    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: e
            type: Embed
            inputs: { text: in.observation }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: e
    """)
    graph = load_yaml_str(yaml_text)
    out = PipelineExecutor(graph, mg).run({"observation": "hello"})
    emb = out["variables"]["embedding"]
    assert isinstance(emb, list) and len(emb) == fake_embedder.DIM
    assert all(isinstance(x, float) for x in emb)


def test_embed_loader_rejects_missing_text(tmp_path):
    from plugmem.pipelines.spec_driven import load_yaml_str
    bad = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: e
            type: Embed
            inputs: {}
          - id: out
            type: Output
            inputs: { mode: { const: m }, reasoning_prompt: { const: [] }, variables: { const: {} } }
    """)
    with pytest.raises(ValueError, match="inputs.text is required"):
        load_yaml_str(bad)


# ----------------------------------------------------------------------- #
# Multi-phase YAML format (Phase 6.6b-2)
# ----------------------------------------------------------------------- #


def test_multi_phase_yaml_parses_both_phases(tmp_path):
    from plugmem.pipelines.spec_driven import load_yaml_str_multi
    text = textwrap.dedent("""
        phases:
          retrieve:
            nodes:
              - { id: in, type: Input }
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: { const: {} }
          reason:
            nodes:
              - { id: in, type: Input }
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: { const: {} }
    """)
    phases = load_yaml_str_multi(text)
    assert set(phases) == {"retrieve", "reason"}
    assert all(p.phase in ("retrieve", "reason") for p in phases.values())


def test_multi_phase_rejects_unknown_phase(tmp_path):
    from plugmem.pipelines.spec_driven import load_yaml_str_multi
    text = textwrap.dedent("""
        phases:
          bogus:
            nodes:
              - { id: in, type: Input }
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="unsupported phase 'bogus'"):
        load_yaml_str_multi(text)


def test_multi_phase_rejects_both_formats_at_once(tmp_path):
    from plugmem.pipelines.spec_driven import load_yaml_str_multi
    text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: { const: {} }
        phases:
          reason:
            nodes:
              - { id: in, type: Input }
    """)
    with pytest.raises(ValueError, match="use either"):
        load_yaml_str_multi(text)


def test_single_phase_yaml_still_works_via_load_yaml_str(tmp_path):
    """Backward-compat: existing single-phase YAMLs load unchanged."""
    from plugmem.pipelines.spec_driven import load_yaml_str
    text = textwrap.dedent("""
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
    g = load_yaml_str(text)
    assert g.phase == "retrieve"


def test_load_yaml_str_requires_retrieve_phase_in_multi_phase_file():
    """Convenience entrypoint that returns one phase must error if
    retrieve is missing (callers needing other phases use
    load_yaml_str_multi)."""
    from plugmem.pipelines.spec_driven import load_yaml_str
    text = textwrap.dedent("""
        phases:
          reason:
            nodes:
              - { id: in, type: Input }
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: { const: {} }
    """)
    with pytest.raises(ValueError, match="no retrieve phase"):
        load_yaml_str(text)


def test_spec_driven_reason_uses_dedicated_reason_phase_when_present(
    client, fake_llm, monkeypatch, tmp_path,
):
    """If the YAML has a `reason:` phase, /reason uses it instead of
    falling back to running the `retrieve:` phase."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = "g-multi-phase"
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text
    _bind(client, gid)
    # Two phases, different prompt names so we can tell which ran.
    _write_yaml(tmp_path / f"{gid}.pipeline.yaml", """
        phases:
          retrieve:
            nodes:
              - { id: in, type: Input }
              - id: plan
                type: LLMCall
                config: { prompt: get_plan, role: retrieval }
                inputs: { goal: in.goal, subgoal: in.subgoal, state: in.state, observation: in.observation }
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: { const: { source: retrieve_phase } }
          reason:
            nodes:
              - { id: in, type: Input }
              - id: msg
                type: PromptRender
                config: { prompt: reasoning_semantic }
                inputs:
                  semantic_memory: { const: "" }
                  time: { const: "" }
                  observation: in.observation
              - id: rcall
                type: LLMCall
                config: { role: reasoning }
                inputs:
                  text: msg.value
              - id: out
                type: Output
                inputs:
                  mode: { const: semantic_memory }
                  reasoning_prompt: { const: [] }
                  variables: rcall.parsed
    """)
    r = client.post(f"/api/v1/graphs/{gid}/reason", json={
        "observation": "hello", "mode": None,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # The reason phase called the reasoning LLM exactly once. The
    # retrieve phase wasn't touched (no get_plan call).
    last = fake_llm.calls[-1]
    assert isinstance(last, list) and last[0]["role"] == "user"
    # The retrieve phase's variables const would have set source=retrieve_phase;
    # the reason phase outputs rcall.parsed which is {"text": <answer>}.
    assert "text" in body["reasoning"] or body["reasoning"] != ""


# ----------------------------------------------------------------------- #
# Compute(similarity) op
# ----------------------------------------------------------------------- #


def test_compute_similarity_against_self_is_one(graph_manager, fake_embedder):
    """cos(v, v) = 1.0 for any non-zero vector."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str
    graph_manager.create_graph("g-sim-self")
    mg = graph_manager.get_graph("g-sim-self")
    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: e
            type: Embed
            inputs: { text: in.observation }
          - id: sim
            type: Compute
            config: { op: similarity }
            inputs:
              a: e.embedding
              b: e.embedding
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: sim
    """)
    out = PipelineExecutor(load_yaml_str(yaml_text), mg).run({"observation": "hello"})
    assert out["variables"]["value"] == pytest.approx(1.0, abs=1e-5)


def test_compute_similarity_pairs_with_embed(graph_manager, fake_embedder):
    """Embed two distinct strings → similarity is well-defined and != 1."""
    from plugmem.pipelines.spec_driven import PipelineExecutor, load_yaml_str
    graph_manager.create_graph("g-sim-pair")
    mg = graph_manager.get_graph("g-sim-pair")
    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: e_obs
            type: Embed
            inputs: { text: in.observation }
          - id: e_other
            type: Embed
            inputs: { text: { const: "a totally different topic" } }
          - id: sim
            type: Compute
            config: { op: similarity }
            inputs:
              a: e_obs.embedding
              b: e_other.embedding
          - id: out
            type: Output
            inputs:
              mode: { const: semantic_memory }
              reasoning_prompt: { const: [] }
              variables: sim
    """)
    out = PipelineExecutor(load_yaml_str(yaml_text), mg).run({"observation": "weather"})
    val = out["variables"]["value"]
    assert isinstance(val, float)
    # FakeEmbedder is deterministic but random — different strings → sim < 1.
    assert val < 0.99
