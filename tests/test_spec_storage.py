"""Tests for the spec-driven YAML version store + the editor routes."""
from __future__ import annotations

import json
import textwrap
import time

import pytest

from plugmem.pipelines import spec_storage
from plugmem.pipelines.spec_driven import load_yaml_str


VALID_YAML = textwrap.dedent("""
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


VALID_YAML_V2 = textwrap.dedent("""
    phase: retrieve
    nodes:
      - { id: in, type: Input }
      - id: msg
        type: PromptRender
        config: { prompt: reasoning_semantic }
        inputs:
          goal: { const: "" }
          subgoal: { const: "" }
          state: { const: "" }
          observation: in.observation
          semantic_memory: { const: "" }
          procedural_memory: { const: "" }
          episodic_memory: { const: "" }
          time: { const: "" }
          information: { const: "" }
          question: in.observation
      - id: out
        type: Output
        inputs:
          mode: { const: semantic_memory }
          reasoning_prompt: msg.messages
          variables: { const: {} }
""")


# --------------------------------------------------------------------- #
# Storage module (pure)
# --------------------------------------------------------------------- #


def test_save_creates_version_and_live_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    v = spec_storage.save_new_version("g1", VALID_YAML, note="first",
                                      validator=load_yaml_str)
    assert v.active is True
    assert v.parent_version_id is None
    assert (tmp_path / "g1.pipeline.yaml").read_text() == VALID_YAML
    hist = tmp_path / ".pipeline_history" / "g1"
    assert (hist / f"{v.version_id}.pipeline.yaml").exists()
    assert (hist / f"{v.version_id}.meta.json").exists()
    assert (hist / "current.txt").read_text().strip() == v.version_id


def test_invalid_yaml_blocks_save(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    bad = "phase: retrieve\nnodes: not-a-list\n"
    with pytest.raises(ValueError):
        spec_storage.save_new_version("g1", bad, validator=load_yaml_str)
    # Nothing was written.
    assert not (tmp_path / "g1.pipeline.yaml").exists()
    assert not (tmp_path / ".pipeline_history" / "g1").exists()


def test_second_save_records_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    v1 = spec_storage.save_new_version("g1", VALID_YAML, validator=load_yaml_str)
    # Sleep just enough to guarantee a distinct version_id timestamp / hex.
    time.sleep(0.01)
    v2 = spec_storage.save_new_version("g1", VALID_YAML_V2, note="v2",
                                       validator=load_yaml_str)
    assert v2.parent_version_id == v1.version_id
    assert v2.version_id != v1.version_id
    # Live file is now v2.
    assert (tmp_path / "g1.pipeline.yaml").read_text() == VALID_YAML_V2


def test_list_versions_newest_first(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    v1 = spec_storage.save_new_version("g1", VALID_YAML, validator=load_yaml_str)
    time.sleep(0.01)
    v2 = spec_storage.save_new_version("g1", VALID_YAML_V2, validator=load_yaml_str)
    versions = spec_storage.list_versions("g1")
    assert [v.version_id for v in versions] == [v2.version_id, v1.version_id]
    actives = [v for v in versions if v.active]
    assert len(actives) == 1 and actives[0].version_id == v2.version_id


def test_rollback_promotes_old_version(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    v1 = spec_storage.save_new_version("g1", VALID_YAML, validator=load_yaml_str)
    time.sleep(0.01)
    spec_storage.save_new_version("g1", VALID_YAML_V2, validator=load_yaml_str)
    rolled = spec_storage.rollback("g1", v1.version_id)
    assert rolled.active is True
    assert (tmp_path / "g1.pipeline.yaml").read_text() == VALID_YAML
    assert spec_storage.get_active_version_id("g1") == v1.version_id


def test_rollback_missing_version_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    spec_storage.save_new_version("g1", VALID_YAML, validator=load_yaml_str)
    with pytest.raises(ValueError):
        spec_storage.rollback("g1", "v_19000101T000000_deadbe")


def test_adopt_existing_live_seeds_history(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    # Hand-written file with no history at all.
    (tmp_path / "g1.pipeline.yaml").write_text(VALID_YAML)
    vid = spec_storage.adopt_existing_live("g1")
    assert vid is not None
    versions = spec_storage.list_versions("g1")
    assert len(versions) == 1
    assert versions[0].version_id == vid
    assert versions[0].active is True


def test_save_after_adopt_records_adopted_as_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    (tmp_path / "g1.pipeline.yaml").write_text(VALID_YAML)
    v = spec_storage.save_new_version("g1", VALID_YAML_V2, validator=load_yaml_str)
    assert v.parent_version_id is not None
    # The adopted version is preserved.
    assert len(spec_storage.list_versions("g1")) == 2


# --------------------------------------------------------------------- #
# Route layer (FastAPI)
# --------------------------------------------------------------------- #


def _create_graph(client) -> str:
    r = client.post("/api/v1/graphs", json={})
    assert r.status_code == 201, r.text
    return r.json()["graph_id"]


def test_route_get_empty_returns_blank(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/spec")
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == ""
    assert body["exists"] is False
    assert body["active_version_id"] is None


def test_route_save_writes_versioned(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML, "note": "init"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"]["active"] is True
    # Read-back returns the same content.
    r2 = client.get(f"/api/v1/graphs/{gid}/pipeline/spec")
    assert r2.status_code == 200
    assert r2.json()["content"] == VALID_YAML


def test_route_save_invalid_returns_422(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": "phase: retrieve\nnodes: not-a-list\n"},
    )
    assert r.status_code == 422
    assert "nodes" in r.json()["detail"].lower()


def test_route_validate_returns_error_payload(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.post(
        f"/api/v1/graphs/{gid}/pipeline/spec/validate",
        json={"content": "phase: retrieve\nnodes: not-a-list\n"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]
    r2 = client.post(
        f"/api/v1/graphs/{gid}/pipeline/spec/validate",
        json={"content": VALID_YAML},
    )
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_route_list_versions_and_rollback(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r1 = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML, "note": "v1"},
    )
    v1_id = r1.json()["version"]["version_id"]
    time.sleep(0.01)
    r2 = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML_V2, "note": "v2"},
    )
    v2_id = r2.json()["version"]["version_id"]

    rlist = client.get(f"/api/v1/graphs/{gid}/pipeline/spec/versions")
    assert rlist.status_code == 200
    rows = rlist.json()["versions"]
    assert {r["version_id"] for r in rows} == {v1_id, v2_id}
    # v2 is active.
    active = [r for r in rows if r["active"]]
    assert len(active) == 1 and active[0]["version_id"] == v2_id

    # Rollback to v1.
    rr = client.post(
        f"/api/v1/graphs/{gid}/pipeline/spec/versions/{v1_id}/rollback",
    )
    assert rr.status_code == 200, rr.text
    assert rr.json()["version"]["version_id"] == v1_id

    # Live file is now v1.
    rget = client.get(f"/api/v1/graphs/{gid}/pipeline/spec")
    assert rget.json()["content"] == VALID_YAML


def test_route_rollback_missing_returns_404(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML},
    )
    r = client.post(
        f"/api/v1/graphs/{gid}/pipeline/spec/versions/"
        f"v_19000101T000000_deadbe/rollback",
    )
    assert r.status_code == 404


def test_route_get_version_content(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r1 = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML, "note": "v1"},
    )
    vid = r1.json()["version"]["version_id"]
    rget = client.get(
        f"/api/v1/graphs/{gid}/pipeline/spec/versions/{vid}",
    )
    assert rget.status_code == 200
    body = rget.json()
    assert body["content"] == VALID_YAML
    assert body["version"]["version_id"] == vid


def test_route_save_runs_through_spec_driven_pipeline(
    client, fake_llm, monkeypatch, tmp_path,
):
    """End-to-end: PUT spec → bind pipeline → /retrieve uses saved YAML."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)

    # Save a runnable spec via the editor route.
    yaml_text = textwrap.dedent("""
        phase: retrieve
        nodes:
          - { id: in, type: Input }
          - id: msgs
            type: PromptRender
            config: { prompt: reasoning_semantic }
            inputs:
              goal: { const: "" }
              subgoal: { const: "" }
              state: { const: "" }
              observation: in.observation
              semantic_memory: { const: "" }
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
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": yaml_text, "note": "init"},
    )
    assert r.status_code == 200

    # Bind to spec-driven.
    rb = client.put(
        f"/api/v1/graphs/{gid}/pipeline",
        json={"pipeline": "spec-driven"},
    )
    assert rb.status_code == 200

    # Retrieve runs the saved pipeline.
    rr = client.post(
        f"/api/v1/graphs/{gid}/retrieve",
        json={"observation": "test observation"},
    )
    assert rr.status_code == 200, rr.text
    body = rr.json()
    assert body["mode"] == "semantic_memory"
    assert body["reasoning_prompt"]


def test_route_invalid_version_id_format_is_404(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": VALID_YAML},
    )
    r = client.get(
        f"/api/v1/graphs/{gid}/pipeline/spec/versions/not_a_version",
    )
    assert r.status_code == 404
