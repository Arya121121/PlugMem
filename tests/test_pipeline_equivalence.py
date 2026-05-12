"""Differential harness: spec-driven YAML ≡ plugmem-default for retrieve+reason.

Goal: prove that the shipped ``plugmem-default`` sample YAML produces
**the same observable behaviour** as ``PlugMemDefaultPipeline`` when
both are fed identical inputs against an identically-seeded graph.

This is the equivalence guarantee that Phase 6.5d's grammar doc + Phase
6.6a's StorageRead nodes were building towards. With this harness in CI,
silent drift between the spec-driven sample and the production pipeline
fails the build.

Two pipelines, one comparison point:

| plugmem-default | spec-driven (sample) |
|---|---|
| ``POST /reason`` | ``POST /retrieve`` |
| 3 LLM calls (get_plan, get_mode, reason) | 3 LLM calls (get_plan, get_mode, reason_llm_call) |
| Response: ``ReasonResponse.reasoning`` | Response: ``variables.text`` |

Both flow through the same ``FakeLLM`` so canned responses are
identical; both call ``MemoryGraph.retrieve_*_nodes`` (default
directly, spec-driven via ``StorageRead``) against the same seeded
chroma collections.
"""
from __future__ import annotations

import pytest

from plugmem.pipelines.sample_specs import PLUGMEM_DEFAULT_RETRIEVE_YAML


SEED_PAYLOAD = {
    "mode": "structured",
    "semantic": [
        {
            "semantic_memory": "FastAPI builds APIs with Pydantic models.",
            "tags": ["python", "fastapi", "api"],
        },
        {
            "semantic_memory": "Docker containers package apps for deployment.",
            "tags": ["docker", "deployment"],
        },
    ],
    "procedural": [
        {"subgoal": "deploy api service",
         "procedural_memory": "Run docker-compose up after pushing the image."},
    ],
}


def _create_graph(client, gid: str) -> None:
    r = client.post("/api/v1/graphs", json={"graph_id": gid})
    assert r.status_code in (200, 201), r.text


def _seed(client, gid: str) -> None:
    r = client.post(f"/api/v1/graphs/{gid}/memories", json=SEED_PAYLOAD)
    assert r.status_code == 200, r.text


def _bind_spec_driven_with_sample(client, gid: str) -> None:
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline",
        json={"pipeline": "spec-driven"},
    )
    assert r.status_code == 200, r.text
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline/spec",
        json={"content": PLUGMEM_DEFAULT_RETRIEVE_YAML, "note": "diff harness"},
    )
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------- #
# Equivalence checks
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("observation", [
    "How do we deploy the API?",
    "What does FastAPI use for validation?",
])
def test_spec_driven_and_default_pick_same_mode(
    client, fake_llm, monkeypatch, tmp_path, observation,
):
    """Both pipelines see the same mode classifier output."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    _create_graph(client, "g-equiv-default-1")
    _create_graph(client, "g-equiv-spec-1")
    _seed(client, "g-equiv-default-1")
    _seed(client, "g-equiv-spec-1")
    _bind_spec_driven_with_sample(client, "g-equiv-spec-1")

    req = {"observation": observation, "goal": "ship", "state": "ok"}
    d = client.post("/api/v1/graphs/g-equiv-default-1/retrieve", json=req)
    s = client.post("/api/v1/graphs/g-equiv-spec-1/retrieve", json=req)
    assert d.status_code == 200 and s.status_code == 200, (d.text, s.text)
    assert d.json()["mode"] == s.json()["mode"]


def test_spec_driven_and_default_reasoning_answer_matches(
    client, fake_llm, monkeypatch, tmp_path,
):
    """spec-driven /retrieve answer == default /reason answer (same FakeLLM)."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    _create_graph(client, "g-equiv-default-2")
    _create_graph(client, "g-equiv-spec-2")
    _seed(client, "g-equiv-default-2")
    _seed(client, "g-equiv-spec-2")
    _bind_spec_driven_with_sample(client, "g-equiv-spec-2")

    req = {"observation": "How do we deploy?", "goal": "ship", "state": "ok"}
    d_calls_before = len(fake_llm.calls)
    d = client.post("/api/v1/graphs/g-equiv-default-2/reason", json=req).json()
    d_llm_count = len(fake_llm.calls) - d_calls_before

    s_calls_before = len(fake_llm.calls)
    s = client.post("/api/v1/graphs/g-equiv-spec-2/retrieve", json=req).json()
    s_llm_count = len(fake_llm.calls) - s_calls_before

    # Both call the same set of LLMs: get_plan + get_mode + reasoning.
    assert d_llm_count == s_llm_count == 3, (d_llm_count, s_llm_count)
    # Same mode picked.
    assert d["mode"] == s["mode"]
    # Same reasoning answer (FakeLLM is deterministic; both hit the same
    # reasoning template with the same retrieved memory string).
    assert d["reasoning"] == s["variables"]["text"]


def test_spec_driven_storage_read_hits_same_chroma_data_as_default(
    client, fake_llm, monkeypatch, tmp_path,
):
    """The reasoning prompt's semantic_memory block (the retrieved facts)
    is identical in both pipelines — proof that StorageRead reaches the
    same backing store as MemoryGraph.retrieve_semantic_nodes."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    _create_graph(client, "g-equiv-default-3")
    _create_graph(client, "g-equiv-spec-3")
    _seed(client, "g-equiv-default-3")
    _seed(client, "g-equiv-spec-3")
    _bind_spec_driven_with_sample(client, "g-equiv-spec-3")

    req = {"observation": "tell me about FastAPI", "goal": "learn",
           "state": "ok", "mode": "semantic_memory"}

    # default /retrieve produces messages WITHOUT a reasoning LLM call,
    # so we can read the semantic_memory string from the messages.
    d = client.post("/api/v1/graphs/g-equiv-default-3/retrieve", json=req).json()
    default_user_msg = "\n".join(
        m["content"] for m in d["reasoning_prompt"] if m["role"] == "user"
    )

    # spec-driven /retrieve also captures the same rendered messages
    # by calling FakeLLM with them; we inspect the LAST FakeLLM call
    # (the reason_llm_call), which received the rendered prompt as a
    # single user message in text-mode.
    fake_llm.calls.clear()
    client.post("/api/v1/graphs/g-equiv-spec-3/retrieve", json=req)
    spec_reason_msg = fake_llm.calls[-1]
    assert isinstance(spec_reason_msg, list) and len(spec_reason_msg) == 1
    spec_user_msg = spec_reason_msg[0]["content"]

    # The "Facts: ..." block from retrieve_semantic_nodes / StorageRead
    # is the load-bearing piece. Strip the surrounding template
    # boilerplate and compare just the fact lines.
    def _facts_block(text: str) -> str:
        lines = [ln for ln in text.split("\n") if ln.startswith("Fact ")]
        return "\n".join(sorted(lines))

    assert _facts_block(default_user_msg) == _facts_block(spec_user_msg), (
        f"Default facts: {_facts_block(default_user_msg)!r}\n"
        f"Spec   facts: {_facts_block(spec_user_msg)!r}"
    )
