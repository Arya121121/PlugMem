"""Runtime check: traces produced by every public route must conform to the spec.

Spins up a TestClient with a deterministic mock LLM that returns a single
multi-section response satisfying every prompt parser in the inference
layer, hits each route end-to-end, then reads back the persisted traces
and asserts each step matches what ``pipeline_spec.STEPS`` declares.

Catches drift the static lint can't see — e.g. a call site that
constructs ``variables`` via ``dict(...)`` instead of a literal, a new
prompt that's invoked but never registered in the spec, or a path that
silently stops firing.
"""
from __future__ import annotations

from typing import Dict, List

import chromadb
import pytest
from fastapi.testclient import TestClient

from plugmem.api import dependencies as deps
from plugmem.api.app import create_app
from plugmem.clients.embedding import PlugMemEmbeddingFunction
from plugmem.clients.llm import LLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.core.pipeline_spec import STEPS
from plugmem.graph_manager import GraphManager
from plugmem.storage.chroma import ChromaStorage

from tests.conftest import FakeEmbedder


# Single response that satisfies *every* prompt parser. Order matters
# because some regexes are greedy under re.S — keep the structured
# sections at the top.
MEGA_RESPONSE = """### Subgoal
inferred subgoal

### Reward
reasonable progress

### State
state summary text

### Memory Type
semantic_memory

### Score
0.5

### Goal
distilled goal
### Experiential Insight
distilled insight

### Next Subgoal
the next subgoal

### Facts
**Statement:** Test fact one.
**Tags:** ["alpha", "beta"]

```json
{"merged_statement": "merged", "relationship": "SAME_TOPIC_MERGE_WELL", "deactivate_earlier": false, "deactivate_later": false, "simple_reasoning": "ok"}
```
"""


class StrongFakeLLM(LLMClient):
    """Deterministic LLM that returns one mega-response for every call.

    Exposes the attributes LLMRouter.role_summary probes (base_url, api_key,
    model, is_azure, azure_api_version) so the inspector model endpoints
    work too if the test wants to hit them.
    """

    base_url = "mock://strong"
    api_key = "mock-key"
    model = "strong-fake"
    is_azure = False
    azure_api_version = ""

    def __init__(self) -> None:
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096) -> str:
        self.calls.append(messages)
        return MEGA_RESPONSE


@pytest.fixture()
def strong_client(tmp_path, monkeypatch):
    """TestClient wired with a deterministic LLM + ephemeral chroma."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path / "prompts"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("PLUGMEM_API_KEY", "")

    deps.reset_singletons()

    llm = StrongFakeLLM()
    embedder = FakeEmbedder()
    chroma = chromadb.EphemeralClient()
    embed_fn = PlugMemEmbeddingFunction(embedder)
    storage = ChromaStorage(client=chroma, embedding_function=embed_fn, embedding_client=embedder)
    from plugmem.prompts.registry import PromptRegistry
    registry = PromptRegistry(prompts_dir=str(tmp_path / "prompts"))
    gm = GraphManager(storage=storage, llm=LLMRouter.from_single_client(llm), embedder=embedder, prompts=registry)

    deps._llm_client = LLMRouter.from_single_client(llm)
    deps._embedding_client = embedder
    deps._graph_manager = gm
    deps._prompt_registry = registry

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    deps.reset_singletons()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _exercise_routes(client: TestClient) -> str:
    """Drive every route that records LLM calls; return the graph_id."""
    r = client.post("/api/v1/graphs", json={"graph_id": "specrt"})
    assert r.status_code in (200, 201), r.text
    gid = r.json()["graph_id"]

    # Trajectory ingest: fires get_subgoal / get_reward / get_state per step,
    # then get_semantic per step + get_procedural per trajectory at close.
    # The first ingest creates a new subgoal node; the second collides on
    # the same get_procedural-derived subgoal and triggers get_new_subgoal.
    for goal in ("test the pipeline", "again — collide on subgoal"):
        r = client.post(f"/api/v1/graphs/{gid}/memories", json={
            "mode": "trajectory",
            "goal": goal,
            "steps": [
                {"observation": "obs one", "action": "act one"},
                {"observation": "obs two", "action": "act two"},
            ],
        })
        assert r.status_code == 200, r.text

    # Retrieve with mode=None: fires get_plan + get_mode (production path).
    r = client.post(f"/api/v1/graphs/{gid}/retrieve", json={
        "observation": "what's next?",
        "mode": None,
    })
    assert r.status_code == 200, r.text

    # Reason: triggers the retrieve flow + the reasoning_llm.complete call.
    r = client.post(f"/api/v1/graphs/{gid}/reason", json={
        "observation": "explain please",
        "mode": None,
    })
    assert r.status_code == 200, r.text

    return gid


def _collect_steps(client: TestClient, gid: str) -> List[Dict]:
    """Walk every persisted trace and flatten all step records."""
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/traces?limit=200")
    assert r.status_code == 200, r.text
    traces = r.json()["traces"]
    all_steps: List[Dict] = []
    for t in traces:
        r2 = client.get(f"/api/v1/graphs/{gid}/pipeline/traces/{t['trace_id']}")
        assert r2.status_code == 200, r2.text
        for s in r2.json().get("steps") or []:
            all_steps.append({**s, "_endpoint": t["endpoint"]})
    return all_steps


SPEC_LLM = {s.id: s for s in STEPS if s.kind == "llm"}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


def test_routes_run_clean(strong_client):
    """End-to-end sanity: every route returns 200 with the mega-response LLM."""
    gid = _exercise_routes(strong_client)
    assert gid


def test_every_recorded_step_is_in_spec(strong_client):
    """No trace step name should be absent from pipeline_spec.STEPS."""
    gid = _exercise_routes(strong_client)
    steps = _collect_steps(strong_client, gid)
    assert steps, "No steps recorded — recorder is broken or routes aren't hitting LLM calls"
    unknown = sorted({s["name"] for s in steps} - set(SPEC_LLM))
    assert not unknown, (
        f"Recorded step names not in spec (kind=llm): {unknown}. "
        f"Either add a spec entry or remove the record_llm_step call."
    )


def test_recorded_variables_match_spec_inputs(strong_client):
    """For every recorded step, variables.keys() must equal spec.inputs."""
    gid = _exercise_routes(strong_client)
    steps = _collect_steps(strong_client, gid)
    bad = []
    for s in steps:
        name = s["name"]
        if name not in SPEC_LLM:
            continue  # caught by other test
        spec_inputs = set(SPEC_LLM[name].inputs)
        recorded_vars = set((s.get("variables") or {}).keys())
        if name == "reason_llm_call":
            # reason_llm_call records {observation, mode} as trace
            # annotations; spec declares ["messages"] (the actual input is
            # the chat messages assembled upstream). Skip.
            continue
        if spec_inputs != recorded_vars:
            extra_in_code = recorded_vars - spec_inputs
            missing_in_code = spec_inputs - recorded_vars
            parts = [f"  {name} (via {s['_endpoint']}):"]
            if extra_in_code:
                parts.append(f"only in trace: {sorted(extra_in_code)}")
            if missing_in_code:
                parts.append(f"only in spec: {sorted(missing_in_code)}")
            bad.append(" ".join(parts))
    bad = sorted(set(bad))
    assert not bad, "\n" + "\n".join(bad)


def test_recorded_parsed_keys_match_spec_outputs(strong_client):
    """For every recorded step, parsed.keys() must equal spec.outputs."""
    gid = _exercise_routes(strong_client)
    steps = _collect_steps(strong_client, gid)
    bad = []
    for s in steps:
        name = s["name"]
        if name not in SPEC_LLM:
            continue
        spec_outputs = set(SPEC_LLM[name].outputs)
        parsed = s.get("parsed")
        if not isinstance(parsed, dict):
            continue  # parsed wasn't a dict literal (e.g. get_new_semantic
                      # passes a returned dict); skip.
        recorded_keys = set(parsed.keys())
        if spec_outputs != recorded_keys:
            parts = [f"  {name} (via {s['_endpoint']}):"]
            extra = recorded_keys - spec_outputs
            missing = spec_outputs - recorded_keys
            if extra:
                parts.append(f"only in trace: {sorted(extra)}")
            if missing:
                parts.append(f"only in spec: {sorted(missing)}")
            bad.append(" ".join(parts))
    bad = sorted(set(bad))
    assert not bad, "\n" + "\n".join(bad)


def test_minimum_coverage_per_phase(strong_client):
    """At least one LLM step from each (non-empty) phase fires under this fixture.

    Sanity-check that the route exerciser is broad enough — if we add a new
    phase and forget to exercise it, this test catches it.
    """
    gid = _exercise_routes(strong_client)
    steps = _collect_steps(strong_client, gid)
    recorded_names = {s["name"] for s in steps}
    phases_with_llm = {s.phase for s in STEPS if s.kind == "llm"}
    # Consolidate is hard to trigger deterministically (depends on
    # similarity thresholds + node tags); exempt it here. The static lint
    # already proves it's in the spec.
    EXEMPT_PHASES = {"consolidate"}
    missing_phases = []
    for phase in phases_with_llm - EXEMPT_PHASES:
        phase_llm_steps = {s.id for s in STEPS if s.kind == "llm" and s.phase == phase}
        if not (phase_llm_steps & recorded_names):
            missing_phases.append((phase, phase_llm_steps))
    assert not missing_phases, (
        f"Phases without any recorded LLM steps: {missing_phases}. "
        f"Either the route exerciser doesn't cover them, or they aren't "
        f"firing as the spec claims."
    )
