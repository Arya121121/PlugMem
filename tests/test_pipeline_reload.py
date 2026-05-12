"""Tests for the pipeline hot-reload endpoint (Phase 6.5c)."""
from __future__ import annotations


def test_reload_blocked_when_env_flag_unset(client, monkeypatch):
    """The hot-reload endpoint is opt-in via env var. Default: 403."""
    monkeypatch.delenv("PLUGMEM_ENABLE_PIPELINE_RELOAD", raising=False)
    r = client.post("/api/v1/pipeline/pipelines/reload")
    assert r.status_code == 403
    assert "PLUGMEM_ENABLE_PIPELINE_RELOAD" in r.json()["detail"]


def test_reload_enabled_reimports_and_repopulates_registry(client, monkeypatch):
    """When enabled, the endpoint reloads modules and the registry still
    holds the three built-ins after the clear-and-repopulate cycle."""
    monkeypatch.setenv("PLUGMEM_ENABLE_PIPELINE_RELOAD", "1")
    r = client.post("/api/v1/pipeline/pipelines/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    # Every built-in module was reloaded.
    assert "plugmem.pipelines" in body["reloaded_modules"]
    assert "plugmem.pipelines.spec_driven" in body["reloaded_modules"]
    # Registry survived the clear-and-rebuild.
    assert set(body["registered"]) >= {"plugmem-default", "naive-rag", "spec-driven"}


def test_reload_module_order_is_children_then_package(client, monkeypatch):
    """Children must be reloaded before the package __init__ so its
    register() calls pick up the fresh class definitions."""
    monkeypatch.setenv("PLUGMEM_ENABLE_PIPELINE_RELOAD", "1")
    r = client.post("/api/v1/pipeline/pipelines/reload")
    mods = r.json()["reloaded_modules"]
    # The package itself must come after every child module.
    pkg_idx = mods.index("plugmem.pipelines")
    for m in mods:
        if m == "plugmem.pipelines" or not m.startswith("plugmem.pipelines."):
            continue
        assert mods.index(m) < pkg_idx, (
            f"Child module {m!r} must be reloaded BEFORE the package, "
            f"got order: {mods}"
        )


def test_reload_idempotent(client, monkeypatch):
    """Calling reload twice in a row leaves the registry in the same state."""
    monkeypatch.setenv("PLUGMEM_ENABLE_PIPELINE_RELOAD", "1")
    r1 = client.post("/api/v1/pipeline/pipelines/reload").json()
    r2 = client.post("/api/v1/pipeline/pipelines/reload").json()
    assert set(r1["registered"]) == set(r2["registered"])


def test_pipelines_still_dispatch_after_reload(client, monkeypatch, tmp_path):
    """Smoke: after a reload, the registry's pipelines still respond to
    /retrieve (i.e. the routes can resolve them via get())."""
    monkeypatch.setenv("PLUGMEM_ENABLE_PIPELINE_RELOAD", "1")
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    # Pre-reload: create a graph + bind to naive-rag.
    r = client.post("/api/v1/graphs", json={"graph_id": "g-reload"})
    assert r.status_code in (200, 201)
    rb = client.put(
        "/api/v1/graphs/g-reload/pipeline",
        json={"pipeline": "naive-rag"},
    )
    assert rb.status_code == 200

    # Reload.
    rr = client.post("/api/v1/pipeline/pipelines/reload")
    assert rr.status_code == 200

    # Post-reload: /retrieve on the same graph still works.
    rget = client.post(
        "/api/v1/graphs/g-reload/retrieve",
        json={"observation": "what?"},
    )
    assert rget.status_code == 200, rget.text
