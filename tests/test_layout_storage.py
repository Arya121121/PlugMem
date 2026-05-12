"""Tests for the canvas layout sidecar (Phase 6.5b)."""
from __future__ import annotations

import json

import pytest

from plugmem.pipelines import layout_storage


# --------------------------------------------------------------------- #
# Storage module
# --------------------------------------------------------------------- #


def test_load_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    layout = layout_storage.load_layout("g1")
    assert layout.positions == {}
    assert layout.viewport is None


def test_save_then_load_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    layout_storage.save_layout(
        "g1",
        {"node_a": {"x": 10, "y": 20}, "node_b": {"x": -5.5, "y": 7.25}},
        viewport={"x": 0, "y": 0, "zoom": 1.5},
    )
    layout = layout_storage.load_layout("g1")
    assert layout.positions["node_a"] == {"x": 10.0, "y": 20.0}
    assert layout.positions["node_b"] == {"x": -5.5, "y": 7.25}
    assert layout.viewport == {"x": 0.0, "y": 0.0, "zoom": 1.5}


def test_save_rejects_non_numeric_coords(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="numeric x and y"):
        layout_storage.save_layout("g1", {"bad": {"x": "ten", "y": 5}})


def test_load_drops_malformed_entries(monkeypatch, tmp_path):
    """A hand-edited JSON with junk values shouldn't crash the loader."""
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    path = layout_storage.layout_path("g1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "positions": {
            "good":  {"x": 1, "y": 2},
            "bad1":  "not-a-dict",
            "bad2":  {"x": "ten", "y": 0},
            "bad3":  {"x": 0},  # missing y
        }
    }))
    layout = layout_storage.load_layout("g1")
    assert set(layout.positions) == {"good"}
    assert layout.positions["good"] == {"x": 1.0, "y": 2.0}


def test_clear_layout_removes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    layout_storage.save_layout("g1", {"a": {"x": 0, "y": 0}})
    assert layout_storage.layout_path("g1").exists()
    assert layout_storage.clear_layout("g1") is True
    assert not layout_storage.layout_path("g1").exists()
    # Idempotent: clearing twice is fine.
    assert layout_storage.clear_layout("g1") is False


# --------------------------------------------------------------------- #
# Route layer
# --------------------------------------------------------------------- #


def _create_graph(client) -> str:
    r = client.post("/api/v1/graphs", json={})
    assert r.status_code == 201, r.text
    return r.json()["graph_id"]


def test_route_get_empty(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.get(f"/api/v1/graphs/{gid}/pipeline/layout")
    assert r.status_code == 200
    body = r.json()
    assert body["positions"] == {}
    assert body["viewport"] is None


def test_route_put_then_get(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    r = client.put(
        f"/api/v1/graphs/{gid}/pipeline/layout",
        json={
            "positions": {
                "in":  {"x": 100, "y": 0},
                "out": {"x": 100, "y": 800},
            },
            "viewport": {"x": 0, "y": 0, "zoom": 0.8},
        },
    )
    assert r.status_code == 200, r.text
    rget = client.get(f"/api/v1/graphs/{gid}/pipeline/layout")
    body = rget.json()
    assert body["positions"]["in"] == {"x": 100.0, "y": 0.0}
    assert body["viewport"]["zoom"] == 0.8


def test_route_delete_clears(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    gid = _create_graph(client)
    client.put(
        f"/api/v1/graphs/{gid}/pipeline/layout",
        json={"positions": {"x": {"x": 0, "y": 0}}},
    )
    r = client.delete(f"/api/v1/graphs/{gid}/pipeline/layout")
    assert r.status_code == 200
    assert r.json()["positions"] == {}
    # And the next GET reflects empty.
    rget = client.get(f"/api/v1/graphs/{gid}/pipeline/layout")
    assert rget.json()["positions"] == {}
