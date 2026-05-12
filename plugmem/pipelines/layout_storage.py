"""Per-graph canvas layout sidecar.

Stores hand-positioned node coordinates so the pipeline canvas can
restore them on next mount, overriding dagre's auto-layout. The YAML
spec stays purely semantic — positions live in a separate JSON file:

    {PROMPTS_DIR}/.pipeline_layout/{graph_id}.json

Schema:

    {
      "positions": { "<step_id>": { "x": <float>, "y": <float> }, ... },
      "viewport":  { "x": <float>, "y": <float>, "zoom": <float> }
    }

``viewport`` is optional. Unknown keys are preserved on save so we can
extend the schema later without breaking already-saved files.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CanvasLayout:
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    viewport: Optional[Dict[str, float]] = None


def _prompts_dir() -> Path:
    return Path(os.getenv("PROMPTS_DIR", "./data/prompts"))


def layout_path(graph_id: str) -> Path:
    return _prompts_dir() / ".pipeline_layout" / f"{graph_id}.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def load_layout(graph_id: str) -> CanvasLayout:
    """Return the saved layout, or an empty layout if none exists."""
    path = layout_path(graph_id)
    if not path.exists():
        return CanvasLayout()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return CanvasLayout()
    positions = raw.get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}
    # Defensive: drop entries that aren't {x: float, y: float}.
    clean: Dict[str, Dict[str, float]] = {}
    for nid, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        x = pos.get("x")
        y = pos.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            clean[str(nid)] = {"x": float(x), "y": float(y)}
    viewport = raw.get("viewport")
    if not isinstance(viewport, dict):
        viewport = None
    return CanvasLayout(positions=clean, viewport=viewport)


def save_layout(
    graph_id: str,
    positions: Dict[str, Dict[str, float]],
    *,
    viewport: Optional[Dict[str, float]] = None,
) -> CanvasLayout:
    """Write the layout atomically. Returns the persisted result."""
    if not isinstance(positions, dict):
        raise ValueError("positions must be a dict mapping node id → {x, y}")
    clean: Dict[str, Dict[str, float]] = {}
    for nid, pos in positions.items():
        if not isinstance(pos, dict):
            raise ValueError(f"positions[{nid!r}] must be a dict, got {type(pos).__name__}")
        x = pos.get("x")
        y = pos.get("y")
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise ValueError(f"positions[{nid!r}] needs numeric x and y, got {pos!r}")
        clean[str(nid)] = {"x": float(x), "y": float(y)}
    clean_viewport: Optional[Dict[str, float]] = None
    if viewport is not None:
        if not isinstance(viewport, dict):
            raise ValueError("viewport must be a dict")
        clean_viewport = {
            k: float(v) for k, v in viewport.items()
            if isinstance(v, (int, float))
        }
    payload: Dict[str, Any] = {"positions": clean}
    if clean_viewport is not None:
        payload["viewport"] = clean_viewport
    _atomic_write(layout_path(graph_id), json.dumps(payload, indent=2))
    return CanvasLayout(positions=clean, viewport=clean_viewport)


def clear_layout(graph_id: str) -> bool:
    """Remove the layout file. Returns True if a file was removed."""
    path = layout_path(graph_id)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
