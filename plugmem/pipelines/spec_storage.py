"""Versioned, non-destructive storage for per-graph spec-driven YAML.

Layout under ``{PROMPTS_DIR}``:

    {graph_id}.pipeline.yaml             # the "live" file the executor reads
    .pipeline_history/{graph_id}/
        current.txt                       # one line: active version_id
        {version_id}.pipeline.yaml        # full source verbatim
        {version_id}.meta.json            # {version_id, ts, parent, note}

Every save (a) validates the YAML via ``load_yaml``, (b) writes a new
version file + meta file, (c) atomically points ``current.txt`` at it,
(d) rewrites the live file to match. Rollback reuses the same path: it
copies the historical content back to the live file and bumps
``current.txt`` to the chosen version id.

The live file path matches what ``SpecDrivenPipeline._load_for_graph``
already reads, so the executor needs no changes.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


VERSION_ID_RE = re.compile(r"^v_[0-9]{8}T[0-9]{6}_[0-9]{6}_[0-9a-f]{6}$")


@dataclass
class SpecVersion:
    version_id: str
    ts: str                      # ISO-8601 UTC
    parent_version_id: Optional[str]
    note: str
    active: bool


def _prompts_dir() -> Path:
    return Path(os.getenv("PROMPTS_DIR", "./data/prompts"))


def live_path(graph_id: str) -> Path:
    return _prompts_dir() / f"{graph_id}.pipeline.yaml"


def history_dir(graph_id: str) -> Path:
    return _prompts_dir() / ".pipeline_history" / graph_id


def _new_version_id() -> str:
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
    micro = f"{int((now - int(now)) * 1_000_000):06d}"
    suffix = uuid.uuid4().hex[:6]
    return f"v_{ts}_{micro}_{suffix}"


def _meta_path(graph_id: str, version_id: str) -> Path:
    return history_dir(graph_id) / f"{version_id}.meta.json"


def _content_path(graph_id: str, version_id: str) -> Path:
    return history_dir(graph_id) / f"{version_id}.pipeline.yaml"


def _current_pointer(graph_id: str) -> Path:
    return history_dir(graph_id) / "current.txt"


def _read_current(graph_id: str) -> Optional[str]:
    p = _current_pointer(graph_id)
    if not p.exists():
        return None
    val = p.read_text().strip()
    return val or None


def _write_current(graph_id: str, version_id: str) -> None:
    p = _current_pointer(graph_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".txt.tmp")
    tmp.write_text(version_id + "\n")
    os.replace(tmp, p)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _read_meta(graph_id: str, version_id: str) -> dict:
    p = _meta_path(graph_id, version_id)
    if not p.exists():
        return {
            "version_id": version_id,
            "ts": "",
            "parent_version_id": None,
            "note": "",
        }
    return json.loads(p.read_text())


def _write_meta(graph_id: str, version_id: str, *, parent: Optional[str], note: str) -> dict:
    meta = {
        "version_id": version_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_version_id": parent,
        "note": note or "",
    }
    _atomic_write(_meta_path(graph_id, version_id), json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------- #


def adopt_existing_live(graph_id: str) -> Optional[str]:
    """If a live YAML exists but no history, seed history with version 0.

    Returns the seeded version_id, or ``None`` if there was nothing to adopt
    or history already exists.
    """
    live = live_path(graph_id)
    if not live.exists():
        return None
    if _read_current(graph_id):
        return None
    content = live.read_text()
    vid = _new_version_id()
    _atomic_write(_content_path(graph_id, vid), content)
    _write_meta(graph_id, vid, parent=None, note="adopted from existing live file")
    _write_current(graph_id, vid)
    return vid


def read_current_content(graph_id: str) -> Optional[str]:
    """Return the YAML source of the active version (or live file if no history)."""
    # Prefer the live file — that's always what the executor uses.
    live = live_path(graph_id)
    if live.exists():
        return live.read_text()
    # Fallback: history-only (live file deleted out of band).
    vid = _read_current(graph_id)
    if vid is None:
        return None
    p = _content_path(graph_id, vid)
    if not p.exists():
        return None
    return p.read_text()


def get_active_version_id(graph_id: str) -> Optional[str]:
    """Return the active version_id, adopting any stray live file."""
    # If there's a live file but no recorded history, seed it now.
    if live_path(graph_id).exists() and _read_current(graph_id) is None:
        return adopt_existing_live(graph_id)
    return _read_current(graph_id)


def list_versions(graph_id: str) -> List[SpecVersion]:
    """Newest first. Includes the adopted-on-read version if applicable."""
    # Make sure an existing-but-untracked live file gets recorded.
    if live_path(graph_id).exists():
        adopt_existing_live(graph_id)

    hdir = history_dir(graph_id)
    if not hdir.exists():
        return []
    active = _read_current(graph_id)
    out: List[SpecVersion] = []
    for meta_file in hdir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        vid = meta.get("version_id") or meta_file.stem.replace(".meta", "")
        out.append(SpecVersion(
            version_id=vid,
            ts=meta.get("ts", ""),
            parent_version_id=meta.get("parent_version_id"),
            note=meta.get("note", ""),
            active=(vid == active),
        ))
    # Sort newest first. version_id includes the microsecond timestamp +
    # a random hex, which is both monotonic and stable across same-second
    # saves; ts (ISO-8601, second precision) ties when saves are < 1s apart.
    out.sort(key=lambda v: v.version_id, reverse=True)
    return out


def read_version(graph_id: str, version_id: str) -> Optional[str]:
    """Return historical YAML content for ``version_id``, or ``None`` if missing."""
    if not VERSION_ID_RE.match(version_id):
        return None
    p = _content_path(graph_id, version_id)
    if not p.exists():
        return None
    return p.read_text()


def save_new_version(
    graph_id: str,
    content: str,
    *,
    note: str = "",
    validator=None,
) -> SpecVersion:
    """Validate, write a new version, promote it to active, update live file.

    ``validator`` is an optional callable invoked as ``validator(content)``
    that raises ``ValueError`` on invalid content. Routes pass
    :func:`plugmem.pipelines.spec_driven.validate_yaml_str` so the bad-YAML
    error surfaces as a 422 before disk is touched.
    """
    if validator is not None:
        validator(content)  # raises ValueError on bad spec

    # If there's a live file with no history, seed it first so we don't
    # lose the prior content silently.
    parent = get_active_version_id(graph_id)

    vid = _new_version_id()
    _atomic_write(_content_path(graph_id, vid), content)
    meta = _write_meta(graph_id, vid, parent=parent, note=note)
    _write_current(graph_id, vid)
    _atomic_write(live_path(graph_id), content)

    return SpecVersion(
        version_id=vid,
        ts=meta["ts"],
        parent_version_id=parent,
        note=meta["note"],
        active=True,
    )


def rollback(graph_id: str, version_id: str) -> SpecVersion:
    """Make ``version_id`` the active version. Live file is rewritten to match.

    Does NOT create a new version — the prior promotion is recoverable by
    rolling back again. Raises ``ValueError`` if the version is missing.
    """
    content = read_version(graph_id, version_id)
    if content is None:
        raise ValueError(f"Version {version_id!r} not found for graph {graph_id!r}")
    _write_current(graph_id, version_id)
    _atomic_write(live_path(graph_id), content)
    meta = _read_meta(graph_id, version_id)
    return SpecVersion(
        version_id=version_id,
        ts=meta.get("ts", ""),
        parent_version_id=meta.get("parent_version_id"),
        note=meta.get("note", ""),
        active=True,
    )


def as_dicts(versions: List[SpecVersion]) -> List[dict]:
    return [asdict(v) for v in versions]
