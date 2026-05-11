"""Snapshot LLM-step data straight from the inference layer.

Walks ``plugmem/inference/*.py`` with ``ast`` and pulls the
``prompt_name`` / ``inputs`` / ``outputs`` for every ``record_llm_step``
call site. That snapshot can be:

- printed for review when adding a new step (``--list``),
- diffed against the committed ``pipeline_spec.STEPS`` (``--check``),
- imported programmatically by tests (the
  :func:`compare_with_spec` function).

Phase / role / description / per / kind / edges are *not* extracted —
those are human-curated metadata that AST analysis can't recover. The
goal here is to make the data-flow leaves (prompt name, inputs,
outputs) impossible to drift, not to autogenerate the whole spec.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from plugmem.core.pipeline_spec import STEPS

REPO = Path(__file__).resolve().parents[2]
INFERENCE_FILES = [
    REPO / "plugmem" / "inference" / "structuring.py",
    REPO / "plugmem" / "inference" / "retrieving.py",
]

# LLM steps whose record_llm_step call deliberately lives outside the
# inference layer — they don't have a prompt template, so the snapshot
# wouldn't see them and the comparison would false-positive.
SNAPSHOT_EXEMPT = {"reason_llm_call"}


# ------------------------------------------------------------------ #
# AST extraction
# ------------------------------------------------------------------ #


@dataclass
class StepSnapshot:
    """Per-call-site view of an LLM step, as the code declares it."""
    id: str
    prompt_name: Optional[str]
    inputs: List[str]
    outputs: List[str]
    parsed_unknown: bool = False  # True if `parsed=` wasn't a dict literal
    file: str = ""
    lineno: int = 0


def _enclosing_func(tree: ast.AST, target: ast.AST) -> Optional[ast.FunctionDef]:
    parents: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur: ast.AST = target
    while id(cur) in parents:
        cur = parents[id(cur)]
        if isinstance(cur, ast.FunctionDef):
            return cur
    return None


def _dict_keys(node: ast.AST) -> Optional[List[str]]:
    if not isinstance(node, ast.Dict):
        return None
    out: List[str] = []
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            out.append(k.value)
    return out


def extract_llm_steps(files: Optional[List[Path]] = None) -> Dict[str, StepSnapshot]:
    """Return ``{step_id: StepSnapshot}`` for every ``record_llm_step`` call site.

    If a single step name is recorded from multiple call sites (unlikely
    given the current design), the *last* one wins. Caller can compare
    against the spec via :func:`compare_with_spec`.
    """
    files = files or INFERENCE_FILES
    out: Dict[str, StepSnapshot] = {}
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "record_llm_step"
            ):
                continue
            name: Optional[str] = None
            parsed_keys: Optional[List[str]] = None
            parsed_unknown = False
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = kw.value.value
                elif kw.arg == "parsed":
                    if isinstance(kw.value, ast.Dict):
                        parsed_keys = _dict_keys(kw.value) or []
                    else:
                        parsed_unknown = True
            if name is None:
                continue
            func = _enclosing_func(tree, node)
            if func is None:
                continue
            variables_keys: List[str] = []
            resolve_arg: Optional[str] = None
            for stmt in ast.walk(func):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "variables"
                ):
                    keys = _dict_keys(stmt.value)
                    if keys is not None:
                        variables_keys = keys
                        break
            for stmt in ast.walk(func):
                if (
                    isinstance(stmt, ast.Call)
                    and isinstance(stmt.func, ast.Name)
                    and stmt.func.id == "_resolve"
                    and stmt.args
                    and isinstance(stmt.args[0], ast.Constant)
                ):
                    resolve_arg = stmt.args[0].value
                    break
            out[name] = StepSnapshot(
                id=name,
                prompt_name=resolve_arg,
                inputs=variables_keys,
                outputs=parsed_keys or [],
                parsed_unknown=parsed_unknown,
                file=path.name,
                lineno=node.lineno,
            )
    return out


# ------------------------------------------------------------------ #
# Spec comparison
# ------------------------------------------------------------------ #


@dataclass
class Mismatch:
    """One difference between a code snapshot and the committed spec."""
    step_id: str
    field: str           # "prompt_name" | "inputs" | "outputs" | "presence"
    in_code: Any
    in_spec: Any
    detail: str = ""


def compare_with_spec(
    snapshot: Optional[Dict[str, StepSnapshot]] = None,
) -> List[Mismatch]:
    """Diff the AST snapshot against the spec's kind=llm entries.

    Returns an empty list iff the code and spec agree. ``parsed_unknown``
    sites are skipped for the outputs check (e.g. ``parsed=<variable>``
    in get_new_semantic — no AST-visible key set).
    """
    snap = snapshot or extract_llm_steps()
    spec_by_id = {s.id: s for s in STEPS if s.kind == "llm"}

    out: List[Mismatch] = []

    # Spec entries the code doesn't claim to record.
    for sid, step in spec_by_id.items():
        if sid in SNAPSHOT_EXEMPT:
            continue
        if sid not in snap:
            out.append(Mismatch(
                step_id=sid, field="presence",
                in_code=None, in_spec="llm step",
                detail="spec lists this step but no record_llm_step call was found",
            ))

    for sid, snap_step in snap.items():
        if sid not in spec_by_id:
            out.append(Mismatch(
                step_id=sid, field="presence",
                in_code=f"{snap_step.file}:{snap_step.lineno}",
                in_spec=None,
                detail="record_llm_step call has no matching kind=llm spec entry",
            ))
            continue
        spec_step = spec_by_id[sid]

        if snap_step.prompt_name is not None and snap_step.prompt_name != spec_step.prompt_name:
            out.append(Mismatch(
                step_id=sid, field="prompt_name",
                in_code=snap_step.prompt_name, in_spec=spec_step.prompt_name,
                detail="_resolve(...) literal differs from spec.prompt_name",
            ))

        if set(snap_step.inputs) != set(spec_step.inputs):
            out.append(Mismatch(
                step_id=sid, field="inputs",
                in_code=sorted(snap_step.inputs), in_spec=sorted(spec_step.inputs),
                detail="variables.keys() differs from spec.inputs",
            ))

        if not snap_step.parsed_unknown and set(snap_step.outputs) != set(spec_step.outputs):
            out.append(Mismatch(
                step_id=sid, field="outputs",
                in_code=sorted(snap_step.outputs), in_spec=sorted(spec_step.outputs),
                detail="parsed={...}.keys() differs from spec.outputs",
            ))

    return out


# ------------------------------------------------------------------ #
# Formatting / CLI
# ------------------------------------------------------------------ #


def format_snapshot(snapshot: Dict[str, StepSnapshot]) -> str:
    """Pretty-print the snapshot for review."""
    lines = ["# Snapshot of LLM steps as the code declares them.",
             "# Use this when adding / refactoring a step to align pipeline_spec.py.",
             ""]
    for sid in sorted(snapshot):
        s = snapshot[sid]
        parsed_note = " (parsed not a dict literal)" if s.parsed_unknown else ""
        lines.append(f"## {sid}  ({s.file}:{s.lineno})")
        lines.append(f"  prompt_name: {s.prompt_name!r}")
        lines.append(f"  inputs:      {s.inputs}")
        lines.append(f"  outputs:     {s.outputs}{parsed_note}")
        lines.append("")
    return "\n".join(lines)


def format_mismatches(mismatches: List[Mismatch]) -> str:
    if not mismatches:
        return "spec is in sync with code (no mismatches)."
    lines = [f"{len(mismatches)} mismatch(es) between code and pipeline_spec.STEPS:\n"]
    for m in mismatches:
        lines.append(f"  • {m.step_id} / {m.field}: {m.detail}")
        lines.append(f"      code: {m.in_code}")
        lines.append(f"      spec: {m.in_spec}")
    lines.append("")
    lines.append(
        "Fix: edit plugmem/core/pipeline_spec.py to match the code, or update the "
        "inference call site if the spec is what you intended."
    )
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot LLM-step data from the inference layer.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the snapshot disagrees with pipeline_spec.STEPS.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the snapshot as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    snap = extract_llm_steps()

    if args.check:
        mismatches = compare_with_spec(snap)
        print(format_mismatches(mismatches))
        return 1 if mismatches else 0

    if args.json:
        payload = {
            sid: {
                "prompt_name": s.prompt_name,
                "inputs": s.inputs,
                "outputs": s.outputs,
                "parsed_unknown": s.parsed_unknown,
                "file": s.file,
                "lineno": s.lineno,
            }
            for sid, s in snap.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(format_snapshot(snap))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
