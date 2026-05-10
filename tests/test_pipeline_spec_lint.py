"""Static lint: pipeline_spec.STEPS must accurately describe inference/*.py.

Walks the inference layer with ``ast`` and asserts that every LLM call
recorded at runtime maps cleanly to a spec entry. Catches:

- Recorded step names that don't exist in the spec.
- Spec entries with kind=llm that aren't recorded anywhere.
- Drift between the call site's ``_resolve("X", ...)`` and the spec's
  ``prompt_name``.
- Drift between the call site's ``variables = {...}`` keys and the spec's
  ``inputs``.

If any of these fail, the visualization is no longer a faithful map of
what the code actually does — fix the spec or fix the code.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from plugmem.core.pipeline_spec import STEPS

REPO = Path(__file__).resolve().parent.parent
INFERENCE_FILES = [
    REPO / "plugmem" / "inference" / "structuring.py",
    REPO / "plugmem" / "inference" / "retrieving.py",
]

# LLM steps recorded outside inference/ (e.g. the reasoning LLM call lives in
# the /reason route + retrieve_and_reason). These get exempted from the
# "every spec step is recorded in inference" check; the runtime trace test
# is responsible for verifying they actually fire.
LLM_STEPS_OUTSIDE_INFERENCE = {"reason_llm_call"}


# ------------------------------------------------------------------ #
# AST helpers
# ------------------------------------------------------------------ #


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


def _find_record_calls(path: Path) -> List[Dict[str, Any]]:
    tree = ast.parse(path.read_text())
    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "record_llm_step"
        ):
            continue
        name = None
        for kw in node.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name = kw.value.value
        if name is None:
            continue
        func = _enclosing_func(tree, node)
        if func is None:
            continue

        # `variables = {...}` literal in same func
        variables_keys: Optional[List[str]] = None
        for stmt in ast.walk(func):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "variables"
                and isinstance(stmt.value, ast.Dict)
            ):
                continue
            keys: List[str] = []
            for k in stmt.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
            variables_keys = keys
            break

        # `_resolve("X", ...)` first-arg literal in same func
        resolve_arg: Optional[str] = None
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

        # `parsed=` kwarg in the record call
        parsed_keys: Optional[List[str]] = None
        for kw in node.keywords:
            if kw.arg == "parsed" and isinstance(kw.value, ast.Dict):
                keys = []
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.append(k.value)
                parsed_keys = keys
                break

        out.append({
            "name": name,
            "func_name": func.name,
            "variables_keys": variables_keys,
            "resolve_arg": resolve_arg,
            "parsed_keys": parsed_keys,
            "file": path.name,
            "lineno": node.lineno,
        })
    return out


def _all_calls() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in INFERENCE_FILES:
        out.extend(_find_record_calls(f))
    return out


SPEC_LLM = {s.id: s for s in STEPS if s.kind == "llm"}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


def test_every_recorded_call_has_spec_entry():
    """record_llm_step(name=X) → X exists in spec as kind=llm."""
    bad = []
    for c in _all_calls():
        if c["name"] not in SPEC_LLM:
            bad.append(
                f"  {c['file']}:{c['lineno']} record_llm_step(name='{c['name']}') "
                f"has no kind=llm step in pipeline_spec.STEPS"
            )
    assert not bad, "\n" + "\n".join(bad)


def test_every_spec_llm_step_is_recorded():
    """Every kind=llm spec step has at least one record_llm_step in inference/."""
    recorded = {c["name"] for c in _all_calls()}
    missing = set(SPEC_LLM) - recorded - LLM_STEPS_OUTSIDE_INFERENCE
    assert not missing, (
        f"Spec lists kind=llm steps not recorded in inference/: {sorted(missing)}. "
        f"Either add a record_llm_step call or move them to "
        f"LLM_STEPS_OUTSIDE_INFERENCE in this test."
    )


def test_prompt_names_align():
    """The literal _resolve('X', ...) matches the spec.prompt_name for that step."""
    bad = []
    for c in _all_calls():
        name = c["name"]
        if name not in SPEC_LLM:
            continue
        spec_prompt = SPEC_LLM[name].prompt_name
        if c["resolve_arg"] is None:
            # No _resolve in this function — likely uses prompts.get directly.
            # Skip rather than false-positive.
            continue
        if c["resolve_arg"] != spec_prompt:
            bad.append(
                f"  {c['file']}:{c['lineno']} {name}: "
                f"code uses _resolve('{c['resolve_arg']}', ...) but "
                f"spec.prompt_name='{spec_prompt}'"
            )
    assert not bad, "\n" + "\n".join(bad)


def test_variables_match_spec_inputs():
    """variables.keys() in the function equals the spec.inputs set."""
    bad = []
    for c in _all_calls():
        name = c["name"]
        if name not in SPEC_LLM:
            continue
        if c["variables_keys"] is None:
            continue  # no variables literal found — skip
        spec_inputs = set(SPEC_LLM[name].inputs)
        recorded_vars = set(c["variables_keys"])
        if spec_inputs != recorded_vars:
            extra_in_code = recorded_vars - spec_inputs
            missing_in_code = spec_inputs - recorded_vars
            parts = [f"  {c['file']}:{c['lineno']} {name}:"]
            if extra_in_code:
                parts.append(f"only in code: {sorted(extra_in_code)}")
            if missing_in_code:
                parts.append(f"only in spec: {sorted(missing_in_code)}")
            bad.append(" ".join(parts))
    assert not bad, "\n" + "\n".join(bad)


def test_parsed_keys_match_spec_outputs():
    """parsed={...}.keys() in record_llm_step equals the spec.outputs set."""
    bad = []
    for c in _all_calls():
        name = c["name"]
        if name not in SPEC_LLM:
            continue
        if c["parsed_keys"] is None:
            continue
        spec_outputs = set(SPEC_LLM[name].outputs)
        recorded_parsed = set(c["parsed_keys"])
        if spec_outputs != recorded_parsed:
            extra_in_code = recorded_parsed - spec_outputs
            missing_in_code = spec_outputs - recorded_parsed
            parts = [f"  {c['file']}:{c['lineno']} {name}:"]
            if extra_in_code:
                parts.append(f"only in code: {sorted(extra_in_code)}")
            if missing_in_code:
                parts.append(f"only in spec: {sorted(missing_in_code)}")
            bad.append(" ".join(parts))
    assert not bad, "\n" + "\n".join(bad)
