"""Regression guard: the AST-extracted snapshot must agree with the spec.

Overlaps with ``test_pipeline_spec_lint.py`` (which catches the same
drift from a different angle) but lives here so:

  - The :mod:`plugmem.tools.spec_snapshot` module is exercised on every
    test run.
  - Running ``python -m plugmem.tools.spec_snapshot --check`` matches
    what CI sees.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from plugmem.tools.spec_snapshot import (
    compare_with_spec,
    extract_llm_steps,
    format_mismatches,
)

REPO = Path(__file__).resolve().parent.parent


def test_snapshot_matches_spec():
    """The committed pipeline_spec.py must match what the AST snapshot reports."""
    mismatches = compare_with_spec()
    assert not mismatches, "\n" + format_mismatches(mismatches)


def test_extracted_steps_cover_every_known_llm_call():
    """Every LLM call in the inference layer is picked up by the extractor.

    Sanity: there should be at least one snapshot entry per kind=llm step
    in inference/ (consolidate's get_new_semantic + every structuring +
    every retrieving prompt).
    """
    snap = extract_llm_steps()
    expected_inside_inference = {
        "get_subgoal", "get_reward", "get_state", "get_semantic",
        "get_procedural", "get_return",
        "get_plan", "get_mode", "get_new_subgoal", "get_new_semantic",
    }
    missing = expected_inside_inference - set(snap)
    assert not missing, f"snapshot didn't pick up: {sorted(missing)}"


def test_cli_check_exit_code_is_clean():
    """`python -m plugmem.tools.spec_snapshot --check` exits 0 when spec is in sync."""
    result = subprocess.run(
        [sys.executable, "-m", "plugmem.tools.spec_snapshot", "--check"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
