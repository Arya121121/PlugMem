"""Ready-to-run sample YAMLs the editor can paste as a starting point.

Each sample is a string that loads cleanly through
``plugmem.pipelines.spec_driven.load_yaml_str``. Useful for the "Insert
template" button so researchers don't have to copy YAML out of docs.
"""
from __future__ import annotations


# ----------------------------------------------------------------------- #
# plugmem-default — retrieve phase, expressed in the spec-driven YAML
# ----------------------------------------------------------------------- #
#
# Mirrors what ``PlugMemDefaultPipeline.retrieve`` does *minus* the storage
# fetches (the MVP YAML grammar has no StorageRead / Embed nodes yet —
# those land in Phase 6.6). The memory-string ports are filled with
# placeholder constants so the rendered prompt makes it clear they are
# stubs, not silently empty.
#
# LLM calls used: get_plan (retrieval), get_mode (retrieval). The mode
# picker runs every time even if the request supplied a mode — to skip
# it conditionally you'd wrap it in a Branch on a Compute node.
PLUGMEM_DEFAULT_RETRIEVE_YAML = """\
# plugmem-default retrieve, expressed as a spec-driven pipeline.
#
# This mirrors PlugMemDefaultPipeline.retrieve except for the storage
# fetches that turn next_subgoal + query_tags into actual memory
# strings. The MVP YAML grammar has no StorageRead / Embed nodes; those
# ports hold placeholder constants so the rendered prompt is honest.
#
# Edit any prompt name, role, or input and Save — the new version is
# kept in history; rollback is one click away.
phase: retrieve

nodes:
  - { id: in, type: Input }

  # Step 1: plan recall — pick query tags + the next subgoal to query.
  - id: plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
    inputs:
      goal: in.goal
      subgoal: in.subgoal
      state: in.state
      observation: in.observation

  # Step 2: pick the memory mode (semantic / procedural / episodic).
  - id: mode_pick
    type: LLMCall
    config: { prompt: get_mode, role: retrieval }
    inputs:
      observation: in.observation
      task_type: in.task_type

  # Step 3: render the reasoning prompt. MVP placeholder memories.
  - id: msgs
    type: PromptRender
    config: { prompt: reasoning_semantic }
    inputs:
      goal: in.goal
      subgoal: plan.parsed.next_subgoal
      state: in.state
      observation: in.observation
      semantic_memory: { const: "(MVP placeholder — StorageRead lands in Phase 6.6)" }
      procedural_memory: { const: "" }
      episodic_memory: { const: "" }
      time: { const: "" }
      information: { const: "" }
      question: in.observation

  # Phase exit: ship mode + messages + a small variables trace.
  - id: out
    type: Output
    inputs:
      mode: mode_pick.parsed.mode
      reasoning_prompt: msgs.messages
      variables: { const: { source: "spec-driven plugmem-default sample" } }
"""


SAMPLES = {
    "plugmem-default": {
        "label": "plugmem-default retrieve",
        "description": (
            "Mirrors PlugMemDefaultPipeline.retrieve: get_plan → get_mode → "
            "render reasoning_semantic. Storage reads are stubbed with "
            "placeholder constants (StorageRead nodes land in Phase 6.6)."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}
