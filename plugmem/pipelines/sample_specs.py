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
# Top-level node IDs intentionally mirror the IDs used by
# ``plugmem.core.pipeline_spec.STEPS`` for the retrieve phase, so the
# spec-driven canvas reads as a near-1-to-1 copy of plugmem-default:
#
#   get_plan                           (LLMCall)
#   get_mode                           (LLMCall)
#   retrieve_semantic_nodes            (Compute placeholder)
#   retrieve_procedural_nodes          (Compute placeholder)
#   retrieve_episodic_nodes            (Compute placeholder)
#   render_reasoning_semantic          (Branch wrapping PromptRender)
#   render_reasoning_procedural        (Branch wrapping PromptRender)
#   render_reasoning_episodic          (Branch wrapping PromptRender)
#
# The MVP YAML grammar has no ``StorageRead`` / ``Embed`` node; the
# ``retrieve_*_nodes`` steps are Compute(concat) stubs that surface
# what data WOULD be fetched in production (query_tags + observation,
# next_subgoal, observation). Real fetches land in Phase 6.6.
PLUGMEM_DEFAULT_RETRIEVE_YAML = """\
# plugmem-default retrieve, expressed as a spec-driven pipeline.
#
# Mirrors MemoryGraph.retrieve_memory() — same IDs as
# plugmem.core.pipeline_spec.STEPS for the retrieve phase.
#
# Flow:
#   get_plan + get_mode (both always run, in parallel by data deps)
#     → retrieve_semantic_nodes  (uses plan.parsed.query_tags + observation)
#     → retrieve_procedural_nodes (uses plan.parsed.next_subgoal)
#     → retrieve_episodic_nodes  (uses observation)
#   → ONE of three reasoning_* templates renders, gated by get_mode.parsed.mode
#   → Output ships {mode, reasoning_prompt, variables}
#
# Storage fetches are Compute(concat) stubs in this MVP — the YAML
# grammar has no StorageRead/Embed nodes yet (Phase 6.6).
phase: retrieve

nodes:
  - { id: in, type: Input }

  # --- get_plan (LLM, always runs) ---
  - id: get_plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
    inputs:
      goal: in.goal
      subgoal: in.subgoal
      state: in.state
      observation: in.observation

  # --- get_mode (LLM, always runs in this MVP version) ---
  - id: get_mode
    type: LLMCall
    config: { prompt: get_mode, role: retrieval }
    inputs:
      observation: in.observation
      task_type: in.task_type

  # --- retrieve_*_nodes — STUBBED via Compute(concat). In real plugmem
  #     these hit ChromaDB through retrieve_semantic_nodes /
  #     retrieve_procedural_nodes / retrieve_episodic_nodes. The MVP
  #     grammar has no StorageRead/Embed node; output is a plausible
  #     string so the rendered prompt is honest. ---
  - id: retrieve_semantic_nodes
    type: Compute
    config: { op: concat }
    inputs:
      a: { const: "Fact 0 (stub — would be retrieved via plan.parsed.query_tags + " }
      b: in.observation

  - id: retrieve_procedural_nodes
    type: Compute
    config: { op: concat }
    inputs:
      a: { const: "Experience 0 (stub — would be retrieved via plan.parsed.next_subgoal=" }
      b: get_plan.parsed.next_subgoal

  - id: retrieve_episodic_nodes
    type: Compute
    config: { op: concat }
    inputs:
      a: { const: "Episode 0 (stub — would be retrieved via observation=" }
      b: in.observation

  # --- 3-way switch by mode: Compute(eq) gate + Branch for each template.
  #     Each Branch's else_value is an empty list, so only the active
  #     branch contributes to the final messages list. ---
  - id: is_semantic
    type: Compute
    config: { op: eq }
    inputs:
      a: get_mode.parsed.mode
      b: { const: semantic_memory }

  - id: is_procedural
    type: Compute
    config: { op: eq }
    inputs:
      a: get_mode.parsed.mode
      b: { const: procedural_memory }

  - id: is_episodic
    type: Compute
    config: { op: eq }
    inputs:
      a: get_mode.parsed.mode
      b: { const: episodic_memory }

  - id: render_reasoning_semantic
    type: Branch
    inputs: { condition: is_semantic.value }
    config:
      outputs: { messages: render.messages }
      else_value: { const: [] }
    body:
      - id: render
        type: PromptRender
        config: { prompt: reasoning_semantic }
        inputs:
          semantic_memory: retrieve_semantic_nodes.value
          time: in.time
          observation: in.observation

  - id: render_reasoning_procedural
    type: Branch
    inputs: { condition: is_procedural.value }
    config:
      outputs: { messages: render.messages }
      else_value: { const: [] }
    body:
      - id: render
        type: PromptRender
        config: { prompt: reasoning_procedural }
        inputs:
          observation: in.observation
          procedural_memory: retrieve_procedural_nodes.value

  - id: render_reasoning_episodic
    type: Branch
    inputs: { condition: is_episodic.value }
    config:
      outputs: { messages: render.messages }
      else_value: { const: [] }
    body:
      - id: render
        type: PromptRender
        config: { prompt: reasoning_episodic }
        inputs:
          information: retrieve_episodic_nodes.value
          time: in.time
          question: in.observation

  # Collapse the three optional branches into one messages list
  # (two will be empty, one will hold the rendered template's messages).
  - id: messages_sem_proc
    type: Compute
    config: { op: concat }
    inputs:
      a: render_reasoning_semantic.messages
      b: render_reasoning_procedural.messages

  - id: messages_all
    type: Compute
    config: { op: concat }
    inputs:
      a: messages_sem_proc.value
      b: render_reasoning_episodic.messages

  # --- Phase exit ---
  - id: out
    type: Output
    inputs:
      mode: get_mode.parsed.mode
      reasoning_prompt: messages_all.value
      variables:
        const:
          source: "spec-driven plugmem-default sample"
"""


SAMPLES = {
    "plugmem-default": {
        "label": "plugmem-default retrieve",
        "description": (
            "Mirrors PlugMemDefaultPipeline.retrieve, with top-level node "
            "IDs matching plugmem.core.pipeline_spec.STEPS for the retrieve "
            "phase. get_plan + get_mode both run; one of three "
            "reasoning_* templates renders by mode. Memory fetches are "
            "stubbed via Compute(concat) (StorageRead lands in Phase 6.6)."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}
