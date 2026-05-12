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
# Mirrors ``MemoryGraph.retrieve_memory`` (core/memory_graph.py:899):
#
#   1. get_plan(LLM) → (next_subgoal, query_tags)
#   2. get_mode(LLM) → mode  (always runs in this sample; in real code
#      it's skipped if the caller already supplied a mode)
#   3. Use plan.query_tags + plan.next_subgoal + observation to fetch
#      semantic / procedural / episodic memory strings (STUBBED — the
#      MVP YAML grammar has no StorageRead / Embed nodes; the strings
#      live as Constants here. Real fetches land in Phase 6.6.)
#   4. Three Branches on mode pick the matching reasoning template; the
#      messages from the chosen branch flow into the response. Branches
#      whose condition is false emit an empty list, so the final
#      concat collapses to whichever single template ran.
PLUGMEM_DEFAULT_RETRIEVE_YAML = """\
# plugmem-default retrieve, expressed as a spec-driven pipeline.
#
# Mirrors MemoryGraph.retrieve_memory():
#   1. get_plan (LLM) — always runs, produces next_subgoal + query_tags
#   2. get_mode (LLM) — always runs, produces mode (semantic/procedural/episodic)
#   3. memory fetch by mode — STUBBED via Constants (the MVP grammar has
#      no StorageRead/Embed nodes; real fetches arrive in Phase 6.6)
#   4. one of three reasoning_* templates is rendered, selected by mode
#
# Edit any prompt / role / input and Save — the new version is kept in
# history; rollback is one click away.
phase: retrieve

nodes:
  - { id: in, type: Input }

  # --- 1. Plan recall (tags + next_subgoal). Always runs. ---
  - id: plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
    inputs:
      goal: in.goal
      subgoal: in.subgoal
      state: in.state
      observation: in.observation

  # --- 2. Mode classifier. Always runs in this sample. ---
  - id: mode_pick
    type: LLMCall
    config: { prompt: get_mode, role: retrieval }
    inputs:
      observation: in.observation
      task_type: in.task_type

  # --- 3. Memory fetch (STUBBED). In real plugmem these are populated
  #       by retrieve_semantic_nodes(plan.query_tags, observation),
  #       retrieve_procedural_nodes(plan.next_subgoal),
  #       retrieve_episodic_nodes(observation). Until StorageRead lands,
  #       the strings are explicit placeholders so the prompt is honest.
  - id: semantic_memory_str
    type: Constant
    config:
      value: "(MVP placeholder — would be facts retrieved via plan.query_tags + observation)"
  - id: procedural_memory_str
    type: Constant
    config:
      value: "(MVP placeholder — would be experiences retrieved via plan.next_subgoal)"
  - id: episodic_memory_str
    type: Constant
    config:
      value: "(MVP placeholder — would be episodic context retrieved via observation)"

  # --- 4. Pick the reasoning template by mode.
  #         Three Compute(eq) flags + three Branches; each Branch outputs
  #         its rendered messages when active and an empty list otherwise.
  #         The final concat picks whichever single one ran. ---
  - id: is_semantic
    type: Compute
    config: { op: eq }
    inputs:
      a: mode_pick.parsed.mode
      b: { const: semantic_memory }

  - id: is_procedural
    type: Compute
    config: { op: eq }
    inputs:
      a: mode_pick.parsed.mode
      b: { const: procedural_memory }

  - id: is_episodic
    type: Compute
    config: { op: eq }
    inputs:
      a: mode_pick.parsed.mode
      b: { const: episodic_memory }

  - id: render_semantic
    type: Branch
    inputs: { condition: is_semantic.value }
    config:
      outputs: { messages: r.messages }
      else_value: { const: [] }
    body:
      - id: r
        type: PromptRender
        config: { prompt: reasoning_semantic }
        inputs:
          semantic_memory: semantic_memory_str.value
          time: in.time
          observation: in.observation

  - id: render_procedural
    type: Branch
    inputs: { condition: is_procedural.value }
    config:
      outputs: { messages: r.messages }
      else_value: { const: [] }
    body:
      - id: r
        type: PromptRender
        config: { prompt: reasoning_procedural }
        inputs:
          observation: in.observation
          procedural_memory: procedural_memory_str.value

  - id: render_episodic
    type: Branch
    inputs: { condition: is_episodic.value }
    config:
      outputs: { messages: r.messages }
      else_value: { const: [] }
    body:
      - id: r
        type: PromptRender
        config: { prompt: reasoning_episodic }
        inputs:
          information: episodic_memory_str.value
          time: in.time
          question: in.observation

  # Collapse the three optional branches into the single messages list
  # that actually fired (the other two are empty lists).
  - id: messages_sem_proc
    type: Compute
    config: { op: concat }
    inputs:
      a: render_semantic.messages
      b: render_procedural.messages

  - id: messages_all
    type: Compute
    config: { op: concat }
    inputs:
      a: messages_sem_proc.value
      b: render_episodic.messages

  # --- Phase exit ---
  - id: out
    type: Output
    inputs:
      mode: mode_pick.parsed.mode
      reasoning_prompt: messages_all.value
      variables:
        const:
          source: "spec-driven plugmem-default sample"
"""


SAMPLES = {
    "plugmem-default": {
        "label": "plugmem-default retrieve",
        "description": (
            "Mirrors PlugMemDefaultPipeline.retrieve: get_plan + get_mode "
            "both run, then one of three reasoning templates is picked by "
            "mode. Memory fetches are stubbed with explicit placeholder "
            "constants (StorageRead nodes land in Phase 6.6)."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}
