"""Ready-to-run sample YAMLs the editor can paste as a starting point.

Each sample is a string that loads cleanly through
``plugmem.pipelines.spec_driven.load_yaml_str``. Useful for the "Insert
template" button so researchers don't have to copy YAML out of docs.
"""
from __future__ import annotations


# ----------------------------------------------------------------------- #
# plugmem-default — retrieve+reason flow, expressed in spec-driven YAML
# ----------------------------------------------------------------------- #
#
# Top-level node IDs intentionally mirror ``plugmem.core.pipeline_spec``:
#
#   get_plan                           (LLMCall)
#   get_mode                           (LLMCall)
#   retrieve_semantic_nodes            (Compute placeholder)
#   retrieve_procedural_nodes          (Compute placeholder)
#   retrieve_episodic_nodes            (Compute placeholder)
#   render_reasoning_semantic          (Branch wrapping PromptRender)
#   render_reasoning_procedural        (Branch wrapping PromptRender)
#   render_reasoning_episodic          (Branch wrapping PromptRender)
#   reason_llm_call                    (LLMCall consuming the rendered messages)
#
# Each ``render_reasoning_*`` body uses a ``PromptRender`` to build the
# messages for that mode. The branches collapse to a single messages
# list via two ``Compute(concat)`` nodes. The final ``reason_llm_call``
# is an LLMCall in **messages mode** — ``inputs.messages`` is wired to
# the merged messages, so the LLM sees exactly the rendered prompt and
# returns a response that flows into the Output's ``variables``.
#
# The MVP grammar has no StorageRead / Embed node; the
# ``retrieve_*_nodes`` steps are Compute(concat) stubs whose inputs
# surface what data WOULD be fetched in production. Real fetches land
# in Phase 6.6.
PLUGMEM_DEFAULT_RETRIEVE_YAML = """\
# plugmem-default retrieve + reasoning, expressed as a spec-driven pipeline.
#
# Flow:
#   get_plan + get_mode (both always run, parallel by data deps)
#     → retrieve_*_nodes (stubs — would hit storage in production)
#     → ONE of three reasoning_* templates renders, gated by mode
#     → reason_llm_call consumes the rendered messages and produces an answer
#     → Output ships {mode, reasoning_prompt: <messages>, variables: <answer>}
#
# The reason_llm_call uses LLMCall's "messages mode": inputs.messages
# is wired to the merged rendered messages, so the LLM call is an
# explicit step in the graph rather than hidden inside the template
# render. Storage fetches are Compute(concat) stubs (Phase 6.6).
phase: retrieve

nodes:
  - { id: in, type: Input }

  # --- Planning + mode classification (both always run) ---
  - id: get_plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
    inputs:
      goal: in.goal
      subgoal: in.subgoal
      state: in.state
      observation: in.observation

  - id: get_mode
    type: LLMCall
    config: { prompt: get_mode, role: retrieval }
    inputs:
      observation: in.observation
      task_type: in.task_type

  # --- Memory fetch — STUBBED via Compute(concat) ---
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

  # --- Mode gates: each Compute(eq) feeds a Branch's condition. ---
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

  # --- Template rendering, one per mode. Body emits `messages`. ---
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

  # --- Collapse the three optional branches into one messages list. ---
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

  # --- The reasoning LLM call: takes the rendered messages directly.
  #     Mirrors plugmem-default's reason_llm_call step. No `config.prompt`
  #     because we want to use the messages built above verbatim. ---
  - id: reason_llm_call
    type: LLMCall
    config: { role: reasoning }
    inputs:
      messages: messages_all.value

  # --- Phase exit. reasoning_prompt = the messages /reason would consume.
  #     variables.text = the LLM's answer (via the LLMCall's parsed dict). ---
  - id: out
    type: Output
    inputs:
      mode: get_mode.parsed.mode
      reasoning_prompt: messages_all.value
      variables: reason_llm_call.parsed
"""


SAMPLES = {
    "plugmem-default": {
        "label": "plugmem-default retrieve + reasoning",
        "description": (
            "Mirrors PlugMemDefaultPipeline.retrieve + reasoning. "
            "get_plan + get_mode both run, then one of three reasoning_* "
            "templates renders by mode, and reason_llm_call consumes the "
            "rendered messages — so the canvas shows the full "
            "Template → LLM → Output arc. Memory fetches are stubbed via "
            "Compute(concat) (StorageRead lands in Phase 6.6)."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}
