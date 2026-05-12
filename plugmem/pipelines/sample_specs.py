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
#   get_plan                           (LLMCall, prompt-mode)
#   get_mode                           (LLMCall, prompt-mode)
#   retrieve_semantic_nodes            (Compute placeholder)
#   retrieve_procedural_nodes          (Compute placeholder)
#   retrieve_episodic_nodes            (Compute placeholder)
#   render_reasoning_semantic          (Branch wrapping PromptRender)
#   render_reasoning_procedural        (Branch wrapping PromptRender)
#   render_reasoning_episodic          (Branch wrapping PromptRender)
#   reason_llm_call                    (LLMCall, text-mode)
#
# Each ``render_reasoning_*`` body is a ``PromptRender`` node. Its
# ``value`` output is the **finished template as a single string** —
# variables substituted in, ready to send to an LLM. The branches
# collapse those strings into one via two ``Compute(concat)`` nodes
# (concat over strings = string concatenation; inactive branches emit
# the empty string, so only the active branch contributes).
#
# ``reason_llm_call`` uses LLMCall's text-mode: ``inputs.text`` is the
# finished rendered string, which the executor wraps as a single user
# message before calling the LLM. The result flows into the Output's
# ``variables``.
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
#     → ONE of three reasoning_* templates renders to a STRING, gated by mode
#     → reason_llm_call (text-mode) wraps that string as a user message
#     → Output ships {mode, reasoning_prompt: <empty>, variables: <answer>}
#
# Each Template node is a `vars → str` function: the `value` port is
# the finished template with all fields filled in. Empty branches emit
# the empty string, so the final concat collapses to the single string
# that the active branch produced.
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

  # --- Template rendering, one per mode.
  #     Body emits ``render.value`` (a STRING — the finished template).
  #     Inactive branches emit the empty string. ---
  - id: render_reasoning_semantic
    type: Branch
    inputs: { condition: is_semantic.value }
    config:
      outputs: { rendered: render.value }
      else_value: { const: "" }
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
      outputs: { rendered: render.value }
      else_value: { const: "" }
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
      outputs: { rendered: render.value }
      else_value: { const: "" }
    body:
      - id: render
        type: PromptRender
        config: { prompt: reasoning_episodic }
        inputs:
          information: retrieve_episodic_nodes.value
          time: in.time
          question: in.observation

  # --- Collapse the three optional renders into one string.
  #     Compute(concat) on strings is string concatenation; two of the
  #     three branches contributed the empty string. ---
  - id: rendered_sem_proc
    type: Compute
    config: { op: concat }
    inputs:
      a: render_reasoning_semantic.rendered
      b: render_reasoning_procedural.rendered

  - id: rendered_all
    type: Compute
    config: { op: concat }
    inputs:
      a: rendered_sem_proc.value
      b: render_reasoning_episodic.rendered

  # --- The reasoning LLM call: takes the rendered STRING directly.
  #     text-mode wraps it as a single user message. Mirrors
  #     plugmem-default's reason_llm_call step. ---
  - id: reason_llm_call
    type: LLMCall
    config: { role: reasoning }
    inputs:
      text: rendered_all.value

  # --- Phase exit. reasoning_prompt stays empty (we send the string
  #     directly); variables.text = the LLM's answer (LLMCall's parsed). ---
  - id: out
    type: Output
    inputs:
      mode: get_mode.parsed.mode
      reasoning_prompt: { const: [] }
      variables: reason_llm_call.parsed
"""


SAMPLES = {
    "plugmem-default": {
        "label": "plugmem-default retrieve + reasoning",
        "description": (
            "Mirrors PlugMemDefaultPipeline.retrieve + reasoning. "
            "Each PromptRender is a `vars → str` function: its `value` "
            "port is the finished template as a string. The active "
            "branch's string flows through reason_llm_call (text-mode) "
            "to produce the answer, which lands in Output.variables. "
            "Memory fetches are stubbed via Compute(concat) (StorageRead "
            "lands in Phase 6.6)."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}
