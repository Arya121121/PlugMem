"""Ready-to-run sample YAMLs the editor can paste as a starting point.

Each sample is a string that loads cleanly through
``plugmem.pipelines.spec_driven.load_yaml_str``. Useful for the "Insert
template" button so researchers don't have to copy YAML out of docs.
"""
from __future__ import annotations


# ----------------------------------------------------------------------- #
# plugmem-default — full retrieve+reason flow in spec-driven YAML
# ----------------------------------------------------------------------- #
#
# This sample is intended to be **functionally equivalent** to
# ``PlugMemDefaultPipeline.retrieve`` (which calls
# ``MemoryGraph.retrieve_memory`` and ``reasoning_llm.complete``).
# Top-level IDs mirror ``plugmem.core.pipeline_spec.STEPS`` so the
# canvas reads as a near-1-to-1 of plugmem-default's retrieve phase.
#
# The pipeline:
#   1. get_plan (LLM, prompt-mode) — produces next_subgoal + query_tags
#   2. get_mode (LLM, prompt-mode) — picks the mode
#   3. Memory fetch by mode (real StorageRead now, no more stubs):
#        - semantic_memory  → retrieve_semantic_nodes (semantic store)
#        - procedural_memory → retrieve_procedural_nodes (procedural store)
#        - episodic_memory   → retrieve_episodic_nodes (episodic context)
#      Each fetch is gated by a Branch on the mode flag.
#   4. ONE reasoning template renders to a STRING, gated by mode.
#   5. reason_llm_call (LLM, text-mode) consumes the rendered string.
#   6. Output ships {mode, reasoning_prompt: [], variables: <LLM parsed>}.
PLUGMEM_DEFAULT_RETRIEVE_YAML = """\
# Functionally equivalent to PlugMemDefaultPipeline.retrieve.
# Top-level IDs match plugmem.core.pipeline_spec for the retrieve phase.
phase: retrieve

nodes:
  - { id: in, type: Input }

  # 1. Plan + mode classification (both always run).
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

  # 2. Mode flags — feed Branch conditions for fetch + render.
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

  # 3. Memory fetch — REAL StorageRead now (no placeholder stubs).
  #    Each gated by its mode flag; inactive ones emit "" (the
  #    matching reasoning template also emits "" so the chain stays
  #    consistent).
  - id: retrieve_semantic_nodes
    type: Branch
    inputs: { condition: is_semantic.value }
    config:
      outputs: { value: read.value }
      else_value: { const: "" }
    body:
      - id: read
        type: StorageRead
        config: { collection: semantic }
        inputs:
          query: in.observation
          tags: get_plan.parsed.query_tags

  - id: retrieve_procedural_nodes
    type: Branch
    inputs: { condition: is_procedural.value }
    config:
      outputs: { value: read.value }
      else_value: { const: "" }
    body:
      - id: read
        type: StorageRead
        config: { collection: procedural }
        inputs:
          subgoal: get_plan.parsed.next_subgoal

  - id: retrieve_episodic_nodes
    type: Branch
    inputs: { condition: is_episodic.value }
    config:
      outputs: { value: read.value }
      else_value: { const: "" }
    body:
      - id: read
        type: StorageRead
        config: { collection: episodic }
        inputs:
          query: in.observation

  # 4. Template rendering, one per mode. Each body emits ``render.value``
  #    (the finished prompt string). Inactive branches emit "".
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

  # 5. Collapse three optional renders into one string. concat over
  #    strings = string concatenation; inactive branches contribute "".
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

  # 6. Reasoning LLM call (text-mode wraps the string as a user message).
  - id: reason_llm_call
    type: LLMCall
    config: { role: reasoning }
    inputs:
      text: rendered_all.value

  # 7. Phase exit. variables.text = the LLM's answer (LLMCall.parsed).
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
            "Functionally equivalent to PlugMemDefaultPipeline.retrieve. "
            "Uses real StorageRead nodes (Phase 6.6) to fetch from the "
            "graph's semantic/procedural/episodic collections — no more "
            "placeholder stubs. Each fetch + reasoning template is gated "
            "by a Branch on the mode flag; the reasoning LLM call uses "
            "text-mode on the rendered prompt string."
        ),
        "content": PLUGMEM_DEFAULT_RETRIEVE_YAML,
    },
}


# ----------------------------------------------------------------------- #
# Node snippets — single-node stubs the editor palette inserts at the
# cursor. Each value is a chunk of valid YAML that, when pasted into the
# `nodes:` list of a pipeline, parses cleanly through the loader (after
# the author fills in the placeholders).
# ----------------------------------------------------------------------- #


NODE_SNIPPETS: dict = {
    "Input": {
        "label": "Input",
        "description": "Phase entry. Outputs are the request body fields.",
        "snippet": "  - { id: in, type: Input }\n",
    },
    "Output": {
        "label": "Output",
        "description": "Phase exit. Inputs become the response payload.",
        "snippet": (
            "  - id: out\n"
            "    type: Output\n"
            "    inputs:\n"
            "      mode: { const: semantic_memory }\n"
            "      reasoning_prompt: { const: [] }\n"
            "      variables: { const: {} }\n"
        ),
    },
    "LLMCall (prompt-mode)": {
        "label": "LLMCall · prompt-mode",
        "description": "Renders a registered prompt with inputs as variables, then calls the LLM.",
        "snippet": (
            "  - id: my_llm_call\n"
            "    type: LLMCall\n"
            "    config: { prompt: <prompt_name>, role: retrieval }\n"
            "    inputs:\n"
            "      observation: in.observation\n"
        ),
    },
    "LLMCall (text-mode)": {
        "label": "LLMCall · text-mode",
        "description": "Wraps a rendered string as a user message; pairs with PromptRender.value.",
        "snippet": (
            "  - id: my_reasoning_call\n"
            "    type: LLMCall\n"
            "    config: { role: reasoning }\n"
            "    inputs:\n"
            "      text: <upstream_node>.value\n"
        ),
    },
    "PromptRender": {
        "label": "PromptRender",
        "description": "Renders a registered prompt to a string (`value`) and a messages list (`messages`).",
        "snippet": (
            "  - id: my_render\n"
            "    type: PromptRender\n"
            "    config: { prompt: <prompt_name> }\n"
            "    inputs:\n"
            "      observation: in.observation\n"
        ),
    },
    "Constant": {
        "label": "Constant",
        "description": "Emits a fixed value (any YAML literal).",
        "snippet": (
            "  - id: my_const\n"
            "    type: Constant\n"
            "    config: { value: \"<your value>\" }\n"
        ),
    },
    "Compute": {
        "label": "Compute",
        "description": "Single op (eq/ne/lt/gt/and/or/not/length/contains/concat/...). Outputs `value`.",
        "snippet": (
            "  - id: my_compute\n"
            "    type: Compute\n"
            "    config: { op: eq }\n"
            "    inputs:\n"
            "      a: <ref_or_const>\n"
            "      b: { const: \"<value>\" }\n"
        ),
    },
    "StorageRead": {
        "label": "StorageRead",
        "description": "Reads from the graph's chroma collection (semantic / procedural / episodic).",
        "snippet": (
            "  - id: my_storage_read\n"
            "    type: StorageRead\n"
            "    config: { collection: semantic }\n"
            "    inputs:\n"
            "      query: in.observation\n"
            "      tags: { const: [] }\n"
        ),
    },
    "Embed": {
        "label": "Embed",
        "description": "Wraps graph.embedder.embed(text); outputs `embedding` (list[float]).",
        "snippet": (
            "  - id: my_embed\n"
            "    type: Embed\n"
            "    inputs:\n"
            "      text: in.observation\n"
        ),
    },
    "ForEach": {
        "label": "ForEach",
        "description": "Iterates a list; body runs once per element; declared outputs collected into lists.",
        "snippet": (
            "  - id: my_loop\n"
            "    type: ForEach\n"
            "    inputs:\n"
            "      items: <list_ref>\n"
            "    config:\n"
            "      item_var: item\n"
            "      outputs:\n"
            "        results: body_node.raw\n"
            "    body:\n"
            "      - id: body_node\n"
            "        type: LLMCall\n"
            "        config: { prompt: <prompt_name>, role: retrieval }\n"
            "        inputs:\n"
            "          observation: item\n"
        ),
    },
    "Branch": {
        "label": "Branch",
        "description": "Conditionally runs a body subgraph; surfaces declared outputs or `else_value`.",
        "snippet": (
            "  - id: my_branch\n"
            "    type: Branch\n"
            "    inputs: { condition: <bool_ref> }\n"
            "    config:\n"
            "      outputs: { result: body_node.value }\n"
            "      else_value: { const: \"\" }\n"
            "    body:\n"
            "      - id: body_node\n"
            "        type: Constant\n"
            "        config: { value: \"ran\" }\n"
        ),
    },
}

