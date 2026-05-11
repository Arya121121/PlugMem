# Spec-driven pipelines (Phase 6.2 MVP)

A graph bound to the `spec-driven` pipeline reads its `retrieve` flow
from a per-graph YAML file at:

```
{PROMPTS_DIR}/{graph_id}.pipeline.yaml
```

(`PROMPTS_DIR` defaults to `./data/prompts`, same env var the prompt
overrides use.)

If the file is missing or invalid, `POST /retrieve` returns 422 with the
exact error. `POST /memories`, `/reason`, and `/consolidate` keep working
because they delegate to `plugmem-default`.

## Quick start

```bash
# 1. Bind a graph to the spec-driven pipeline.
curl -X PUT localhost:8000/api/v1/graphs/myg/pipeline \
     -H 'content-type: application/json' \
     -d '{"pipeline": "spec-driven"}'

# 2. Drop a YAML in PROMPTS_DIR.
mkdir -p data/prompts && cat > data/prompts/myg.pipeline.yaml <<'YAML'
phase: retrieve
nodes:
  - { id: in, type: Input }
  - id: plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
    inputs: { goal: in.goal, subgoal: in.subgoal, state: in.state, observation: in.observation }
  - id: msgs
    type: PromptRender
    config: { prompt: reasoning_semantic }
    inputs:
      goal: in.goal
      subgoal: in.subgoal
      state: in.state
      observation: in.observation
      semantic_memory: { const: "(none — placeholder for MVP)" }
      procedural_memory: { const: "" }
      episodic_memory: { const: "" }
      time: { const: "" }
      information: { const: "" }
      question: in.observation
  - id: out
    type: Output
    inputs:
      mode: { const: semantic_memory }
      reasoning_prompt: msgs.messages
      variables: { const: {} }
YAML

# 3. Call /retrieve — it runs your pipeline.
curl -X POST localhost:8000/api/v1/graphs/myg/retrieve \
     -H 'content-type: application/json' \
     -d '{"observation": "what now?"}'
```

## Schema

```yaml
phase: retrieve            # MVP only supports "retrieve"
nodes:
  - id: <unique-id>
    type: Input | Output | LLMCall | PromptRender | Constant
    config: { ... }        # node-type-specific
    inputs: { <port>: <ref>, ... }
```

References (`<ref>`) come in two shapes:

| Shape | Meaning |
|---|---|
| `"<node_id>.<port>"` (dot path) | Pull `port` from another node's outputs. Can be nested, e.g. `plan.parsed.next_subgoal`. |
| `{ const: <value> }` | An inline literal. Use for hard-coded inputs without declaring a Constant node. |

The loader rejects:

- Cycles, self-references, unknown node ids, duplicate ids.
- More or fewer than one `Input` and one `Output` node.
- `phase` other than `retrieve` (MVP scope).

## Node types

### `Input`

Phase entry. Outputs are the request body fields:

```
in.observation   in.goal   in.subgoal   in.state
in.task_type     in.time   in.mode
```

No `config`, no `inputs`. Exactly one Input per pipeline.

### `Output`

Phase exit. Its `inputs` become the response. Required ports:

```
mode               # str
reasoning_prompt   # list of chat messages [{"role": "...", "content": "..."}]
variables          # dict (free-form trace metadata)
```

### `LLMCall`

Calls a registered prompt through the LLM router; recorded into the
trace recorder under the **node's id**, not the prompt's name.

```yaml
- id: plan
  type: LLMCall
  config:
    prompt: get_plan      # name in PromptRegistry
    role: retrieval       # LLMRouter role (default/structuring/retrieval/reasoning/consolidation)
  inputs: { ... }         # template variables (must match what the prompt expects)
```

Outputs:

| Port | Type | Meaning |
|---|---|---|
| `raw` | str | The full LLM response. |
| `parsed.<key>` | varies | Best-effort structured output. Currently parsed: `get_plan` → `{next_subgoal, query_tags}`, `get_mode` → `{mode}`. Other prompts: `parsed.text` = the raw response. |

There is a hard cap of **20 LLM calls per pipeline run** (config knob will
follow if anyone hits it).

### `PromptRender`

Renders a template into chat messages **without** an LLM call.

```yaml
- id: msgs
  type: PromptRender
  config: { prompt: reasoning_semantic }
  inputs: { observation: in.observation, semantic_memory: { const: "..." } }
```

Output: `messages` — a `list[{"role": str, "content": str}]`.

### `Constant`

Emits a fixed value.

```yaml
- id: mode
  type: Constant
  config: { value: semantic_memory }
```

Output: `value`. (For one-offs prefer the inline `{ const: ... }` shorthand.)

## Tracing

Every `LLMCall` executes inside the existing `PipelineTraceRecorder`
context, so the Pipeline tab's "Recent runs" panel and the per-step
stats footer work for spec-driven pipelines too. The trace step's
`name` is the node id, so two nodes with the same prompt (e.g. two
`get_plan` calls with different variables) show up as distinct rows.

## Out of scope for the MVP

- Phases other than `retrieve` (delegate to `plugmem-default`).
- Branch / loop / Compute / StorageRead / Embed nodes.
- xyflow-based visual editor — write YAML by hand for now.
- Auto-upload via API — the YAML lives on the server filesystem.
- Per-prompt parser plugins — only `get_plan` + `get_mode` are parsed
  today. Future iterations will read parsers from the registry.
