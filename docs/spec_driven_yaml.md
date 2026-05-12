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

### `ForEach`

Iterates over a list input. Each element is bound to a per-iteration
variable; the `body` subgraph runs once per element; declared outputs
are collected across iterations into lists.

```yaml
- id: per_tag
  type: ForEach
  inputs:
    items: plan.parsed.query_tags        # ref to a list-valued port
  config:
    item_var: tag                        # name bound inside the body
    outputs:
      analyses: analyze.raw              # → list[str]
  body:
    - id: analyze
      type: LLMCall
      config: { prompt: get_subgoal, role: retrieval }
      inputs:
        goal: tag                        # bare item_var ref → whole element
        state: in.state                  # outer-scope refs still work
        observation: in.observation
        action: { const: "" }
```

**Inputs:**

| Port | Type | Meaning |
|---|---|---|
| `items` | `list` | The list to iterate. Resolved once before the loop. |

**Config:**

| Key | Type | Meaning |
|---|---|---|
| `item_var` | str | Name bound to each element inside the body. References to `<item_var>` (bare) get the whole element; `<item_var>.<field>` walks dict keys. |
| `outputs` | dict | `{<output_name>: "<body_node>.<port>"}`. After every iteration, each declared port is collected into a list under `<output_name>` at the outer scope. |

**Outputs:** the keys named in `config.outputs`. Each is a list with one
entry per iteration, in input order.

**Body semantics:**

- Body is its own closed subgraph (own topological sort).
- Body nodes may reference outer-scope nodes (e.g. `in.observation`,
  `plan.parsed.next_subgoal`) and the item var (e.g. `tag`).
- Body cannot contain `Input` or `Output` nodes (those only exist at the
  top level).
- Nested `ForEach` is supported.

**Trace recording:** `LLMCall` inside a body is recorded as
`<body_node_id>#<iter_idx>` so each iteration's call appears as a
distinct step in the Pipeline tab's traces panel. Nested loops chain:
`<id>#<outer_i>#<inner_i>`.

**Caps:**

- A `ForEach` is rejected at runtime if `len(items) > MAX_LOOP_ITERATIONS`
  (1000). The cap surfaces as a 422 from the route.
- The global `LLM_CALL_CAP` (20 per pipeline run) still applies — a loop
  that fans out an LLM call hits it sooner than a pure-compute loop.

### `Compute`

A single-op function. Lets the spec produce booleans (for `Branch`
conditions), compare values, and do basic list / string operations
without escaping into Python.

```yaml
- id: above_threshold
  type: Compute
  config: { op: gt }
  inputs:
    a: score.parsed.value
    b: { const: 0.5 }
# → above_threshold.value is bool
```

Every op produces a single output port named `value`. Unknown ops are
rejected at load time.

**Op table** (MVP whitelist):

| Op | Inputs | Output | Notes |
|---|---|---|---|
| `eq` | `a`, `b` | bool | `a == b` |
| `ne` | `a`, `b` | bool | `a != b` |
| `lt` | `a`, `b` | bool | `a < b` |
| `gt` | `a`, `b` | bool | `a > b` |
| `lte` | `a`, `b` | bool | `a <= b` |
| `gte` | `a`, `b` | bool | `a >= b` |
| `and` | `a`, `b` | bool | truthiness of both |
| `or` | `a`, `b` | bool | truthiness of either |
| `not` | `a` | bool | negates truthiness |
| `length` | `list` | int | `len(list)` |
| `contains` | `list`, `item` | bool | `item in list` |
| `concat` | `a`, `b` | str or list | `a + b` (strings or lists) |

Out of scope for this MVP: `filter_by`, `top_k`, `similarity` — they
need predicate sub-specs or embedder access. Open a request if needed.

### `Branch`

Conditionally run a body subgraph. Truthy → execute the body and surface
its declared outputs. Falsy → skip the body and emit `else_value` for
every declared output.

```yaml
- id: maybe_merge
  type: Branch
  inputs:
    condition: above_threshold.value      # any value; coerced via bool(x)
  config:
    outputs:
      result: merge.parsed.merged_statement   # ref → body node port
    else_value: { const: null }               # what each output is when false
  body:
    - id: merge
      type: LLMCall
      config: { prompt: get_new_semantic, role: consolidation }
      inputs: { memory_earlier: in.left, memory_later: in.right }
```

**Inputs:**

| Port | Type | Meaning |
|---|---|---|
| `condition` | any | Coerced to bool via Python `bool(x)`. |

**Config:**

| Key | Type | Meaning |
|---|---|---|
| `outputs` | dict | `{<output_name>: "<body_node>.<port>"}`. When the condition is truthy, each declared port is sourced from the body's env. Required and non-empty. |
| `else_value` | ref | Optional. When the condition is falsy, every declared output gets this value. Defaults to `{const: null}`. Accepts a ref string or a `{const: ...}` literal. |

**Body semantics:**

- Same scoping rules as `ForEach.body`: body nodes may reference outer
  scope; cannot contain `Input` / `Output`; cycles rejected at load.
- LLM calls inside the body run only when the condition is truthy — so
  a `Branch` with a falsy condition is a cheap no-op (no LLM, no
  embedder calls).
- `Branch` and `ForEach` can nest inside each other freely.

**Single-body design:** the MVP supports only a body that runs when
truthy. To express full if/else, chain two `Branch` nodes with inverted
conditions (use `Compute` with `op: not`), or use `else_value` to
encode the alternative path's result.

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
