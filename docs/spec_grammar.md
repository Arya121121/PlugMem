# Spec-driven pipeline grammar (formal reference)

Canonical reference for the YAML grammar consumed by
`plugmem.pipelines.spec_driven.load_yaml_str`. Reader-oriented examples
live in [`spec_driven_yaml.md`](spec_driven_yaml.md); this file is the
**normative** spec — every rule listed here is enforced by the loader,
and every loader rule is listed here.

This document is the source-of-truth for downstream tooling (JSON
Schema, editor autocomplete, the spec_view converter). When the
grammar changes, update this file in the same PR.

---

## 1. Top-level structure (EBNF-ish)

```ebnf
Document   = "phase:" Phase  "nodes:" NodeList

Phase      = "retrieve"                   (* MVP — other phases delegate *)

NodeList   = "[" Node ("," Node)* "]"

Node       = InputNode
           | OutputNode
           | LLMCallNode
           | PromptRenderNode
           | ConstantNode
           | ComputeNode
           | ForEachNode
           | BranchNode

Common     = id: <NodeId>          (required, non-empty string, unique in scope)
             type: <NodeType>      (required, one of the variants above)
             config: <Object>      (optional, type-specific schema)
             inputs: <PortMap>     (optional, type-specific schema)
             body: <NodeList>      (only for ForEach + Branch)

PortMap    = { <PortName>: <Ref> }     (PortName: non-empty string)

Ref        = NodeRef | LiteralRef
NodeRef    = "<NodeId>" | "<NodeId>.<path>"
LiteralRef = { const: <YAML value> }

NodeId     = [A-Za-z_][A-Za-z0-9_]*     (informal; the loader accepts any
                                         non-empty string but `.` in ids
                                         conflicts with ref dot-paths
                                         and is discouraged.)
```

### Top-level invariants (enforced)

- `phase` must equal `"retrieve"`. Other values raise `ValueError`.
- `nodes` must be a list (not a mapping).
- Exactly one node with `type: Input`.
- Exactly one node with `type: Output`.
- Node ids must be unique within the top-level scope.
- The top-level graph must be a DAG (cycle detection via Kahn's
  algorithm). Self-references raise immediately.

### Reference resolution

When resolving `<NodeId>[.<path>]`:

1. If the value is `{ const: <x> }` → emit `<x>` literally.
2. Otherwise the head is a plain string. Lookup order:
   1. **Item vars** (only inside a `ForEach.body`): if the head matches
      `config.item_var`, bind the whole per-iteration element. Bare refs
      (no dot) are valid **only** when they name an item var.
   2. **Local scope**: nodes in the current scope (top-level OR
      siblings inside the same body).
   3. **Outer scope**: top-level nodes are visible from any body.
3. Walk dotted path segments through `dict`s. Missing keys → `ValueError`.

### Reference vs literal

| Form | Meaning |
|---|---|
| `"some_node"` | Whole output dict of `some_node`. (Rarely used directly.) |
| `"some_node.port"` | One named port of `some_node`. |
| `"some_node.port.subkey"` | Walks dict keys inside the port. |
| `{ const: <value> }` | Inline literal — string, number, list, dict, etc. |

---

## 2. Node-type schemas

Every node-type sub-section below documents:

- **config**  — allowed keys + types
- **inputs**  — allowed/required port names + value shapes
- **outputs** — port names emitted at runtime (used by the spec_view
  converter and by consumers)

### 2.1 `Input`

The phase entry point. Exactly one per pipeline.

| Field    | Required | Schema |
|----------|----------|--------|
| `config` | no       | must be absent or `{}` |
| `inputs` | no       | must be absent or `{}` |

**Outputs:** the request-body fields, available as `<input_id>.<field>`:

```
observation, goal, subgoal, state, task_type, time, mode
```

### 2.2 `Output`

The phase exit. Exactly one per pipeline. Its `inputs` dict becomes the
response payload.

| Field    | Required | Schema |
|----------|----------|--------|
| `config` | no       | must be absent or `{}` |
| `inputs` | **yes**  | must include `mode`, `reasoning_prompt`, `variables` |

**Required input ports** (their types are enforced by the pydantic
response model, not the YAML loader):

| Port | Type |
|------|------|
| `mode` | `str` (one of `semantic_memory`/`procedural_memory`/`episodic_memory`) |
| `reasoning_prompt` | `list[ { "role": str, "content": str } ]` |
| `variables` | `dict[str, Any]` |

### 2.3 `LLMCall`

Calls the LLM router. Exactly one of three input modes must be set:

| Mode | Set by | Behaviour |
|------|--------|-----------|
| Prompt | `config.prompt: <str>` | renders the named registered prompt with `inputs` as template variables |
| Text | `inputs.text: <str>` | wraps the string as `[{role: "user", content: <text>}]` |
| Messages | `inputs.messages: <list>` | passes the list verbatim |

Setting more than one is a load/runtime error.

| Field | Required | Schema |
|-------|----------|--------|
| `config.prompt` | conditional | non-empty string when prompt-mode |
| `config.role`   | no | one of `default`, `structuring`, `retrieval`, `reasoning`, `consolidation` (defaults to `default`) |
| `inputs.text`     | conditional | string when text-mode |
| `inputs.messages` | conditional | list of `{role, content}` dicts when messages-mode |
| `inputs.<other>`  | prompt-mode only | template variables for the named prompt |

**Outputs:**

| Port | Type | Meaning |
|------|------|---------|
| `raw` | `str` | full LLM response |
| `parsed.<key>` | varies | prompt-specific parsed dict; defaults to `{ text: <raw> }` for messages/text mode and prompts without a registered parser |

Registered parsers: `get_plan` → `{next_subgoal, query_tags}`,
`get_mode` → `{mode}`. All other prompts return `{text: <raw>}`.

### 2.4 `PromptRender`

Renders a registered prompt to messages **without** an LLM call. A
`vars → str` function.

| Field | Required | Schema |
|-------|----------|--------|
| `config.prompt` | **yes** | non-empty string, name in `PromptRegistry` |
| `inputs.<v>` | yes | template variables for the prompt |

**Outputs:**

| Port | Type | Meaning |
|------|------|---------|
| `value` | `str` | the finished template — each message's content joined by `\n\n` |
| `messages` | `list[{role, content}]` | structured form for callers that need role info |

### 2.5 `Constant`

Emits a fixed value.

| Field | Required | Schema |
|-------|----------|--------|
| `config.value` | **yes** | any YAML value (string, number, list, dict, bool, null) |
| `inputs` | no | must be absent or `{}` |

**Outputs:**

| Port | Meaning |
|------|---------|
| `value` | the configured value |

For one-off literals, prefer the inline `{ const: <value> }` shorthand
in any input slot — it's equivalent without declaring a node.

### 2.6 `Compute`

Single-op pure compute. Whitelist below; unknown ops are rejected at
load time.

| Field | Required | Schema |
|-------|----------|--------|
| `config.op` | **yes** | one of the ops in the table |
| `inputs.<port>` | per op | exact required ports listed below |

**Outputs:** always a single port `value`.

**Op table** (MVP whitelist):

| Op | Inputs | Output | Notes |
|----|--------|--------|-------|
| `eq` | `a`, `b` | `bool` | `a == b` |
| `ne` | `a`, `b` | `bool` | `a != b` |
| `lt` | `a`, `b` | `bool` | `a < b` |
| `gt` | `a`, `b` | `bool` | `a > b` |
| `lte` | `a`, `b` | `bool` | `a <= b` |
| `gte` | `a`, `b` | `bool` | `a >= b` |
| `and` | `a`, `b` | `bool` | truthiness of both |
| `or` | `a`, `b` | `bool` | truthiness of either |
| `not` | `a` | `bool` | negates truthiness |
| `length` | `list` | `int` | `len(list)` |
| `contains` | `list`, `item` | `bool` | `item in list` |
| `concat` | `a`, `b` | `str` or `list` | `a + b` (strings or lists; identical-type required at runtime) |

### 2.7 `ForEach`

Iterates a list. Body runs once per element; declared outputs are
collected into lists across iterations.

| Field | Required | Schema |
|-------|----------|--------|
| `config.item_var` | **yes** | non-empty string, the per-iteration binding name |
| `config.outputs` | no | `{ <out_name>: "<body_node>.<port>" }` |
| `inputs.items` | **yes** | a ref that resolves to `list` |
| `body` | **yes** | non-empty `NodeList` (no `Input`/`Output` allowed) |

**Outputs:** the keys named in `config.outputs`. Each is a list with one
entry per iteration, in input order.

**Caps:** `MAX_LOOP_ITERATIONS = 1000` per ForEach instance.

### 2.8 `Branch`

Conditionally executes a body. Truthy condition → body runs and its
declared outputs surface; falsy → every declared output receives
`else_value`.

| Field | Required | Schema |
|-------|----------|--------|
| `config.outputs` | **yes** | non-empty `{ <out_name>: "<body_node>.<port>" }` |
| `config.else_value` | no | a `Ref` or `{ const: ... }`; defaults to `{ const: null }` |
| `inputs.condition` | **yes** | any ref (coerced via Python `bool()`) |
| `body` | **yes** | non-empty `NodeList` (no `Input`/`Output` allowed) |

**Outputs:** the keys named in `config.outputs`.

---

## 3. Body subgraph rules

Body nodes (under `ForEach.body` or `Branch.body`) are a closed
subgraph with their own topological sort.

- May not contain `Input` or `Output` nodes.
- Body nodes may reference outer-scope top-level node ids and (for
  `ForEach`) the `item_var`.
- Body-local ids must be unique within that body. Across different
  bodies, ids may repeat (each body is its own scope).
- Cycles inside the body are rejected at load time.
- Nested `ForEach`/`Branch` are allowed and recursively validated.

---

## 4. Trace recording

Every `LLMCall` is recorded via `record_llm_step`, named by the node's
id. Inside a `ForEach.body`, the recorded name is suffixed with the
iteration index: `<node_id>#<iter_idx>`. Nested loops chain:
`<id>#<outer_i>#<inner_i>`.

---

## 5. Runtime caps

| Cap | Limit | Source |
|-----|-------|--------|
| LLM calls per pipeline run | 20 | `LLM_CALL_CAP` |
| ForEach iterations per instance | 1000 | `MAX_LOOP_ITERATIONS` |

Both raise `ValueError` (mapped to HTTP 422 by the route).

---

## 6. Out of scope (planned)

| Item | Phase |
|------|-------|
| `StorageRead` / `Embed` node types | 6.6 |
| `close` / `insert` / `consolidate` phases | 6.6 |
| `Switch` (n-way branch) | TBD — gauge with use case |
| `filter_by`, `top_k`, `similarity` ops | TBD |
| Per-prompt parser plugins | TBD |
| Visual editor (drag-and-drop) | 6.5b |
| Python hot-reload for forked pipelines | 6.5c |
