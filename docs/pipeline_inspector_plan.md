# Pipeline Inspector — Plan & Status

Last updated 2026-05-11.

The Pipeline tab in the Memory Inspector lets users (a) visualize PlugMem's
processing graph, (b) edit the prompts and LLM models used at each step,
(c) inspect every run via captured traces, and (d) eventually replace the
algorithm itself with a custom pipeline.

## Status

| Phase | Scope | State |
|---|---|---|
| 1   | Read-only pipeline diagram | shipped |
| 1.5 | Exact spec (kinds, edges, branches, loops) + xyflow rewrite | shipped |
| 2   | Per-graph prompt editing (CRUD + preview + side editor) | shipped |
| 3   | Per-role model swap (`LLMRouter.set_role` + Test connection + env-var refs) | shipped |
| 4   | Pipeline trace capture + display (configurable cap; default 100; 0 = unlimited) | shipped |
| 5   | Polish (diff view, sparklines, recent calls, re-run with current prompts) | shipped |
| 5.5 | Pluggable memory-pipeline implementations + `naive-rag` baseline | shipped |
| 6.0 | Spec ↔ code drift checks (static AST lint + runtime trace check) | shipped |
| 6.1 | Direction A — code→spec snapshot tool with `--check` for CI | shipped |
| 6.2 | Direction B — `SpecDrivenPipeline` MVP: retrieve-only, YAML-driven | shipped |
| 6.3 | Loops in spec-driven pipelines (`ForEach` node) | shipped |
| 6.4 | Branch + Compute nodes in spec-driven pipelines | shipped |
| 6.5a | Non-destructive YAML versioning + YAML editor UI + version history | shipped |
| 6.5d | Formal grammar spec doc + JSON Schema | shipped |
| 6.6a | StorageRead + Embed nodes + sample drops placeholder stubs | shipped |
| 6.6a-diff | Differential harness: spec-driven sample ≡ plugmem-default | shipped |
| 6.6b-1 | `/reason` via spec-driven (reuses retrieve YAML, reshapes output) | shipped |
| 6.5b | Interactive xyflow canvas (drag-and-drop palette, port wiring) | future |
| 6.5c | Python hot-reload for forked pipeline modules | future |
| 6.6b-2 | Spec-driven coverage of close / insert / consolidate phases | future |

## Cross-cutting principles

These apply across phases. Codify behavior, not implementation.

### Builtins are read-only

The shipped Python (inference functions, `plugmem-default` pipeline,
`PromptRegistry` builtins, `_defaults.yaml`) is **never modified by the UI
or API**. All user customization lives in per-graph YAML overlays or in
forked pipeline copies.

Enforcement: the prompt PUT endpoint refuses writes to anything except the
per-graph layer; spec-driven YAMLs live alongside prompt overrides; new
"user pipelines" must be assigned a fresh name, not overwrite a built-in.

### Edits are non-destructive

> "When a user makes changes through the visualization tool, make a copy
> of the previous code and apply the changes there."

When a user edits an existing pipeline (YAML or Python) via the editor:

1. The current source is **copied** into a new versioned artifact.
2. Edits land on the copy.
3. The user can promote the copy to the graph's binding, keep it as a
   draft, or discard it.
4. The previous version remains on disk and is reachable for rollback.

For Phase 6.2 (`SpecDrivenPipeline`), the YAML is on disk and easy to
version. For Phase 6.5 (Python-backed pipelines via the visual editor),
this requires real-time fork + hot-reload — see Phase 6.5 for the design.

### Spec is the source of truth (when it agrees with code)

The static lint (`tests/test_pipeline_spec_lint.py`) and runtime trace
check (`tests/test_pipeline_spec_runtime.py`) guarantee the visualization
is a 1-to-1 map of what the production code actually does. CI fails on
drift; the snapshot tool (`plugmem/tools/spec_snapshot.py`) tells the
developer exactly what to fix.

### Pipelines are pluggable

Multiple memory algorithms can be hosted side-by-side. The
`MemoryPipeline` protocol + `plugmem.pipelines` registry are the contract.
Adding a baseline (RAG, naive, custom) is a class + a `register()` call;
no edits to routes or the inspector.

---

## Phase 6.3 — Loops (`ForEach`)

**Goal**: let spec-driven pipelines iterate. Unlocks per-tag fan-out in
retrieve (e.g. one LLM call per `get_plan.parsed.query_tags`) and is the
foundation for close / insert / consolidate phases (which all loop in the
production code).

### Node type

```yaml
- id: per_tag
  type: ForEach
  config:
    items: plan.parsed.query_tags        # ref to a list-valued port
    item_var: tag                        # name bound inside the body
    outputs:                             # what to collect across iterations
      analyses: analyze.raw              # → list[str]
  body:
    - id: analyze
      type: LLMCall
      config: { prompt: get_subgoal, role: retrieval }
      inputs: { goal: tag, state: in.state, observation: in.observation, action: { const: "" } }
```

### Semantics

- `items` resolves to a list. Each element gets bound to `item_var` for
  one iteration.
- `body` is a closed subgraph with its own topo-sort. Body nodes can
  reference outer nodes (e.g. `in.observation`, `plan.parsed.next_subgoal`)
  and the per-iter var (e.g. `tag`).
- `outputs` declares which body-node outputs to collect; each becomes a
  list at the outer scope. `per_tag.analyses` is `list[str]` with length
  equal to the number of iterations.
- Empty `items` → outputs are empty lists (no error).
- Body nodes cannot be `Input` or `Output` (those only exist at the top
  level).

### Implementation

**`plugmem/pipelines/spec_driven.py` (~+150 LoC)**
- Extend `NodeSpec` to allow a nested `body` field.
- Loader: recognize `ForEach`; validate `items` ref, non-empty `item_var`,
  body is a `list[NodeSpec]`, body topologically sortable, body has no
  nested `Input`/`Output`, outer-scope refs allowed.
- Executor: `_run_foreach`. For each item, build a sub-env that includes
  outer-env + item_var binding, run body in topo order, collect declared
  outputs into outer-env lists.
- Trace step naming: body `LLMCall` recorded as `<body_node_id>#<iter_idx>`
  so iterations don't collide in the trace UI.
- Cap: `MAX_LOOP_ITERATIONS = 1000` per `ForEach` instance (in addition
  to the existing `LLM_CALL_CAP = 20`).

**`tests/test_spec_driven_pipeline.py` (~+100 LoC, 6–7 new tests)**
- Loader accepts a valid `ForEach`.
- Loader rejects `Input` / `Output` nodes nested in `body`.
- Loader rejects cycles within `body`.
- Executor iterates a `Constant` list (deterministic, no LLM).
- Executor iterates an `LLMCall.parsed.<list_port>` (chains with `get_plan`).
- Nested `ForEach` works (body contains another `ForEach`).
- Iteration cap raises a clear error.

**`docs/spec_driven_yaml.md` (~+40 lines)**: add `ForEach` to the node-type
reference + a worked example.

### Out of scope for 6.3

- `Branch` node (Phase 6.4).
- `Compute` node for list manipulation (filter / map / topk) (Phase 6.4).
- Other accumulators (`last`, `count`) — only "collect into list" supported.
- Body nodes leaking arbitrary references outward — only declared `outputs`
  cross the loop boundary.

---

## Phase 6.4 — Branch + Compute

**Goal**: cover `insert` and `consolidate` phases, which both have
conditional gating (subgoal-exists?, score≥threshold?) and need list
operations (top-k, filter).

### Node types

- **`Branch`** — config: `condition` (ref to a bool-valued port).
  Outputs: `then.*`, `else.*` route to different downstream subgraphs.
  Validation: the two branches converge at some downstream node or each
  branch terminates at `Output`.
- **`Compute`** — config: `op` from a whitelist:
  `similarity`, `top_k`, `filter_by`, `concat`, `count`, `slice`, `length`,
  `head`, `tail`, `not`, `eq`, `lt`, `gt`, `and`, `or`.
  Inputs / outputs per op; each op is ~5–10 LoC.

### Why a whitelist (not arbitrary expressions)

User-authored Python in the YAML is a security risk and a debugging
nightmare. A small whitelist covers PlugMem's actual ops; extending it is
one Python function each.

### Open question

`Branch` vs `Switch` (n-way). `Branch` first; `Switch` only if a real
multi-way decision shows up in close/insert.

---

## Phase 6.5a — Non-destructive YAML versioning + editor UI (shipped)

A versioned filesystem store under `{PROMPTS_DIR}/.pipeline_history/{graph_id}/`
holds every saved spec; the "live" YAML at `{PROMPTS_DIR}/{graph_id}.pipeline.yaml`
is always a mirror of the active version. The executor still reads the live
file — no executor change needed.

**Storage module:** `plugmem/pipelines/spec_storage.py`

- `save_new_version(graph_id, content, *, note, validator)` — validate via
  `load_yaml_str`, write a new `{version_id}.pipeline.yaml` + `.meta.json`,
  atomically flip `current.txt`, rewrite the live file.
- `list_versions(graph_id)` — newest first; sorted by microsecond-precision
  `version_id` so same-second saves preserve order.
- `read_version(graph_id, version_id)` — historical content.
- `rollback(graph_id, version_id)` — point `current.txt` at the chosen
  version and rewrite the live file; no new version row.
- `adopt_existing_live(graph_id)` — if a user dropped a YAML on disk
  out-of-band, seed history with it on first list/save so prior content
  is never destroyed.

`version_id` format: `v_YYYYMMDDTHHMMSS_uuuuuu_xxxxxx` (UTC, microseconds, hex).

**API routes:** all under `/api/v1/graphs/{gid}/pipeline/spec`

| Method | Path | Purpose |
|---|---|---|
| GET    | `/pipeline/spec`                                       | Current YAML + active version id + live path |
| PUT    | `/pipeline/spec`                                       | Validate + save a new version (422 on bad spec) |
| POST   | `/pipeline/spec/validate`                              | Dry-run validation, no disk write |
| GET    | `/pipeline/spec/versions`                              | List versions newest first |
| GET    | `/pipeline/spec/versions/{vid}`                        | Historical content |
| POST   | `/pipeline/spec/versions/{vid}/rollback`               | Promote vid as active |

**UI:** Pipeline tab now exposes a "Bound pipeline" picker in the sidebar
and a "Spec editor" panel above the canvas (only shown when bound to
`spec-driven`). The panel has:

- A YAML textarea pre-loaded with the active version.
- `Validate` button (POST `/spec/validate`).
- `Save new version` button (PUT `/spec`) with an optional note.
- `Discard changes` to restore the active version.
- Collapsible "Version history" list with per-row `View` (load read-only)
  and `Rollback` actions.

**Test coverage:** `tests/test_spec_storage.py` (17 tests) — storage
behaviors (atomic save, parent tracking, rollback, adopt) + route layer
(get/put/validate/list/rollback/end-to-end retrieve).

### Pipeline-aware visualization (6.5a follow-up)

The Pipeline tab canvas now reflects the bound pipeline of the current
graph, not a static dump of `plugmem-default`:

| Binding | Source |
|---|---|
| `plugmem-default` | `plugmem.core.pipeline_spec.to_dict()` (hand-curated, 6 phases) |
| `naive-rag` | Hand-coded 3-phase spec in `pipeline_views.naive_rag_spec()` |
| `spec-driven` | Live conversion of the saved YAML → PipelineSpec shape (`pipeline_views.spec_driven_spec`) |

Route: `GET /api/v1/graphs/{gid}/pipeline/spec_view` — dispatches via
`pipeline_views.view_for_pipeline(name, graph_id=gid)`. The UI calls
this on initial load, on every binding swap, on every save, and on
every rollback so the canvas stays in lock-step with the running spec.

### Starter YAML samples (6.5a follow-up)

`GET /api/v1/pipeline/samples` returns ready-to-paste sample YAMLs from
`plugmem.pipelines.sample_specs`. The Spec editor has an "Insert
template" button + select that pastes the chosen sample into the
textarea. Shipped sample:

- **plugmem-default retrieve** — mirrors `PlugMemDefaultPipeline.retrieve`
  (`get_plan` → `get_mode` → render `reasoning_semantic`). Storage reads
  are stubbed with explicit placeholder constants since the MVP grammar
  has no `StorageRead` / `Embed` nodes yet (those land in Phase 6.6).

---

## Phase 6.5b / 6.5c — Interactive canvas + Python hot-reload (deferred)

These remain on the roadmap but were deferred from 6.5a so the
versioning foundation could ship and be tested first.

## Phase 6.5d — Formal grammar spec + JSON Schema (shipped)

Canonical source-of-truth for the YAML grammar:

- [`docs/spec_grammar.md`](spec_grammar.md) — normative reference.
  Every loader-enforced rule is listed; every node type, config key,
  input/output port is documented. Reader-oriented examples stay in
  [`docs/spec_driven_yaml.md`](spec_driven_yaml.md).
- [`plugmem/pipelines/spec_schema.json`](../plugmem/pipelines/spec_schema.json)
  — JSON Schema (draft 2020-12). Mirrors the static shape; the loader
  still owns rules JSON Schema can't express (cycles, ref resolution,
  body scoping, mutually-exclusive LLMCall input modes).
- `tests/test_spec_grammar.py` — drift checks (every loader node type
  + every Compute op is documented in both the schema enum and the
  grammar doc; shipped sample validates against both).

### A. Visual editor

xyflow editor mode toggled from the Pipeline tab sidebar:

- Drag-from-palette to add a node.
- Click-and-drag to connect ports.
- Per-node config in the existing right side panel (re-using prompt/model
  editors where appropriate).
- "Test run" button: execute the in-progress spec with a sample input,
  show the trace inline.
- "Save" / "Discard" / "Diff vs current" actions.

### B. Non-destructive edits

When the user clicks "Save", the system does **not** overwrite the bound
pipeline's source. Instead:

1. Read the current pipeline source (Python module **or** YAML).
2. Compute a unique name like `<base>-fork-<timestamp>` or
   `<base>-{graph_id}-{short_hash}`.
3. Write a **copy** to a versioned location:
   - YAML: `{prompts_dir}/.history/{name}.pipeline.yaml`
   - Python: `data/pipelines/{name}/__init__.py` (full module)
4. Apply the user's edits to the copy.
5. Register the copy with the pipeline registry under the new name.
6. (Atomically) flip the graph's binding to the new name.
7. The previous version remains addressable for rollback.

Rollback: a new endpoint `POST /graphs/{gid}/pipeline/rollback` reverts
the graph's binding to the previous version in the chain. Audit log of
"who saved what when" lives in chroma settings collection (already exists
for trace cap; extend the schema).

### C. Python hot-reload

> "This means we need to be able to write and reload python code in real time."

Implementing (B) for Python-backed pipelines requires loading and
reloading Python modules at runtime without restarting the server.

**Approach**:

1. **Pipeline source layout**. Forked Python pipelines live under
   `data/pipelines/{name}/` as importable packages. `__init__.py` exposes
   a `Pipeline` class subclassing `MemoryPipeline`.
2. **Dynamic registration**. A bootstrap scan at app startup imports
   every package in `data/pipelines/` and registers it. New packages can
   be added between requests.
3. **Hot-reload endpoint**. `POST /api/v1/pipelines/reload` (or
   `/api/v1/pipelines/{name}/reload`) calls `importlib.reload` on the
   target package(s) and re-registers. Use this when:
   - A developer edits Python on disk (no server restart needed).
   - The visual editor just wrote a new fork.
4. **Source-of-truth choice per pipeline**. A pipeline can declare its
   storage format: `spec_yaml` (loaded each call) or `python` (loaded at
   import, reloadable on demand).
5. **Safety**: forked Python is sandboxed only to the extent that it
   imports from `plugmem.*`. We do **not** attempt to block arbitrary
   imports — researchers using their own server. Document the threat
   model clearly.

### Risks / open questions

- **Arbitrary code execution**. Forked Python runs in-process with full
  server privileges. Document and accept for the researcher use case;
  add a `PLUGMEM_ENABLE_PYTHON_FORKS` env flag (default off in production
  deployments).
- **Module-level state**. `LLMRouter`, `PromptRegistry` singletons are
  module-level. If a forked pipeline imports `plugmem.api.dependencies`,
  hot-reload must not stomp the singletons. Keep singletons in a small,
  rarely-reloaded module; forks import from there.
- **Stale references**. After `importlib.reload`, existing references
  to old classes still point at old code. The registry must re-fetch
  the class on reload.
- **Editor-to-Python mapping**. The visual editor produces YAML (Phase
  6.2-6.4 model). Generating Python from the same model = an additional
  emitter. Or: the editor only edits YAML, and the "Python fork" path is
  reserved for hand-written code with a separate ergonomics surface.
  TBD; lean toward "editor → YAML, Python fork is manual".

### Phasing

- **6.5a** — non-destructive edit infrastructure for spec-driven YAML
  (versioned filesystem layout, rollback endpoint, audit log).
- **6.5b** — visual editor MVP for retrieve-phase YAMLs (read-only canvas
  becomes interactive; "Save" creates a versioned YAML).
- **6.5c** — Python-backed forked pipelines + hot-reload.

---

## Phase 6.6 — Spec-driven for non-retrieve phases

Once `ForEach` (6.3) and `Branch` + `Compute` (6.4) exist, extend the
spec-driven YAML to model:

- `close` — outer loop over trajectories, inner loop over steps, calls
  `get_semantic` per step and `get_procedural` per trajectory.
- `insert` — loop over procedural memories with a `Branch` on "subgoal
  exists?".
- `consolidate` — nested loop with a `Branch` on the merge-threshold
  comparator.

For each phase, drop the corresponding `_default.<phase>()` delegation
in `SpecDrivenPipeline`.

---

## Cross-cutting roadmap items

| Item | Phase |
|---|---|
| Drift checks block CI | 6.0 (shipped) |
| Snapshot tool with `--check` | 6.1 (shipped) |
| YAML-defined retrieve | 6.2 (shipped) |
| Loops in YAML | 6.3 (in progress) |
| Branch + Compute | 6.4 (shipped) |
| Non-destructive YAML versioning + editor UI | 6.5a (shipped) |
| Visual editor (xyflow drag-and-drop) | 6.5b |
| Python hot-reload | 6.5c |
| Spec-driven for close / insert / consolidate | 6.6 |

## Decisions logged

- Spec is hand-curated for `phase`, `role`, `description`, `per`, `kind`,
  `edges`. AST-extractable fields (`prompt_name`, `inputs`, `outputs`)
  are guarded by the lint + snapshot tool.
- YAML lives alongside prompt overrides under `PROMPTS_DIR`.
- Trace recorder is the single backbone for every pipeline implementation
  (default, naive-rag, spec-driven). Adding a new pipeline gets the trace
  panel + sparklines for free.
- Pipeline binding is per-graph and persisted in the existing
  `{graph_id}_pipeline_settings` chroma collection.
- The "20 LLM calls per run" cap on `SpecDrivenPipeline` is hardcoded for
  the MVP. Configurable later if needed.
