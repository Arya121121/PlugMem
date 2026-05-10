# Pipeline Inspector — Plan

A new **Pipeline** tab in the inspector that:

1. Visualizes PlugMem's processing pipeline as a step-by-step diagram
   (subgoal → reward → state → segment → semantic → procedural, plus
   retrieval and consolidation paths).
2. Per step: view/edit the prompt template and pick the LLM model.
3. Shows recent traces of pipeline runs with the actual LLM I/O at each step.

## Resolved design decisions

- **Prompt persistence (Phase 2).** Edits persist to the per-graph YAML
  layer at `{prompts_dir}/{graph_id}.yaml`. The shipped builtins (Python
  classes in `plugmem/prompts/*.py`) and the service-wide
  `{prompts_dir}/_defaults.yaml` are **read-only from the UI** — the API
  must refuse writes to either. Reset = remove the entry from the per-graph
  YAML so resolution falls back to the builtin.
- **Runtime model swap (Phase 3).** Add `LLMRouter.set_role(role, cfg)`;
  takes effect on the next call from that role. The model editor must offer
  a **"Test connection"** action that issues a small one-off `complete()`
  call against the candidate `{base_url, api_key, model}` *before* the user
  commits the swap, and shows success/failure (latency + first chunk).
- **Trace capture (Phase 4).** Always-on. The retention cap is a
  **user-configurable per-graph setting** with a recommended default of
  **N=100** runs, persisted to chroma collection
  `{graph_id}_pipeline_trace`. Setting the cap to `0` means **unlimited**
  (intended for testing — log everything). Settings UI lives in the
  Pipeline tab.
- **Visualization library (Phase 1+).** **xyflow** (`@xyflow/react`) +
  React 18. Adopted now (not bespoke HTML) so the future "build your own
  memory pipeline" feature has a real node-editor foundation. Cytoscape
  stays reserved for the data graph.
- **Distribution.** Vendored: a one-time esbuild step in
  `vendor-build/` produces
  `plugmem/api/static/inspector/vendor/xyflow-bundle.{js,css}`, both
  committed. No runtime CDN, no build step in CI. Re-run on version bump.

---

## Phase 1 — Read-only pipeline diagram

Foundation: the new tab renders the pipeline shape from a single source of
truth. No editing yet.

**Backend**

- New `plugmem/core/pipeline_spec.py` — declarative list of steps
  `(id, label, prompt_name, role, inputs, outputs, phase)` for
  `append | close | retrieve | consolidate`.
- New `plugmem/api/routes/pipeline.py` with `GET /api/v1/pipeline/spec`.
  Wire into [app.py](../plugmem/api/app.py).
- Add `PipelineSpec` / `PipelineStep` to
  [schemas.py](../plugmem/api/schemas.py).

**Frontend**

- New "Pipeline" tab in
  [index.html](../plugmem/api/static/inspector/index.html); mount from
  [app.js](../plugmem/api/static/inspector/app.js).
- New `plugmem/api/static/inspector/pipeline.js`: phase columns, step cards,
  edges. Click a card → empty right-panel placeholder.
- Add `getPipelineSpec` to
  [api.js](../plugmem/api/static/inspector/api.js).

**Done when**: opening the tab shows the full pipeline DAG and the side
panel opens (empty) on click.

---

## Phase 2 — Prompt editing

Decision needed before starting: prompt persistence (see above).

**Backend**

- `GET    /api/v1/graphs/{graph_id}/pipeline/prompts` — list with builtin +
  override + effective text.
- `PUT    /api/v1/graphs/{graph_id}/pipeline/prompts/{name}` —
  `{system, user, persist?}`; updates `PromptRegistry.set` (graph layer);
  writes YAML if `persist`.
- `POST   /api/v1/graphs/{graph_id}/pipeline/prompts/{name}/reset` — clears
  override.
- `POST   /api/v1/graphs/{graph_id}/pipeline/prompts/{name}/preview` —
  renders `PromptBase.build_messages(variables)` without calling the LLM.

**Frontend**

- Side panel "Prompt" tab: editable system + user textareas, variable list,
  **Preview rendered**, Save / Reset / Save to YAML.
- Status badge on each step card: builtin / overridden.

**Done when**: editing a prompt changes the next pipeline run's behavior
without restart, and Reset restores the builtin.

---

## Phase 3 — Model swap per role

Decision needed: runtime model swap (see above).

**Backend**

- `LLMRouter.set_role(role, {base_url, api_key, model})` in
  [llm_router.py](../plugmem/clients/llm_router.py): atomically constructs
  and swaps the client.
- `GET /api/v1/pipeline/models` — current roles, model, base_url (no key).
- `PUT /api/v1/graphs/{graph_id}/pipeline/models/{role}` — body
  `{base_url, api_key, model}`.
- `POST /api/v1/pipeline/models/test` — body
  `{base_url, api_key, model, prompt?}`. Issues a single short `complete()`
  against the candidate config and returns
  `{ok, latency_ms, sample, error?}`. Does **not** mutate router state.

**Frontend**

- Side panel "Model" tab: role dropdown + base_url/model/api_key inputs.
- Step cards display the bound model in a badge.

**Done when**: changing a role's model in the UI causes the next call from
that role to hit the new endpoint.

---

## Phase 4 — Pipeline trace capture & display

Decision needed: trace capture cap & opt-in (see above).

**Backend**

- New `plugmem/core/pipeline_trace.py` with `PipelineTraceRecorder`
  (`start_run`, `step`, `finish`).
- Thread an optional recorder through
  [inference/structuring.py](../plugmem/inference/structuring.py) and
  [inference/retrieving.py](../plugmem/inference/retrieving.py); orchestrate
  from [core/memory.py](../plugmem/core/memory.py).
- New chroma collection `{graph_id}_pipeline_trace` mirroring the
  `add_recall` pattern in
  [storage/chroma.py](../plugmem/storage/chroma.py).
- `GET /api/v1/graphs/{graph_id}/pipeline/traces?limit=20`
- `GET /api/v1/graphs/{graph_id}/pipeline/traces/{trace_id}`

**Frontend**

- Bottom traces strip: timestamps + outcome. Click → expandable timeline of
  steps with input vars / raw response / parsed output / latency / model.

**Done when**: every ingest and retrieval call produces a trace visible and
inspectable in the UI.

---

## Phase 6 — Future: user-defined memory pipelines

Out of scope for the current PR — captured here so design decisions in
Phases 1-5 stay compatible with it.

- Let users **add / remove / reorder steps** and wire arbitrary dataflow.
- Each step = an executable node. Build a small dataflow executor on top
  of xyflow (~150-300 LoC: topo sort by edges → call each node's `compute`
  → propagate). Add caching / async / partial recompute as actually
  needed, instead of inheriting a generic engine.
- Persist pipeline shape per graph alongside prompt + model overrides.
- Versioning / templates so users can fork a pipeline.

## Phase 5 — Polish

Independent improvements; pick what's worth shipping.

- Diff-against-builtin view in the prompt editor.
- Variable placeholder highlighting in textareas.
- "Recent calls" tab inside the side panel (filtered traces for that step).
- Latency / token sparklines per step on the diagram.
- "Re-run trace with current prompts" button on a past trace.

---

## Out of scope

- Editing the pipeline shape itself (adding/reordering steps).
- Multi-tenant prompt versioning / history (only current + reset-to-builtin).
- Cross-harness sharing of prompt overrides — graphs are isolated per
  harness.
