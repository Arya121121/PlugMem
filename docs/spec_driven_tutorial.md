# Authoring a custom memory pipeline (tutorial)

This walkthrough builds a working custom retrieve pipeline from scratch
using the spec-driven YAML grammar — no Python edits, no server
restarts. By the end you'll have a per-graph pipeline that runs against
your data and produces a real LLM-generated answer.

If you want the reference instead of the walkthrough, see:

- [`spec_driven_yaml.md`](spec_driven_yaml.md) — reader-oriented reference.
- [`spec_grammar.md`](spec_grammar.md) — normative grammar (EBNF + JSON Schema).

---

## Prerequisites

- The PlugMem server running locally:

  ```bash
  uvicorn plugmem.api.app:create_app --factory --reload
  ```

- A graph to experiment on. You can create one via the API:

  ```bash
  curl -X POST localhost:8000/api/v1/graphs \
       -H 'content-type: application/json' \
       -d '{"graph_id": "tutorial"}'
  ```

  Or click **Load demo graph** at the top of `http://localhost:8000/inspector/`.

- (Optional but recommended) Seed some memories so retrieve has
  something to find:

  ```bash
  curl -X POST localhost:8000/api/v1/graphs/tutorial/memories \
       -H 'content-type: application/json' \
       -d '{
         "mode": "structured",
         "semantic": [
           {"semantic_memory": "FastAPI uses Pydantic for validation.",
            "tags": ["fastapi", "python"]},
           {"semantic_memory": "Docker packages apps for deployment.",
            "tags": ["docker", "deployment"]}
         ]
       }'
  ```

---

## 1. Bind the graph to the spec-driven pipeline

The bindings determines which `MemoryPipeline` runs for the graph. Open
the **Pipeline** tab in the inspector, then change **Bound pipeline**
from `plugmem-default` to `spec-driven`.

Equivalent via API:

```bash
curl -X PUT localhost:8000/api/v1/graphs/tutorial/pipeline \
     -H 'content-type: application/json' \
     -d '{"pipeline": "spec-driven"}'
```

The Spec editor panel appears above the canvas. The canvas shows a
placeholder ("No YAML saved — use the Spec editor below.") because the
graph has no pipeline YAML yet.

---

## 2. Start from the shipped sample

In the editor's **Start from a sample** row, leave
`plugmem-default retrieve + reasoning` selected and click **Insert
template**. The textarea fills with a YAML pipeline functionally
equivalent to `PlugMemDefaultPipeline.retrieve`.

Click **Validate** — you should see "Valid." Click **Save new version**
— the canvas redraws, showing the full DAG from `Input` through
`get_plan` / `get_mode`, the three Branch-gated reasoning templates,
and finally `reason_llm_call` flowing into `Output`.

Run it against your seeded data:

```bash
curl -X POST localhost:8000/api/v1/graphs/tutorial/retrieve \
     -H 'content-type: application/json' \
     -d '{"observation": "How do we deploy?", "goal": "ship"}'
```

You should get back:

```json
{
  "mode": "semantic_memory",
  "reasoning_prompt": [],
  "variables": { "text": "<LLM answer>" }
}
```

`variables.text` is the reasoning LLM's response.

`POST /reason` also works on a spec-driven graph — it runs the same
YAML and reshapes the output to the `ReasonResponse` schema (`mode` +
`reasoning` + `reasoning_prompt`).

---

## 3. Make your first customization

Switch the **retrieval** model from `default` to a different
LLMRouter role on a single node. In the textarea, find:

```yaml
  - id: get_plan
    type: LLMCall
    config: { prompt: get_plan, role: retrieval }
```

Add a `note:` to the Save dialog so you can find this version later
("trying structuring role for get_plan"), change `role: retrieval` to
`role: structuring`, and click **Save new version**.

Open **Version history** — you'll see two rows now. The new one is
**active** (left-bordered green); the original is one click away via
**Rollback**.

Try a few retrieves. If anything looks broken, rollback. The bound
pipeline source on disk is updated atomically every save, so the next
`/retrieve` call uses the new spec immediately — no server restart.

---

## 4. Author from scratch with the palette

For a custom pipeline, click **Discard changes** to clear the editor
(the active version stays on disk), then write your own minimal
pipeline.

Open the **Insert a node…** palette. Each chip drops a stub YAML
fragment at your cursor. Build the smallest possible pipeline:

1. Click **Input** — pastes `- { id: in, type: Input }`.
2. Type the top-level skeleton above it:

   ```yaml
   phase: retrieve

   nodes:
   ```

3. Click **PromptRender** — pastes a `my_render` stub. Fill in
   `prompt: reasoning_semantic` and wire its inputs:

   ```yaml
     - id: my_render
       type: PromptRender
       config: { prompt: reasoning_semantic }
       inputs:
         semantic_memory: { const: "FastAPI uses Pydantic." }
         time: { const: "" }
         observation: in.observation
   ```

4. Click **LLMCall · text-mode** — pastes a `my_reasoning_call` stub.
   Wire it to your template:

   ```yaml
     - id: my_reasoning_call
       type: LLMCall
       config: { role: reasoning }
       inputs:
         text: my_render.value
   ```

5. Click **Output** — pastes the stub. Wire it:

   ```yaml
     - id: out
       type: Output
       inputs:
         mode: { const: semantic_memory }
         reasoning_prompt: { const: [] }
         variables: my_reasoning_call.parsed
   ```

**Validate**, **Save new version**, and run `/retrieve`. The canvas
shows your hand-authored DAG: a single Template feeding a single LLM
into the Output.

---

## 5. When something's wrong

The validator gives precise errors. Try saving an invalid spec:

```yaml
phase: retrieve
nodes:
  - { id: in, type: Input }
  - id: ghost
    type: LLMCall
    config: { role: reasoning }
    inputs:
      text: nonexistent.value
  - { id: out, type: Output, inputs: { mode: { const: x }, reasoning_prompt: { const: [] }, variables: { const: {} } } }
```

You'll see a 422 with the exact message:

```
top-level: node 'ghost' input 'text': references unknown node 'nonexistent'
```

Common rejections the validator catches:

- Cycles (`load_yaml`: "Pipeline has a cycle")
- Self-references (`Node 'x' references itself`)
- Wrong node type / unknown Compute op
- LLMCall with both `config.prompt` AND `inputs.text` (or no input mode at all)
- StorageRead with `collection: tags` (not a valid collection)

The runtime adds:

- LLM call cap (`LLM_CALL_CAP = 20` per pipeline run)
- ForEach iteration cap (`MAX_LOOP_ITERATIONS = 1000` per instance)

---

## 6. Going deeper

| You want to… | Look here |
|---|---|
| Browse every node type's schema | [`spec_grammar.md`](spec_grammar.md) |
| Read a reader-oriented overview | [`spec_driven_yaml.md`](spec_driven_yaml.md) |
| Replace plugmem-default end-to-end | `plugmem/pipelines/sample_specs.py` — `PLUGMEM_DEFAULT_RETRIEVE_YAML` |
| Verify your spec matches plugmem-default | `tests/test_pipeline_equivalence.py` (differential harness) |
| Understand the bigger roadmap | [`pipeline_inspector_plan.md`](pipeline_inspector_plan.md) |
| Use a JSON Schema-aware editor | `plugmem/pipelines/spec_schema.json` |

---

## What's not supported yet

| Capability | Status |
|---|---|
| `close` / `insert` / `consolidate` phases as YAML | Phase 6.6b-2 (future) |
| Drag-and-drop canvas editor | Phase 6.5b (future) |
| Python hot-reload for forked pipelines | Phase 6.5c (future) |
| `Switch` (n-way branch), `filter_by`, `top_k`, `similarity` ops | future |

Those phases / endpoints currently delegate to `PlugMemDefaultPipeline`.
`/memories` and `/consolidate` ignore the YAML; `/retrieve` and `/reason`
run it.
