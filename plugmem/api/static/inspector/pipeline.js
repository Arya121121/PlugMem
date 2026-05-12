// Pipeline tab — exact, accurate xyflow visualization of PlugMem.
//
// Renders every LLM call, embedding, compute, storage write, branch, and
// loop boundary that the production code path actually executes, organised
// by phase (append / close / insert / retrieve / reason / consolidate).
//
// Per-phase layout is computed with dagre; phases are laid out left-to-right.
// React + xyflow + dagre live in ./vendor/xyflow-bundle.js.

import { api } from "./api.js";

const PHASE_GAP = 120;            // px between phase columns
const STAGE_OFFSET_Y = 120;       // stage trigger sits above the phase entry
const NODE_DIMS = {               // [width, height] used by dagre — actual
  llm:               [280, 200],  // rendered height is css-controlled
  template_render:   [280, 160],
  embed:             [240, 100],
  compute:           [240, 100],
  storage:           [240, 100],
  branch:            [300, 130],
  loop_marker:       [220, 80],
};

const PER_LABELS = {
  per_step: "per step",
  per_trajectory: "per trajectory",
  per_call: "per call",
};

const EDITABLE_KINDS = ["llm", "template_render"];
const ALL_KINDS = ["llm", "template_render", "embed", "compute", "storage", "branch", "loop_marker"];


// ------------------------- helpers (pure, no closure deps) ------------------ //

function _escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function lineDiff(a, b) {
  const al = (a || "").split("\n");
  const bl = (b || "").split("\n");
  const m = al.length, n = bl.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = al[i - 1] === bl[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  const out = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && al[i - 1] === bl[j - 1]) {
      out.push({ op: "=", line: al[i - 1] }); i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      out.push({ op: "+", line: bl[j - 1] }); j--;
    } else {
      out.push({ op: "-", line: al[i - 1] }); i--;
    }
  }
  out.reverse();
  return out;
}

function renderDiffHtml(before, after) {
  const diff = lineDiff(before, after);
  if (diff.every((d) => d.op === "=")) {
    return `<div class="hint">No differences from builtin.</div>`;
  }
  return diff.map((d) => {
    const cls = d.op === "+" ? "diff-add" : d.op === "-" ? "diff-del" : "diff-eq";
    const sign = d.op === "+" ? "+" : d.op === "-" ? "-" : " ";
    return `<div class="diff-row ${cls}"><span class="diff-sign">${sign}</span><span class="diff-line">${_escapeHtml(d.line) || "&nbsp;"}</span></div>`;
  }).join("");
}

const KIND_LABELS = {
  llm: "LLM",
  template_render: "TEMPLATE",
  embed: "EMBED",
  compute: "COMPUTE",
  storage: "STORAGE",
  branch: "BRANCH",
  loop_marker: "LOOP",
};

let xyflowModP = null;
async function loadXyflow() {
  if (!xyflowModP) {
    xyflowModP = import("./vendor/xyflow-bundle.js");
  }
  return xyflowModP;
}

export function mountPipeline({ container, getGraphId, toast }) {
  const els = {
    canvas: container.querySelector("#pipeline-canvas"),
    empty: container.querySelector("#pipeline-empty"),
    phaseList: container.querySelector("#pipeline-phase-list"),
    roleLegend: container.querySelector("#pipeline-role-legend"),
    detail: container.querySelector("#pipeline-detail"),
    detailTitle: container.querySelector("#pipeline-detail-title"),
    detailBody: container.querySelector("#pipeline-detail-body"),
    detailClose: container.querySelector("#pipeline-detail-close"),
    tracesList: container.querySelector("#pipeline-traces-list"),
    tracesRefresh: container.querySelector("#pipeline-traces-refresh"),
    tracesCap: container.querySelector("#pipeline-traces-cap"),
    tracesCapSave: container.querySelector("#pipeline-traces-cap-save"),
    bindingSelect: container.querySelector("#pipeline-binding-select"),
    bindingDesc: container.querySelector("#pipeline-binding-desc"),
    specEditor: container.querySelector("#pipeline-spec-editor"),
    specEditorPath: container.querySelector("#pipeline-spec-editor-path"),
    specEditorActive: container.querySelector("#pipeline-spec-editor-active"),
    specEditorToggle: container.querySelector("#pipeline-spec-editor-toggle"),
    specEditorTextarea: container.querySelector("#pipeline-spec-editor-textarea"),
    specEditorNote: container.querySelector("#pipeline-spec-editor-note"),
    specEditorValidate: container.querySelector("#pipeline-spec-editor-validate"),
    specEditorDiscard: container.querySelector("#pipeline-spec-editor-discard"),
    specEditorSave: container.querySelector("#pipeline-spec-editor-save"),
    specEditorStatus: container.querySelector("#pipeline-spec-editor-status"),
    specVersionsList: container.querySelector("#pipeline-spec-versions-list"),
    specSampleSelect: container.querySelector("#pipeline-spec-sample-select"),
    specSampleInsert: container.querySelector("#pipeline-spec-sample-insert"),
  };

  let spec = null;
  let loaded = false;
  let reactRoot = null;
  let xyflowMod = null;
  let visibleKinds = new Set(ALL_KINDS);
  let currentGraphId = null;
  const promptInfoByName = new Map();
  const bindingByRole = new Map();
  const statsByName = new Map();
  let activePromptStep = null;
  // Phase 6.5a: pipeline binding + spec editor state
  let availablePipelines = [];
  let currentBinding = null;
  let savedSpecContent = "";
  let savedSpecVersionId = null;
  let savedSpecPath = "";
  let availableSamples = [];

  els.detailClose.addEventListener("click", () => {
    els.detail.hidden = true;
  });

  els.tracesRefresh?.addEventListener("click", async () => {
    await Promise.all([refreshTraces(), refreshStepStats()]);
    if (xyflowMod) mountReactApp(xyflowMod);
  });
  els.tracesCapSave?.addEventListener("click", async () => {
    const gid = getGraphId();
    if (!gid) return;
    const cap = parseInt(els.tracesCap.value, 10);
    if (Number.isNaN(cap) || cap < 0) {
      toast("Cap must be a non-negative integer.", "warn");
      return;
    }
    try {
      const r = await api.setTraceCap(gid, cap);
      toast(`Trace cap → ${r.cap === 0 ? "unlimited" : r.cap}`);
    } catch (err) {
      toast(`Save cap: ${err.message}`, "error");
    }
  });

  const foldToggle = container.querySelector("#pipeline-fold-toggle");
  if (foldToggle) {
    foldToggle.addEventListener("change", () => {
      visibleKinds = foldToggle.checked
        ? new Set(EDITABLE_KINDS)
        : new Set(ALL_KINDS);
      if (xyflowMod && spec) mountReactApp(xyflowMod);
    });
  }

  async function fetchSpec() {
    const gid = getGraphId();
    if (gid) {
      try {
        return await api.getPipelineSpecView(gid);
      } catch (err) {
        console.warn("spec_view failed, falling back to static spec:", err);
      }
    }
    return await api.getPipelineSpec();
  }

  async function load() {
    if (loaded) return;
    els.empty.hidden = false;
    els.empty.textContent = "Loading pipeline…";
    try {
      const [s, mod] = await Promise.all([fetchSpec(), loadXyflow()]);
      spec = s;
      xyflowMod = mod;
      currentGraphId = getGraphId();
      await Promise.all([
        refreshPromptInfo(),
        refreshModelInfo(),
        refreshStepStats(),
        refreshAvailablePipelines(),
        refreshSamples(),
      ]);
      renderSidebar();
      void refreshTraces();
      void refreshBinding();
      mountReactApp(mod);
      els.empty.hidden = true;
      loaded = true;
    } catch (err) {
      els.empty.hidden = false;
      els.empty.textContent = `Error: ${err.message}`;
      toast(`pipeline: ${err.message}`, "error");
    }
  }

  async function refreshPromptInfo() {
    promptInfoByName.clear();
    const gid = getGraphId();
    if (!gid) return;
    try {
      const res = await api.listPipelinePrompts(gid);
      for (const p of res.prompts || []) promptInfoByName.set(p.name, p);
    } catch (err) {
      // Fail soft: missing override info just means no badges + no editor.
      console.warn("listPipelinePrompts failed:", err);
    }
  }

  async function refreshModelInfo() {
    bindingByRole.clear();
    try {
      const res = await api.listPipelineModels();
      for (const b of res.bindings || []) bindingByRole.set(b.role, b);
    } catch (err) {
      console.warn("listPipelineModels failed:", err);
    }
  }

  async function refreshStepStats() {
    statsByName.clear();
    const gid = getGraphId();
    if (!gid) return;
    try {
      const res = await api.getPipelineStats(gid);
      for (const [name, s] of Object.entries(res.stats || {})) {
        statsByName.set(name, s);
      }
    } catch (err) {
      console.warn("getPipelineStats failed:", err);
    }
  }

  async function refreshTraces() {
    if (!els.tracesList) return;
    const gid = getGraphId();
    if (!gid) {
      els.tracesList.innerHTML = `<div class="hint">No graph selected.</div>`;
      return;
    }
    els.tracesList.innerHTML = `<div class="hint">Loading…</div>`;
    try {
      const res = await api.listPipelineTraces(gid, { limit: 25 });
      if (els.tracesCap && typeof res.cap === "number") {
        els.tracesCap.value = String(res.cap);
      }
      if (!res.traces || res.traces.length === 0) {
        els.tracesList.innerHTML = `<div class="hint">No runs yet — call /memories, /retrieve, /reason, or /consolidate to generate one.</div>`;
        return;
      }
      els.tracesList.innerHTML = "";
      for (const t of res.traces) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = `pipeline-trace-row ${t.ok ? "ok" : "fail"}`;
        row.dataset.traceId = t.trace_id;
        row.innerHTML = `
          <div class="pipeline-trace-row-head">
            <span class="pipeline-trace-endpoint">${escapeHtml(t.endpoint)}</span>
            <span class="hint">${escapeHtml(t.ts || "")}</span>
          </div>
          <div class="pipeline-trace-row-meta">
            <span class="pipeline-trace-steps">${t.num_steps} ${t.num_steps === 1 ? "step" : "steps"}</span>
            <span class="hint">${t.duration_ms} ms</span>
            ${t.ok ? "" : `<span class="pipeline-card-flag" title="failed">err</span>`}
          </div>
        `;
        row.addEventListener("click", () => selectTrace(t.trace_id));
        els.tracesList.appendChild(row);
      }
    } catch (err) {
      els.tracesList.innerHTML = `<div class="hint">Error: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function selectTrace(traceId) {
    const gid = getGraphId();
    if (!gid) return;
    activePromptStep = null;
    els.detail.hidden = false;
    els.detailTitle.textContent = "Run trace";
    els.detailBody.innerHTML = `<div class="hint">Loading trace…</div>`;
    try {
      const t = await api.getPipelineTrace(gid, traceId);
      renderTraceDetail(t);
    } catch (err) {
      els.detailBody.innerHTML = `<div class="prompt-error">${escapeHtml(err.message)}</div>`;
    }
  }

  function renderTraceDetail(t) {
    els.detailTitle.textContent = `Run · ${t.endpoint}`;
    const totalMs = t.duration_ms || 0;
    let stepsHtml = "";
    for (const s of t.steps || []) {
      const widthPct = totalMs > 0 ? Math.max(2, Math.round((s.latency_ms / totalMs) * 100)) : 0;
      const offsetPct = totalMs > 0 ? Math.round((s.ts_offset_ms / totalMs) * 100) : 0;
      const parsedJson = s.parsed != null ? JSON.stringify(s.parsed, null, 2) : null;
      const varsJson = s.variables ? JSON.stringify(s.variables, null, 2) : "{}";
      stepsHtml += `
        <details class="trace-step ${s.error ? "fail" : "ok"}">
          <summary>
            <span class="trace-step-name">${escapeHtml(s.name)}</span>
            <span class="trace-step-bar"><span class="trace-step-bar-fill" style="margin-left:${offsetPct}%; width:${widthPct}%;"></span></span>
            <span class="hint">${s.latency_ms} ms${s.model ? ` · ${escapeHtml(s.model)}` : ""}</span>
            ${s.error ? `<span class="pipeline-card-flag">err</span>` : ""}
          </summary>
          <div class="trace-step-body">
            <div class="trace-step-block"><div class="prompt-block-label">Variables</div><pre class="prompt-readonly">${escapeHtml(varsJson)}</pre></div>
            ${parsedJson != null ? `<div class="trace-step-block"><div class="prompt-block-label">Parsed</div><pre class="prompt-readonly">${escapeHtml(parsedJson)}</pre></div>` : ""}
            <div class="trace-step-block"><div class="prompt-block-label">Raw response</div><pre class="prompt-readonly">${escapeHtml(s.response || "")}</pre></div>
            ${s.error ? `<div class="prompt-error">${escapeHtml(s.error)}</div>` : ""}
          </div>
        </details>
      `;
    }
    els.detailBody.innerHTML = `
      <dl class="pipeline-kv">
        <dt>Trace ID</dt><dd><code>${escapeHtml(t.trace_id)}</code></dd>
        <dt>Endpoint</dt><dd><code>${escapeHtml(t.endpoint)}</code></dd>
        <dt>Started</dt><dd>${escapeHtml(t.ts || "")}</dd>
        <dt>Duration</dt><dd>${t.duration_ms} ms</dd>
        <dt>LLM steps</dt><dd>${t.num_steps}</dd>
        <dt>Status</dt><dd>${t.ok ? `<span class="pipeline-trace-ok">ok</span>` : `<span class="pipeline-trace-fail">failed</span>`}</dd>
        ${t.session_id ? `<dt>Session</dt><dd><code>${escapeHtml(t.session_id)}</code></dd>` : ""}
        ${t.error ? `<dt>Error</dt><dd class="pipeline-trace-fail">${escapeHtml(t.error)}</dd>` : ""}
      </dl>
      <div class="trace-actions">
        <button type="button" class="btn trace-rerun">Re-render with current prompts</button>
        <span class="hint">Renders each step's prompt template against the original variables. No LLM calls.</span>
      </div>
      <div class="pipeline-detail-section"><strong>Step timeline</strong>${stepsHtml || `<div class="hint">No LLM steps recorded.</div>`}</div>
    `;

    const rerunBtn = els.detailBody.querySelector(".trace-rerun");
    rerunBtn?.addEventListener("click", () => { void rerunTrace(t); });
  }

  async function rerunTrace(t) {
    if (!t || !Array.isArray(t.steps)) return;
    const gid = getGraphId();
    if (!gid) { toast("Pick a graph first.", "warn"); return; }
    if (!spec) { toast("Pipeline spec not loaded.", "warn"); return; }

    // step.name → prompt_name (for both kind=llm and kind=template_render).
    const promptByStepName = new Map();
    for (const s of spec.steps) {
      if (s.prompt_name) promptByStepName.set(s.id, s.prompt_name);
    }

    const stepEls = els.detailBody.querySelectorAll(".trace-step");
    for (let i = 0; i < t.steps.length; i++) {
      const step = t.steps[i];
      const stepEl = stepEls[i];
      if (!stepEl) continue;

      const promptName = promptByStepName.get(step.name);
      let block = stepEl.querySelector(".trace-rerun-out");
      if (!block) {
        block = document.createElement("div");
        block.className = "trace-rerun-out trace-step-block";
        stepEl.querySelector(".trace-step-body")?.appendChild(block);
      }

      if (!promptName) {
        block.innerHTML = `<div class="prompt-block-label">Re-rendered (current prompts)</div><div class="hint">No registered prompt for <code>${escapeHtml(step.name)}</code> — the LLM call uses messages assembled upstream (e.g. reason_llm_call).</div>`;
        continue;
      }

      block.innerHTML = `<div class="prompt-block-label">Re-rendered (current prompts)</div><div class="hint">Loading…</div>`;
      try {
        const res = await api.previewPipelinePrompt(gid, promptName, {
          variables: step.variables || {},
        });
        let msgs = "";
        for (const m of res.messages || []) {
          msgs += `
            <div class="prompt-preview-msg">
              <div class="prompt-preview-role">${escapeHtml(m.role)}</div>
              <pre class="prompt-preview-content">${escapeHtml(m.content)}</pre>
            </div>`;
        }
        block.innerHTML = `<div class="prompt-block-label">Re-rendered (current prompts)</div>${msgs || `<div class="hint">No messages returned.</div>`}`;
      } catch (err) {
        block.innerHTML = `<div class="prompt-block-label">Re-rendered (current prompts)</div><div class="prompt-error">${escapeHtml(err.message)}</div>`;
      }
    }
  }

  function renderSidebar() {
    els.phaseList.innerHTML = "";
    els.roleLegend.innerHTML = "";
    for (const phase of spec.phases) {
      const summary = document.createElement("div");
      summary.className = "pipeline-phase-summary";
      const stepCount = spec.steps.filter((s) => s.phase === phase.id).length;
      const triggeredBy = (phase.triggered_by || []).map((t) => `<code>${escapeHtml(t)}</code>`).join(" ");
      summary.innerHTML = `
        <div class="pipeline-phase-summary-head">
          <strong>${escapeHtml(phase.label)}</strong>
          <span class="pipeline-phase-count">${stepCount}</span>
        </div>
        <div class="hint">${escapeHtml(phase.description)}</div>
        ${phase.trigger ? `<div class="pipeline-phase-trigger">⚡ ${escapeHtml(phase.trigger)}</div>` : ""}
        ${triggeredBy ? `<div class="pipeline-phase-triggered-by">${triggeredBy}</div>` : ""}
      `;
      els.phaseList.appendChild(summary);
    }
    for (const role of spec.roles) {
      const chip = document.createElement("span");
      chip.className = "pipeline-role-badge";
      chip.dataset.role = role;
      chip.textContent = role;
      els.roleLegend.appendChild(chip);
    }
    if (spec.kinds && spec.kinds.length) {
      const sep = document.createElement("div");
      sep.className = "pipeline-legend-section";
      sep.textContent = "Step kinds";
      els.roleLegend.appendChild(sep);
      for (const k of spec.kinds) {
        const chip = document.createElement("span");
        chip.className = `pipeline-kind-chip kind-${k}`;
        chip.textContent = KIND_LABELS[k] || k;
        els.roleLegend.appendChild(chip);
      }
    }
  }

  function mountReactApp(mod) {
    const {
      React,
      ReactDOMClient,
      ReactFlow,
      Background,
      Controls,
      MiniMap,
      Handle,
      Position,
      MarkerType,
      dagre,
    } = mod;
    const e = React.createElement;
    const { useMemo, useCallback } = React;

    const handleStyle = {
      background: "transparent",
      border: "none",
      width: 1,
      height: 1,
      pointerEvents: "none",
    };
    const Hin = e(Handle, { type: "target", position: Position.Top, isConnectable: false, style: handleStyle });
    const Hout = e(Handle, { type: "source", position: Position.Bottom, isConnectable: false, style: handleStyle });

    function ioBlock(data) {
      const row = (which, label, vars) =>
        e("div", { className: `pipeline-io-row pipeline-io-${which}` },
          e("span", { className: "pipeline-io-prefix" }, label),
          e("span", { className: "pipeline-io-vars" },
            (vars && vars.length)
              ? vars.map((v, i) =>
                  e(React.Fragment, { key: v + i },
                    i > 0 ? e("span", { className: "pipeline-io-sep" }, "·") : null,
                    e("code", { className: "pipeline-io-var" }, v),
                  ),
                )
              : e("span", { className: "pipeline-io-empty" }, "—"),
          ),
        );
      return e("div", { className: "pipeline-io" },
        row("in", "in", data.inputs),
        row("out", "out", data.outputs),
      );
    }

    function renderSparkline(e, latencies) {
      if (!latencies || latencies.length === 0) return null;
      const w = 80;
      const h = 18;
      const max = Math.max(1, ...latencies);
      const slot = w / latencies.length;
      const barW = Math.max(2, slot - 1);
      return e("svg", {
        className: "pipeline-sparkline",
        width: w, height: h,
        viewBox: `0 0 ${w} ${h}`,
        preserveAspectRatio: "none",
        "aria-hidden": "true",
        title: `recent latencies: ${latencies.map((v) => v + "ms").join(", ")}`,
      }, latencies.map((v, i) => {
        const barH = Math.max(1, Math.round((v / max) * (h - 2)));
        const x = i * slot;
        const y = h - barH;
        return e("rect", {
          key: i, x, y, width: barW, height: barH,
          fill: "currentColor",
        });
      }));
    }

    function StepNode({ data, selected }) {
      const cls = ["pipeline-card", `kind-${data.kind}`,
                   selected ? "is-selected" : "",
                   data.optional ? "is-optional" : "",
                   data.has_graph_override ? "is-overridden" : ""].filter(Boolean).join(" ");
      return e("div", { className: cls },
        Hin,
        e("div", { className: "pipeline-card-row" },
          e("span", { className: "pipeline-card-label" }, data.label),
          data.role
            ? e("span", { className: "pipeline-role-badge", "data-role": data.role }, data.role)
            : e("span", { className: `pipeline-kind-chip kind-${data.kind}` }, KIND_LABELS[data.kind] || data.kind),
        ),
        e("div", { className: "pipeline-card-desc" }, data.description),
        ioBlock(data),
        e("div", { className: "pipeline-card-meta" },
          data.prompt_name
            ? e("code", { className: "pipeline-card-prompt" }, data.prompt_name)
            : null,
          data.per && data.per !== "per_call"
            ? e("span", { className: "pipeline-card-per" }, PER_LABELS[data.per] || data.per)
            : null,
          data.optional ? e("span", { className: "pipeline-card-flag" }, "optional") : null,
          data.has_graph_override ? e("span", { className: "pipeline-override-flag" }, "overridden") : null,
        ),
        data.stats && data.stats.count > 0
          ? e("div", { className: "pipeline-card-stats" },
              e("span", { className: "pipeline-stat" },
                e("span", { className: "pipeline-stat-label" }, "calls"),
                e("span", { className: "pipeline-stat-val" }, String(data.stats.count)),
              ),
              e("span", { className: "pipeline-stat" },
                e("span", { className: "pipeline-stat-label" }, "avg"),
                e("span", { className: "pipeline-stat-val" }, `${data.stats.mean_latency_ms}ms`),
              ),
              data.stats.errors > 0
                ? e("span", { className: "pipeline-stat is-err" },
                    e("span", { className: "pipeline-stat-label" }, "err"),
                    e("span", { className: "pipeline-stat-val" }, String(data.stats.errors)),
                  )
                : null,
              renderSparkline(e, data.stats.recent_latencies),
            )
          : null,
        Hout,
      );
    }

    function MiniNode({ data, selected }) {
      const cls = ["pipeline-mini", `kind-${data.kind}`,
                   selected ? "is-selected" : "",
                   data.optional ? "is-optional" : ""].filter(Boolean).join(" ");
      return e("div", { className: cls },
        Hin,
        e("div", { className: "pipeline-mini-row" },
          e("span", { className: `pipeline-kind-chip kind-${data.kind}` }, KIND_LABELS[data.kind] || data.kind),
          e("span", { className: "pipeline-mini-label" }, data.label),
          data.optional ? e("span", { className: "pipeline-card-flag" }, "opt") : null,
        ),
        e("div", { className: "pipeline-mini-desc" }, data.description),
        Hout,
      );
    }

    function BranchNode({ data, selected }) {
      const cls = ["pipeline-branch", selected ? "is-selected" : ""].filter(Boolean).join(" ");
      return e("div", { className: cls },
        Hin,
        e("div", { className: "pipeline-branch-head" },
          e("span", { className: "pipeline-branch-icon", "aria-hidden": "true" }, "◇"),
          e("span", { className: "pipeline-branch-label" }, data.label),
        ),
        data.branch_condition
          ? e("div", { className: "pipeline-branch-cond" }, data.branch_condition)
          : null,
        (data.branch_outcomes && data.branch_outcomes.length)
          ? e("ul", { className: "pipeline-branch-outcomes" },
              data.branch_outcomes.map((o, i) =>
                e("li", { key: i }, o),
              ),
            )
          : null,
        Hout,
      );
    }

    function LoopNode({ data, selected }) {
      const cls = ["pipeline-loop", selected ? "is-selected" : ""].filter(Boolean).join(" ");
      return e("div", { className: cls },
        Hin,
        e("div", { className: "pipeline-loop-row" },
          e("span", { className: "pipeline-loop-icon", "aria-hidden": "true" }, "↻"),
          e("span", { className: "pipeline-loop-label" }, data.label),
        ),
        data.loop_scope
          ? e("div", { className: "pipeline-loop-scope" }, data.loop_scope)
          : null,
        Hout,
      );
    }

    function StageNode({ data }) {
      return e("div", { className: "pipeline-stage", "data-phase": data.phase },
        e("div", { className: "pipeline-stage-row" },
          e("span", { className: "pipeline-stage-icon", "aria-hidden": "true" }, "⚡"),
          e("span", { className: "pipeline-stage-label" }, data.label),
        ),
        data.trigger
          ? e("div", { className: "pipeline-stage-when" }, data.trigger)
          : null,
        (data.triggered_by && data.triggered_by.length)
          ? e("div", { className: "pipeline-stage-by" },
              data.triggered_by.map((t, i) =>
                e("code", { key: i, className: "pipeline-stage-by-chip" }, t),
              ),
            )
          : null,
        e(Handle, { type: "source", position: Position.Bottom, isConnectable: false, style: handleStyle }),
      );
    }

    const nodeTypes = {
      llm: StepNode,
      template_render: StepNode,
      embed: MiniNode,
      compute: MiniNode,
      storage: MiniNode,
      branch: BranchNode,
      loop_marker: LoopNode,
      stage: StageNode,
    };

    const { nodes, edges } = buildGraph(spec, dagre, MarkerType, visibleKinds, promptInfoByName, statsByName);

    function App() {
      const onNodeClick = useCallback((_, node) => {
        if (node.data?.step) selectStep(node.data.step);
      }, []);

      return e(ReactFlow, {
        nodes,
        edges,
        nodeTypes,
        onNodeClick,
        nodesDraggable: false,
        nodesConnectable: false,
        edgesFocusable: false,
        elementsSelectable: true,
        fitView: true,
        fitViewOptions: { padding: 0.1 },
        minZoom: 0.15,
        maxZoom: 1.5,
        defaultEdgeOptions: { animated: false },
      },
        e(Background, { gap: 24, size: 1 }),
        e(Controls, { showInteractive: false }),
        e(MiniMap, { pannable: true, zoomable: true, ariaLabel: "Pipeline mini-map" }),
      );
    }

    if (reactRoot) reactRoot.unmount();
    reactRoot = ReactDOMClient.createRoot(els.canvas);
    reactRoot.render(e(App));
  }

  function selectStep(step) {
    els.detail.hidden = false;
    els.detailTitle.textContent = step.label;
    const inputs = step.inputs.length
      ? step.inputs.map((v) => `<code>${escapeHtml(v)}</code>`).join(" ")
      : `<span class="hint">—</span>`;
    const outputs = step.outputs.length
      ? step.outputs.map((v) => `<code>${escapeHtml(v)}</code>`).join(" ")
      : `<span class="hint">—</span>`;

    let kindLine = "";
    if (step.kind === "llm" || step.kind === "template_render") {
      kindLine = `<dt>Prompt</dt><dd>${step.prompt_name ? `<code>${escapeHtml(step.prompt_name)}</code>` : `<span class="hint">— (uses messages from upstream template)</span>`}</dd>`;
      if (step.role) {
        kindLine += `<dt>Role</dt><dd><span class="pipeline-role-badge" data-role="${escapeAttr(step.role)}">${escapeHtml(step.role)}</span></dd>`;
      }
    }
    let extra = "";
    if (step.kind === "branch") {
      extra = `<div class="pipeline-detail-section"><strong>Condition</strong><div>${escapeHtml(step.branch_condition || step.description)}</div></div>`;
      if (step.branch_outcomes?.length) {
        extra += `<div class="pipeline-detail-section"><strong>Outcomes</strong><ul>${step.branch_outcomes.map((o) => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
      }
    }
    if (step.kind === "loop_marker" && step.loop_scope) {
      extra = `<div class="pipeline-detail-section"><strong>Loop scope</strong><div>${escapeHtml(step.loop_scope)}</div></div>`;
    }

    let editorHtml = "";
    const isEditable = (step.kind === "llm" || step.kind === "template_render") && !!step.prompt_name;
    if (isEditable) {
      editorHtml = renderPromptEditorHtml(step);
      activePromptStep = step;
    } else {
      activePromptStep = null;
    }

    let modelHtml = "";
    const showModelEditor = step.kind === "llm" && !!step.role;
    if (showModelEditor) {
      modelHtml = renderModelEditorHtml(step);
    }

    els.detailBody.innerHTML = `
      <dl class="pipeline-kv">
        <dt>Step ID</dt><dd><code>${escapeHtml(step.id)}</code></dd>
        <dt>Kind</dt><dd><span class="pipeline-kind-chip kind-${escapeAttr(step.kind)}">${escapeHtml(KIND_LABELS[step.kind] || step.kind)}</span></dd>
        ${kindLine}
        <dt>Phase</dt><dd>${escapeHtml(step.phase)}</dd>
        <dt>Cadence</dt><dd>${escapeHtml(PER_LABELS[step.per] || step.per)}</dd>
        <dt>Inputs</dt><dd class="pipeline-kv-row">${inputs}</dd>
        <dt>Outputs</dt><dd class="pipeline-kv-row">${outputs}</dd>
        ${step.optional ? `<dt>Status</dt><dd><span class="pipeline-card-flag">optional</span></dd>` : ""}
      </dl>
      ${extra}
      ${editorHtml}
      ${modelHtml}
    `;

    if (isEditable) wirePromptEditor(step);
    if (showModelEditor) wireModelEditor(step);
  }

  function renderModelEditorHtml(step) {
    const binding = bindingByRole.get(step.role);
    if (!binding) {
      return `
        <section class="model-editor">
          <h3 class="model-editor-title">Bound LLM</h3>
          <p class="hint">Could not load model bindings.</p>
        </section>
      `;
    }
    const fallbackNote = binding.falls_back_to_default
      ? `<div class="hint">This role currently falls back to <code>default</code>. Saving creates an explicit binding for <code>${escapeHtml(step.role)}</code>.</div>`
      : "";
    return `
      <section class="model-editor" data-role="${escapeAttr(step.role)}">
        <h3 class="model-editor-title">Bound LLM
          <span class="pipeline-role-badge" data-role="${escapeAttr(step.role)}">${escapeHtml(step.role)}</span>
        </h3>
        <div class="model-status">
          <code>${escapeHtml(binding.model || "—")}</code>
          <span class="hint">@</span>
          <code>${escapeHtml(binding.base_url || "—")}</code>
          ${binding.is_azure ? `<span class="pipeline-card-flag">azure</span>` : ""}
          ${binding.has_api_key ? "" : `<span class="pipeline-card-flag" title="No API key currently set">no key</span>`}
        </div>
        ${fallbackNote}
        <label class="prompt-field">
          <span class="prompt-field-label">Base URL</span>
          <input class="model-base-url" type="url" value="${escapeAttr(binding.base_url || "")}" autocomplete="off">
        </label>
        <label class="prompt-field">
          <span class="prompt-field-label">Model</span>
          <input class="model-name" type="text" value="${escapeAttr(binding.model || "")}" autocomplete="off">
        </label>
        <label class="prompt-field">
          <span class="prompt-field-label">API key
            <span class="hint">(leave empty to keep current; <code>\${VAR}</code> reads from server env)</span>
          </span>
          <input class="model-api-key" type="text" value="" autocomplete="new-password" placeholder="sk-…  or  \${OPENAI_API_KEY}">
        </label>
        <label class="check">
          <input type="checkbox" class="model-azure" ${binding.is_azure ? "checked" : ""}>
          <span>Azure OpenAI endpoint</span>
        </label>
        <label class="prompt-field">
          <span class="prompt-field-label">Azure API version</span>
          <input class="model-azure-version" type="text" value="${escapeAttr(binding.azure_api_version || "2024-05-01-preview")}" autocomplete="off">
        </label>
        <div class="prompt-actions">
          <button type="button" class="btn model-test">Test connection</button>
          <button type="button" class="btn btn-primary model-apply">Apply</button>
        </div>
        <div class="model-test-output" hidden></div>
        <div class="prompt-error model-error" hidden></div>
      </section>
    `;
  }

  function wireModelEditor(step) {
    const root = els.detailBody.querySelector(".model-editor");
    if (!root) return;
    const baseEl = root.querySelector(".model-base-url");
    const modelEl = root.querySelector(".model-name");
    const keyEl = root.querySelector(".model-api-key");
    const azureEl = root.querySelector(".model-azure");
    const azVerEl = root.querySelector(".model-azure-version");
    const testBtn = root.querySelector(".model-test");
    const applyBtn = root.querySelector(".model-apply");
    const testOut = root.querySelector(".model-test-output");
    const errEl = root.querySelector(".model-error");

    function showError(msg) { errEl.textContent = msg; errEl.hidden = false; }
    function clearError() { errEl.hidden = true; errEl.textContent = ""; }

    testBtn?.addEventListener("click", async () => {
      clearError();
      testBtn.disabled = true;
      testOut.hidden = true;
      try {
        const body = {
          base_url: baseEl.value.trim(),
          model: modelEl.value.trim(),
          is_azure: azureEl.checked,
          azure_api_version: azVerEl.value.trim() || "2024-05-01-preview",
          role: step.role,
          prompt: "ping",
          max_tokens: 16,
        };
        const typedKey = keyEl.value;
        if (typedKey) body.api_key = typedKey;
        const res = await api.testPipelineModel(body);
        const cls = res.ok ? "ok" : "error";
        testOut.className = `model-test-output ${cls}`;
        testOut.innerHTML = `
          <div class="model-test-line">
            <strong>${res.ok ? "OK" : "Failed"}</strong>
            <span class="hint">${res.latency_ms} ms</span>
          </div>
          ${res.sample ? `<pre class="model-test-sample">${escapeHtml(res.sample)}</pre>` : ""}
          ${res.error ? `<div class="model-test-err">${escapeHtml(res.error)}</div>` : ""}
        `;
        testOut.hidden = false;
      } catch (err) {
        showError(`Test failed: ${err.message}`);
      } finally {
        testBtn.disabled = false;
      }
    });

    applyBtn?.addEventListener("click", async () => {
      clearError();
      applyBtn.disabled = true;
      try {
        const body = {
          base_url: baseEl.value.trim(),
          model: modelEl.value.trim(),
          is_azure: azureEl.checked,
          azure_api_version: azVerEl.value.trim() || "2024-05-01-preview",
        };
        const typedKey = keyEl.value;
        if (typedKey) body.api_key = typedKey;
        const res = await api.updatePipelineModel(step.role, body);
        toast(`Bound ${res.role} → ${res.binding.model}`);
        await refreshModelInfo();
        // Re-render the editor with the new state.
        selectStep(step);
      } catch (err) {
        showError(`Apply failed: ${err.message}`);
      } finally {
        applyBtn.disabled = false;
      }
    });
  }

  function renderPromptEditorHtml(step) {
    const info = promptInfoByName.get(step.prompt_name);
    if (!info) {
      return `
        <section class="prompt-editor">
          <h3 class="prompt-editor-title">Prompt</h3>
          <p class="hint">Could not load prompt details for <code>${escapeHtml(step.prompt_name)}</code>${getGraphId() ? "" : " — pick a graph first"}.</p>
        </section>
      `;
    }
    const overridden = info.has_graph_override;
    const startSystem = (overridden ? info.graph : info.builtin)?.system || "";
    const startUser = (overridden ? info.graph : info.builtin)?.user || "";
    return `
      <section class="prompt-editor" data-prompt="${escapeAttr(step.prompt_name)}">
        <h3 class="prompt-editor-title">Prompt template</h3>
        <div class="prompt-editor-status">
          <span class="prompt-status-badge" data-status="${overridden ? "graph" : "builtin"}">
            ${overridden ? "graph override" : "builtin"}
          </span>
          ${info.service ? `<span class="prompt-status-badge" data-status="service">service default loaded</span>` : ""}
        </div>
        <details class="prompt-builtin">
          <summary>Builtin (read-only)</summary>
          <div class="prompt-builtin-block">
            <div class="prompt-block-label">System</div>
            <pre class="prompt-readonly">${escapeHtml(info.builtin?.system || "")}</pre>
          </div>
          <div class="prompt-builtin-block">
            <div class="prompt-block-label">User</div>
            <pre class="prompt-readonly">${escapeHtml(info.builtin?.user || "")}</pre>
          </div>
        </details>
        <label class="prompt-field">
          <span class="prompt-field-label">System template</span>
          <textarea class="prompt-system" rows="6" spellcheck="false">${escapeHtml(startSystem)}</textarea>
          <div class="prompt-detected-vars" data-source="system"></div>
        </label>
        <label class="prompt-field">
          <span class="prompt-field-label">User template</span>
          <textarea class="prompt-user" rows="12" spellcheck="false">${escapeHtml(startUser)}</textarea>
          <div class="prompt-detected-vars" data-source="user"></div>
        </label>
        <label class="prompt-field">
          <span class="prompt-field-label">Preview variables (JSON)</span>
          <textarea class="prompt-vars" rows="4" spellcheck="false" placeholder='{"goal": "...", "observation": "..."}'></textarea>
          <div class="hint">Templates use <code>{var}</code> placeholders; provide JSON to substitute on preview. Missing keys raise a 400.</div>
        </label>
        <div class="prompt-actions">
          <button type="button" class="btn btn-primary prompt-save">Save</button>
          <button type="button" class="btn prompt-reset" ${overridden ? "" : "disabled"}>Reset to builtin</button>
          <button type="button" class="btn prompt-preview">Preview rendered</button>
        </div>
        <div class="prompt-preview-output" hidden>
          <h4>Rendered messages</h4>
          <div class="prompt-preview-list"></div>
        </div>
        <details class="prompt-diff">
          <summary>Diff vs builtin <span class="hint">(line-based)</span></summary>
          <div class="prompt-diff-section">
            <div class="prompt-block-label">System</div>
            <div class="prompt-diff-body" data-target="system"></div>
          </div>
          <div class="prompt-diff-section">
            <div class="prompt-block-label">User</div>
            <div class="prompt-diff-body" data-target="user"></div>
          </div>
        </details>
        <details class="prompt-step-traces">
          <summary>Recent calls (this step)</summary>
          <div class="prompt-step-traces-list"><div class="hint">Loading…</div></div>
        </details>
        <div class="prompt-error" hidden></div>
      </section>
    `;
  }

  function wirePromptEditor(step) {
    const root = els.detailBody.querySelector(".prompt-editor");
    if (!root) return;
    const sysEl = root.querySelector(".prompt-system");
    const usrEl = root.querySelector(".prompt-user");
    const varsEl = root.querySelector(".prompt-vars");
    const saveBtn = root.querySelector(".prompt-save");
    const resetBtn = root.querySelector(".prompt-reset");
    const previewBtn = root.querySelector(".prompt-preview");
    const previewOut = root.querySelector(".prompt-preview-output");
    const previewList = root.querySelector(".prompt-preview-list");
    const errorEl = root.querySelector(".prompt-error");
    const sysVarsEl = root.querySelector(".prompt-detected-vars[data-source='system']");
    const usrVarsEl = root.querySelector(".prompt-detected-vars[data-source='user']");

    function extractVarNames(text) {
      const set = new Set();
      const re = /\{(\w+)\}/g;
      let m;
      while ((m = re.exec(text || "")) !== null) set.add(m[1]);
      return [...set];
    }
    function ensureVarInPreview(name) {
      let parsed = {};
      const raw = varsEl.value.trim();
      if (raw) {
        try { parsed = JSON.parse(raw); }
        catch { /* leave existing text alone — user is mid-edit */ return; }
      }
      if (!Object.prototype.hasOwnProperty.call(parsed, name)) {
        parsed[name] = "";
        varsEl.value = JSON.stringify(parsed, null, 2);
      }
    }
    function renderDetected(target, textarea) {
      if (!target) return;
      const names = extractVarNames(textarea.value);
      target.innerHTML = "";
      if (names.length === 0) {
        target.innerHTML = `<span class="hint">No {placeholders} detected.</span>`;
        return;
      }
      const head = document.createElement("span");
      head.className = "hint";
      head.textContent = "vars: ";
      target.appendChild(head);
      for (const n of names) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "prompt-var-chip";
        chip.textContent = n;
        chip.title = `Click to add "${n}" to preview variables`;
        chip.addEventListener("click", () => ensureVarInPreview(n));
        target.appendChild(chip);
      }
    }
    renderDetected(sysVarsEl, sysEl);
    renderDetected(usrVarsEl, usrEl);
    sysEl.addEventListener("input", () => renderDetected(sysVarsEl, sysEl));
    usrEl.addEventListener("input", () => renderDetected(usrVarsEl, usrEl));

    // Live diff vs builtin
    const info = promptInfoByName.get(step.prompt_name);
    const builtinSys = info?.builtin?.system || "";
    const builtinUsr = info?.builtin?.user || "";
    const diffSysEl = root.querySelector(".prompt-diff-body[data-target='system']");
    const diffUsrEl = root.querySelector(".prompt-diff-body[data-target='user']");
    function refreshDiff() {
      if (diffSysEl) diffSysEl.innerHTML = renderDiffHtml(builtinSys, sysEl.value);
      if (diffUsrEl) diffUsrEl.innerHTML = renderDiffHtml(builtinUsr, usrEl.value);
    }
    refreshDiff();
    sysEl.addEventListener("input", refreshDiff);
    usrEl.addEventListener("input", refreshDiff);

    // Recent calls (this step) — fetch on demand once the editor mounts.
    const stepTracesEl = root.querySelector(".prompt-step-traces-list");
    if (stepTracesEl) {
      const gid = getGraphId();
      if (!gid) {
        stepTracesEl.innerHTML = `<div class="hint">No graph selected.</div>`;
      } else {
        api.listStepTraces(gid, step.id, { limit: 10 }).then((res) => {
          if (!res.traces || res.traces.length === 0) {
            stepTracesEl.innerHTML = `<div class="hint">No recorded calls yet for <code>${escapeHtml(step.id)}</code>.</div>`;
            return;
          }
          stepTracesEl.innerHTML = "";
          for (const t of res.traces) {
            const row = document.createElement("button");
            row.type = "button";
            row.className = `pipeline-trace-row ${t.ok ? "ok" : "fail"}`;
            row.innerHTML = `
              <div class="pipeline-trace-row-head">
                <span class="pipeline-trace-endpoint">${escapeHtml(t.endpoint)}</span>
                <span class="hint">${escapeHtml(t.ts || "")}</span>
              </div>
              <div class="pipeline-trace-row-meta">
                <span class="pipeline-trace-steps">${t.step_latency_ms} ms (this step)</span>
                <span class="hint">${t.duration_ms} ms (total)</span>
                ${t.step_error ? `<span class="pipeline-card-flag" title="${escapeAttr(t.step_error)}">err</span>` : ""}
              </div>
            `;
            row.addEventListener("click", () => selectTrace(t.trace_id));
            stepTracesEl.appendChild(row);
          }
        }).catch((err) => {
          stepTracesEl.innerHTML = `<div class="hint">${escapeHtml(err.message)}</div>`;
        });
      }
    }

    function showError(msg) {
      errorEl.textContent = msg;
      errorEl.hidden = false;
    }
    function clearError() {
      errorEl.hidden = true;
      errorEl.textContent = "";
    }

    saveBtn?.addEventListener("click", async () => {
      const gid = getGraphId();
      if (!gid) { showError("Pick a graph first."); return; }
      clearError();
      saveBtn.disabled = true;
      try {
        await api.updatePipelinePrompt(gid, step.prompt_name, {
          system: sysEl.value,
          user: usrEl.value,
        });
        toast(`Saved override for ${step.prompt_name}`);
        await refreshPromptInfo();
        if (xyflowMod) mountReactApp(xyflowMod);
        // Re-render the editor so badges + reset-button state refresh.
        if (activePromptStep && activePromptStep.id === step.id) selectStep(step);
      } catch (err) {
        showError(`Save failed: ${err.message}`);
      } finally {
        saveBtn.disabled = false;
      }
    });

    resetBtn?.addEventListener("click", async () => {
      const gid = getGraphId();
      if (!gid) { showError("Pick a graph first."); return; }
      if (!confirm(`Reset ${step.prompt_name} to the builtin template?`)) return;
      clearError();
      resetBtn.disabled = true;
      try {
        await api.resetPipelinePrompt(gid, step.prompt_name);
        toast(`Reset ${step.prompt_name} to builtin`);
        await refreshPromptInfo();
        if (xyflowMod) mountReactApp(xyflowMod);
        if (activePromptStep && activePromptStep.id === step.id) selectStep(step);
      } catch (err) {
        showError(`Reset failed: ${err.message}`);
      } finally {
        resetBtn.disabled = false;
      }
    });

    previewBtn?.addEventListener("click", async () => {
      const gid = getGraphId();
      if (!gid) { showError("Pick a graph first."); return; }
      let variables = {};
      const raw = varsEl.value.trim();
      if (raw) {
        try { variables = JSON.parse(raw); }
        catch (err) { showError(`Variables must be valid JSON: ${err.message}`); return; }
      }
      clearError();
      previewBtn.disabled = true;
      previewList.innerHTML = "";
      previewOut.hidden = true;
      try {
        const res = await api.previewPipelinePrompt(gid, step.prompt_name, {
          system: sysEl.value,
          user: usrEl.value,
          variables,
        });
        for (const msg of res.messages || []) {
          const block = document.createElement("div");
          block.className = "prompt-preview-msg";
          block.innerHTML = `
            <div class="prompt-preview-role">${escapeHtml(msg.role)}</div>
            <pre class="prompt-preview-content">${escapeHtml(msg.content)}</pre>
          `;
          previewList.appendChild(block);
        }
        previewOut.hidden = false;
      } catch (err) {
        showError(`Preview failed: ${err.message}`);
      } finally {
        previewBtn.disabled = false;
      }
    });
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  // --------------------------------------------------------------------- //
  // Pipeline binding + spec editor (Phase 6.5a)
  // --------------------------------------------------------------------- //

  async function refreshAvailablePipelines() {
    if (availablePipelines.length) return;
    try {
      const res = await api.listPipelines();
      availablePipelines = res.pipelines || [];
    } catch (err) {
      availablePipelines = [];
      console.warn("listPipelines failed:", err);
    }
  }

  function renderBindingSelect() {
    if (!els.bindingSelect) return;
    const cur = currentBinding || "";
    els.bindingSelect.innerHTML = "";
    for (const p of availablePipelines) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      if (p.name === cur) opt.selected = true;
      els.bindingSelect.appendChild(opt);
    }
    const cur_p = availablePipelines.find((p) => p.name === cur);
    if (els.bindingDesc) {
      els.bindingDesc.textContent = cur_p?.description || "";
    }
  }

  async function refreshBinding() {
    const gid = getGraphId();
    if (!gid) {
      currentBinding = null;
      renderBindingSelect();
      toggleSpecEditorVisible(false);
      return;
    }
    try {
      const res = await api.getGraphPipeline(gid);
      currentBinding = res.pipeline;
    } catch (err) {
      currentBinding = null;
      console.warn("getGraphPipeline failed:", err);
    }
    renderBindingSelect();
    toggleSpecEditorVisible(currentBinding === "spec-driven");
    if (currentBinding === "spec-driven") {
      await Promise.all([refreshSpecEditor(), refreshSpecVersions()]);
    }
  }

  function toggleSpecEditorVisible(visible) {
    if (!els.specEditor) return;
    els.specEditor.hidden = !visible;
  }

  function setEditorStatus(text, kind = "") {
    if (!els.specEditorStatus) return;
    els.specEditorStatus.textContent = text || "";
    els.specEditorStatus.className = `hint ${kind}`.trim();
  }

  function markDirty() {
    if (!els.specEditorTextarea) return;
    const dirty = els.specEditorTextarea.value !== savedSpecContent;
    els.specEditorTextarea.classList.toggle("dirty", dirty);
    els.specEditorSave.disabled = false;
    els.specEditorDiscard.disabled = !dirty;
  }

  async function refreshSpecEditor() {
    const gid = getGraphId();
    if (!gid || !els.specEditorTextarea) return;
    setEditorStatus("Loading…");
    try {
      const res = await api.getPipelineSpecYaml(gid);
      savedSpecContent = res.content || "";
      savedSpecVersionId = res.active_version_id || null;
      savedSpecPath = res.live_path || "";
      els.specEditorTextarea.value = savedSpecContent;
      els.specEditorPath.textContent = savedSpecPath;
      els.specEditorActive.textContent = savedSpecVersionId
        ? `active: ${savedSpecVersionId}`
        : "no saved version yet";
      els.specEditorTextarea.classList.remove("dirty");
      els.specEditorDiscard.disabled = true;
      setEditorStatus(savedSpecContent ? "Loaded." : "No YAML yet — write one and save.", "ok");
    } catch (err) {
      setEditorStatus(`Load failed: ${err.message}`, "err");
    }
  }

  async function refreshSpecVersions() {
    const gid = getGraphId();
    if (!gid || !els.specVersionsList) return;
    els.specVersionsList.innerHTML = `<div class="hint">Loading versions…</div>`;
    try {
      const res = await api.listPipelineSpecVersions(gid);
      const versions = res.versions || [];
      if (!versions.length) {
        els.specVersionsList.innerHTML = `<div class="hint">No versions yet — saving creates one.</div>`;
        return;
      }
      els.specVersionsList.innerHTML = "";
      for (const v of versions) {
        const row = document.createElement("div");
        row.className = `pipeline-spec-version-row${v.active ? " active" : ""}`;
        const ts = v.ts || v.version_id;
        const note = v.note || (v.active ? "active" : "");
        row.innerHTML = `
          <code title="${escapeAttr(v.version_id)}">${escapeHtml(ts)}</code>
          <span class="pipeline-spec-version-note" title="${escapeAttr(v.note || "")}">${escapeHtml(note)}</span>
          <button type="button" class="btn" data-action="view" data-version="${escapeAttr(v.version_id)}">View</button>
          <button type="button" class="btn" data-action="rollback" data-version="${escapeAttr(v.version_id)}" ${v.active ? "disabled" : ""}>${v.active ? "Active" : "Rollback"}</button>
        `;
        els.specVersionsList.appendChild(row);
      }
      els.specVersionsList.querySelectorAll("button[data-action='view']").forEach((b) => {
        b.addEventListener("click", () => viewVersion(b.dataset.version));
      });
      els.specVersionsList.querySelectorAll("button[data-action='rollback']").forEach((b) => {
        b.addEventListener("click", () => rollbackVersion(b.dataset.version));
      });
    } catch (err) {
      els.specVersionsList.innerHTML = `<div class="hint">Error: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function viewVersion(vid) {
    const gid = getGraphId();
    if (!gid) return;
    try {
      const res = await api.getPipelineSpecVersion(gid, vid);
      els.specEditorTextarea.value = res.content || "";
      els.specEditorTextarea.classList.toggle("dirty",
        els.specEditorTextarea.value !== savedSpecContent);
      els.specEditorDiscard.disabled = false;
      setEditorStatus(`Viewing ${vid} — Save creates a new version. Discard to restore active.`, "ok");
    } catch (err) {
      setEditorStatus(`View failed: ${err.message}`, "err");
    }
  }

  async function rollbackVersion(vid) {
    const gid = getGraphId();
    if (!gid) return;
    if (!confirm(`Rollback to ${vid}? The live YAML will be replaced with that version's contents.`)) return;
    try {
      await api.rollbackPipelineSpecVersion(gid, vid);
      toast(`Rolled back to ${vid.slice(0, 24)}…`);
      await Promise.all([refreshSpecEditor(), refreshSpecVersions()]);
      if (currentBinding === "spec-driven") await reloadCanvasForBinding();
    } catch (err) {
      toast(`Rollback failed: ${err.message}`, "error");
    }
  }

  async function validateEditor() {
    const gid = getGraphId();
    if (!gid) return;
    setEditorStatus("Validating…");
    try {
      const res = await api.validatePipelineSpecYaml(gid, els.specEditorTextarea.value);
      if (res.ok) {
        setEditorStatus("Valid.", "ok");
      } else {
        setEditorStatus(res.error || "Invalid YAML.", "err");
      }
    } catch (err) {
      setEditorStatus(`Validate failed: ${err.message}`, "err");
    }
  }

  async function saveEditor() {
    const gid = getGraphId();
    if (!gid) return;
    const content = els.specEditorTextarea.value;
    const note = els.specEditorNote.value || "";
    els.specEditorSave.disabled = true;
    setEditorStatus("Saving…");
    try {
      const res = await api.savePipelineSpecYaml(gid, content, note);
      savedSpecContent = content;
      savedSpecVersionId = res.version.version_id;
      els.specEditorActive.textContent = `active: ${savedSpecVersionId}`;
      els.specEditorTextarea.classList.remove("dirty");
      els.specEditorDiscard.disabled = true;
      els.specEditorNote.value = "";
      setEditorStatus(`Saved as ${savedSpecVersionId}.`, "ok");
      toast(`Spec saved (${savedSpecVersionId.slice(0, 22)}…)`);
      await refreshSpecVersions();
      if (currentBinding === "spec-driven") await reloadCanvasForBinding();
    } catch (err) {
      setEditorStatus(err.message, "err");
    } finally {
      els.specEditorSave.disabled = false;
    }
  }

  function discardEditorChanges() {
    if (!els.specEditorTextarea) return;
    els.specEditorTextarea.value = savedSpecContent;
    els.specEditorTextarea.classList.remove("dirty");
    els.specEditorDiscard.disabled = true;
    setEditorStatus("Reverted to active version.", "ok");
  }

  async function reloadCanvasForBinding() {
    try {
      spec = await fetchSpec();
      if (xyflowMod) mountReactApp(xyflowMod);
      renderSidebar();
    } catch (err) {
      console.warn("reloadCanvasForBinding failed:", err);
    }
  }

  els.bindingSelect?.addEventListener("change", async (e) => {
    const gid = getGraphId();
    if (!gid) { toast("Pick a graph first.", "warn"); return; }
    const next = e.target.value;
    if (next === currentBinding) return;
    try {
      await api.setGraphPipeline(gid, next);
      currentBinding = next;
      toast(`Pipeline → ${next}`);
      const cur_p = availablePipelines.find((p) => p.name === next);
      if (els.bindingDesc) els.bindingDesc.textContent = cur_p?.description || "";
      toggleSpecEditorVisible(next === "spec-driven");
      if (next === "spec-driven") {
        await Promise.all([refreshSpecEditor(), refreshSpecVersions()]);
      }
      await reloadCanvasForBinding();
    } catch (err) {
      toast(`set pipeline: ${err.message}`, "error");
      // Revert UI selection.
      e.target.value = currentBinding || "";
    }
  });

  async function refreshSamples() {
    if (availableSamples.length || !els.specSampleSelect) return;
    try {
      const res = await api.listPipelineSpecSamples();
      availableSamples = res.samples || [];
    } catch (err) {
      availableSamples = [];
      console.warn("listPipelineSpecSamples failed:", err);
    }
    els.specSampleSelect.innerHTML = "";
    for (const s of availableSamples) {
      const opt = document.createElement("option");
      opt.value = s.key;
      opt.textContent = s.label;
      opt.title = s.description || "";
      els.specSampleSelect.appendChild(opt);
    }
  }

  function insertSelectedSample() {
    const key = els.specSampleSelect?.value;
    const sample = availableSamples.find((s) => s.key === key);
    if (!sample || !els.specEditorTextarea) return;
    const cur = els.specEditorTextarea.value.trim();
    if (cur && !confirm("Replace the current editor contents with the sample?")) return;
    els.specEditorTextarea.value = sample.content;
    markDirty();
    setEditorStatus(`Inserted sample: ${sample.label}. Click Save new version to persist.`, "ok");
  }

  els.specEditorTextarea?.addEventListener("input", markDirty);
  els.specEditorValidate?.addEventListener("click", validateEditor);
  els.specEditorSave?.addEventListener("click", saveEditor);
  els.specEditorDiscard?.addEventListener("click", discardEditorChanges);
  els.specSampleInsert?.addEventListener("click", insertSelectedSample);
  els.specEditorToggle?.addEventListener("click", () => {
    const isCollapsed = els.specEditor.classList.toggle("collapsed");
    els.specEditorToggle.textContent = isCollapsed ? "Expand" : "Collapse";
  });

  return {
    async refresh({ graphId } = {}) {
      if (!loaded) {
        await load();
        return;
      }
      if (graphId !== undefined && graphId !== currentGraphId) {
        currentGraphId = graphId;
        // Close any open prompt editor for the previous graph.
        els.detail.hidden = true;
        activePromptStep = null;
        await Promise.all([refreshPromptInfo(), refreshStepStats(), refreshBinding()]);
        void refreshTraces();
        // Binding may have changed — refetch the spec view + re-render.
        await reloadCanvasForBinding();
      }
    },
  };
}


function buildGraph(spec, dagre, MarkerType, visibleKinds, promptByName, statsByName) {
  const kinds = visibleKinds || new Set(ALL_KINDS);
  // Filter + contract: build effective steps + edges from the visible-kinds set.
  const { steps: effectiveSteps, edges: effectiveEdges } = contractGraph(
    spec.steps, spec.edges, kinds,
  );
  const effectiveCrossPhase = contractGraph(
    spec.steps, spec.cross_phase_edges || [], kinds,
  ).edges;

  const nodes = [];
  const edges = [];

  // Per-phase layout: dagre over only seq+branch+alt edges within the phase.
  let cursorX = 0;
  const phaseBounds = {};
  for (const phase of spec.phases) {
    const phaseSteps = effectiveSteps.filter((s) => s.phase === phase.id);
    if (phaseSteps.length === 0) continue;
    const stepIds = new Set(phaseSteps.map((s) => s.id));
    const layoutEdges = effectiveEdges.filter(
      (ed) => stepIds.has(ed.source) && stepIds.has(ed.target) && ed.kind !== "loop_back",
    );

    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "TB", nodesep: 36, ranksep: 56, marginx: 12, marginy: 12 });
    g.setDefaultEdgeLabel(() => ({}));

    for (const s of phaseSteps) {
      const dims = NODE_DIMS[s.kind] || NODE_DIMS.compute;
      g.setNode(s.id, { width: dims[0], height: dims[1] });
    }
    for (const ed of layoutEdges) {
      g.setEdge(ed.source, ed.target);
    }

    dagre.layout(g);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const id of g.nodes()) {
      const n = g.node(id);
      const x = n.x - n.width / 2;
      const y = n.y - n.height / 2;
      if (x < minX) minX = x;
      if (x + n.width > maxX) maxX = x + n.width;
      if (y < minY) minY = y;
      if (y + n.height > maxY) maxY = y + n.height;
    }
    if (!isFinite(minX)) { minX = 0; maxX = 0; minY = 0; maxY = 0; }

    const phaseWidth = maxX - minX;

    // Stage trigger node sits above the phase, centered on its width.
    const stageId = `__stage:${phase.id}`;
    nodes.push({
      id: stageId,
      type: "stage",
      position: {
        x: cursorX + Math.max(0, (phaseWidth - 320) / 2),
        y: -STAGE_OFFSET_Y,
      },
      data: {
        phase: phase.id,
        label: phase.label,
        trigger: phase.trigger,
        triggered_by: phase.triggered_by,
      },
      style: { width: 320 },
      selectable: false,
    });

    // Step nodes.
    for (const step of phaseSteps) {
      const n = g.node(step.id);
      if (!n) continue;
      const dims = NODE_DIMS[step.kind] || NODE_DIMS.compute;
      const promptInfo = step.prompt_name
        ? (promptByName?.get(step.prompt_name) || null)
        : null;
      const stats = statsByName?.get(step.id) || null;
      nodes.push({
        id: step.id,
        type: step.kind,
        position: {
          x: cursorX + (n.x - n.width / 2 - minX),
          y: (n.y - n.height / 2 - minY),
        },
        data: {
          ...step,
          step,
          has_graph_override: !!promptInfo?.has_graph_override,
          stats,
        },
        style: { width: dims[0] },
      });
    }

    // Stage → entry edge: pick the step with no incoming intra-phase seq/branch edge.
    const incoming = new Set(layoutEdges.map((e) => e.target));
    const entryStep = phaseSteps.find((s) => !incoming.has(s.id));
    if (entryStep) {
      edges.push({
        id: `${stageId}->${entryStep.id}`,
        source: stageId,
        target: entryStep.id,
        type: "smoothstep",
        style: { stroke: "var(--accent)", strokeDasharray: "4 4" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "var(--accent)" },
      });
    }

    phaseBounds[phase.id] = {
      xStart: cursorX,
      xEnd: cursorX + phaseWidth,
      width: phaseWidth,
    };
    cursorX += phaseWidth + PHASE_GAP;
  }

  const renderedNodeIds = new Set(nodes.map((n) => n.id));

  // Intra-phase edges (now rendered, including loop_back edges).
  for (const ed of effectiveEdges) {
    if (!renderedNodeIds.has(ed.source) || !renderedNodeIds.has(ed.target)) continue;
    edges.push(makeEdge(ed, MarkerType));
  }

  // Cross-phase ghost edges (data-flow hints between phases).
  for (const ed of effectiveCrossPhase) {
    if (!renderedNodeIds.has(ed.source) || !renderedNodeIds.has(ed.target)) continue;
    edges.push(makeEdge(ed, MarkerType, { ghost: true }));
  }

  return { nodes, edges };
}


// Filter steps by visibleKinds and contract edges through hidden steps.
// Walks each visible step's outgoing edges; whenever the target is hidden,
// jumps through it and re-attaches to the next visible target. Loop-back
// edges are preserved only when both endpoints survive filtering — this
// avoids spurious back-edges when the loop boundary itself is hidden.
function contractGraph(allSteps, allEdges, visibleKinds) {
  const stepById = new Map(allSteps.map((s) => [s.id, s]));
  const isVisible = (id) => {
    const s = stepById.get(id);
    return s ? visibleKinds.has(s.kind) : false;
  };
  const visibleSteps = allSteps.filter((s) => visibleKinds.has(s.kind));

  // Forward adjacency, excluding loop_back edges from the contraction walk
  // (those are folded only when both endpoints stay visible).
  const fwd = new Map();
  for (const ed of allEdges) {
    if (ed.kind === "loop_back") continue;
    if (!fwd.has(ed.source)) fwd.set(ed.source, []);
    fwd.get(ed.source).push(ed);
  }

  const out = [];
  const seen = new Set();
  const push = (edge) => {
    const key = `${edge.source}->${edge.target}:${edge.kind}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(edge);
  };

  for (const src of visibleSteps) {
    const visited = new Set();
    const queue = [];
    for (const ed of fwd.get(src.id) || []) {
      queue.push({ target: ed.target, kind: ed.kind, label: ed.label });
    }
    while (queue.length) {
      const cur = queue.shift();
      if (visited.has(cur.target)) continue;
      visited.add(cur.target);
      if (isVisible(cur.target)) {
        push({ source: src.id, target: cur.target, kind: cur.kind, label: cur.label || "" });
      } else {
        for (const next of fwd.get(cur.target) || []) {
          if (visited.has(next.target)) continue;
          queue.push({
            target: next.target,
            kind: cur.kind === "seq" && next.kind !== "seq" ? next.kind : cur.kind,
            label: cur.label || next.label,
          });
        }
      }
    }
  }

  // Preserve loop_back edges only when both endpoints are visible.
  for (const ed of allEdges) {
    if (ed.kind !== "loop_back") continue;
    if (isVisible(ed.source) && isVisible(ed.target)) {
      push({ source: ed.source, target: ed.target, kind: ed.kind, label: ed.label });
    }
  }

  return { steps: visibleSteps, edges: out };
}


function makeEdge(ed, MarkerType, { ghost = false } = {}) {
  let style = {};
  let labelStyle = { fontSize: 11, fill: "var(--fg-muted)" };
  let labelBg = { fill: "var(--bg-elev)", fillOpacity: 0.85 };
  let stroke = "var(--border-strong)";

  if (ed.kind === "branch") {
    stroke = "var(--node-procedural)";
  } else if (ed.kind === "loop_back") {
    stroke = "var(--warn)";
    style.strokeDasharray = "5 4";
  } else if (ed.kind === "alt") {
    stroke = "var(--fg-faint)";
    style.strokeDasharray = "3 4";
  }
  if (ghost) {
    style.strokeDasharray = "2 6";
    style.opacity = 0.5;
    stroke = "var(--fg-faint)";
  }
  style.stroke = stroke;

  return {
    id: `${ed.source}->${ed.target}:${ed.kind}`,
    source: ed.source,
    target: ed.target,
    type: ed.kind === "loop_back" ? "smoothstep" : "smoothstep",
    label: ed.label || undefined,
    style,
    labelStyle,
    labelBgStyle: labelBg,
    labelBgPadding: [3, 4],
    labelBgBorderRadius: 3,
    markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
  };
}
