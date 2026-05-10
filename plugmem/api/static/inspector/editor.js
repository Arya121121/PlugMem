// Shared inline editor used by Browse and Graph detail panels.
//
// renderEditor({ type, node, getGraphId, toast, onSaved, onCancel }) returns
// a DOM node that contains a small form for the editable fields of the node
// type, plus Save/Cancel buttons. onSaved is called with the patched
// NodeDetailResponse so the caller can refresh its row + re-render the
// detail body.

import { api } from "./api.js";

const PATCH = {
  semantic: (gid, id, body) => api.patchSemantic(gid, id, body),
  procedural: (gid, id, body) => api.patchProcedural(gid, id, body),
  tag: (gid, id, body) => api.patchTag(gid, id, body),
  subgoal: (gid, id, body) => api.patchSubgoal(gid, id, body),
  episodic: (gid, id, body) => api.patchEpisodic(gid, id, body),
};

export function renderEditor({ type, node, getGraphId, toast, onSaved, onCancel }) {
  const wrap = document.createElement("div");
  wrap.className = "detail-editor";

  const fields = buildFields(type, node);
  for (const f of fields) wrap.appendChild(f.row);

  const actions = document.createElement("div");
  actions.className = "detail-actions";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "btn btn-primary";
  saveBtn.textContent = "Save";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn btn-ghost";
  cancelBtn.textContent = "Cancel";
  actions.appendChild(saveBtn);
  actions.appendChild(cancelBtn);
  wrap.appendChild(actions);

  saveBtn.addEventListener("click", async () => {
    const gid = getGraphId();
    if (!gid) return;
    const body = {};
    for (const f of fields) {
      const value = f.read();
      if (value === undefined) continue;
      body[f.payloadKey] = value;
    }
    if (!Object.keys(body).length) {
      toast("nothing changed", "info");
      return;
    }
    saveBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      const res = await PATCH[type](gid, node.id, body);
      toast(`${type} ${node.id} updated`);
      if (typeof onSaved === "function") onSaved(res);
    } catch (err) {
      toast(`save: ${err.message}`, "error");
      saveBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });

  cancelBtn.addEventListener("click", () => {
    if (typeof onCancel === "function") onCancel();
  });

  return wrap;
}

function buildFields(type, node) {
  if (type === "semantic") {
    return [
      textareaField("text", "text", node.text || "", { rows: 5 }),
      tagsField("tags", "tags", node.tags || []),
      numberField("credibility", "credibility", node.credibility ?? 10, { step: 1 }),
    ];
  }
  if (type === "procedural") {
    return [
      textareaField("text", "text", node.text || "", { rows: 5 }),
      numberField("return", "return", node.return ?? 0, { step: "any" }),
    ];
  }
  if (type === "tag") {
    return [
      inputField("tag", "tag", node.tag || ""),
      numberField("importance", "importance", node.importance ?? 1, { step: 1 }),
    ];
  }
  if (type === "subgoal") {
    return [textareaField("subgoal", "subgoal", node.subgoal || "", { rows: 3 })];
  }
  if (type === "episodic") {
    return [
      textareaField("observation", "observation", node.observation || "", { rows: 3 }),
      textareaField("action", "action", node.action || "", { rows: 3 }),
      inputField("subgoal", "subgoal", node.subgoal || ""),
      inputField("state", "state", node.state || ""),
      inputField("reward", "reward", node.reward || ""),
    ];
  }
  return [];
}

function fieldShell(label) {
  const row = document.createElement("label");
  row.className = "form-row";
  const span = document.createElement("span");
  span.className = "form-label";
  span.textContent = label;
  row.appendChild(span);
  return row;
}

function inputField(label, payloadKey, initial) {
  const row = fieldShell(label);
  const input = document.createElement("input");
  input.type = "text";
  input.value = initial == null ? "" : String(initial);
  row.appendChild(input);
  return {
    row,
    payloadKey,
    read() {
      const v = input.value;
      return v === String(initial ?? "") ? undefined : v;
    },
  };
}

function textareaField(label, payloadKey, initial, { rows = 3 } = {}) {
  const row = fieldShell(label);
  const ta = document.createElement("textarea");
  ta.rows = rows;
  ta.value = initial == null ? "" : String(initial);
  row.appendChild(ta);
  return {
    row,
    payloadKey,
    read() {
      const v = ta.value;
      return v === String(initial ?? "") ? undefined : v;
    },
  };
}

function numberField(label, payloadKey, initial, { step = "any" } = {}) {
  const row = fieldShell(label);
  const input = document.createElement("input");
  input.type = "number";
  input.step = String(step);
  input.value = initial == null ? "" : String(initial);
  row.appendChild(input);
  return {
    row,
    payloadKey,
    read() {
      const v = input.value;
      if (v === "" || v === String(initial ?? "")) return undefined;
      const parsed = step === 1 ? parseInt(v, 10) : parseFloat(v);
      if (Number.isNaN(parsed)) return undefined;
      return parsed;
    },
  };
}

function tagsField(label, payloadKey, initial) {
  const row = fieldShell(label);
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "comma-separated";
  const initialStr = (initial || []).join(", ");
  input.value = initialStr;
  row.appendChild(input);
  return {
    row,
    payloadKey,
    read() {
      const v = input.value;
      if (v === initialStr) return undefined;
      return v
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    },
  };
}
