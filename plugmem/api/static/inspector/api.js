// Tiny fetch wrapper for the PlugMem API.
//
// - X-API-Key is read from localStorage (key: "plugmem_api_key") if set.
// - JSON requests/responses by default.
// - Errors surface as thrown Error with .status, .body for callers to handle.

const BASE = "/api/v1";
const KEY_STORAGE = "plugmem_api_key";

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(key) {
  if (key) localStorage.setItem(KEY_STORAGE, key);
  else localStorage.removeItem(KEY_STORAGE);
}

async function request(method, path, { query, body } = {}) {
  let url = BASE + path;
  if (query) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === "") continue;
      params.append(k, v);
    }
    const qs = params.toString();
    if (qs) url += `?${qs}`;
  }

  const headers = { "Accept": "application/json" };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(url, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  let payload = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    payload = await res.json().catch(() => null);
  } else {
    payload = await res.text().catch(() => null);
  }

  if (!res.ok) {
    const detail = (payload && payload.detail) || payload || res.statusText;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.body = payload;
    throw err;
  }
  return payload;
}

export const api = {
  listGraphs: () => request("GET", "/graphs"),
  getStats: (gid) => request("GET", `/graphs/${encodeURIComponent(gid)}/stats`),
  search: (gid, { q, node_type, limit, only_active } = {}) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/search`, {
      query: { q, node_type, limit, only_active },
    }),
  getNode: (gid, type, id) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/node/${type}/${id}`),
  patchSemantic: (gid, sid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/semantic/${sid}`, { body }),
  patchProcedural: (gid, pid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/procedural/${pid}`, { body }),
  patchTag: (gid, tid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/tag/${tid}`, { body }),
  patchSubgoal: (gid, sgid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/subgoal/${sgid}`, { body }),
  patchEpisodic: (gid, eid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/episodic/${eid}`, { body }),
  seedDemo: ({ graph_id, reset } = {}) =>
    request("POST", "/demo/seed", { query: { graph_id, reset } }),
  recallTrace: (gid, body) =>
    request("POST", `/graphs/${encodeURIComponent(gid)}/recall_trace`, { body }),
  topology: (gid, { include_episodic, include_inactive, node_limit, tag_min_importance } = {}) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/topology`, {
      query: { include_episodic, include_inactive, node_limit, tag_min_importance },
    }),
  listSessions: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/sessions`),
  sessionTimeline: (gid, sessionId) =>
    request(
      "GET",
      `/graphs/${encodeURIComponent(gid)}/sessions/${encodeURIComponent(sessionId)}`,
    ),
  getPipelineSpec: () => request("GET", "/pipeline/spec"),
  listPipelinePrompts: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/prompts`),
  updatePipelinePrompt: (gid, name, body) =>
    request("PUT", `/graphs/${encodeURIComponent(gid)}/pipeline/prompts/${encodeURIComponent(name)}`, { body }),
  resetPipelinePrompt: (gid, name) =>
    request("POST", `/graphs/${encodeURIComponent(gid)}/pipeline/prompts/${encodeURIComponent(name)}/reset`),
  previewPipelinePrompt: (gid, name, body) =>
    request("POST", `/graphs/${encodeURIComponent(gid)}/pipeline/prompts/${encodeURIComponent(name)}/preview`, { body }),
  listPipelineModels: () => request("GET", "/pipeline/models"),
  updatePipelineModel: (role, body) =>
    request("PUT", `/pipeline/models/${encodeURIComponent(role)}`, { body }),
  testPipelineModel: (body) =>
    request("POST", "/pipeline/models/test", { body }),
  listPipelineTraces: (gid, { limit } = {}) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/traces`, {
      query: { limit },
    }),
  getPipelineTrace: (gid, traceId) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/traces/${encodeURIComponent(traceId)}`),
  getTraceCap: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/traces/cap`),
  setTraceCap: (gid, cap) =>
    request("PUT", `/graphs/${encodeURIComponent(gid)}/pipeline/traces/cap`, { body: { cap } }),
  getPipelineStats: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/stats`),
  listStepTraces: (gid, stepName, { limit } = {}) =>
    request(
      "GET",
      `/graphs/${encodeURIComponent(gid)}/pipeline/steps/${encodeURIComponent(stepName)}/traces`,
      { query: { limit } },
    ),
  // Pipeline binding + spec-driven YAML editor (Phase 5.5 / 6.5a)
  listPipelines: () => request("GET", "/pipeline/pipelines"),
  getGraphPipeline: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline`),
  setGraphPipeline: (gid, name) =>
    request("PUT", `/graphs/${encodeURIComponent(gid)}/pipeline`, { body: { pipeline: name } }),
  getPipelineSpecYaml: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/spec`),
  savePipelineSpecYaml: (gid, content, note = "") =>
    request("PUT", `/graphs/${encodeURIComponent(gid)}/pipeline/spec`, {
      body: { content, note },
    }),
  validatePipelineSpecYaml: (gid, content) =>
    request("POST", `/graphs/${encodeURIComponent(gid)}/pipeline/spec/validate`, {
      body: { content },
    }),
  listPipelineSpecVersions: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/pipeline/spec/versions`),
  getPipelineSpecVersion: (gid, vid) =>
    request(
      "GET",
      `/graphs/${encodeURIComponent(gid)}/pipeline/spec/versions/${encodeURIComponent(vid)}`,
    ),
  rollbackPipelineSpecVersion: (gid, vid) =>
    request(
      "POST",
      `/graphs/${encodeURIComponent(gid)}/pipeline/spec/versions/${encodeURIComponent(vid)}/rollback`,
    ),
};
