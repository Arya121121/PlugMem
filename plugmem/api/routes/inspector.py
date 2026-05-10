"""Memory Inspector endpoints — read/search/inspect/deactivate."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_embedder, get_graph_manager
from plugmem.api.schemas import (
    EpisodicUpdateRequest,
    NodeDetailResponse,
    ProceduralUpdateRequest,
    RecallAuditEntry,
    RecallListResponse,
    RecallTraceRequest,
    RecallTraceResponse,
    SearchResponse,
    SemanticUpdateRequest,
    SessionEvent,
    SessionListResponse,
    SubgoalUpdateRequest,
    TagUpdateRequest,
    SessionTimelineResponse,
    TopologyResponse,
)
from plugmem.core.graph_node import TagNode
from plugmem.graph_manager import GraphManager

router = APIRouter(prefix="/graphs", tags=["inspector"], dependencies=[Depends(require_api_key)])

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _manager() -> GraphManager:
    return get_graph_manager()


def _get_graph(graph_id: str):
    gm = _manager()
    try:
        return gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")


def _check_type(node_type: str) -> None:
    if node_type not in NODE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_type '{node_type}'. Must be one of: {list(NODE_TYPES)}",
        )


# ------------------------------------------------------------------ #
# Serializers — single source of truth shared by /search and /node
# ------------------------------------------------------------------ #


def _serialize_episodic(n) -> Dict[str, Any]:
    return {
        "id": n.episodic_id,
        "episodic_id": n.episodic_id,
        "observation": n.observation,
        "action": n.action,
        "subgoal": n.subgoal,
        "state": n.state,
        "reward": n.reward,
        "session_id": n.session_id,
        "time": n.time,
    }


def _serialize_semantic(n) -> Dict[str, Any]:
    return {
        "id": n.semantic_id,
        "semantic_id": n.semantic_id,
        "text": n.get_semantic_memory(),
        "tags": [t.tag for t in n.tag_nodes] or list(n.tags),
        "is_active": n.is_active,
        "credibility": getattr(n, "Credibility", 10),
        "session_id": getattr(n, "session_id", None),
        "date": getattr(n, "date", ""),
        "time": n.time,
        "n_tags": len(n.tag_nodes),
        "n_episodics": len(n.episodic_nodes),
        "n_bro": len(n.bro_semantic_nodes),
    }


def _serialize_tag(n) -> Dict[str, Any]:
    return {
        "id": n.tag_id,
        "tag_id": n.tag_id,
        "tag": n.tag,
        "importance": n.importance,
        "time": n.time,
        "n_semantics": len(n.semantic_nodes),
    }


def _serialize_subgoal(n) -> Dict[str, Any]:
    return {
        "id": n.subgoal_id,
        "subgoal_id": n.subgoal_id,
        "subgoal": n.subgoal,
        "time": n.time,
        "n_procedurals": len(n.procedural_nodes),
        "activated": n.activate,
    }


def _serialize_procedural(n) -> Dict[str, Any]:
    return {
        "id": n.procedural_id,
        "procedural_id": n.procedural_id,
        "text": n.get_procedural_memory(),
        "subgoals": [s.subgoal for s in n.subgoal_nodes] or list(n.subgoals),
        "return": n.Return,
        "time": n.time,
        "session_id": getattr(n, "session_id", None),
        "n_episodics": len(n.episodic_nodes),
    }


SERIALIZERS = {
    "episodic": _serialize_episodic,
    "semantic": _serialize_semantic,
    "tag": _serialize_tag,
    "subgoal": _serialize_subgoal,
    "procedural": _serialize_procedural,
}


def _node_text(node_type: str, node) -> str:
    """Return the searchable text field for a node."""
    if node_type == "semantic":
        return node.get_semantic_memory() or ""
    if node_type == "procedural":
        return node.get_procedural_memory() or ""
    if node_type == "tag":
        return node.tag or ""
    if node_type == "subgoal":
        return node.subgoal or ""
    if node_type == "episodic":
        return f"{node.observation}\n{node.action}"
    return ""


def _list_for_type(graph, node_type: str) -> List:
    return {
        "episodic": graph.episodic_nodes,
        "semantic": graph.semantic_nodes,
        "tag": graph.tag_nodes,
        "subgoal": graph.subgoal_nodes,
        "procedural": graph.procedural_nodes,
    }[node_type]


def _lookup_for_type(graph, node_type: str):
    return {
        "episodic": graph.episodic_id2node,
        "semantic": graph.semantic_id2node,
        "tag": graph.tag_id2node,
        "subgoal": graph.subgoal_id2node,
        "procedural": graph.procedural_id2node,
    }[node_type]


# ------------------------------------------------------------------ #
# /search — substring filter on the node text field
# ------------------------------------------------------------------ #


@router.get("/{graph_id}/search", response_model=SearchResponse)
async def search_nodes(
    graph_id: str,
    q: str = "",
    node_type: str = "semantic",
    limit: int = 50,
    only_active: bool = False,
) -> SearchResponse:
    _check_type(node_type)
    graph = _get_graph(graph_id)

    nodes = _list_for_type(graph, node_type)
    serializer = SERIALIZERS[node_type]
    needle = q.casefold().strip()

    matches: List[Dict[str, Any]] = []
    for node in nodes:
        if only_active and node_type == "semantic" and not node.is_active:
            continue
        if needle and needle not in _node_text(node_type, node).casefold():
            continue
        matches.append(serializer(node))

    # sort newest first
    matches.sort(key=lambda d: d.get("time") if isinstance(d.get("time"), (int, float)) else 0, reverse=True)
    truncated = matches[: max(0, limit)]

    return SearchResponse(
        graph_id=graph_id,
        node_type=node_type,
        query=q,
        count=len(matches),
        nodes=truncated,
    )


# ------------------------------------------------------------------ #
# /node/{type}/{id} — single node + one-hop edges
# ------------------------------------------------------------------ #


@router.get("/{graph_id}/node/{node_type}/{node_id}", response_model=NodeDetailResponse)
async def get_node_detail(graph_id: str, node_type: str, node_id: int) -> NodeDetailResponse:
    _check_type(node_type)
    graph = _get_graph(graph_id)

    lookup = _lookup_for_type(graph, node_type)
    node = lookup.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"{node_type} node {node_id} not found")

    serializer = SERIALIZERS[node_type]
    edges: Dict[str, List[Dict[str, Any]]] = {}

    if node_type == "semantic":
        edges["tags"] = [_serialize_tag(t) for t in node.tag_nodes]
        edges["episodics"] = [_serialize_episodic(e) for e in node.episodic_nodes]
        edges["bro_semantics"] = [_serialize_semantic(s) for s in node.bro_semantic_nodes]
        edges["son_semantics"] = [_serialize_semantic(s) for s in getattr(node, "son_semantic", [])]
    elif node_type == "tag":
        edges["semantics"] = [_serialize_semantic(s) for s in node.semantic_nodes]
    elif node_type == "subgoal":
        edges["procedurals"] = [_serialize_procedural(p) for p in node.procedural_nodes]
    elif node_type == "procedural":
        edges["subgoals"] = [_serialize_subgoal(s) for s in node.subgoal_nodes]
        edges["episodics"] = [_serialize_episodic(e) for e in node.episodic_nodes]
    elif node_type == "episodic":
        # reverse lookup: which semantics linked back?
        linked = [s for s in graph.semantic_nodes if any(e.episodic_id == node.episodic_id for e in s.episodic_nodes)]
        edges["semantics"] = [_serialize_semantic(s) for s in linked]

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type=node_type,
        node=serializer(node),
        edges=edges,
    )


# ------------------------------------------------------------------ #
# PATCH semantic node — currently only is_active is mutable
# ------------------------------------------------------------------ #


@router.post("/{graph_id}/recall_trace", response_model=RecallTraceResponse)
async def recall_trace(graph_id: str, body: RecallTraceRequest) -> RecallTraceResponse:
    """Run the retrieval pipeline with full instrumentation.

    Without ``auto_plan=true`` the LLM is not called — sensible defaults are
    used (mode=semantic_memory, no tags, subgoal=observation), so the demo
    runs end-to-end without an LLM service.
    """
    graph = _get_graph(graph_id)
    try:
        result = graph.retrieve_with_trace(
            observation=body.observation,
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            query_tags=body.query_tags,
            next_subgoal=body.next_subgoal,
            auto_plan=body.auto_plan,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recall failed: {exc}") from exc

    # Audit the trace call so it shows up in the Sessions view alongside
    # /retrieve and /reason. Best-effort — never fails the response.
    try:
        from datetime import datetime, timezone

        graph.storage.add_recall(
            graph_id,
            endpoint="recall_trace",
            ts=datetime.now(timezone.utc).isoformat(),
            graph_time=graph.semantic_time,
            session_id=body.session_id,
            observation=body.observation or "",
            goal=body.goal or "",
            subgoal=body.subgoal or "",
            state=body.state or "",
            task_type=body.task_type or "",
            mode=result.get("mode", ""),
            next_subgoal=(result.get("plan") or {}).get("next_subgoal", "") or "",
            query_tags=(result.get("plan") or {}).get("query_tags", []) or [],
            selected_semantic_ids=(result.get("selected") or {}).get("semantic_ids", []) or [],
            selected_procedural_ids=(result.get("selected") or {}).get("procedural_ids", []) or [],
            n_messages=len(result.get("rendered_prompt") or []),
        )
    except Exception:
        pass

    return RecallTraceResponse(**result)


# ------------------------------------------------------------------ #
# /topology — cytoscape-shaped node/edge dump for the Graph view
# ------------------------------------------------------------------ #


_UID_PREFIX = {
    "semantic": "sem",
    "tag": "tag",
    "procedural": "proc",
    "subgoal": "sg",
    "episodic": "epis",
}
_TYPE_PRIORITY = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _uid(node_type: str, node_id: int) -> str:
    return f"{_UID_PREFIX[node_type]}-{node_id}"


def _short(s: str, n: int = 60) -> str:
    if not s:
        return ""
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _topology_node(node_type: str, node) -> Dict[str, Any]:
    if node_type == "semantic":
        return {
            "data": {
                "id": _uid("semantic", node.semantic_id),
                "type": "semantic",
                "node_id": node.semantic_id,
                "label": _short(node.get_semantic_memory()),
                "is_active": node.is_active,
                "credibility": getattr(node, "Credibility", 10),
                "time": node.time,
            },
            "classes": "semantic" if node.is_active else "semantic inactive",
        }
    if node_type == "tag":
        return {
            "data": {
                "id": _uid("tag", node.tag_id),
                "type": "tag",
                "node_id": node.tag_id,
                "label": node.tag,
                "importance": node.importance,
                "time": node.time,
            },
            "classes": "tag",
        }
    if node_type == "subgoal":
        return {
            "data": {
                "id": _uid("subgoal", node.subgoal_id),
                "type": "subgoal",
                "node_id": node.subgoal_id,
                "label": _short(node.subgoal, 40),
                "activated": node.activate,
                "time": node.time,
            },
            "classes": "subgoal",
        }
    if node_type == "procedural":
        return {
            "data": {
                "id": _uid("procedural", node.procedural_id),
                "type": "procedural",
                "node_id": node.procedural_id,
                "label": _short(node.get_procedural_memory()),
                "return": node.Return,
                "time": node.time,
            },
            "classes": "procedural",
        }
    # episodic
    return {
        "data": {
            "id": _uid("episodic", node.episodic_id),
            "type": "episodic",
            "node_id": node.episodic_id,
            "label": _short(node.observation or node.action or "(empty)", 40),
            "session_id": node.session_id,
            "time": node.time,
        },
        "classes": "episodic",
    }


def _edge(source: str, target: str, kind: str) -> Dict[str, Any]:
    return {
        "data": {
            "id": f"{source}__{kind}__{target}",
            "source": source,
            "target": target,
            "kind": kind,
        },
        "classes": kind,
    }


def _build_topology(
    graph,
    *,
    include_episodic: bool,
    include_inactive: bool,
    node_limit: int,
    tag_min_importance: int,
) -> Dict[str, Any]:
    pools: Dict[str, List] = {
        "semantic": list(graph.semantic_nodes),
        "tag": [t for t in graph.tag_nodes if t.importance >= tag_min_importance],
        "procedural": list(graph.procedural_nodes),
        "subgoal": list(graph.subgoal_nodes),
        "episodic": list(graph.episodic_nodes) if include_episodic else [],
    }
    if not include_inactive:
        pools["semantic"] = [s for s in pools["semantic"] if s.is_active]

    for pool in pools.values():
        pool.sort(key=lambda n: getattr(n, "time", 0) or 0, reverse=True)

    total = sum(len(p) for p in pools.values())
    truncated = total > node_limit

    if truncated:
        # Equal split across types that have items, then distribute leftovers
        # by priority order.
        active_types = [t for t in _TYPE_PRIORITY if pools[t]]
        share = node_limit // max(1, len(active_types))
        chosen = {t: [] for t in pools}
        budget = node_limit
        for t in active_types:
            take = min(share, len(pools[t]))
            chosen[t] = pools[t][:take]
            budget -= take
        for t in _TYPE_PRIORITY:
            if budget <= 0:
                break
            already = len(chosen[t])
            extra = min(budget, len(pools[t]) - already)
            if extra > 0:
                chosen[t] = pools[t][: already + extra]
                budget -= extra
        pools = chosen

    # Emit nodes
    out_nodes: List[Dict[str, Any]] = []
    included: Dict[str, set] = {t: set() for t in pools}
    for node_type in _TYPE_PRIORITY:
        for n in pools[node_type]:
            out_nodes.append(_topology_node(node_type, n))
            nid = getattr(n, f"{'episodic' if node_type == 'episodic' else node_type}_id")
            included[node_type].add(nid)

    # Emit edges, only between included nodes; dedupe by edge id
    out_edges: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(source: str, target: str, kind: str) -> None:
        eid = f"{source}__{kind}__{target}"
        if eid in seen:
            return
        seen.add(eid)
        out_edges.append(_edge(source, target, kind))

    for s in pools["semantic"]:
        sid = _uid("semantic", s.semantic_id)
        for t in s.tag_nodes:
            if t.tag_id in included["tag"]:
                _add(sid, _uid("tag", t.tag_id), "tagged")
        for e in s.episodic_nodes:
            if e.episodic_id in included["episodic"]:
                _add(sid, _uid("episodic", e.episodic_id), "evidenced_by")
        for bro in s.bro_semantic_nodes:
            if bro is None or bro.semantic_id not in included["semantic"]:
                continue
            # Symmetric: emit once with the lower id as source.
            a, b = sorted((s.semantic_id, bro.semantic_id))
            _add(_uid("semantic", a), _uid("semantic", b), "related")
        for son in getattr(s, "son_semantic", []):
            if son is None or son.semantic_id not in included["semantic"]:
                continue
            _add(sid, _uid("semantic", son.semantic_id), "derived_from")

    for p in pools["procedural"]:
        pid = _uid("procedural", p.procedural_id)
        for sg in p.subgoal_nodes:
            if sg.subgoal_id in included["subgoal"]:
                _add(pid, _uid("subgoal", sg.subgoal_id), "grouped_by")
        for e in p.episodic_nodes:
            if e.episodic_id in included["episodic"]:
                _add(pid, _uid("episodic", e.episodic_id), "from_session")

    counts = {t: len(pools[t]) for t in pools}
    counts["total_nodes"] = len(out_nodes)
    counts["total_edges"] = len(out_edges)
    return {
        "nodes": out_nodes,
        "edges": out_edges,
        "counts": counts,
        "truncated": truncated,
    }


@router.get("/{graph_id}/topology", response_model=TopologyResponse)
async def get_topology(
    graph_id: str,
    include_episodic: bool = False,
    include_inactive: bool = True,
    node_limit: int = 500,
    tag_min_importance: int = 0,
) -> TopologyResponse:
    """Cytoscape-shaped node/edge dump for the Graph tab.

    Episodics are off by default — they're numerous and noisy. Inactive
    semantics are kept by default but rendered with a faded class so users
    can see what was deactivated.
    """
    if node_limit <= 0:
        raise HTTPException(status_code=400, detail="node_limit must be positive")
    graph = _get_graph(graph_id)
    payload = _build_topology(
        graph,
        include_episodic=include_episodic,
        include_inactive=include_inactive,
        node_limit=node_limit,
        tag_min_importance=tag_min_importance,
    )
    return TopologyResponse(
        graph_id=graph_id,
        nodes=payload["nodes"],
        edges=payload["edges"],
        counts=payload["counts"],
        truncated=payload["truncated"],
        node_limit=node_limit,
    )


@router.patch("/{graph_id}/semantic/{semantic_id}", response_model=NodeDetailResponse)
async def update_semantic(
    graph_id: str,
    semantic_id: int,
    body: SemanticUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.semantic_id2node.get(semantic_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"semantic node {semantic_id} not found")

    metadata_updates: Dict[str, Any] = {}
    new_text: Optional[str] = None
    new_embedding = None
    touched = False

    if body.is_active is not None:
        node.is_active = bool(body.is_active)
        metadata_updates["is_active"] = node.is_active
        touched = True

    if body.credibility is not None:
        node.Credibility = int(body.credibility)
        metadata_updates["credibility"] = node.Credibility
        touched = True

    if body.text is not None:
        new_text = body.text
        node.semantic_memory_str = new_text
        try:
            new_embedding = get_embedder().embed(new_text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
        node.embedding = new_embedding
        touched = True

    if body.tags is not None:
        _reconcile_semantic_tags(graph, node, body.tags)
        metadata_updates["tags"] = list(node.tags)
        metadata_updates["tag_ids"] = [t.tag_id for t in node.tag_nodes]
        touched = True

    if not touched:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    embedding_arg = None
    if new_embedding is not None:
        if hasattr(new_embedding, "tolist"):
            embedding_arg = new_embedding.tolist()
        else:
            embedding_arg = list(new_embedding)

    graph.storage.update_semantic(
        graph_id,
        semantic_id,
        text=new_text,
        embedding=embedding_arg,
        metadata_updates=metadata_updates or None,
    )

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="semantic",
        node=_serialize_semantic(node),
        edges={
            "tags": [_serialize_tag(t) for t in node.tag_nodes],
            "episodics": [_serialize_episodic(e) for e in node.episodic_nodes],
            "bro_semantics": [_serialize_semantic(s) for s in node.bro_semantic_nodes],
        },
    )


def _reconcile_semantic_tags(graph, sem_node, new_tag_strs: List[str]) -> None:
    """Diff tag list, attach/detach as needed, persist tag-side changes."""
    embedder = get_embedder()
    # Preserve order while deduping and dropping empty strings.
    seen: set = set()
    cleaned: List[str] = []
    for t in new_tag_strs:
        s = (t or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    new_set = set(cleaned)
    old_set = {t.tag for t in sem_node.tag_nodes}

    # Detach from removed tags
    for tag_node in list(sem_node.tag_nodes):
        if tag_node.tag in new_set:
            continue
        sem_node.tag_nodes.remove(tag_node)
        if sem_node in tag_node.semantic_nodes:
            tag_node.semantic_nodes.remove(sem_node)
        graph.storage.update_tag(
            graph.graph_id,
            tag_node.tag_id,
            metadata_updates={
                "semantic_ids": [s.semantic_id for s in tag_node.semantic_nodes],
            },
        )

    # Attach to added tags (create new tag nodes if needed)
    for tag_str in cleaned:
        if tag_str in old_set:
            continue
        tag_node = graph.tag2node.get(tag_str)
        if tag_node is None:
            try:
                emb = embedder.embed(tag_str)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"tag embedding failed: {exc}") from exc
            tag_id = len(graph.tag_nodes)
            tag_node = TagNode(
                tag=tag_str,
                tag_id=tag_id,
                embedding=emb,
                time=graph.semantic_time,
            )
            graph.tag_nodes.append(tag_node)
            graph.tag2node[tag_str] = tag_node
            graph.tag_id2node[tag_id] = tag_node
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            graph.storage.add_tag(
                graph.graph_id,
                tag_id=tag_id,
                tag=tag_str,
                embedding=emb_list,
                semantic_ids=[sem_node.semantic_id],
                time=graph.semantic_time,
            )
        else:
            if sem_node not in tag_node.semantic_nodes:
                tag_node.semantic_nodes.append(sem_node)
            graph.storage.update_tag(
                graph.graph_id,
                tag_node.tag_id,
                metadata_updates={
                    "semantic_ids": [s.semantic_id for s in tag_node.semantic_nodes],
                },
            )
        sem_node.tag_nodes.append(tag_node)

    # Re-sync the cached string list to match canonical order
    sem_node.tags = [t.tag for t in sem_node.tag_nodes]


@router.patch("/{graph_id}/procedural/{procedural_id}", response_model=NodeDetailResponse)
async def update_procedural(
    graph_id: str,
    procedural_id: int,
    body: ProceduralUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.procedural_id2node.get(procedural_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"procedural node {procedural_id} not found")

    metadata_updates: Dict[str, Any] = {}
    new_text: Optional[str] = None
    new_embedding = None
    touched = False

    if body.text is not None:
        new_text = body.text
        node.procedural_memory_str = new_text
        try:
            new_embedding = get_embedder().embed(new_text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
        node.embedding = new_embedding
        touched = True

    if body.return_value is not None:
        node.Return = float(body.return_value)
        metadata_updates["return"] = node.Return
        touched = True

    if not touched:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    embedding_arg = None
    if new_embedding is not None:
        embedding_arg = (
            new_embedding.tolist() if hasattr(new_embedding, "tolist") else list(new_embedding)
        )

    graph.storage.update_procedural(
        graph_id,
        procedural_id,
        text=new_text,
        embedding=embedding_arg,
        metadata_updates=metadata_updates or None,
    )

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="procedural",
        node=_serialize_procedural(node),
        edges={
            "subgoals": [_serialize_subgoal(s) for s in node.subgoal_nodes],
            "episodics": [_serialize_episodic(e) for e in node.episodic_nodes],
        },
    )


@router.patch("/{graph_id}/tag/{tag_id}", response_model=NodeDetailResponse)
async def update_tag(
    graph_id: str,
    tag_id: int,
    body: TagUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.tag_id2node.get(tag_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"tag node {tag_id} not found")

    metadata_updates: Dict[str, Any] = {}
    new_text: Optional[str] = None
    new_embedding = None
    touched = False

    if body.tag is not None:
        new_text = body.tag.strip()
        if not new_text:
            raise HTTPException(status_code=422, detail="tag string cannot be empty")
        if new_text != node.tag and new_text in graph.tag2node:
            raise HTTPException(
                status_code=409,
                detail=f"tag '{new_text}' already exists (id={graph.tag2node[new_text].tag_id})",
            )
        try:
            new_embedding = get_embedder().embed(new_text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc

        # Update lookup tables and propagate to semantic nodes that hold the tag string.
        old_tag = node.tag
        if old_tag in graph.tag2node and graph.tag2node[old_tag] is node:
            del graph.tag2node[old_tag]
        node.tag = new_text
        node.embedding = new_embedding
        graph.tag2node[new_text] = node
        for sem_node in node.semantic_nodes:
            sem_node.tags = [new_text if t == old_tag else t for t in sem_node.tags]
            graph.storage.update_semantic(
                graph_id,
                sem_node.semantic_id,
                metadata_updates={"tags": list(sem_node.tags)},
            )
        touched = True

    if body.importance is not None:
        node.importance = int(body.importance)
        metadata_updates["importance"] = node.importance
        touched = True

    if not touched:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    embedding_arg = None
    if new_embedding is not None:
        embedding_arg = (
            new_embedding.tolist() if hasattr(new_embedding, "tolist") else list(new_embedding)
        )

    graph.storage.update_tag(
        graph_id,
        tag_id,
        tag=new_text,
        embedding=embedding_arg,
        metadata_updates=metadata_updates or None,
    )

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="tag",
        node=_serialize_tag(node),
        edges={"semantics": [_serialize_semantic(s) for s in node.semantic_nodes]},
    )


@router.patch("/{graph_id}/subgoal/{subgoal_id}", response_model=NodeDetailResponse)
async def update_subgoal(
    graph_id: str,
    subgoal_id: int,
    body: SubgoalUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.subgoal_id2node.get(subgoal_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"subgoal node {subgoal_id} not found")

    if body.subgoal is None:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    new_text = body.subgoal.strip()
    if not new_text:
        raise HTTPException(status_code=422, detail="subgoal string cannot be empty")
    if new_text != node.subgoal and new_text in graph.subgoal2node:
        raise HTTPException(
            status_code=409,
            detail=f"subgoal '{new_text}' already exists (id={graph.subgoal2node[new_text].subgoal_id})",
        )

    try:
        new_embedding = get_embedder().embed(new_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc

    old_subgoal = node.subgoal
    if old_subgoal in graph.subgoal2node and graph.subgoal2node[old_subgoal] is node:
        del graph.subgoal2node[old_subgoal]
    node.subgoal = new_text
    node.embedding = new_embedding
    graph.subgoal2node[new_text] = node

    # Procedurals cache subgoal strings on the node and in storage metadata.
    # Episodics that fed those procedurals carry the same subgoal string as a
    # free-form copy — rename them too so the detail panels stay consistent.
    touched_episodics: set = set()
    for proc_node in node.procedural_nodes:
        proc_node.subgoals = [new_text if s == old_subgoal else s for s in proc_node.subgoals]
        graph.storage.update_procedural(
            graph_id,
            proc_node.procedural_id,
            metadata_updates={"subgoal": new_text},
        )
        for epis_node in proc_node.episodic_nodes:
            if epis_node.episodic_id in touched_episodics:
                continue
            if epis_node.subgoal != old_subgoal:
                continue
            epis_node.subgoal = new_text
            graph.storage.update_episodic(
                graph_id,
                epis_node.episodic_id,
                metadata_updates={"subgoal": new_text},
            )
            touched_episodics.add(epis_node.episodic_id)

    embedding_arg = (
        new_embedding.tolist() if hasattr(new_embedding, "tolist") else list(new_embedding)
    )
    graph.storage.update_subgoal(
        graph_id,
        subgoal_id,
        subgoal=new_text,
        embedding=embedding_arg,
    )

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="subgoal",
        node=_serialize_subgoal(node),
        edges={"procedurals": [_serialize_procedural(p) for p in node.procedural_nodes]},
    )


@router.patch("/{graph_id}/episodic/{episodic_id}", response_model=NodeDetailResponse)
async def update_episodic(
    graph_id: str,
    episodic_id: int,
    body: EpisodicUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.episodic_id2node.get(episodic_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"episodic node {episodic_id} not found")

    metadata_updates: Dict[str, Any] = {}
    document_changed = False
    touched = False

    if body.observation is not None:
        node.observation = body.observation
        metadata_updates["observation"] = node.observation
        document_changed = True
        touched = True
    if body.action is not None:
        node.action = body.action
        metadata_updates["action"] = node.action
        document_changed = True
        touched = True
    if body.subgoal is not None:
        node.subgoal = body.subgoal
        metadata_updates["subgoal"] = node.subgoal
        touched = True
    if body.state is not None:
        node.state = body.state
        metadata_updates["state"] = node.state
        touched = True
    if body.reward is not None:
        node.reward = body.reward
        metadata_updates["reward"] = node.reward
        touched = True

    if not touched:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    document = None
    if document_changed:
        document = (
            f"{node.observation}\n{node.action}" if (node.observation or node.action) else ""
        )

    graph.storage.update_episodic(
        graph_id,
        episodic_id,
        document=document,
        metadata_updates=metadata_updates or None,
    )

    linked_semantics = [
        s for s in graph.semantic_nodes if any(e.episodic_id == node.episodic_id for e in s.episodic_nodes)
    ]
    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="episodic",
        node=_serialize_episodic(node),
        edges={"semantics": [_serialize_semantic(s) for s in linked_semantics]},
    )


# ------------------------------------------------------------------ #
# /recalls — audit log of /retrieve, /reason, /recall_trace calls
# ------------------------------------------------------------------ #


@router.get("/{graph_id}/recalls", response_model=RecallListResponse)
async def list_recalls(
    graph_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> RecallListResponse:
    graph = _get_graph(graph_id)
    rows = graph.storage.list_recalls(graph_id, session_id=session_id, limit=limit)
    return RecallListResponse(
        graph_id=graph_id,
        session_id=session_id,
        count=len(rows),
        recalls=[RecallAuditEntry(**row) for row in rows],
    )


@router.get("/{graph_id}/sessions", response_model=SessionListResponse)
async def list_sessions(graph_id: str) -> SessionListResponse:
    _get_graph(graph_id)  # 404 if missing
    sessions = _manager().storage.list_sessions(graph_id)
    return SessionListResponse(graph_id=graph_id, sessions=sessions)


def _coerce_int_time(t: Any) -> int:
    if isinstance(t, int):
        return t
    try:
        return int(t)
    except (TypeError, ValueError):
        return 0


@router.get(
    "/{graph_id}/sessions/{session_id}",
    response_model=SessionTimelineResponse,
)
async def get_session_timeline(
    graph_id: str,
    session_id: str,
) -> SessionTimelineResponse:
    """Chronological merge of inserts + recalls for one session.

    Sorted by time ascending. Inserts come before recalls at the same
    time (the recall reads what the insert just produced).
    """
    graph = _get_graph(graph_id)
    events: List[Dict[str, Any]] = []

    for n in graph.episodic_nodes:
        if getattr(n, "session_id", None) != session_id:
            continue
        text = "\n".join(s for s in (n.observation, n.action) if s)
        events.append({
            "kind": "insert",
            "node_type": "episodic",
            "node_id": n.episodic_id,
            "time": _coerce_int_time(n.time),
            "label": _short_event_text(n.observation or n.action or "(empty)"),
            "text": text,
            "subgoal": n.subgoal or None,
        })

    for n in graph.semantic_nodes:
        if getattr(n, "session_id", None) != session_id:
            continue
        events.append({
            "kind": "insert",
            "node_type": "semantic",
            "node_id": n.semantic_id,
            "time": _coerce_int_time(n.time),
            "label": _short_event_text(n.get_semantic_memory()),
            "text": n.get_semantic_memory(),
            "is_active": n.is_active,
            "credibility": getattr(n, "Credibility", 10),
        })

    for n in graph.procedural_nodes:
        if getattr(n, "session_id", None) != session_id:
            continue
        events.append({
            "kind": "insert",
            "node_type": "procedural",
            "node_id": n.procedural_id,
            "time": _coerce_int_time(n.time),
            "label": _short_event_text(n.get_procedural_memory()),
            "text": n.get_procedural_memory(),
            "return_value": n.Return,
        })

    for r in graph.storage.list_recalls(graph_id, session_id=session_id, limit=10_000):
        events.append({
            "kind": "recall",
            "endpoint": r.get("endpoint"),
            "recall_id": r.get("recall_id"),
            "time": _coerce_int_time(r.get("graph_time", 0)),
            "ts": r.get("ts"),
            "observation": r.get("observation", ""),
            "mode": r.get("mode", ""),
            "next_subgoal": r.get("next_subgoal", "") or "",
            "query_tags": r.get("query_tags", []) or [],
            "selected_semantic_ids": r.get("selected_semantic_ids", []) or [],
            "selected_procedural_ids": r.get("selected_procedural_ids", []) or [],
            "n_messages": r.get("n_messages", 0),
        })

    # Sort: time asc, then insert-before-recall, then by id for stability.
    def _sort_key(e: Dict[str, Any]):
        kind_rank = 0 if e["kind"] == "insert" else 1
        ident = e.get("node_id") if e["kind"] == "insert" else e.get("recall_id")
        return (e["time"], kind_rank, ident or 0)

    events.sort(key=_sort_key)

    return SessionTimelineResponse(
        graph_id=graph_id,
        session_id=session_id,
        count=len(events),
        events=[SessionEvent(**e) for e in events],
    )


def _short_event_text(s: str, n: int = 100) -> str:
    if not s:
        return ""
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
