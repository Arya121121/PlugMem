"""ChromaDB storage wrapper for PlugMem memory graphs.

Replaces all file-based save_*/update_* functions with ChromaDB operations.
Each memory graph gets 5 collections: semantic, procedural, tag, subgoal, episodic.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

logger = logging.getLogger(__name__)

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _collection_name(graph_id: str, node_type: str) -> str:
    return f"{graph_id}_{node_type}"


def _to_list(v: Any) -> Optional[List[float]]:
    """Convert numpy arrays or lists to plain float lists for ChromaDB."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    if isinstance(v, list):
        return v
    return list(v)


def _serialize_list(v: Any) -> str:
    """Serialize a Python list to JSON string for ChromaDB metadata."""
    if v is None:
        return "[]"
    return json.dumps(v)


def _deserialize_list(s: str) -> list:
    """Deserialize a JSON string back to a Python list."""
    if not s:
        return []
    return json.loads(s)


class ChromaStorage:
    """Manages ChromaDB collections for PlugMem memory graphs."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        embedding_function=None,
        embedding_client=None,
    ):
        self._client = client
        # Build embedding function: prefer explicit, then wrap client, else None
        if embedding_function is not None:
            self._embedding_fn = embedding_function
        elif embedding_client is not None:
            from plugmem.clients.embedding import PlugMemEmbeddingFunction
            self._embedding_fn = PlugMemEmbeddingFunction(embedding_client)
        else:
            self._embedding_fn = None

    # ------------------------------------------------------------------ #
    # Graph lifecycle
    # ------------------------------------------------------------------ #

    def create_graph(self, graph_id: str) -> None:
        """Create 5 collections for a new memory graph."""
        for node_type in NODE_TYPES:
            name = _collection_name(graph_id, node_type)
            self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        logger.info("Created graph %s with 5 collections", graph_id)

    def delete_graph(self, graph_id: str) -> None:
        """Delete all collections for a memory graph (incl. recall audit)."""
        for node_type in NODE_TYPES:
            try:
                self._client.delete_collection(_collection_name(graph_id, node_type))
            except Exception:
                pass
        try:
            self._client.delete_collection(f"{graph_id}_recall_audit")
        except Exception:
            pass
        try:
            self._client.delete_collection(f"{graph_id}_pipeline_trace")
        except Exception:
            pass
        try:
            self._client.delete_collection(f"{graph_id}_pipeline_settings")
        except Exception:
            pass

    def list_graphs(self) -> List[str]:
        """List all graph IDs by inspecting collection names."""
        collections = self._client.list_collections()
        graph_ids: set[str] = set()
        for col in collections:
            # col may be a Collection object or a string depending on chromadb version.
            # In chromadb >= 0.6 list_collections returns names (strings); accessing
            # `.name` on the proxy raises NotImplementedError despite hasattr being True.
            if isinstance(col, str):
                col_name = col
            else:
                try:
                    col_name = col.name
                except (AttributeError, NotImplementedError):
                    col_name = str(col)
            for nt in NODE_TYPES:
                suffix = f"_{nt}"
                if col_name.endswith(suffix):
                    graph_ids.add(col_name[: -len(suffix)])
                    break
        return sorted(graph_ids)

    def graph_exists(self, graph_id: str) -> bool:
        """Check if a graph exists."""
        try:
            self._client.get_collection(_collection_name(graph_id, "semantic"))
            return True
        except Exception:
            return False

    def get_graph_stats(self, graph_id: str) -> Dict[str, int]:
        """Return node counts per type."""
        stats = {}
        for node_type in NODE_TYPES:
            col = self._client.get_collection(
                _collection_name(graph_id, node_type),
                embedding_function=self._embedding_fn,
            )
            stats[node_type] = col.count()
        return stats

    # ------------------------------------------------------------------ #
    # Collection accessors
    # ------------------------------------------------------------------ #

    def _col(self, graph_id: str, node_type: str):
        return self._client.get_collection(
            _collection_name(graph_id, node_type),
            embedding_function=self._embedding_fn,
        )

    # ------------------------------------------------------------------ #
    # Episodic nodes
    # ------------------------------------------------------------------ #

    def add_episodic(
        self,
        graph_id: str,
        episodic_id: int,
        observation: str = "",
        action: str = "",
        time: Any = "",
        session_id: Optional[str] = None,
        subgoal: str = "",
        state: str = "",
        reward: str = "",
        embedding: Optional[List[float]] = None,
    ) -> None:
        doc = f"{observation}\n{action}" if observation or action else ""
        metadata: Dict[str, Any] = {
            "episodic_id": episodic_id,
            "observation": observation,
            "action": action,
            "time": str(time),
            "subgoal": subgoal,
            "state": state,
            "reward": reward,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        col = self._col(graph_id, "episodic")
        kwargs: Dict[str, Any] = {
            "ids": [str(episodic_id)],
            "documents": [doc],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_episodic(
        self,
        graph_id: str,
        episodic_id: int,
        document: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "episodic")
        kwargs: Dict[str, Any] = {"ids": [str(episodic_id)]}
        if document is not None:
            kwargs["documents"] = [document]
        if metadata_updates:
            processed: Dict[str, Any] = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                elif k == "time":
                    processed[k] = str(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def get_episodic(self, graph_id: str, episodic_id: int) -> Optional[Dict]:
        col = self._col(graph_id, "episodic")
        result = col.get(ids=[str(episodic_id)], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return result["metadatas"][0]

    def get_all_episodic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "episodic")
        return col.get(include=["documents", "metadatas"])

    # ------------------------------------------------------------------ #
    # Semantic nodes
    # ------------------------------------------------------------------ #

    def add_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        tag_ids: Optional[List[int]] = None,
        time: int = 0,
        is_active: bool = True,
        episodic_ids: Optional[List[int]] = None,
        bro_semantic_ids: Optional[List[int]] = None,
        son_semantic_ids: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        credibility: int = 10,
        date: str = "",
    ) -> None:
        metadata: Dict[str, Any] = {
            "semantic_id": semantic_id,
            "tags": _serialize_list(tags or []),
            "tag_ids": _serialize_list(tag_ids or []),
            "time": time,
            "is_active": is_active,
            "episodic_ids": _serialize_list(episodic_ids or []),
            "bro_semantic_ids": _serialize_list(bro_semantic_ids or []),
            "son_semantic_ids": _serialize_list(son_semantic_ids or []),
            "credibility": credibility,
            "date": date,
        }
        if session_id is not None:
            metadata["session_id"] = session_id

        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "ids": [str(semantic_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {"ids": [str(semantic_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            # Serialize list values
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_semantic(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> Dict:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "query_embeddings": [_to_list(query_embedding)],
            "n_results": min(n_results, max(col.count(), 1)),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def get_all_semantic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "semantic")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Tag nodes
    # ------------------------------------------------------------------ #

    def add_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: str,
        embedding: Optional[List[float]] = None,
        semantic_ids: Optional[List[int]] = None,
        time: int = 0,
        importance: int = 1,
    ) -> None:
        metadata: Dict[str, Any] = {
            "tag_id": tag_id,
            "semantic_ids": _serialize_list(semantic_ids or []),
            "time": time,
            "importance": importance,
        }
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {
            "ids": [str(tag_id)],
            "documents": [tag],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {"ids": [str(tag_id)]}
        if tag is not None:
            kwargs["documents"] = [tag]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_tag(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "tag")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_tags(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "tag")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Subgoal nodes
    # ------------------------------------------------------------------ #

    def add_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: str,
        embedding: Optional[List[float]] = None,
        procedural_ids: Optional[List[int]] = None,
        time: int = 0,
    ) -> None:
        metadata: Dict[str, Any] = {
            "subgoal_id": subgoal_id,
            "procedural_ids": _serialize_list(procedural_ids or []),
            "time": time,
        }
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {
            "ids": [str(subgoal_id)],
            "documents": [subgoal],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {"ids": [str(subgoal_id)]}
        if subgoal is not None:
            kwargs["documents"] = [subgoal]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_subgoal(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_subgoals(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Procedural nodes
    # ------------------------------------------------------------------ #

    def add_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        subgoal: str = "",
        subgoal_id: Optional[int] = None,
        episodic_ids: Optional[List[int]] = None,
        time: int = 0,
        return_value: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        metadata: Dict[str, Any] = {
            "procedural_id": procedural_id,
            "subgoal": subgoal,
            "time": time,
            "return": return_value,
            "episodic_ids": _serialize_list(episodic_ids or []),
        }
        if subgoal_id is not None:
            metadata["subgoal_id"] = subgoal_id
        if session_id is not None:
            metadata["session_id"] = session_id
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {
            "ids": [str(procedural_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {"ids": [str(procedural_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_procedural(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_procedural(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Recall audit log
    # ------------------------------------------------------------------ #
    #
    # A separate collection per graph (`{graph_id}_recall_audit`) records
    # every /retrieve, /reason, and /recall_trace call. Lazily created on
    # first append; not enumerated by `list_graphs` because the suffix is
    # outside NODE_TYPES.

    def _recall_col(self, graph_id: str):
        return self._client.get_or_create_collection(
            name=f"{graph_id}_recall_audit",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    def add_recall(
        self,
        graph_id: str,
        *,
        endpoint: str,
        observation: str,
        ts: str,
        graph_time: int = 0,
        session_id: Optional[str] = None,
        goal: str = "",
        subgoal: str = "",
        state: str = "",
        task_type: str = "",
        mode: str = "",
        next_subgoal: str = "",
        query_tags: Optional[List[str]] = None,
        selected_semantic_ids: Optional[List[int]] = None,
        selected_procedural_ids: Optional[List[int]] = None,
        n_messages: int = 0,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Append one recall to the audit log. Returns the assigned recall_id."""
        col = self._recall_col(graph_id)
        recall_id = col.count()
        metadata: Dict[str, Any] = {
            "recall_id": recall_id,
            "endpoint": endpoint,
            "ts": ts,
            "graph_time": graph_time,
            "observation": observation,
            "goal": goal,
            "subgoal": subgoal,
            "state": state,
            "task_type": task_type,
            "mode": mode,
            "next_subgoal": next_subgoal,
            "query_tags": _serialize_list(query_tags or []),
            "selected_semantic_ids": _serialize_list(selected_semantic_ids or []),
            "selected_procedural_ids": _serialize_list(selected_procedural_ids or []),
            "n_messages": n_messages,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        kwargs: Dict[str, Any] = {
            "ids": [str(recall_id)],
            "documents": [observation or ""],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)
        return recall_id

    def list_recalls(
        self,
        graph_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return audit rows newest-first, optionally filtered by session_id."""
        col = self._recall_col(graph_id)
        kwargs: Dict[str, Any] = {"include": ["metadatas"]}
        if session_id is not None:
            kwargs["where"] = {"session_id": session_id}
        data = col.get(**kwargs)
        rows = list(data.get("metadatas") or [])
        for row in rows:
            row["query_tags"] = _deserialize_list(row.get("query_tags", "[]"))
            row["selected_semantic_ids"] = _deserialize_list(row.get("selected_semantic_ids", "[]"))
            row["selected_procedural_ids"] = _deserialize_list(row.get("selected_procedural_ids", "[]"))
        rows.sort(key=lambda r: r.get("recall_id", 0), reverse=True)
        return rows[: max(0, limit)]

    # ------------------------------------------------------------------ #
    # Pipeline trace log (Phase 4)
    # ------------------------------------------------------------------ #
    #
    # Per-graph collection ``{graph_id}_pipeline_trace`` stores one document
    # per pipeline run; the full step list is JSON-serialized into the
    # document body. A small sister collection ``{graph_id}_pipeline_settings``
    # holds the retention cap.

    DEFAULT_TRACE_CAP = 100

    def _trace_col(self, graph_id: str):
        return self._client.get_or_create_collection(
            name=f"{graph_id}_pipeline_trace",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    def _trace_settings_col(self, graph_id: str):
        return self._client.get_or_create_collection(
            name=f"{graph_id}_pipeline_settings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    def get_pipeline_trace_cap(self, graph_id: str) -> int:
        col = self._trace_settings_col(graph_id)
        data = col.get(ids=["trace_cap"], include=["metadatas"])
        metas = data.get("metadatas") or []
        if not metas:
            return self.DEFAULT_TRACE_CAP
        try:
            return int(metas[0].get("cap", self.DEFAULT_TRACE_CAP))
        except (TypeError, ValueError):
            return self.DEFAULT_TRACE_CAP

    def set_pipeline_trace_cap(self, graph_id: str, cap: int) -> int:
        col = self._trace_settings_col(graph_id)
        cap = max(0, int(cap))
        try:
            col.delete(ids=["trace_cap"])
        except Exception:
            pass
        col.add(
            ids=["trace_cap"],
            documents=["pipeline_trace_cap"],
            metadatas=[{"cap": cap}],
        )
        return cap

    def get_pipeline_name(self, graph_id: str, *, default: str = "plugmem-default") -> str:
        """Per-graph memory-pipeline binding (e.g. plugmem-default, naive-rag).

        Stored alongside the trace cap in the ``_pipeline_settings`` collection.
        """
        col = self._trace_settings_col(graph_id)
        data = col.get(ids=["pipeline_name"], include=["metadatas"])
        metas = data.get("metadatas") or []
        if not metas:
            return default
        return str(metas[0].get("name") or default)

    def set_pipeline_name(self, graph_id: str, name: str) -> str:
        col = self._trace_settings_col(graph_id)
        name = (name or "").strip() or "plugmem-default"
        try:
            col.delete(ids=["pipeline_name"])
        except Exception:
            pass
        col.add(
            ids=["pipeline_name"],
            documents=["pipeline_binding"],
            metadatas=[{"name": name}],
        )
        return name

    def add_pipeline_trace(self, graph_id: str, record) -> str:
        """Persist one trace; enforce the retention cap by deleting oldest.

        ``record`` is a :class:`plugmem.core.pipeline_trace.TraceRecord`. The
        full step list is serialized to JSON into the document body.
        """
        from dataclasses import asdict
        col = self._trace_col(graph_id)

        steps_json = json.dumps([asdict(s) for s in record.steps], default=str)
        metadata: Dict[str, Any] = {
            "trace_id": record.trace_id,
            "ts": record.ts,
            "endpoint": record.endpoint,
            "duration_ms": record.duration_ms,
            "ok": record.ok,
            "num_steps": record.num_steps,
            "session_id": record.session_id or "",
            "error": record.error or "",
            "meta_json": json.dumps(record.meta or {}, default=str),
        }
        col.add(
            ids=[record.trace_id],
            documents=[steps_json],
            metadatas=[metadata],
        )

        cap = self.get_pipeline_trace_cap(graph_id)
        if cap > 0:
            data = col.get(include=["metadatas"])
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            if len(ids) > cap:
                paired = sorted(
                    zip(ids, metas),
                    key=lambda p: p[1].get("ts", "") or "",
                )
                excess = len(ids) - cap
                old_ids = [p[0] for p in paired[:excess]]
                if old_ids:
                    try:
                        col.delete(ids=old_ids)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Trace eviction failed: %s", e)
        return record.trace_id

    def list_pipeline_traces(
        self, graph_id: str, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return trace summaries (no steps) newest-first."""
        col = self._trace_col(graph_id)
        data = col.get(include=["metadatas"])
        metas = list(data.get("metadatas") or [])
        rows = []
        for m in metas:
            rows.append({
                "trace_id": m.get("trace_id", ""),
                "ts": m.get("ts", ""),
                "endpoint": m.get("endpoint", ""),
                "duration_ms": int(m.get("duration_ms") or 0),
                "ok": bool(m.get("ok")),
                "num_steps": int(m.get("num_steps") or 0),
                "session_id": m.get("session_id") or None,
                "error": m.get("error") or None,
            })
        rows.sort(key=lambda r: r.get("ts", "") or "", reverse=True)
        return rows[: max(0, limit)]

    def get_pipeline_trace(
        self, graph_id: str, trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return one full trace (with step list) or None."""
        col = self._trace_col(graph_id)
        data = col.get(ids=[trace_id], include=["metadatas", "documents"])
        ids = data.get("ids") or []
        if not ids:
            return None
        meta = (data.get("metadatas") or [{}])[0]
        doc = (data.get("documents") or [""])[0]
        try:
            steps = json.loads(doc) if doc else []
        except json.JSONDecodeError:
            steps = []
        try:
            meta_dict = json.loads(meta.get("meta_json", "") or "{}")
        except json.JSONDecodeError:
            meta_dict = {}
        return {
            "trace_id": meta.get("trace_id", trace_id),
            "ts": meta.get("ts", ""),
            "endpoint": meta.get("endpoint", ""),
            "duration_ms": int(meta.get("duration_ms") or 0),
            "ok": bool(meta.get("ok")),
            "num_steps": int(meta.get("num_steps") or 0),
            "session_id": meta.get("session_id") or None,
            "error": meta.get("error") or None,
            "meta": meta_dict,
            "steps": steps,
        }

    def aggregate_step_stats(self, graph_id: str) -> Dict[str, Dict[str, Any]]:
        """Walk all stored traces and aggregate per-step-name stats.

        Returns a dict keyed by ``StepRecord.name`` with::

            {count, mean_latency_ms, errors, last_ts, last_latency_ms,
             recent_latencies}

        ``recent_latencies`` is the last N (default 20) latency values, ordered
        oldest → newest, suitable for sparkline rendering.
        """
        col = self._trace_col(graph_id)
        try:
            data = col.get(include=["documents", "metadatas"])
        except Exception:
            return {}
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        SPARK_LIMIT = 20

        running: Dict[str, Dict[str, Any]] = {}
        for doc, meta in zip(docs, metas):
            try:
                steps = json.loads(doc) if doc else []
            except json.JSONDecodeError:
                continue
            ts = (meta or {}).get("ts", "") or ""
            for s in steps:
                name = s.get("name") or ""
                if not name:
                    continue
                entry = running.setdefault(name, {
                    "count": 0, "total": 0, "errors": 0,
                    "last_ts": "", "last_latency_ms": 0,
                    "pairs": [],
                })
                latency = int(s.get("latency_ms") or 0)
                entry["count"] += 1
                entry["total"] += latency
                if s.get("error"):
                    entry["errors"] += 1
                if ts > entry["last_ts"]:
                    entry["last_ts"] = ts
                    entry["last_latency_ms"] = latency
                entry["pairs"].append((ts, latency))

        out: Dict[str, Dict[str, Any]] = {}
        for name, e in running.items():
            e["pairs"].sort(key=lambda p: p[0])
            recent = [int(p[1]) for p in e["pairs"][-SPARK_LIMIT:]]
            out[name] = {
                "name": name,
                "count": e["count"],
                "mean_latency_ms": int(e["total"] / e["count"]) if e["count"] else 0,
                "errors": e["errors"],
                "last_ts": e["last_ts"],
                "last_latency_ms": e["last_latency_ms"],
                "recent_latencies": recent,
            }
        return out

    def list_traces_for_step(
        self, graph_id: str, step_name: str, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return recent trace summaries that include the named step."""
        col = self._trace_col(graph_id)
        try:
            data = col.get(include=["documents", "metadatas"])
        except Exception:
            return []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        rows: List[Dict[str, Any]] = []
        for doc, meta in zip(docs, metas):
            try:
                steps = json.loads(doc) if doc else []
            except json.JSONDecodeError:
                continue
            match = next((s for s in steps if s.get("name") == step_name), None)
            if match is None:
                continue
            rows.append({
                "trace_id": meta.get("trace_id", ""),
                "ts": meta.get("ts", ""),
                "endpoint": meta.get("endpoint", ""),
                "duration_ms": int(meta.get("duration_ms") or 0),
                "ok": bool(meta.get("ok")),
                "step_latency_ms": int(match.get("latency_ms") or 0),
                "step_error": match.get("error") or None,
            })
        rows.sort(key=lambda r: r.get("ts", "") or "", reverse=True)
        return rows[: max(0, limit)]

    def list_sessions(self, graph_id: str) -> List[str]:
        """Distinct session_ids that appear anywhere in the graph (nodes or recalls)."""
        seen: set = set()
        for node_type in ("episodic", "semantic", "procedural"):
            col = self._col(graph_id, node_type)
            data = col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        try:
            audit_col = self._recall_col(graph_id)
            data = audit_col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        except Exception:
            pass
        return sorted(seen)
