"""Multi-LLM router — assigns different LLM clients to different operations.

Reads a YAML configuration that maps operation roles (structuring, retrieval,
reasoning, consolidation) to separate LLM endpoints/models.  Any role not
explicitly configured inherits from ``default``.

Example YAML (``llm_config.yaml``)::

    default:
      base_url: "http://localhost:8000/v1"
      api_key: "tok-xxx"
      model: "qwen-2.5-32b-instruct"

    structuring:
      model: "qwen-2.5-32b-instruct"

    retrieval:
      model: "gpt-4o-mini"

    reasoning:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o"

    consolidation:
      # omitted → uses default

Environment variable references like ``${VAR}`` are expanded at load time.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from plugmem.clients.llm import LLMClient, OpenAICompatibleLLMClient

logger = logging.getLogger(__name__)

ROLES = ("default", "structuring", "retrieval", "reasoning", "consolidation")

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` references with environment variable values."""
    if not isinstance(value, str):
        return value
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def expand_env_vars(value: str) -> str:
    """Public wrapper for ``${VAR}`` expansion.

    Used by the inspector's model-swap routes so the user can paste
    ``${OPENAI_API_KEY}`` into the API key field instead of the raw secret.
    Missing variables expand to empty strings (same semantics as YAML
    config loading).
    """
    return _expand_env(value)


def _expand_dict(d: dict) -> dict:
    return {k: _expand_env(v) if isinstance(v, str) else v for k, v in d.items()}


class LLMRouter:
    """Holds per-role LLMClient instances.

    Use ``for_role(role)`` to get the client for a specific operation.
    Implements ``LLMClient`` itself so it can be used as a drop-in replacement
    (calls go to the ``default`` role).
    """

    def __init__(self, clients: Dict[str, LLMClient]):
        if "default" not in clients:
            raise ValueError("LLMRouter requires a 'default' client")
        self._clients = clients

    # -- LLMClient protocol (delegates to default) -----------------------

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        return self._clients["default"].complete(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )

    # -- Role-based access -----------------------------------------------

    def for_role(self, role: str) -> LLMClient:
        """Return the client for *role*, falling back to ``default``."""
        return self._clients.get(role, self._clients["default"])

    @property
    def structuring(self) -> LLMClient:
        return self.for_role("structuring")

    @property
    def retrieval(self) -> LLMClient:
        return self.for_role("retrieval")

    @property
    def reasoning(self) -> LLMClient:
        return self.for_role("reasoning")

    @property
    def consolidation(self) -> LLMClient:
        return self.for_role("consolidation")

    # -- Factory ---------------------------------------------------------

    # -- Introspection / runtime swap (used by the inspector) -----------

    def has_role(self, role: str) -> bool:
        """True if *role* has its own client (not falling back to default)."""
        return role in self._clients

    def role_summary(self) -> Dict[str, Dict[str, object]]:
        """Snapshot of current per-role bindings for UI display.

        Never includes the api_key (only ``has_api_key``). Roles that fall
        back to default are reported with ``falls_back_to_default=True`` and
        the default's settings are echoed so the UI can show the effective
        binding.
        """
        out: Dict[str, Dict[str, object]] = {}
        default = self._clients["default"]
        for role in ROLES:
            client = self._clients.get(role, default)
            falls_back = role != "default" and role not in self._clients
            out[role] = {
                "role": role,
                "base_url": getattr(client, "base_url", "") or "",
                "model": getattr(client, "model", "") or "",
                "has_api_key": bool(getattr(client, "api_key", "")),
                "is_azure": bool(getattr(client, "is_azure", False)),
                "azure_api_version": getattr(client, "azure_api_version", ""),
                "falls_back_to_default": falls_back,
            }
        return out

    def set_role(
        self,
        role: str,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        is_azure: bool = False,
        azure_api_version: str = "2024-05-01-preview",
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ) -> None:
        """Atomically swap the client bound to *role*.

        ``api_key=None`` keeps the existing key (or the default's key if the
        role wasn't explicitly bound). ``api_key=""`` explicitly clears it.
        Construction is done first; the swap only runs if it succeeds, so a
        bad config doesn't half-update state.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}'. Allowed: {list(ROLES)}")

        # Match YAML loader behavior: ``${VAR}`` references in api_key /
        # base_url are resolved against the process environment at write time.
        if api_key is None:
            existing = self._clients.get(role) or self._clients["default"]
            api_key = getattr(existing, "api_key", "") or ""
        else:
            api_key = _expand_env(api_key)
        base_url = _expand_env(base_url)

        new_client = OpenAICompatibleLLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
            is_azure=is_azure,
            azure_api_version=azure_api_version,
        )
        self._clients[role] = new_client
        logger.info("LLMRouter: role '%s' bound to %s (model=%s)", role, base_url, model)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMRouter":
        """Load router from a YAML config file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Empty config file: {path}")

        default_cfg = _expand_dict(raw.get("default", {}))
        if not default_cfg:
            raise ValueError("Config must contain a 'default' section")

        clients: Dict[str, LLMClient] = {}
        for role in ROLES:
            if role == "default":
                clients[role] = _build_client(default_cfg)
            elif role in raw:
                # Merge: role-specific values override default
                merged = {**default_cfg, **_expand_dict(raw[role])}
                clients[role] = _build_client(merged)
            # else: not configured → for_role() falls back to default

        configured = ["default"] + [r for r in ROLES[1:] if r in clients]
        logger.info("LLMRouter loaded from %s — roles: %s", path, configured)
        return cls(clients)

    @classmethod
    def from_single_client(cls, client: LLMClient) -> "LLMRouter":
        """Wrap a single client as a router (all roles use the same client)."""
        return cls({"default": client})


def _build_client(cfg: dict) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
        max_retries=int(cfg.get("max_retries", 5)),
        retry_delay=float(cfg.get("retry_delay", 5.0)),
        is_azure=bool(cfg.get("is_azure", False)),
        azure_api_version=cfg.get("azure_api_version", "2024-05-01-preview"),
        token_usage_file=cfg.get("token_usage_file"),
    )
