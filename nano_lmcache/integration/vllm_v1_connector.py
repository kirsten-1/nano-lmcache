"""Scheduler-side vLLM 0.19 connector logic for nano-lmcache.

This module keeps the connector aligned with the project's own
segment-based radix-tree design. Prefix lookups go through the existing
``SegmentSplitter`` and ``SegmentRadixTree`` instead of adding a second index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

from nano_lmcache.index import MatchResult
from nano_lmcache.integration.vllm_v1_adapter import NanoLMCacheVLLMAdapter


logger = logging.getLogger(__name__)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
except Exception:  # pragma: no cover - exercised only without vLLM installed.
    class KVConnectorMetadata:  # type: ignore[no-redef]
        """Fallback metadata base when vLLM is not importable."""

    class KVConnectorRole(Enum):  # type: ignore[no-redef]
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            self._connector_metadata = None
            return

        def bind_connector_metadata(self, connector_metadata: Any) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

    @dataclass
    class KVConnectorStats:  # type: ignore[no-redef]
        data: dict[str, Any] = field(default_factory=dict)


@dataclass
class NanoLMCacheConnectorRequestMetadata:
    """Minimal request metadata for worker-side KV load/store decisions."""

    request_id: str
    token_ids: list[int]
    block_ids: tuple[list[int], ...]
    matched_token_count: int
    candidate_token_count: int
    is_store: bool
    load_kv_async: bool = False


@dataclass
class NanoLMCacheConnectorMetadata(KVConnectorMetadata):
    """Metadata emitted by the scheduler-side connector."""

    requests: list[NanoLMCacheConnectorRequestMetadata] = field(default_factory=list)


@dataclass
class NanoLMCacheConnectorStats(KVConnectorStats):
    """Simple serializable connector stats for dry-run observability."""

    data: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        self.data.clear()

    def aggregate(self, other: "NanoLMCacheConnectorStats") -> "NanoLMCacheConnectorStats":
        merged = dict(self.data)
        for key, value in other.data.items():
            merged[key] = merged.get(key, 0) + value
        return NanoLMCacheConnectorStats(data=merged)

    def reduce(self) -> dict[str, int | float]:
        return dict(self.data)

    def is_empty(self) -> bool:
        return not self.data


_ENGINE_REGISTRY: dict[str, Any] = {}


def register_nano_lmcache_engine(registry_key: str, engine: Any) -> None:
    """Register a NanoLMCache engine for scheduler-side connector lookup."""
    _ENGINE_REGISTRY[registry_key] = engine


def unregister_nano_lmcache_engine(registry_key: str) -> None:
    """Remove a previously registered NanoLMCache engine."""
    _ENGINE_REGISTRY.pop(registry_key, None)


def build_vllm_kv_transfer_config(
    *,
    registry_key: str,
    kv_role: str = "kv_both",
    engine_id: str = "nano-lmcache-v1",
    connector_name: str = "NanoLMCacheConnectorV1",
    connector_module_path: str = "nano_lmcache.integration.vllm_v1_connector",
    enable_external_matching: bool = False,
    enable_probe_logging: bool = False,
    kv_connector_extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a vLLM KVTransferConfig-compatible dictionary.

    The actual NanoLMCache engine stays in a local registry. The config only
    carries a string key so it remains serializable across spawned workers.
    """
    extra_config = {
        "nano_lmcache_registry_key": registry_key,
        "nano_lmcache_enable_external_matching": enable_external_matching,
        "nano_lmcache_probe": enable_probe_logging,
    }
    if kv_connector_extra_config:
        extra_config.update(kv_connector_extra_config)
    return {
        "kv_connector": connector_name,
        "kv_connector_module_path": connector_module_path,
        "kv_role": kv_role,
        "engine_id": engine_id,
        "kv_connector_extra_config": extra_config,
    }


class NanoLMCacheIndexMatcher:
    """Prefix matcher backed by nano-lmcache's splitter and radix tree."""

    def __init__(self, splitter: Any, index: Any):
        self._splitter = splitter
        self._index = index

    @classmethod
    def from_engine(cls, engine: Any) -> "NanoLMCacheIndexMatcher":
        return cls(engine.splitter, engine.index)

    def match_prefix(self, token_ids: list[int]) -> MatchResult:
        if not token_ids:
            return MatchResult(
                matched_tokens=0,
                matched_segments=[],
                remaining_tokens=0,
            )
        segments = self._splitter.split(token_ids)
        return self._index.match_prefix(segments)


class NanoLMCacheVLLMConnectorCore:
    """Pure scheduler-side connector logic independent of the vLLM runtime."""

    def __init__(
        self,
        matcher: NanoLMCacheIndexMatcher,
        *,
        block_size: int = 16,
        load_kv_async: bool = False,
        enable_external_matching: bool = False,
        probe_enabled: bool = False,
    ):
        self.matcher = matcher
        self.block_size = block_size
        self.load_kv_async = load_kv_async
        self.enable_external_matching = enable_external_matching
        self.probe_enabled = probe_enabled
        self.adapter = NanoLMCacheVLLMAdapter(block_size=block_size)
        self._requests: dict[str, Any] = {}
        self._candidate_tokens: dict[str, int] = {}

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = list(getattr(request, "prompt_token_ids", None) or [])
        if len(token_ids) <= 1:
            return 0, False

        match = self.matcher.match_prefix(token_ids)
        result = self.adapter.plan_external_match(
            request_id=request.request_id,
            prompt_token_ids=token_ids,
            local_computed_tokens=num_computed_tokens,
            candidate_cached_tokens=match.matched_tokens,
            load_kv_async=self.load_kv_async,
        )
        self._requests[request.request_id] = request
        self._candidate_tokens[request.request_id] = result.new_external_tokens
        actual_external_tokens = (
            result.new_external_tokens if self.enable_external_matching else 0
        )
        if self.probe_enabled:
            logger.info(
                "nano-lmcache connector match req=%s prompt_tokens=%d "
                "local_tokens=%d candidate_external_tokens=%d actual_external_tokens=%d",
                request.request_id,
                len(token_ids),
                num_computed_tokens,
                result.new_external_tokens,
                actual_external_tokens,
            )
        return actual_external_tokens, (
            self.load_kv_async and actual_external_tokens > 0
        )

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        if hasattr(blocks, "get_block_ids"):
            block_ids = blocks.get_block_ids()
        else:
            block_ids = blocks
        self.adapter.update_state_after_alloc(
            request_id=request.request_id,
            block_ids=block_ids,
            num_external_tokens=num_external_tokens,
        )
        self._requests[request.request_id] = request

    def build_connector_meta(self, scheduler_output: Any) -> NanoLMCacheConnectorMetadata:
        step_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        step_req_ids.update(scheduler_output.scheduled_cached_reqs.req_ids)

        requests: list[NanoLMCacheConnectorRequestMetadata] = []
        for metadata in self.adapter.build_connector_meta().requests:
            if metadata.request_id not in step_req_ids:
                continue
            token_ids = self._resolve_token_ids(metadata.request_id, scheduler_output)
            requests.append(
                NanoLMCacheConnectorRequestMetadata(
                    request_id=metadata.request_id,
                    token_ids=token_ids,
                    block_ids=metadata.block_ids,
                    matched_token_count=metadata.matched_tokens,
                    candidate_token_count=self._candidate_tokens.get(
                        metadata.request_id,
                        metadata.matched_tokens,
                    ),
                    is_store=metadata.is_store,
                    load_kv_async=metadata.load_kv_async,
                )
            )
        if self.probe_enabled and requests:
            logger.info(
                "nano-lmcache connector meta requests=%s",
                [
                    {
                        "req_id": request.request_id,
                        "candidate": request.candidate_token_count,
                        "actual": request.matched_token_count,
                        "is_store": request.is_store,
                    }
                    for request in requests
                ],
            )
        return NanoLMCacheConnectorMetadata(requests=requests)

    def update_connector_output(self, connector_output: Any) -> None:
        self.adapter.update_connector_output(connector_output)

    def request_finished(
        self,
        request: Any,
        block_ids: list[int] | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        self._requests.pop(request.request_id, None)
        self._candidate_tokens.pop(request.request_id, None)
        return self.adapter.request_finished(request.request_id, block_ids)

    def _resolve_token_ids(self, request_id: str, scheduler_output: Any) -> list[int]:
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id == request_id:
                return list(request.prompt_token_ids or [])

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if request_id in cached_reqs.all_token_ids:
            return list(cached_reqs.all_token_ids[request_id])

        request = self._requests.get(request_id)
        if request is None:
            return []

        all_token_ids = list(getattr(request, "all_token_ids", []) or [])
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id != request_id:
                continue
            total_tokens = (
                cached_reqs.num_computed_tokens[idx]
                + scheduler_output.num_scheduled_tokens.get(request_id, 0)
            )
            return all_token_ids[:total_tokens]

        return list(getattr(request, "prompt_token_ids", None) or [])


class NanoLMCacheConnectorV1(KVConnectorBase_V1):
    """Thin vLLM-compatible wrapper around the scheduler-side connector core."""

    def __init__(
        self,
        vllm_config: Any = None,
        role: Any = None,
        kv_cache_config: Any = None,
        *,
        matcher: NanoLMCacheIndexMatcher | None = None,
        core: NanoLMCacheVLLMConnectorCore | None = None,
        block_size: int | None = None,
        load_kv_async: bool = False,
    ):
        if vllm_config is not None and role is not None:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )

        if core is None:
            matcher = matcher or self._extract_matcher(vllm_config)
            if role == KVConnectorRole.SCHEDULER and matcher is None:
                raise ValueError(
                    "NanoLMCacheConnectorV1 requires a matcher or a "
                    "nano_lmcache_matcher/nano_lmcache_registry_key entry in "
                    "kv_transfer_config.extra_config."
                )
            if matcher is not None:
                if block_size is None:
                    block_size = getattr(
                        getattr(vllm_config, "cache_config", None),
                        "block_size",
                        16,
                    )
                core = NanoLMCacheVLLMConnectorCore(
                    matcher,
                    block_size=block_size,
                    load_kv_async=load_kv_async,
                    enable_external_matching=self._get_enable_external_matching(
                        vllm_config
                    ),
                    probe_enabled=self._get_probe_enabled(vllm_config),
                )

        self.core = core
        self._last_stats = NanoLMCacheConnectorStats()

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.core is None:
            return 0, False
        return self.core.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        if self.core is None:
            return
        self.core.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> NanoLMCacheConnectorMetadata:
        if self.core is None:
            return NanoLMCacheConnectorMetadata()
        return self.core.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        if self.core is None:
            return
        self.core.update_connector_output(connector_output)

    def request_finished(
        self,
        request: Any,
        block_ids: list[int] | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.core is None:
            return False, None
        return self.core.request_finished(request, block_ids)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self):
        return

    def get_kv_connector_stats(self) -> NanoLMCacheConnectorStats | None:
        metadata = self._connector_metadata
        if not isinstance(metadata, NanoLMCacheConnectorMetadata):
            return None

        total_candidate_tokens = sum(
            request.candidate_token_count for request in metadata.requests
        )
        total_actual_tokens = sum(
            request.matched_token_count for request in metadata.requests
        )
        matched_reqs = sum(
            1 for request in metadata.requests if request.candidate_token_count > 0
        )
        stats = NanoLMCacheConnectorStats(
            data={
                "nano_lmcache_candidate_tokens": total_candidate_tokens,
                "nano_lmcache_actual_external_tokens": total_actual_tokens,
                "nano_lmcache_matched_requests": matched_reqs,
            }
        )
        self._last_stats = stats
        if getattr(self.core, "probe_enabled", False) and not stats.is_empty():
            logger.info("nano-lmcache connector stats=%s", stats.data)
        return stats if not stats.is_empty() else None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> NanoLMCacheConnectorStats | None:
        if not data:
            return None
        return NanoLMCacheConnectorStats(data=dict(data))

    @staticmethod
    def _extract_matcher(vllm_config: Any) -> NanoLMCacheIndexMatcher | None:
        if vllm_config is None or getattr(vllm_config, "kv_transfer_config", None) is None:
            return None

        transfer_config = vllm_config.kv_transfer_config
        get_extra = getattr(transfer_config, "get_from_extra_config", None)
        if get_extra is None:
            return None

        matcher = get_extra("nano_lmcache_matcher", None)
        if matcher is not None:
            return matcher

        registry_key = get_extra("nano_lmcache_registry_key", None)
        if registry_key is not None:
            engine = _ENGINE_REGISTRY.get(registry_key)
            if engine is not None:
                return NanoLMCacheIndexMatcher.from_engine(engine)

        engine = get_extra("nano_lmcache_engine", None)
        if engine is not None:
            return NanoLMCacheIndexMatcher.from_engine(engine)

        return None

    @staticmethod
    def _get_enable_external_matching(vllm_config: Any) -> bool:
        if vllm_config is None or getattr(vllm_config, "kv_transfer_config", None) is None:
            return False
        get_extra = getattr(vllm_config.kv_transfer_config, "get_from_extra_config", None)
        if get_extra is None:
            return False
        return bool(get_extra("nano_lmcache_enable_external_matching", False))

    @staticmethod
    def _get_probe_enabled(vllm_config: Any) -> bool:
        if vllm_config is None or getattr(vllm_config, "kv_transfer_config", None) is None:
            return False
        get_extra = getattr(vllm_config.kv_transfer_config, "get_from_extra_config", None)
        if get_extra is None:
            return False
        return bool(get_extra("nano_lmcache_probe", False))
