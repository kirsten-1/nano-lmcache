"""vLLM V1 connector semantics modeled for nano-lmcache.

This module does not depend on vLLM imports. It provides a small state machine
that mirrors the scheduler/worker handshake used by ``KVConnectorBase_V1`` so
that nano-lmcache can evolve toward a real connector without pretending that
block injection already works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _align_down(value: int, block_size: int) -> int:
    """Align a token count down to the nearest full vLLM block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return value - (value % block_size)


@dataclass(frozen=True)
class ExternalMatchResult:
    """External prefix match candidate after block alignment."""

    prompt_tokens: int
    local_computed_tokens: int
    matched_tokens: int
    block_size: int

    @property
    def new_external_tokens(self) -> int:
        """Tokens the external connector can contribute beyond local cache."""
        return max(0, self.matched_tokens - self.local_computed_tokens)


@dataclass
class SchedulerPendingRequest:
    """Scheduler-side request summary aligned with vLLM connector semantics."""

    request_id: str
    prompt_token_ids: list[int]
    block_ids: tuple[list[int], ...] | None = None
    local_computed_tokens: int = 0
    external_matched_tokens: int = 0
    external_tokens_after_alloc: int = 0
    needs_remote_load: bool = False
    load_kv_async: bool = False
    worker_acknowledged: bool = False


@dataclass
class ConnectorRequestMetadata:
    """Metadata that scheduler would send to a worker connector for one request."""

    request_id: str
    prompt_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    matched_tokens: int
    is_store: bool
    load_kv_async: bool = False


@dataclass
class ConnectorMeta:
    """Batch of request metadata for one scheduler step."""

    requests: list[ConnectorRequestMetadata] = field(default_factory=list)


@dataclass
class WorkerConnectorOutput:
    """Minimal worker-to-scheduler feedback for one scheduler step."""

    finished_load_req_ids: set[str] = field(default_factory=set)
    finished_store_req_ids: set[str] = field(default_factory=set)


class NanoLMCacheVLLMAdapter:
    """State machine that mirrors the vLLM V1 connector lifecycle.

    The adapter intentionally stops at metadata production. It does not try to
    inject/export KV blocks because that remains runtime- and backend-specific.
    """

    def __init__(self, block_size: int = 16):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._pending: dict[str, SchedulerPendingRequest] = {}

    def plan_external_match(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        local_computed_tokens: int,
        candidate_cached_tokens: int,
        *,
        load_kv_async: bool = False,
    ) -> ExternalMatchResult:
        """Compute block-aligned external match state for a request.

        vLLM only reuses full blocks and must leave the final token to be
        recomputed for logits, so both the candidate match length and the
        locally computed prefix are aligned down to the connector block size.
        """
        prompt_tokens = len(prompt_token_ids)
        if prompt_tokens == 0:
            raise ValueError("prompt_token_ids must not be empty")

        # The runtime probe against vLLM 0.19 showed that a fully reusable
        # 2085-token prompt only surfaces as a 2064-token cache hit. The
        # scheduler/allocator therefore behaves more conservatively than a
        # plain "leave one token for logits" rule and effectively keeps the
        # last full block on the recompute side.
        max_cache_hit_length = max(0, prompt_tokens - 1 - self.block_size)
        aligned_local = min(
            _align_down(max(0, local_computed_tokens), self.block_size),
            _align_down(max_cache_hit_length, self.block_size),
        )
        aligned_match = min(
            _align_down(max(0, candidate_cached_tokens), self.block_size),
            _align_down(max_cache_hit_length, self.block_size),
        )
        if aligned_match < aligned_local:
            aligned_match = aligned_local

        pending = SchedulerPendingRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            local_computed_tokens=aligned_local,
            external_matched_tokens=aligned_match,
            needs_remote_load=aligned_match > aligned_local,
            load_kv_async=load_kv_async and aligned_match > aligned_local,
        )
        self._pending[request_id] = pending
        return ExternalMatchResult(
            prompt_tokens=prompt_tokens,
            local_computed_tokens=aligned_local,
            matched_tokens=aligned_match,
            block_size=self.block_size,
        )

    def update_state_after_alloc(
        self,
        request_id: str,
        block_ids: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> None:
        """Mirror vLLM's post-allocation connector callback."""
        pending = self._pending[request_id]
        pending.block_ids = tuple(list(group) for group in block_ids)
        pending.external_tokens_after_alloc = max(0, num_external_tokens)
        pending.needs_remote_load = pending.external_tokens_after_alloc > 0

    def build_connector_meta(self) -> ConnectorMeta:
        """Build worker metadata without mutating external scheduler objects."""
        meta = ConnectorMeta()
        for pending in self._pending.values():
            if pending.block_ids is None:
                continue
            meta.requests.append(
                ConnectorRequestMetadata(
                    request_id=pending.request_id,
                    prompt_token_ids=list(pending.prompt_token_ids),
                    block_ids=tuple(list(group) for group in pending.block_ids),
                    matched_tokens=(
                        pending.external_matched_tokens
                        if pending.needs_remote_load
                        else 0
                    ),
                    is_store=not pending.needs_remote_load,
                    load_kv_async=pending.load_kv_async,
                )
            )
        return meta

    def update_connector_output(self, output: WorkerConnectorOutput) -> None:
        """Consume minimal worker acknowledgements for scheduled requests."""
        finished = output.finished_load_req_ids | output.finished_store_req_ids
        for request_id in finished:
            pending = self._pending.get(request_id)
            if pending is not None:
                pending.worker_acknowledged = True
                pending.needs_remote_load = False

    def request_finished(
        self,
        request_id: str,
        block_ids: list[int] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Mirror vLLM's request_finished API with explicit non-ownership."""
        self._pending.pop(request_id, None)
        return False, None

    def get_pending_request(self, request_id: str) -> SchedulerPendingRequest | None:
        """Return tracked scheduler state for tests and debugging."""
        return self._pending.get(request_id)
