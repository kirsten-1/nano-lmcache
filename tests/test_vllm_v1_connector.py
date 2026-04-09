"""Tests for the scheduler-side vLLM connector core."""

from types import SimpleNamespace

from nano_lmcache.index import SegmentRadixTree, StorageTier
from nano_lmcache.integration.vllm_v1_connector import (
    NanoLMCacheIndexMatcher,
    NanoLMCacheVLLMConnectorCore,
)
from nano_lmcache.segment import SegmentSplitter


class FakeBlocks:
    def __init__(self, block_ids):
        self._block_ids = block_ids

    def get_block_ids(self):
        return self._block_ids


def build_cached_matcher(tokens, *, max_segment_length=64, min_segment_length=16):
    splitter = SegmentSplitter(
        max_length=max_segment_length,
        min_length=min_segment_length,
    )
    index = SegmentRadixTree()
    segments = splitter.split(tokens)
    index.insert(segments, StorageTier.CPU)
    return NanoLMCacheIndexMatcher(splitter, index)


def make_request(request_id, prompt_token_ids, all_token_ids=None):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        all_token_ids=list(all_token_ids or prompt_token_ids),
    )


def make_scheduler_output(new_reqs, cached_reqs, num_scheduled_tokens):
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
    )


def test_core_uses_segment_radix_tree_for_prefix_matching():
    prefix = list(range(128))
    matcher = build_cached_matcher(prefix)
    core = NanoLMCacheVLLMConnectorCore(matcher, block_size=16)

    request = make_request("req-1", prefix + [999, 1000])
    new_tokens, load_async = core.get_num_new_matched_tokens(request, 0)

    assert new_tokens == 112
    assert load_async is False


def test_core_builds_load_metadata_for_new_requests():
    prefix = list(range(128))
    matcher = build_cached_matcher(prefix)
    core = NanoLMCacheVLLMConnectorCore(matcher, block_size=16)

    request = make_request("req-2", prefix + [999, 1000])
    core.get_num_new_matched_tokens(request, 0)
    core.update_state_after_alloc(request, FakeBlocks(([20, 21, 22, 23, 24, 25, 26, 27],)), 112)

    scheduler_output = make_scheduler_output(
        new_reqs=[
            SimpleNamespace(
                req_id="req-2",
                prompt_token_ids=request.prompt_token_ids,
            )
        ],
        cached_reqs=SimpleNamespace(
            req_ids=[],
            all_token_ids={},
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={"req-2": len(request.prompt_token_ids) - 112},
    )

    meta = core.build_connector_meta(scheduler_output)

    assert len(meta.requests) == 1
    assert meta.requests[0].request_id == "req-2"
    assert meta.requests[0].matched_token_count == 112
    assert meta.requests[0].is_store is False
    assert meta.requests[0].token_ids == request.prompt_token_ids


def test_core_builds_store_metadata_when_no_external_match_exists():
    matcher = build_cached_matcher(list(range(64)))
    core = NanoLMCacheVLLMConnectorCore(matcher, block_size=16)

    request = make_request("req-3", list(range(1000, 1064)))
    core.get_num_new_matched_tokens(request, 0)
    core.update_state_after_alloc(request, FakeBlocks(([0, 1, 2, 3],)), 0)

    scheduler_output = make_scheduler_output(
        new_reqs=[
            SimpleNamespace(
                req_id="req-3",
                prompt_token_ids=request.prompt_token_ids,
            )
        ],
        cached_reqs=SimpleNamespace(
            req_ids=[],
            all_token_ids={},
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={"req-3": len(request.prompt_token_ids)},
    )

    meta = core.build_connector_meta(scheduler_output)

    assert len(meta.requests) == 1
    assert meta.requests[0].matched_token_count == 0
    assert meta.requests[0].is_store is True


def test_core_uses_all_token_ids_for_resumed_cached_requests():
    prefix = list(range(128))
    all_token_ids = prefix + [5000, 5001, 5002]
    matcher = build_cached_matcher(prefix)
    core = NanoLMCacheVLLMConnectorCore(matcher, block_size=16)

    request = make_request("req-4", prefix, all_token_ids=all_token_ids)
    core.get_num_new_matched_tokens(request, 0)
    core.update_state_after_alloc(request, FakeBlocks(([30, 31, 32, 33, 34, 35, 36, 37],)), 96)

    scheduler_output = make_scheduler_output(
        new_reqs=[],
        cached_reqs=SimpleNamespace(
            req_ids=["req-4"],
            all_token_ids={},
            num_computed_tokens=[96],
        ),
        num_scheduled_tokens={"req-4": 3},
    )

    meta = core.build_connector_meta(scheduler_output)

    assert len(meta.requests) == 1
    assert meta.requests[0].token_ids == all_token_ids[:99]
    assert meta.requests[0].matched_token_count == 96
