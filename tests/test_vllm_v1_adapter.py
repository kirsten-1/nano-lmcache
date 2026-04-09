"""Tests for the vLLM V1 connector adapter scaffold."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nano_lmcache"
    / "integration"
    / "vllm_v1_adapter.py"
)
SPEC = spec_from_file_location("nano_lmcache.integration.vllm_v1_adapter", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

NanoLMCacheVLLMAdapter = MODULE.NanoLMCacheVLLMAdapter
WorkerConnectorOutput = MODULE.WorkerConnectorOutput


def test_plan_external_match_aligns_to_full_blocks():
    adapter = NanoLMCacheVLLMAdapter(block_size=16)

    result = adapter.plan_external_match(
        request_id="req-1",
        prompt_token_ids=list(range(2085)),
        local_computed_tokens=0,
        candidate_cached_tokens=2085,
    )

    assert result.prompt_tokens == 2085
    assert result.local_computed_tokens == 0
    assert result.matched_tokens == 2064
    assert result.new_external_tokens == 2064


def test_plan_external_match_never_regresses_below_local_hit():
    adapter = NanoLMCacheVLLMAdapter(block_size=16)

    result = adapter.plan_external_match(
        request_id="req-2",
        prompt_token_ids=list(range(2107)),
        local_computed_tokens=16,
        candidate_cached_tokens=0,
    )

    assert result.local_computed_tokens == 16
    assert result.matched_tokens == 16
    assert result.new_external_tokens == 0


def test_build_connector_meta_marks_load_and_store_paths():
    adapter = NanoLMCacheVLLMAdapter(block_size=16)

    external = adapter.plan_external_match(
        request_id="load-req",
        prompt_token_ids=list(range(2085)),
        local_computed_tokens=0,
        candidate_cached_tokens=2085,
    )
    adapter.update_state_after_alloc(
        request_id="load-req",
        block_ids=([264 + i for i in range(129)],),
        num_external_tokens=external.new_external_tokens,
    )

    store = adapter.plan_external_match(
        request_id="store-req",
        prompt_token_ids=list(range(128)),
        local_computed_tokens=0,
        candidate_cached_tokens=0,
    )
    adapter.update_state_after_alloc(
        request_id="store-req",
        block_ids=([i for i in range(8)],),
        num_external_tokens=store.new_external_tokens,
    )

    meta = adapter.build_connector_meta()
    by_id = {request.request_id: request for request in meta.requests}

    assert by_id["load-req"].is_store is False
    assert by_id["load-req"].matched_tokens == 2064
    assert by_id["store-req"].is_store is True
    assert by_id["store-req"].matched_tokens == 0


def test_worker_ack_clears_pending_remote_load():
    adapter = NanoLMCacheVLLMAdapter(block_size=16)
    result = adapter.plan_external_match(
        request_id="req-3",
        prompt_token_ids=list(range(2085)),
        local_computed_tokens=0,
        candidate_cached_tokens=2085,
        load_kv_async=True,
    )
    adapter.update_state_after_alloc(
        request_id="req-3",
        block_ids=([264 + i for i in range(129)],),
        num_external_tokens=result.new_external_tokens,
    )

    pending = adapter.get_pending_request("req-3")
    assert pending is not None
    assert pending.needs_remote_load is True
    assert pending.load_kv_async is True

    adapter.update_connector_output(
        WorkerConnectorOutput(finished_load_req_ids={"req-3"})
    )

    pending = adapter.get_pending_request("req-3")
    assert pending is not None
    assert pending.worker_acknowledged is True
    assert pending.needs_remote_load is False


def test_request_finished_drops_state_and_keeps_block_lifecycle_unowned():
    adapter = NanoLMCacheVLLMAdapter(block_size=16)
    adapter.plan_external_match(
        request_id="req-4",
        prompt_token_ids=list(range(64)),
        local_computed_tokens=0,
        candidate_cached_tokens=0,
    )

    owns_blocks, transfer_params = adapter.request_finished("req-4")

    assert owns_blocks is False
    assert transfer_params is None
    assert adapter.get_pending_request("req-4") is None
