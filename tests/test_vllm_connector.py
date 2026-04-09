"""Tests for the vLLM integration connector."""

import tempfile
import shutil

import pytest
import torch

from nano_lmcache.integration import VLLMConnector, NanoLMCacheVLLMConfig


def make_connector():
    temp_dir = tempfile.mkdtemp()
    config = NanoLMCacheVLLMConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        gpu_size_gb=0.0,
        cpu_size_gb=0.01,
        disk_size_gb=0.1,
        disk_cache_dir=temp_dir,
        enable_gpu_cache=False,
        enable_async_write=False,
        max_segment_length=16,
        min_segment_length=4,
    )
    connector = VLLMConnector(config)
    return connector, temp_dir


def make_kv(seq_len: int) -> torch.Tensor:
    return torch.randn(4, 2, seq_len, 2, 8, dtype=torch.float16)


def test_direct_tensor_store_and_retrieve():
    connector, temp_dir = make_connector()
    try:
        tokens = list(range(32))
        connector.store(tokens, make_kv(32), async_write=False)

        retrieved, matched = connector.retrieve(tokens + list(range(32, 40)))

        assert matched == 32
        assert retrieved is not None
        assert retrieved.shape == (4, 2, 32, 2, 8)
    finally:
        connector.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_integration_status_reports_block_path_not_implemented():
    connector, temp_dir = make_connector()
    try:
        status = connector.integration_status()
        assert status["direct_tensor_store_supported"] is True
        assert status["direct_tensor_retrieve_supported"] is True
        assert status["vllm_block_conversion_supported"] is False
    finally:
        connector.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_block_conversion_methods_fail_loudly():
    connector, temp_dir = make_connector()
    try:
        key_cache = torch.randn(1, 2, 2, 4, 4, dtype=torch.float16)
        value_cache = torch.randn(1, 2, 8, 4, dtype=torch.float16)
        slot_mapping = torch.zeros(4, dtype=torch.int32)

        with pytest.raises(NotImplementedError):
            connector.store_from_vllm_blocks([1, 2, 3, 4], key_cache, value_cache, slot_mapping)

        with pytest.raises(NotImplementedError):
            connector.retrieve_to_vllm_blocks([1, 2, 3, 4], key_cache, value_cache, slot_mapping)
    finally:
        connector.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
