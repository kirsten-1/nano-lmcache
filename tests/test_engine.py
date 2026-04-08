"""Tests for the main NanoLMCache engine."""

import pytest
import torch
import tempfile
import shutil

from nano_lmcache import (
    NanoLMCache,
    NanoLMCacheConfig,
    StorageConfig,
    SegmentConfig,
)


def make_kv_cache(
    num_layers: int = 4,
    num_heads: int = 4,
    head_dim: int = 32,
    seq_len: int = 100,
) -> torch.Tensor:
    """Create a mock KV cache tensor."""
    return torch.randn(
        num_layers, 2, seq_len, num_heads, head_dim, dtype=torch.float16
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def config(temp_dir):
    """Create a test configuration."""
    return NanoLMCacheConfig(
        storage=StorageConfig(
            cpu_size_gb=0.01,  # 10 MB
            disk_cache_dir=temp_dir,
            disk_size_gb=0.1,  # 100 MB
        ),
        segment=SegmentConfig(
            max_segment_length=50,
            min_segment_length=10,
        ),
        enable_async_write=False,
        enable_logging=False,
    )


class TestNanoLMCache:
    """Tests for NanoLMCache."""

    def test_init(self, config):
        """Test initialization."""
        cache = NanoLMCache(config)
        assert cache is not None
        cache.shutdown()

    def test_context_manager(self, config):
        """Test context manager usage."""
        with NanoLMCache(config) as cache:
            assert cache is not None

    def test_store_retrieve_exact(self, config):
        """Test storing and retrieving with exact match."""
        with NanoLMCache(config) as cache:
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)

            # Store
            keys = cache.store(tokens, kv)
            assert len(keys) > 0

            # Retrieve
            retrieved, matched = cache.retrieve(tokens)

            assert matched == 100
            assert retrieved is not None
            assert retrieved.shape[2] == 100  # seq_len dimension

    def test_store_retrieve_prefix(self, config):
        """Test storing and retrieving with prefix match."""
        with NanoLMCache(config) as cache:
            # Store original sequence
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            # Retrieve with extended sequence (prefix should match)
            extended_tokens = tokens + [999, 998, 997]
            retrieved, matched = cache.retrieve(extended_tokens)

            assert matched > 0
            assert matched <= 100

    def test_retrieve_no_match(self, config):
        """Test retrieval with no match."""
        with NanoLMCache(config) as cache:
            # Store something
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            # Try to retrieve completely different sequence
            different_tokens = list(range(1000, 1100))
            retrieved, matched = cache.retrieve(different_tokens)

            assert matched == 0
            assert retrieved is None

    def test_empty_input(self, config):
        """Test with empty input."""
        with NanoLMCache(config) as cache:
            # Empty store
            keys = cache.store([], make_kv_cache(seq_len=0))
            assert keys == []

            # Empty retrieve
            retrieved, matched = cache.retrieve([])
            assert matched == 0
            assert retrieved is None

    def test_multiple_sequences(self, config):
        """Test storing multiple different sequences."""
        with NanoLMCache(config) as cache:
            # Store sequence 1
            tokens1 = list(range(100))
            kv1 = make_kv_cache(seq_len=100)
            cache.store(tokens1, kv1)

            # Store sequence 2
            tokens2 = list(range(200, 300))
            kv2 = make_kv_cache(seq_len=100)
            cache.store(tokens2, kv2)

            # Both should be retrievable
            _, matched1 = cache.retrieve(tokens1)
            _, matched2 = cache.retrieve(tokens2)

            assert matched1 > 0
            assert matched2 > 0

    def test_prefix_sharing(self, config):
        """Test that common prefix is shared."""
        with NanoLMCache(config) as cache:
            # Store sequence with prefix
            prefix = list(range(50))
            suffix1 = list(range(1000, 1050))
            suffix2 = list(range(2000, 2050))

            tokens1 = prefix + suffix1
            tokens2 = prefix + suffix2

            kv1 = make_kv_cache(seq_len=100)
            kv2 = make_kv_cache(seq_len=100)

            cache.store(tokens1, kv1)

            # Second sequence should match the prefix
            _, matched = cache.retrieve(tokens2)

            # Should match at least part of the prefix
            assert matched > 0

    def test_stats(self, config):
        """Test statistics."""
        with NanoLMCache(config) as cache:
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            stats = cache.stats()

            assert "index" in stats
            assert "storage" in stats
            assert stats["index"]["total_segments"] > 0

    def test_clear(self, config):
        """Test clearing the cache."""
        with NanoLMCache(config) as cache:
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            # Verify data exists
            _, matched = cache.retrieve(tokens)
            assert matched > 0

            # Clear
            cache.clear()

            # Verify data is gone
            _, matched = cache.retrieve(tokens)
            assert matched == 0

    def test_evict(self, config):
        """Test manual eviction."""
        with NanoLMCache(config) as cache:
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            initial_segments = cache.stats()["index"]["total_segments"]

            # Evict
            evicted = cache.evict(n=1)

            assert evicted > 0
            assert cache.stats()["index"]["total_segments"] < initial_segments

    def test_repr(self, config):
        """Test string representation."""
        with NanoLMCache(config) as cache:
            repr_str = repr(cache)
            assert "NanoLMCache" in repr_str

    def test_default_config(self, temp_dir):
        """Test with default configuration."""
        # Modify default config to use temp dir
        config = NanoLMCacheConfig()
        config.storage.disk_cache_dir = temp_dir
        config.enable_logging = False
        config.enable_async_write = False

        with NanoLMCache(config) as cache:
            tokens = list(range(100))
            kv = make_kv_cache(seq_len=100)
            cache.store(tokens, kv)

            _, matched = cache.retrieve(tokens)
            assert matched > 0
