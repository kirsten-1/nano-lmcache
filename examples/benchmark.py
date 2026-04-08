"""
Benchmark example for NanoLMCache.

This example measures:
1. Store throughput
2. Retrieve latency
3. Cache hit rate
"""

import torch
import time
import random
from typing import List, Tuple
from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig


def create_mock_kv_cache(seq_len: int) -> torch.Tensor:
    """Create mock KV cache."""
    return torch.randn(32, 2, seq_len, 8, 128, dtype=torch.float16)


def generate_requests(
    num_requests: int,
    seq_lengths: List[int],
    prefix_reuse_ratio: float = 0.5,
) -> List[Tuple[List[int], bool]]:
    """
    Generate mock requests.

    Args:
        num_requests: Number of requests to generate
        seq_lengths: Possible sequence lengths
        prefix_reuse_ratio: Ratio of requests that reuse a common prefix

    Returns:
        List of (tokens, is_prefix_reuse)
    """
    requests = []
    common_prefix = list(range(256))  # Common system prompt

    for i in range(num_requests):
        seq_len = random.choice(seq_lengths)

        if random.random() < prefix_reuse_ratio:
            # Reuse common prefix
            suffix = [1000 + j for j in range(seq_len - len(common_prefix))]
            tokens = common_prefix + suffix
            is_prefix_reuse = True
        else:
            # New sequence
            tokens = [2000 + j for j in range(seq_len)]
            is_prefix_reuse = False

        requests.append((tokens, is_prefix_reuse))

    return requests


def benchmark_store(cache: NanoLMCache, requests: List[Tuple[List[int], bool]]):
    """Benchmark store performance."""
    print("\n=== Store Benchmark ===")

    total_tokens = 0
    total_time = 0

    for tokens, _ in requests:
        kv = create_mock_kv_cache(len(tokens))

        start = time.perf_counter()
        cache.store(tokens, kv, async_write=False)
        elapsed = time.perf_counter() - start

        total_tokens += len(tokens)
        total_time += elapsed

    tokens_per_sec = total_tokens / total_time
    print(f"Total tokens stored: {total_tokens:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")


def benchmark_retrieve(cache: NanoLMCache, requests: List[Tuple[List[int], bool]]):
    """Benchmark retrieve performance."""
    print("\n=== Retrieve Benchmark ===")

    total_queries = 0
    total_hits = 0
    total_tokens_queried = 0
    total_tokens_matched = 0
    total_time = 0

    for tokens, is_prefix_reuse in requests:
        start = time.perf_counter()
        _, matched = cache.retrieve(tokens)
        elapsed = time.perf_counter() - start

        total_queries += 1
        total_tokens_queried += len(tokens)
        total_tokens_matched += matched
        total_time += elapsed

        if matched > 0:
            total_hits += 1

    hit_rate = total_hits / total_queries
    token_hit_rate = total_tokens_matched / total_tokens_queried
    avg_latency_ms = (total_time / total_queries) * 1000

    print(f"Total queries: {total_queries}")
    print(f"Cache hit rate: {hit_rate:.1%}")
    print(f"Token hit rate: {token_hit_rate:.1%}")
    print(f"Average latency: {avg_latency_ms:.2f}ms")


def main():
    print("=" * 60)
    print("NanoLMCache Benchmark")
    print("=" * 60)

    # Configuration
    config = NanoLMCacheConfig(
        storage=StorageConfig(
            cpu_size_gb=2.0,
            disk_size_gb=5.0,
            disk_cache_dir="/tmp/nano_lmcache_bench",
        ),
        segment=SegmentConfig(
            max_segment_length=256,
            min_segment_length=64,
        ),
        enable_async_write=False,
        enable_logging=False,
    )

    # Generate requests
    print("\nGenerating requests...")
    num_requests = 100
    requests = generate_requests(
        num_requests=num_requests,
        seq_lengths=[256, 512, 1024],
        prefix_reuse_ratio=0.6,
    )
    print(f"Generated {num_requests} requests")

    # Run benchmarks
    with NanoLMCache(config) as cache:
        # First pass: store all
        benchmark_store(cache, requests)

        # Second pass: retrieve (should hit cache for prefix reuse)
        benchmark_retrieve(cache, requests)

        # Print final stats
        print("\n=== Final Stats ===")
        stats = cache.stats()
        print(f"Total segments: {stats['index']['total_segments']}")
        print(f"Total cached tokens: {stats['index']['total_tokens']}")
        print(f"CPU usage: {stats['storage']['cpu']['size_bytes'] / 1e6:.1f} MB")
        print(f"Disk usage: {stats['storage']['disk']['size_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
