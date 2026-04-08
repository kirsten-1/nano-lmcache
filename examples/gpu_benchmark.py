"""
GPU Benchmark for nano-lmcache.

This script measures GPU KV cache performance:
1. GPU <-> CPU transfer throughput
2. Cache hit/miss latency
3. Comparison with different tier configurations

Requirements:
- CUDA-capable GPU (4090/5090 recommended)
- PyTorch with CUDA support

Usage:
    python examples/gpu_benchmark.py
"""

import torch
import time
import argparse
from typing import List, Tuple
import statistics

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This benchmark requires a GPU.")
    print("Please run on a machine with CUDA support.")
    exit(1)

from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig
from nano_lmcache.index import StorageTier


def get_gpu_info():
    """Print GPU information."""
    print("\n=== GPU Information ===")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - SM Count: {props.multi_processor_count}")


def create_kv_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Create a mock KV cache tensor on GPU."""
    return torch.randn(
        num_layers, 2, seq_len, num_heads, head_dim,
        dtype=dtype, device=device,
    )


def benchmark_gpu_to_cpu_transfer(
    tensor_sizes_mb: List[float],
    num_iterations: int = 10,
) -> List[Tuple[float, float, float]]:
    """
    Benchmark GPU to CPU transfer throughput.

    Returns:
        List of (size_mb, throughput_gbps, latency_ms)
    """
    print("\n=== GPU -> CPU Transfer Benchmark ===")
    results = []

    for size_mb in tensor_sizes_mb:
        num_elements = int(size_mb * 1e6 / 2)  # float16 = 2 bytes
        tensor = torch.randn(num_elements, dtype=torch.float16, device="cuda:0")

        # Warmup
        for _ in range(3):
            _ = tensor.cpu()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cpu_tensor = tensor.cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        throughput = size_mb / avg_time / 1000  # GB/s
        latency = avg_time * 1000  # ms

        print(f"  {size_mb:6.1f} MB: {throughput:6.2f} GB/s, {latency:6.2f} ms")
        results.append((size_mb, throughput, latency))

    return results


def benchmark_cpu_to_gpu_transfer(
    tensor_sizes_mb: List[float],
    num_iterations: int = 10,
    use_pinned: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Benchmark CPU to GPU transfer throughput.
    """
    print(f"\n=== CPU -> GPU Transfer Benchmark (pinned={use_pinned}) ===")
    results = []

    for size_mb in tensor_sizes_mb:
        num_elements = int(size_mb * 1e6 / 2)

        if use_pinned:
            tensor = torch.randn(num_elements, dtype=torch.float16, pin_memory=True)
        else:
            tensor = torch.randn(num_elements, dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = tensor.to("cuda:0")
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            gpu_tensor = tensor.to("cuda:0", non_blocking=use_pinned)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        throughput = size_mb / avg_time / 1000
        latency = avg_time * 1000

        print(f"  {size_mb:6.1f} MB: {throughput:6.2f} GB/s, {latency:6.2f} ms")
        results.append((size_mb, throughput, latency))

    return results


def benchmark_cache_operations(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_lengths: List[int] = [256, 512, 1024, 2048],
    num_iterations: int = 10,
):
    """
    Benchmark cache store/retrieve operations.
    """
    print("\n=== Cache Operations Benchmark ===")
    print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")

    # Calculate KV size per token
    bytes_per_token = 2 * num_layers * num_heads * head_dim * 2  # float16
    print(f"KV cache size per token: {bytes_per_token / 1024:.2f} KB")

    # Initialize cache with GPU support
    config = NanoLMCacheConfig(
        storage=StorageConfig(
            gpu_size_gb=2.0,
            cpu_size_gb=4.0,
            disk_size_gb=10.0,
            disk_cache_dir="/tmp/nano_lmcache_bench",
        ),
        segment=SegmentConfig(
            max_segment_length=256,
            min_segment_length=64,
        ),
        enable_async_write=False,
        enable_logging=False,
    )

    with NanoLMCache(config) as cache:
        for seq_len in seq_lengths:
            tokens = list(range(seq_len))
            kv_cache = create_kv_cache(num_layers, num_heads, head_dim, seq_len)
            kv_size_mb = kv_cache.numel() * 2 / 1e6

            # Benchmark store
            store_times = []
            for i in range(num_iterations):
                # Use different tokens each time to avoid cache effects
                test_tokens = [t + i * 10000 for t in tokens]
                torch.cuda.synchronize()
                start = time.perf_counter()
                cache.store(test_tokens, kv_cache, async_write=False)
                torch.cuda.synchronize()
                store_times.append(time.perf_counter() - start)

            # Benchmark retrieve (cache hit)
            retrieve_times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _, matched = cache.retrieve(tokens, target_device="cuda:0")
                torch.cuda.synchronize()
                retrieve_times.append(time.perf_counter() - start)

            store_avg = statistics.mean(store_times) * 1000
            retrieve_avg = statistics.mean(retrieve_times) * 1000
            store_throughput = kv_size_mb / statistics.mean(store_times) / 1000

            print(f"\nSeq length: {seq_len} ({kv_size_mb:.2f} MB)")
            print(f"  Store:    {store_avg:6.2f} ms ({store_throughput:.2f} GB/s)")
            print(f"  Retrieve: {retrieve_avg:6.2f} ms")

            # Clear for next iteration
            cache.clear()


def benchmark_prefix_matching(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    prefix_length: int = 1024,
    suffix_lengths: List[int] = [128, 256, 512],
    num_requests: int = 100,
):
    """
    Benchmark prefix matching scenarios (like system prompt reuse).
    """
    print("\n=== Prefix Matching Benchmark ===")
    print(f"Prefix length: {prefix_length} tokens")

    config = NanoLMCacheConfig(
        storage=StorageConfig(
            gpu_size_gb=2.0,
            cpu_size_gb=4.0,
            disk_cache_dir="/tmp/nano_lmcache_prefix",
        ),
        segment=SegmentConfig(
            max_segment_length=256,
            min_segment_length=64,
        ),
        enable_async_write=False,
        enable_logging=False,
    )

    with NanoLMCache(config) as cache:
        # Store prefix once (simulate system prompt)
        prefix_tokens = list(range(prefix_length))
        prefix_kv = create_kv_cache(num_layers, num_heads, head_dim, prefix_length)
        cache.store(prefix_tokens, prefix_kv, async_write=False)

        for suffix_len in suffix_lengths:
            hit_times = []
            miss_times = []
            hit_count = 0

            for i in range(num_requests):
                # Create query with shared prefix + unique suffix
                suffix = [prefix_length + i * suffix_len + j for j in range(suffix_len)]
                query_tokens = prefix_tokens + suffix

                torch.cuda.synchronize()
                start = time.perf_counter()
                _, matched = cache.retrieve(query_tokens, target_device="cuda:0")
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000

                if matched > 0:
                    hit_times.append(elapsed)
                    hit_count += 1
                else:
                    miss_times.append(elapsed)

            hit_rate = hit_count / num_requests * 100
            avg_hit_time = statistics.mean(hit_times) if hit_times else 0
            avg_miss_time = statistics.mean(miss_times) if miss_times else 0

            print(f"\nSuffix length: {suffix_len}")
            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Avg hit latency:  {avg_hit_time:.2f} ms")
            print(f"  Avg miss latency: {avg_miss_time:.2f} ms")
            if hit_times:
                print(f"  Tokens matched: {prefix_length} (prefix)")


def benchmark_tier_promotion(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 512,
    num_iterations: int = 10,
):
    """
    Benchmark tier promotion/demotion.
    """
    print("\n=== Tier Promotion Benchmark ===")

    config = NanoLMCacheConfig(
        storage=StorageConfig(
            gpu_size_gb=0.5,  # Small GPU cache to force promotion
            cpu_size_gb=2.0,
            disk_cache_dir="/tmp/nano_lmcache_tier",
        ),
        enable_logging=False,
    )

    with NanoLMCache(config) as cache:
        tokens = list(range(seq_len))
        kv_cache = create_kv_cache(num_layers, num_heads, head_dim, seq_len)

        # Store to CPU first
        cache.store(tokens, kv_cache, async_write=False)

        # Benchmark CPU -> GPU promotion
        promote_times = []
        for _ in range(num_iterations):
            # Force to CPU
            cache.storage.demote(list(cache.storage._locations.keys())[0])

            torch.cuda.synchronize()
            start = time.perf_counter()
            cache.prefetch(tokens, StorageTier.GPU)
            torch.cuda.synchronize()
            promote_times.append(time.perf_counter() - start)

        # Benchmark GPU -> CPU demotion
        demote_times = []
        for _ in range(num_iterations):
            # Ensure on GPU
            cache.prefetch(tokens, StorageTier.GPU)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for key in list(cache.storage._locations.keys()):
                cache.storage.demote(key)
            torch.cuda.synchronize()
            demote_times.append(time.perf_counter() - start)

        kv_size_mb = kv_cache.numel() * 2 / 1e6
        print(f"KV cache size: {kv_size_mb:.2f} MB")
        print(f"CPU -> GPU promotion: {statistics.mean(promote_times)*1000:.2f} ms")
        print(f"GPU -> CPU demotion:  {statistics.mean(demote_times)*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark for nano-lmcache")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--transfer", action="store_true", help="Run transfer benchmarks")
    parser.add_argument("--cache", action="store_true", help="Run cache operation benchmarks")
    parser.add_argument("--prefix", action="store_true", help="Run prefix matching benchmarks")
    parser.add_argument("--tier", action="store_true", help="Run tier promotion benchmarks")
    parser.add_argument("--layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.transfer, args.cache, args.prefix, args.tier]):
        args.all = True

    print("=" * 60)
    print("nano-lmcache GPU Benchmark")
    print("=" * 60)

    get_gpu_info()

    if args.all or args.transfer:
        sizes = [1, 10, 50, 100, 500]
        benchmark_gpu_to_cpu_transfer(sizes)
        benchmark_cpu_to_gpu_transfer(sizes, use_pinned=True)
        benchmark_cpu_to_gpu_transfer(sizes, use_pinned=False)

    if args.all or args.cache:
        benchmark_cache_operations(
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.head_dim,
        )

    if args.all or args.prefix:
        benchmark_prefix_matching(
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.head_dim,
        )

    if args.all or args.tier:
        benchmark_tier_promotion(
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.head_dim,
        )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
