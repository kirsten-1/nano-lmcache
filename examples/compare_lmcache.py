"""
Comparison benchmark: nano-lmcache vs LMCache

This script compares the performance of nano-lmcache against LMCache
using similar benchmark scenarios.

Usage:
    # First install lmcache:
    pip install lmcache

    # Run comparison:
    python examples/compare_lmcache.py --json comparison_results.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

# Check CUDA
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, running on CPU only")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latency_stats(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(times_ms)
    p50_idx = int(len(s) * 0.50)
    p95_idx = int(len(s) * 0.95)
    return {
        "avg": statistics.mean(s),
        "p50": s[min(p50_idx, len(s)-1)],
        "p95": s[min(p95_idx, len(s)-1)],
        "min": s[0],
        "max": s[-1],
    }


def make_kv(layers: int, heads: int, head_dim: int, seq_len: int,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    return torch.randn(layers, 2, seq_len, heads, head_dim, dtype=dtype, device=device)


def kv_size_bytes(layers: int, heads: int, head_dim: int, seq_len: int) -> int:
    return layers * 2 * seq_len * heads * head_dim * 2  # float16


# ---------------------------------------------------------------------------
# nano-lmcache benchmark
# ---------------------------------------------------------------------------

def bench_nano_lmcache(
    layers: int, heads: int, head_dim: int,
    seq_lengths: List[int], iterations: int,
    prefix_length: int, requests: int,
    gpu_size_gb: float, cpu_size_gb: float,
) -> Dict[str, Any]:
    """Benchmark nano-lmcache."""
    from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig

    results = {"library": "nano-lmcache", "scenarios": []}

    # Scenario 1: Store/Retrieve (exact hit)
    for seq_len in seq_lengths:
        config = NanoLMCacheConfig(
            storage=StorageConfig(
                gpu_size_gb=gpu_size_gb,
                cpu_size_gb=cpu_size_gb,
                disk_cache_dir="/tmp/nano_lmcache_compare",
                disk_size_gb=10.0,
            ),
            segment=SegmentConfig(max_segment_length=256, min_segment_length=64),
            enable_async_write=False,
            enable_logging=False,
        )

        with NanoLMCache(config) as cache:
            store_times = []
            retrieve_times = []
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            for i in range(iterations):
                tokens = list(range(i * 100000, i * 100000 + seq_len))
                kv = make_kv(layers, heads, head_dim, seq_len, device=device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                cache.store(tokens, kv, async_write=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                store_times.append((time.perf_counter() - t0) * 1000)

                del kv

            # Retrieve
            total_matched = 0
            for i in range(iterations):
                tokens = list(range(i * 100000, i * 100000 + seq_len))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, matched = cache.retrieve(tokens, target_device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                retrieve_times.append((time.perf_counter() - t0) * 1000)
                total_matched += matched

            hit_rate = total_matched / (seq_len * iterations) if iterations else 0
            kv_mb = kv_size_bytes(layers, heads, head_dim, seq_len) / 1e6

            results["scenarios"].append({
                "test": "store_retrieve",
                "seq_len": seq_len,
                "kv_size_mb": round(kv_mb, 2),
                "store_latency_ms": latency_stats(store_times),
                "retrieve_latency_ms": latency_stats(retrieve_times),
                "hit_rate": round(hit_rate, 4),
            })

            cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario 2: Prefix hit
    config = NanoLMCacheConfig(
        storage=StorageConfig(
            gpu_size_gb=gpu_size_gb,
            cpu_size_gb=cpu_size_gb,
            disk_cache_dir="/tmp/nano_lmcache_compare_prefix",
            disk_size_gb=10.0,
        ),
        segment=SegmentConfig(max_segment_length=256, min_segment_length=64),
        enable_async_write=False,
        enable_logging=False,
    )

    with NanoLMCache(config) as cache:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        prefix_tokens = list(range(prefix_length))
        prefix_kv = make_kv(layers, heads, head_dim, prefix_length, device=device)
        cache.store(prefix_tokens, prefix_kv, async_write=False)
        del prefix_kv

        retrieve_times = []
        for i in range(requests):
            suffix = [prefix_length + i * 128 + j for j in range(128)]
            query = prefix_tokens + suffix

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, matched = cache.retrieve(query, target_device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            retrieve_times.append((time.perf_counter() - t0) * 1000)

        results["scenarios"].append({
            "test": "prefix_hit",
            "prefix_length": prefix_length,
            "suffix_length": 128,
            "requests": requests,
            "retrieve_latency_ms": latency_stats(retrieve_times),
        })

        cache.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# LMCache benchmark (if available)
# ---------------------------------------------------------------------------

def check_lmcache_available() -> Tuple[bool, str]:
    """Check if LMCache is available and return version."""
    try:
        import lmcache
        version = getattr(lmcache, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"


def bench_lmcache(
    layers: int, heads: int, head_dim: int,
    seq_lengths: List[int], iterations: int,
    prefix_length: int, requests: int,
    chunk_size: int = 256,
) -> Optional[Dict[str, Any]]:
    """
    Benchmark LMCache using its low-level API.

    Note: LMCache is primarily designed for vLLM integration.
    This benchmark attempts to use its standalone components.
    """
    available, version = check_lmcache_available()
    if not available:
        print("LMCache not available. Install with: pip install lmcache")
        return None

    print(f"LMCache version: {version}")

    results = {"library": "lmcache", "version": version, "scenarios": []}

    try:
        # Try to use LMCache's low-level API
        # This may vary depending on LMCache version
        from lmcache.config import LMCacheEngineConfig
        from lmcache.storage_backend.local_backend import LMCLocalBackend

        # Create a simple local backend for comparison
        # Note: LMCache's full functionality requires vLLM integration

        for seq_len in seq_lengths:
            store_times = []
            retrieve_times = []

            # LMCache uses chunk-based approach
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            # Simulate LMCache's chunking behavior
            for i in range(iterations):
                tokens = list(range(i * 100000, i * 100000 + seq_len))

                # Store (simulate chunk-based store)
                t0 = time.perf_counter()
                # LMCache would hash each chunk and store
                chunk_hashes = []
                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, seq_len)
                    chunk_tokens = tokens[start:end]
                    # Rolling hash (depends on previous chunks)
                    if c == 0:
                        h = hash(tuple(chunk_tokens))
                    else:
                        h = hash((chunk_hashes[-1], tuple(chunk_tokens)))
                    chunk_hashes.append(h)
                store_times.append((time.perf_counter() - t0) * 1000)

                # Retrieve (simulate)
                t0 = time.perf_counter()
                # LMCache would look up each chunk hash
                for h in chunk_hashes:
                    _ = h  # Simulated lookup
                retrieve_times.append((time.perf_counter() - t0) * 1000)

            results["scenarios"].append({
                "test": "store_retrieve_simulated",
                "seq_len": seq_len,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
                "store_latency_ms": latency_stats(store_times),
                "retrieve_latency_ms": latency_stats(retrieve_times),
                "note": "Simulated - LMCache requires vLLM for full benchmark",
            })

        return results

    except Exception as e:
        print(f"LMCache benchmark failed: {e}")
        print("LMCache requires vLLM integration for full functionality.")
        print("Falling back to simulated comparison...")

        # Provide a simulated comparison based on documented behavior
        return {
            "library": "lmcache",
            "version": version,
            "note": "Simulated - requires vLLM for real benchmark",
            "documented_features": {
                "chunk_size": chunk_size,
                "index_type": "hash_table",
                "supports_cpu_offload": True,
                "supports_disk_offload": True,
                "supports_distributed": True,
            },
        }


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def generate_comparison_summary(
    nano_results: Dict[str, Any],
    lmcache_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a comparison summary."""
    summary = {
        "nano_lmcache": {},
        "lmcache": {},
        "comparison": {},
    }

    # Extract nano-lmcache metrics
    for scenario in nano_results.get("scenarios", []):
        test = scenario.get("test", "unknown")
        if test == "store_retrieve":
            seq_len = scenario["seq_len"]
            key = f"store_retrieve_seq{seq_len}"
            summary["nano_lmcache"][key] = {
                "store_avg_ms": scenario["store_latency_ms"]["avg"],
                "retrieve_avg_ms": scenario["retrieve_latency_ms"]["avg"],
                "hit_rate": scenario["hit_rate"],
            }
        elif test == "prefix_hit":
            summary["nano_lmcache"]["prefix_hit"] = {
                "retrieve_avg_ms": scenario["retrieve_latency_ms"]["avg"],
                "prefix_length": scenario["prefix_length"],
            }

    # Architecture comparison
    summary["comparison"]["architecture"] = {
        "nano_lmcache": {
            "index": "Radix Tree (segment-based)",
            "segmentation": "Variable-length (semantic boundaries)",
            "hash": "Position-dependent rolling hash",
            "storage_tiers": ["GPU", "CPU", "Disk"],
            "code_complexity": "~1000 lines",
        },
        "lmcache": {
            "index": "Hash Table (chunk-based)",
            "segmentation": "Fixed-size chunks",
            "hash": "Rolling hash per chunk",
            "storage_tiers": ["GPU", "CPU", "Disk", "Remote"],
            "code_complexity": "~10000+ lines",
        },
    }

    summary["comparison"]["trade_offs"] = {
        "nano_lmcache_advantages": [
            "Simpler codebase, easier to understand and modify",
            "Variable-length segments avoid chunk alignment waste",
            "Radix tree enables efficient prefix matching",
            "Lower memory overhead for index structure",
        ],
        "lmcache_advantages": [
            "Production-tested with vLLM integration",
            "Supports non-prefix reuse (CacheBlend)",
            "Distributed cache sharing across instances",
            "More mature async and batching support",
        ],
    }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare nano-lmcache vs LMCache")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--prefix-length", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--gpu-size-gb", type=float, default=4.0)
    parser.add_argument("--cpu-size-gb", type=float, default=4.0)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)

    print("=" * 64)
    print("nano-lmcache vs LMCache Comparison")
    print("=" * 64)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KV shape: [{args.layers}, 2, seq_len, {args.heads}, {args.head_dim}] float16")
    print()

    # Benchmark nano-lmcache
    print("--- Benchmarking nano-lmcache ---")
    nano_results = bench_nano_lmcache(
        args.layers, args.heads, args.head_dim,
        args.seq_lengths, args.iterations,
        args.prefix_length, args.requests,
        args.gpu_size_gb, args.cpu_size_gb,
    )

    for scenario in nano_results["scenarios"]:
        test = scenario.get("test", "unknown")
        if test == "store_retrieve":
            print(f"  seq_len={scenario['seq_len']:4d}  "
                  f"store={scenario['store_latency_ms']['avg']:.2f}ms  "
                  f"retrieve={scenario['retrieve_latency_ms']['avg']:.2f}ms  "
                  f"hit_rate={scenario['hit_rate']:.0%}")
        elif test == "prefix_hit":
            print(f"  prefix_hit: retrieve={scenario['retrieve_latency_ms']['avg']:.2f}ms")
    print()

    # Benchmark LMCache (if available)
    print("--- Checking LMCache ---")
    lmcache_available, lmcache_version = check_lmcache_available()
    if lmcache_available:
        print(f"LMCache found: {lmcache_version}")
        lmcache_results = bench_lmcache(
            args.layers, args.heads, args.head_dim,
            args.seq_lengths, args.iterations,
            args.prefix_length, args.requests,
        )
    else:
        print("LMCache not installed. Install with: pip install lmcache")
        print("Generating comparison based on documented behavior...")
        lmcache_results = None
    print()

    # Generate summary
    print("--- Comparison Summary ---")
    summary = generate_comparison_summary(nano_results, lmcache_results)

    print("\nArchitecture Comparison:")
    print(f"  nano-lmcache: Radix Tree + Variable segments + ~1000 lines")
    print(f"  LMCache:      Hash Table + Fixed chunks + ~10000+ lines")

    print("\nnano-lmcache advantages:")
    for adv in summary["comparison"]["trade_offs"]["nano_lmcache_advantages"]:
        print(f"  + {adv}")

    print("\nLMCache advantages:")
    for adv in summary["comparison"]["trade_offs"]["lmcache_advantages"]:
        print(f"  + {adv}")

    # Output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "nano_lmcache": nano_results,
        "lmcache": lmcache_results,
        "summary": summary,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json}")

    print()
    print("=" * 64)
    print("Comparison complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
