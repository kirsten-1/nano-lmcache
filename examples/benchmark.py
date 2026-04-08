"""
Reproducible cache microbenchmark for nano-lmcache.

Measures store/retrieve performance across three scenarios:
  - exact_hit:  store a sequence, retrieve the same sequence
  - prefix_hit: store a base sequence, retrieve with shared prefix + new suffix
  - miss:       retrieve a sequence that was never stored

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --requests 200 --seq-lengths 512 1024 2048 --json results.json
    python examples/benchmark.py --scenarios exact_hit prefix_hit
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def latency_stats(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(times_ms)
    return {
        "avg": statistics.mean(s),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "min": s[0],
        "max": s[-1],
    }


def make_kv(layers: int, heads: int, head_dim: int, seq_len: int,
            dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(layers, 2, seq_len, heads, head_dim, dtype=dtype)


def env_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
    return meta


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_exact_hit(cache: NanoLMCache, seq_lengths: List[int],
                  requests: int, layers: int, heads: int,
                  head_dim: int, dtype: torch.dtype) -> Dict[str, Any]:
    """Store a sequence, then retrieve the exact same sequence."""
    store_times: List[float] = []
    retrieve_times: List[float] = []
    total_matched = 0
    total_tokens = 0

    for i in range(requests):
        seq_len = seq_lengths[i % len(seq_lengths)]
        tokens = list(range(i * 10000, i * 10000 + seq_len))
        kv = make_kv(layers, heads, head_dim, seq_len, dtype)

        # store
        t0 = time.perf_counter()
        cache.store(tokens, kv, async_write=False)
        store_times.append((time.perf_counter() - t0) * 1000)

        # retrieve
        t0 = time.perf_counter()
        _, matched = cache.retrieve(tokens)
        retrieve_times.append((time.perf_counter() - t0) * 1000)

        total_matched += matched
        total_tokens += seq_len

    return {
        "scenario": "exact_hit",
        "requests": requests,
        "total_tokens": total_tokens,
        "total_matched": total_matched,
        "token_hit_rate": total_matched / total_tokens if total_tokens else 0,
        "store_latency_ms": latency_stats(store_times),
        "retrieve_latency_ms": latency_stats(retrieve_times),
    }


def run_prefix_hit(cache: NanoLMCache, seq_lengths: List[int],
                   prefix_length: int, requests: int,
                   layers: int, heads: int, head_dim: int,
                   dtype: torch.dtype) -> Dict[str, Any]:
    """Store a base sequence once, then retrieve prefix + unique suffix."""
    # store the prefix/base once
    prefix_tokens = list(range(prefix_length))
    base_kv = make_kv(layers, heads, head_dim, prefix_length, dtype)
    cache.store(prefix_tokens, base_kv, async_write=False)

    retrieve_times: List[float] = []
    total_matched = 0
    total_tokens = 0

    for i in range(requests):
        suffix_len = seq_lengths[i % len(seq_lengths)]
        suffix = [prefix_length + i * 10000 + j for j in range(suffix_len)]
        query = prefix_tokens + suffix

        t0 = time.perf_counter()
        _, matched = cache.retrieve(query)
        retrieve_times.append((time.perf_counter() - t0) * 1000)

        total_matched += matched
        total_tokens += len(query)

    return {
        "scenario": "prefix_hit",
        "requests": requests,
        "prefix_length": prefix_length,
        "total_tokens": total_tokens,
        "total_matched": total_matched,
        "token_hit_rate": total_matched / total_tokens if total_tokens else 0,
        "retrieve_latency_ms": latency_stats(retrieve_times),
    }


def run_miss(cache: NanoLMCache, seq_lengths: List[int],
             requests: int, layers: int, heads: int,
             head_dim: int, dtype: torch.dtype) -> Dict[str, Any]:
    """Retrieve sequences that were never stored."""
    # pre-populate with some data so the index is not empty
    base_tokens = list(range(1024))
    base_kv = make_kv(layers, heads, head_dim, 1024, dtype)
    cache.store(base_tokens, base_kv, async_write=False)

    retrieve_times: List[float] = []

    for i in range(requests):
        seq_len = seq_lengths[i % len(seq_lengths)]
        tokens = [9000000 + i * 10000 + j for j in range(seq_len)]

        t0 = time.perf_counter()
        _, matched = cache.retrieve(tokens)
        retrieve_times.append((time.perf_counter() - t0) * 1000)

        assert matched == 0

    return {
        "scenario": "miss",
        "requests": requests,
        "retrieve_latency_ms": latency_stats(retrieve_times),
    }


SCENARIOS = {
    "exact_hit": run_exact_hit,
    "prefix_hit": run_prefix_hit,
    "miss": run_miss,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nano-lmcache reproducible benchmark")
    p.add_argument("--scenarios", nargs="+", default=["exact_hit", "prefix_hit", "miss"],
                   choices=list(SCENARIOS.keys()), help="Scenarios to run")
    p.add_argument("--requests", type=int, default=100, help="Requests per scenario")
    p.add_argument("--seq-lengths", nargs="+", type=int, default=[512, 1024, 2048],
                   help="Sequence lengths to cycle through")
    p.add_argument("--prefix-length", type=int, default=512,
                   help="Prefix length for prefix_hit scenario")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=8, help="Number of KV heads")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--cpu-size-gb", type=float, default=4.0)
    p.add_argument("--disk-size-gb", type=float, default=10.0)
    p.add_argument("--max-segment-length", type=int, default=256)
    p.add_argument("--min-segment-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", type=str, default=None, help="Path to write JSON results")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations (discarded)")
    return p


def main():
    args = build_parser().parse_args()
    set_seeds(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    config = NanoLMCacheConfig(
        storage=StorageConfig(
            cpu_size_gb=args.cpu_size_gb,
            disk_size_gb=args.disk_size_gb,
            disk_cache_dir="/tmp/nano_lmcache_bench",
        ),
        segment=SegmentConfig(
            max_segment_length=args.max_segment_length,
            min_segment_length=args.min_segment_length,
        ),
        num_layers=args.layers,
        num_kv_heads=args.heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        enable_async_write=False,
        enable_logging=False,
    )

    print("=" * 64)
    print("nano-lmcache benchmark")
    print("=" * 64)

    env = env_metadata()
    print(f"Platform : {env['platform']}")
    print(f"Python   : {env['python']}")
    print(f"PyTorch  : {env['torch']}")
    print(f"CUDA     : {env.get('gpu', 'N/A')}")
    print(f"Seed     : {args.seed}")
    print(f"KV shape : [{args.layers}, 2, seq_len, {args.heads}, {args.head_dim}] {args.dtype}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print()

    all_results: List[Dict[str, Any]] = []

    for scenario_name in args.scenarios:
        with NanoLMCache(config) as cache:
            # warmup
            if args.warmup > 0:
                wt = list(range(256))
                wkv = make_kv(args.layers, args.heads, args.head_dim, 256, dtype)
                for _ in range(args.warmup):
                    cache.store(wt, wkv, async_write=False)
                    cache.retrieve(wt)
                cache.clear()

            # run
            if scenario_name == "prefix_hit":
                result = run_prefix_hit(
                    cache, args.seq_lengths, args.prefix_length,
                    args.requests, args.layers, args.heads, args.head_dim, dtype,
                )
            elif scenario_name == "miss":
                result = run_miss(
                    cache, args.seq_lengths, args.requests,
                    args.layers, args.heads, args.head_dim, dtype,
                )
            else:
                result = run_exact_hit(
                    cache, args.seq_lengths, args.requests,
                    args.layers, args.heads, args.head_dim, dtype,
                )

            all_results.append(result)

            # console output
            print(f"--- {scenario_name} ---")
            if "store_latency_ms" in result:
                sl = result["store_latency_ms"]
                print(f"  Store    avg={sl['avg']:.2f}ms  p50={sl['p50']:.2f}ms  p95={sl['p95']:.2f}ms")
            rl = result["retrieve_latency_ms"]
            print(f"  Retrieve avg={rl['avg']:.2f}ms  p50={rl['p50']:.2f}ms  p95={rl['p95']:.2f}ms")
            if "token_hit_rate" in result:
                print(f"  Token hit rate: {result['token_hit_rate']:.1%}")
            print()

    # structured output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": env,
        "args": {
            "scenarios": args.scenarios,
            "requests": args.requests,
            "seq_lengths": args.seq_lengths,
            "prefix_length": args.prefix_length,
            "layers": args.layers,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "cpu_size_gb": args.cpu_size_gb,
            "disk_size_gb": args.disk_size_gb,
            "max_segment_length": args.max_segment_length,
            "min_segment_length": args.min_segment_length,
            "seed": args.seed,
            "warmup": args.warmup,
        },
        "results": all_results,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.json}")

    print("=" * 64)
    print("Benchmark complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
