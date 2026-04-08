"""
Reproducible GPU benchmark for nano-lmcache.

Measures GPU-specific cache performance:
  1. GPU <-> CPU transfer throughput
  2. Cache store/retrieve with GPU tier
  3. Prefix-hit latency with GPU tier
  4. Tier promotion/demotion cost

Requirements:
  - CUDA-capable GPU
  - PyTorch with CUDA support

Usage:
    python examples/gpu_benchmark.py
    python examples/gpu_benchmark.py --all --json gpu_results.json
    python examples/gpu_benchmark.py --cache --prefix --layers 32 --heads 32
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Guard: CUDA required
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This benchmark requires a GPU.")
    sys.exit(1)

from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig
from nano_lmcache.index import StorageTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            dtype: torch.dtype = torch.float16,
            device: str = "cuda:0") -> torch.Tensor:
    return torch.randn(layers, 2, seq_len, heads, head_dim, dtype=dtype, device=device)


def env_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        meta["gpus"].append({
            "index": i,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
        })
    return meta


def print_latency(label: str, stats: Dict[str, float]):
    print(f"  {label:20s}  avg={stats['avg']:7.2f}ms  "
          f"p50={stats['p50']:7.2f}ms  p95={stats['p95']:7.2f}ms")


# ---------------------------------------------------------------------------
# Benchmark: raw transfer
# ---------------------------------------------------------------------------

def bench_transfer(sizes_mb: List[float], iterations: int,
                   warmup: int) -> Dict[str, Any]:
    """Measure raw GPU<->CPU memcpy throughput."""
    results: Dict[str, List[Dict[str, Any]]] = {"gpu_to_cpu": [], "cpu_to_gpu_pinned": [],
                                                  "cpu_to_gpu_paged": []}

    for size_mb in sizes_mb:
        n = max(1, int(size_mb * 1e6 / 2))  # float16

        # GPU -> CPU
        gpu_t = torch.randn(n, dtype=torch.float16, device="cuda:0")
        for _ in range(warmup):
            _ = gpu_t.cpu()
        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = gpu_t.cpu()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(times)
        throughput = size_mb / avg_ms * 1000 / 1000  # GB/s
        results["gpu_to_cpu"].append({"size_mb": size_mb, "throughput_gbps": round(throughput, 2),
                                      "latency_ms": latency_stats(times)})

        # CPU (pinned) -> GPU
        cpu_p = torch.randn(n, dtype=torch.float16, pin_memory=True)
        for _ in range(warmup):
            _ = cpu_p.to("cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = cpu_p.to("cuda:0", non_blocking=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(times)
        throughput = size_mb / avg_ms * 1000 / 1000
        results["cpu_to_gpu_pinned"].append({"size_mb": size_mb, "throughput_gbps": round(throughput, 2),
                                             "latency_ms": latency_stats(times)})

        # CPU (paged) -> GPU
        cpu_pg = torch.randn(n, dtype=torch.float16)
        for _ in range(warmup):
            _ = cpu_pg.to("cuda:0")
        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = cpu_pg.to("cuda:0")
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(times)
        throughput = size_mb / avg_ms * 1000 / 1000
        results["cpu_to_gpu_paged"].append({"size_mb": size_mb, "throughput_gbps": round(throughput, 2),
                                            "latency_ms": latency_stats(times)})

    return {"benchmark": "transfer", **results}


# ---------------------------------------------------------------------------
# Benchmark: cache store/retrieve (exact hit)
# ---------------------------------------------------------------------------

def bench_cache_ops(layers: int, heads: int, head_dim: int,
                    seq_lengths: List[int], iterations: int,
                    warmup: int, gpu_size_gb: float,
                    cpu_size_gb: float) -> Dict[str, Any]:
    """Store + exact-retrieve with GPU tier."""
    bytes_per_token = 2 * layers * heads * head_dim * 2  # float16
    per_scenario: List[Dict[str, Any]] = []

    for seq_len in seq_lengths:
        config = NanoLMCacheConfig(
            storage=StorageConfig(gpu_size_gb=gpu_size_gb, cpu_size_gb=cpu_size_gb,
                                  disk_cache_dir="/tmp/nano_lmcache_gpu_bench",
                                  disk_size_gb=10.0),
            segment=SegmentConfig(max_segment_length=256, min_segment_length=64),
            enable_async_write=False, enable_logging=False,
        )

        with NanoLMCache(config) as cache:
            # warmup
            wt = list(range(256))
            wkv = make_kv(layers, heads, head_dim, 256)
            for _ in range(warmup):
                cache.store(wt, wkv, async_write=False)
                cache.retrieve(wt, target_device="cuda:0")
            cache.clear()

            store_times: List[float] = []
            retrieve_times: List[float] = []

            for i in range(iterations):
                tokens = [t + i * 100000 for t in range(seq_len)]
                kv = make_kv(layers, heads, head_dim, seq_len)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cache.store(tokens, kv, async_write=False)
                torch.cuda.synchronize()
                store_times.append((time.perf_counter() - t0) * 1000)

            # retrieve the last stored sequence
            tokens_last = [t + (iterations - 1) * 100000 for t in range(seq_len)]
            for _ in range(warmup):
                cache.retrieve(tokens_last, target_device="cuda:0")
            torch.cuda.synchronize()

            for _ in range(iterations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, matched = cache.retrieve(tokens_last, target_device="cuda:0")
                torch.cuda.synchronize()
                retrieve_times.append((time.perf_counter() - t0) * 1000)

            kv_size_mb = round(seq_len * bytes_per_token / 1e6, 2)
            per_scenario.append({
                "seq_len": seq_len,
                "kv_size_mb": kv_size_mb,
                "store_latency_ms": latency_stats(store_times),
                "retrieve_latency_ms": latency_stats(retrieve_times),
            })

    return {"benchmark": "cache_ops", "bytes_per_token": bytes_per_token,
            "scenarios": per_scenario}


# ---------------------------------------------------------------------------
# Benchmark: prefix hit
# ---------------------------------------------------------------------------

def bench_prefix(layers: int, heads: int, head_dim: int,
                 prefix_length: int, suffix_lengths: List[int],
                 requests: int, warmup: int,
                 gpu_size_gb: float, cpu_size_gb: float) -> Dict[str, Any]:
    """Shared-prefix retrieval latency (system prompt reuse scenario)."""
    config = NanoLMCacheConfig(
        storage=StorageConfig(gpu_size_gb=gpu_size_gb, cpu_size_gb=cpu_size_gb,
                              disk_cache_dir="/tmp/nano_lmcache_gpu_prefix",
                              disk_size_gb=10.0),
        segment=SegmentConfig(max_segment_length=256, min_segment_length=64),
        enable_async_write=False, enable_logging=False,
    )
    per_suffix: List[Dict[str, Any]] = []

    with NanoLMCache(config) as cache:
        prefix_tokens = list(range(prefix_length))
        prefix_kv = make_kv(layers, heads, head_dim, prefix_length)
        cache.store(prefix_tokens, prefix_kv, async_write=False)

        for suffix_len in suffix_lengths:
            retrieve_times: List[float] = []
            hit_count = 0

            for i in range(requests):
                suffix = [prefix_length + i * suffix_len + j for j in range(suffix_len)]
                query = prefix_tokens + suffix

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, matched = cache.retrieve(query, target_device="cuda:0")
                torch.cuda.synchronize()
                retrieve_times.append((time.perf_counter() - t0) * 1000)

                if matched > 0:
                    hit_count += 1

            per_suffix.append({
                "suffix_len": suffix_len,
                "total_query_len": prefix_length + suffix_len,
                "hit_rate": hit_count / requests if requests else 0,
                "retrieve_latency_ms": latency_stats(retrieve_times),
            })

    return {"benchmark": "prefix_hit", "prefix_length": prefix_length,
            "requests": requests, "scenarios": per_suffix}


# ---------------------------------------------------------------------------
# Benchmark: tier promotion
# ---------------------------------------------------------------------------

def bench_tier(layers: int, heads: int, head_dim: int,
               seq_len: int, iterations: int,
               gpu_size_gb: float, cpu_size_gb: float) -> Dict[str, Any]:
    """Measure CPU->GPU promotion and GPU->CPU demotion cost."""
    config = NanoLMCacheConfig(
        storage=StorageConfig(gpu_size_gb=gpu_size_gb, cpu_size_gb=cpu_size_gb,
                              disk_cache_dir="/tmp/nano_lmcache_gpu_tier"),
        enable_async_write=False, enable_logging=False,
    )

    promote_times: List[float] = []
    demote_times: List[float] = []

    with NanoLMCache(config) as cache:
        tokens = list(range(seq_len))
        kv = make_kv(layers, heads, head_dim, seq_len)
        cache.store(tokens, kv, async_write=False)

        for _ in range(iterations):
            # demote all to CPU
            for key in list(cache.storage._locations.keys()):
                cache.storage.demote(key)

            # promote
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cache.prefetch(tokens, StorageTier.GPU)
            torch.cuda.synchronize()
            promote_times.append((time.perf_counter() - t0) * 1000)

        for _ in range(iterations):
            cache.prefetch(tokens, StorageTier.GPU)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for key in list(cache.storage._locations.keys()):
                cache.storage.demote(key)
            torch.cuda.synchronize()
            demote_times.append((time.perf_counter() - t0) * 1000)

    kv_size_mb = round(seq_len * 2 * layers * heads * head_dim * 2 / 1e6, 2)
    return {
        "benchmark": "tier_promotion",
        "seq_len": seq_len,
        "kv_size_mb": kv_size_mb,
        "cpu_to_gpu_ms": latency_stats(promote_times),
        "gpu_to_cpu_ms": latency_stats(demote_times),
    }


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nano-lmcache GPU benchmark")
    p.add_argument("--all", action="store_true", help="Run all benchmarks")
    p.add_argument("--transfer", action="store_true", help="Run transfer benchmarks")
    p.add_argument("--cache", action="store_true", help="Run cache op benchmarks")
    p.add_argument("--prefix", action="store_true", help="Run prefix-hit benchmarks")
    p.add_argument("--tier", action="store_true", help="Run tier promotion benchmarks")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq-lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    p.add_argument("--prefix-length", type=int, default=1024)
    p.add_argument("--suffix-lengths", nargs="+", type=int, default=[128, 256, 512])
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--transfer-sizes", nargs="+", type=float, default=[1, 10, 50, 100, 500])
    p.add_argument("--gpu-size-gb", type=float, default=2.0)
    p.add_argument("--cpu-size-gb", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", type=str, default=None, help="Path to write JSON results")
    return p


def main():
    args = build_parser().parse_args()
    set_seeds(args.seed)

    if not any([args.all, args.transfer, args.cache, args.prefix, args.tier]):
        args.all = True

    env = env_metadata()

    print("=" * 64)
    print("nano-lmcache GPU benchmark")
    print("=" * 64)
    for gpu in env["gpus"]:
        print(f"GPU {gpu['index']}: {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"CUDA   : {env['cuda_version']}")
    print(f"PyTorch: {env['torch']}")
    print(f"Seed   : {args.seed}")
    print(f"KV     : [{args.layers}, 2, seq_len, {args.heads}, {args.head_dim}] float16")
    print()

    all_results: List[Dict[str, Any]] = []

    # --- transfer ---
    if args.all or args.transfer:
        print("--- Transfer throughput ---")
        res = bench_transfer(args.transfer_sizes, args.iterations, args.warmup)
        all_results.append(res)
        for direction in ["gpu_to_cpu", "cpu_to_gpu_pinned", "cpu_to_gpu_paged"]:
            print(f"  {direction}:")
            for entry in res[direction]:
                ls = entry["latency_ms"]
                print(f"    {entry['size_mb']:6.0f} MB  {entry['throughput_gbps']:6.2f} GB/s  "
                      f"avg={ls['avg']:.2f}ms  p95={ls['p95']:.2f}ms")
        print()

    # --- cache ops ---
    if args.all or args.cache:
        print("--- Cache store/retrieve (exact hit) ---")
        res = bench_cache_ops(args.layers, args.heads, args.head_dim,
                              args.seq_lengths, args.iterations, args.warmup,
                              args.gpu_size_gb, args.cpu_size_gb)
        all_results.append(res)
        for sc in res["scenarios"]:
            print(f"  seq_len={sc['seq_len']:5d}  kv={sc['kv_size_mb']:.1f}MB")
            print_latency("Store", sc["store_latency_ms"])
            print_latency("Retrieve", sc["retrieve_latency_ms"])
        print()

    # --- prefix ---
    if args.all or args.prefix:
        print(f"--- Prefix hit (prefix={args.prefix_length}) ---")
        res = bench_prefix(args.layers, args.heads, args.head_dim,
                           args.prefix_length, args.suffix_lengths,
                           args.requests, args.warmup,
                           args.gpu_size_gb, args.cpu_size_gb)
        all_results.append(res)
        for sc in res["scenarios"]:
            print(f"  suffix={sc['suffix_len']:5d}  hit_rate={sc['hit_rate']:.0%}")
            print_latency("Retrieve", sc["retrieve_latency_ms"])
        print()

    # --- tier ---
    if args.all or args.tier:
        print("--- Tier promotion/demotion ---")
        res = bench_tier(args.layers, args.heads, args.head_dim,
                         512, args.iterations, args.gpu_size_gb, args.cpu_size_gb)
        all_results.append(res)
        print(f"  KV size: {res['kv_size_mb']} MB")
        print_latency("CPU->GPU promote", res["cpu_to_gpu_ms"])
        print_latency("GPU->CPU demote", res["gpu_to_cpu_ms"])
        print()

    # --- structured output ---
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": env,
        "args": {k: v for k, v in vars(args).items() if k != "json"},
        "results": all_results,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.json}")

    print("=" * 64)
    print("GPU benchmark complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
