#!/usr/bin/env python3
"""Run nano-lmcache's vLLM connector in dry-run mode.

This script mounts ``NanoLMCacheConnectorV1`` into a real vLLM 0.19.0 run but
keeps ``enable_external_matching=False`` so correctness is not affected. The
goal is to verify that:

1. vLLM successfully imports and constructs the connector
2. scheduler-side candidate prefix matches are computed from nano-lmcache
3. connector stats/logs show those candidate hits during real generation
"""

from __future__ import annotations

import argparse
import os

import torch

from nano_lmcache import NanoLMCache, NanoLMCacheConfig, SegmentConfig, StorageConfig
from nano_lmcache.integration import (
    build_vllm_kv_transfer_config,
    register_nano_lmcache_engine,
    unregister_nano_lmcache_engine,
)

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are a careful assistant.

Follow these rules:
- answer precisely
- avoid speculation
- preserve formatting
- explain tradeoffs clearly
"""


def count_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def build_target_prompt(tokenizer, base_prompt: str, target_tokens: int) -> str:
    prompt = base_prompt.strip()
    filler = "\nPolicy note: be accurate, concise, and explicit about uncertainty."
    while count_tokens(tokenizer, prompt) < target_tokens:
        prompt += filler
    return prompt


def format_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_dummy_kv(seq_len: int) -> torch.Tensor:
    return torch.zeros(1, 2, seq_len, 1, 1, dtype=torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description="nano-lmcache vLLM connector dry-run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-system-prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--registry-key", default="demo-engine")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "0")

    nano_cache = NanoLMCache(
        NanoLMCacheConfig(
            storage=StorageConfig(
                gpu_size_gb=0.0,
                cpu_size_gb=0.01,
                disk_size_gb=0.1,
                disk_cache_dir="/tmp/nano_lmcache_vllm_connector_demo",
            ),
            segment=SegmentConfig(
                max_segment_length=64,
                min_segment_length=16,
            ),
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            enable_async_write=False,
            enable_logging=False,
        )
    )

    register_nano_lmcache_engine(args.registry_key, nano_cache)
    llm = None
    try:
        kv_transfer_config = build_vllm_kv_transfer_config(
            registry_key=args.registry_key,
            enable_external_matching=False,
            enable_probe_logging=True,
        )

        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=False,
            kv_transfer_config=kv_transfer_config,
        )

        tokenizer = llm.get_tokenizer()
        system_prompt = build_target_prompt(
            tokenizer,
            SYSTEM_PROMPT,
            args.target_system_prompt_tokens,
        )

        warm_prompt = format_prompt(
            system_prompt,
            "Summarize why prefix caching reduces TTFT.",
        )
        cold_prompt = format_prompt(
            system_prompt,
            "Write one short sentence about apples.",
        )

        warm_token_ids = tokenizer(warm_prompt, add_special_tokens=False)["input_ids"]
        nano_cache.store(warm_token_ids, make_dummy_kv(len(warm_token_ids)), async_write=False)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_tokens,
        )

        print("=" * 60)
        print("Connector dry-run: cold request")
        print("=" * 60)
        llm.generate([cold_prompt], sampling_params)
        llm.llm_engine.do_log_stats()

        print("=" * 60)
        print("Connector dry-run: warm request seeded in nano-lmcache")
        print("=" * 60)
        llm.generate([warm_prompt], sampling_params)
        llm.llm_engine.do_log_stats()

        print("=" * 60)
        print("Expected log signals")
        print("=" * 60)
        print("- 'nano-lmcache connector match ... candidate_external_tokens=...'")
        print("- 'nano-lmcache connector stats={...}'")
        print("- 'KV Transfer metrics: nano_lmcache_candidate_tokens=...'")
        print("Actual external tokens should stay at 0 in dry-run mode.")
    finally:
        unregister_nano_lmcache_engine(args.registry_key)
        if llm is not None:
            del llm
        nano_cache.shutdown()


if __name__ == "__main__":
    main()
