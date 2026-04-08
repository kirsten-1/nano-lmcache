"""
vLLM Integration Example for nano-lmcache.

This script demonstrates how to use nano-lmcache with vLLM for KV cache reuse.

Requirements:
- vLLM >= 0.4.0
- CUDA-capable GPU

Usage:
    python examples/vllm_integration_example.py --model meta-llama/Llama-2-7b-hf
"""

import argparse
import time
from typing import List, Optional

import torch

# Check dependencies
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("WARNING: vLLM not installed. Install with: pip install vllm")

from nano_lmcache.integration import VLLMConnector, NanoLMCacheVLLMConfig


def simulate_kv_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Simulate a KV cache tensor."""
    return torch.randn(
        num_layers, 2, seq_len, num_kv_heads, head_dim,
        dtype=torch.float16, device=device,
    )


def demo_basic_usage():
    """Demonstrate basic VLLMConnector usage."""
    print("\n=== Basic VLLMConnector Usage ===")

    # Model configuration (Llama-2-7B)
    config = NanoLMCacheVLLMConfig(
        num_layers=32,
        num_kv_heads=32,  # Llama-2-7B has 32 KV heads
        head_dim=128,
        gpu_size_gb=1.0,
        cpu_size_gb=4.0,
        enable_gpu_cache=torch.cuda.is_available(),
    )

    with VLLMConnector(config) as connector:
        # Simulate a system prompt
        system_prompt_tokens = list(range(256))  # 256 tokens

        # Create KV cache for system prompt
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kv_cache = simulate_kv_cache(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            len(system_prompt_tokens),
            device=device,
        )

        # Store system prompt KV cache
        print(f"Storing system prompt: {len(system_prompt_tokens)} tokens")
        connector.store(system_prompt_tokens, kv_cache)

        # Simulate multiple user requests with same system prompt
        for i in range(3):
            # User request = system prompt + user query
            user_query = [1000 + i * 100 + j for j in range(64)]  # 64 tokens
            full_request = system_prompt_tokens + user_query

            # Try to retrieve cached KV
            start = time.perf_counter()
            cached_kv, matched = connector.retrieve(full_request)
            elapsed = (time.perf_counter() - start) * 1000

            if matched > 0:
                print(f"  Request {i+1}: Cache HIT - {matched} tokens matched in {elapsed:.2f}ms")
            else:
                print(f"  Request {i+1}: Cache MISS in {elapsed:.2f}ms")

        # Print statistics
        stats = connector.stats()
        print(f"\nStatistics:")
        print(f"  Hit rate: {stats['connector']['hit_rate']:.1%}")
        print(f"  Total tokens stored: {stats['connector']['total_tokens_stored']}")
        print(f"  Total tokens hit: {stats['connector']['total_tokens_hit']}")


def demo_prefix_caching_scenario():
    """Demonstrate prefix caching in a realistic scenario."""
    print("\n=== Prefix Caching Scenario ===")

    config = NanoLMCacheVLLMConfig(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        gpu_size_gb=2.0,
        cpu_size_gb=4.0,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with VLLMConnector(config) as connector:
        # Scenario: RAG with multiple questions about the same document

        # 1. Store document context (1024 tokens)
        document_tokens = list(range(1024))
        document_kv = simulate_kv_cache(
            config.num_layers, config.num_kv_heads, config.head_dim,
            len(document_tokens), device,
        )
        connector.store(document_tokens, document_kv)
        print(f"Stored document context: {len(document_tokens)} tokens")

        # 2. Multiple questions about the document
        questions = [
            list(range(2000, 2064)),   # Question 1: 64 tokens
            list(range(3000, 3128)),   # Question 2: 128 tokens
            list(range(4000, 4096)),   # Question 3: 96 tokens
        ]

        print("\nProcessing questions:")
        total_saved = 0

        for i, question in enumerate(questions):
            full_query = document_tokens + question

            start = time.perf_counter()
            _, matched = connector.retrieve(full_query, target_device=device)
            elapsed = (time.perf_counter() - start) * 1000

            if matched > 0:
                # Calculate compute savings
                saved_compute = matched / len(full_query) * 100
                total_saved += matched
                print(f"  Q{i+1}: Matched {matched}/{len(full_query)} tokens "
                      f"({saved_compute:.1f}% compute saved) in {elapsed:.2f}ms")
            else:
                print(f"  Q{i+1}: No match")

        print(f"\nTotal tokens saved from recomputation: {total_saved}")


def demo_with_vllm():
    """Demonstrate integration with actual vLLM (requires vLLM installed)."""
    if not HAS_VLLM:
        print("\n=== vLLM Integration Demo (SKIPPED - vLLM not installed) ===")
        return

    print("\n=== vLLM Integration Demo ===")
    # This would be the actual vLLM integration
    # For now, we show the intended usage pattern

    print("""
    # Intended usage with vLLM:

    from vllm import LLM, SamplingParams
    from nano_lmcache.integration import VLLMConnector

    # Initialize vLLM
    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    # Initialize nano-lmcache connector
    connector = VLLMConnector(
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
    )

    # Before inference, check cache
    tokens = tokenizer.encode(prompt)
    cached_kv, matched_len = connector.retrieve(tokens)

    if matched_len > 0:
        # Use cached KV for prefix
        # (requires custom vLLM integration)
        pass

    # After inference, store KV cache
    # (requires access to vLLM's internal KV cache)
    connector.store(tokens, kv_cache)
    """)


def main():
    parser = argparse.ArgumentParser(description="vLLM Integration Example")
    parser.add_argument("--basic", action="store_true", help="Run basic demo")
    parser.add_argument("--prefix", action="store_true", help="Run prefix caching demo")
    parser.add_argument("--vllm", action="store_true", help="Run vLLM demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")

    args = parser.parse_args()

    if not any([args.basic, args.prefix, args.vllm, args.all]):
        args.all = True

    print("=" * 60)
    print("nano-lmcache vLLM Integration Examples")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available (running on CPU)")

    if args.all or args.basic:
        demo_basic_usage()

    if args.all or args.prefix:
        demo_prefix_caching_scenario()

    if args.all or args.vllm:
        demo_with_vllm()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
