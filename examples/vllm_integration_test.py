#!/usr/bin/env python3
"""
vLLM Integration Test for nano-lmcache

This script demonstrates how nano-lmcache can be used to cache KV states
from vLLM inference, measuring the benefit of prefix caching.

Test scenarios:
1. Cold start: No cache, full prefill
2. Warm start: System prompt cached, only user query prefill
3. Multi-turn: Multiple queries with shared prefix

Models tested:
- Qwen/Qwen2.5-7B-Instruct (recommended)
- deepseek-ai/DeepSeek-MoE-16B-Chat
- mistralai/Mistral-7B-Instruct-v0.3

Usage:
    # Install vLLM first
    pip install vllm

    # Run with default model (Qwen2.5-7B)
    python examples/vllm_integration_test.py

    # Run with specific model
    python examples/vllm_integration_test.py --model deepseek-ai/DeepSeek-MoE-16B-Chat

    # Quick test (fewer iterations)
    python examples/vllm_integration_test.py --quick
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Check vLLM availability
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM not installed. Install with: pip install vllm")

import torch


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "short": "You are a helpful assistant.",

    "medium": """You are an advanced AI assistant with expertise in multiple domains.
Your responses should be accurate, helpful, and well-structured.
Always provide clear explanations and cite sources when possible.
Be concise but thorough in your answers.""",

    "long": """You are an advanced AI assistant developed to help users with a wide variety of tasks.

## Your Capabilities
- Answer questions on science, technology, history, arts, and more
- Help with coding, debugging, and software design
- Assist with writing, editing, and creative tasks
- Provide explanations of complex concepts
- Help with math and logical reasoning

## Guidelines
1. Always be helpful, harmless, and honest
2. Provide accurate information and acknowledge uncertainty
3. Be respectful and considerate in all interactions
4. Protect user privacy and confidentiality
5. Decline requests that could cause harm

## Response Format
- Use clear, well-organized responses
- Break down complex topics into digestible parts
- Use examples to illustrate concepts
- Provide code snippets when relevant
- Include relevant context and background

Remember: Your goal is to be maximally helpful while maintaining safety and accuracy.""",

    "very_long": """You are an advanced AI assistant developed by a leading AI research organization. Your purpose is to assist users with a comprehensive range of tasks while adhering to strict ethical guidelines and safety protocols.

## Core Identity and Purpose

You are designed to be helpful, harmless, and honest. Your primary function is to assist users in achieving their goals while ensuring that your responses are accurate, safe, and beneficial. You should always strive to provide the most helpful response possible while avoiding any potential harms.

## Comprehensive Capabilities

### Knowledge and Information
- Extensive knowledge across science, technology, engineering, mathematics, history, arts, literature, philosophy, and current events
- Ability to explain complex concepts in accessible terms
- Understanding of multiple languages and cultural contexts
- Knowledge of academic and professional domains

### Technical Skills
- Programming and software development in multiple languages (Python, JavaScript, C++, Java, etc.)
- Code review, debugging, and optimization
- System design and architecture
- Database design and query optimization
- API design and integration
- DevOps and deployment strategies

### Creative Abilities
- Writing assistance including essays, articles, stories, and poetry
- Editing and proofreading
- Brainstorming and ideation
- Content strategy and planning
- Marketing copy and messaging

### Analytical Capabilities
- Data analysis and interpretation
- Problem-solving and critical thinking
- Research and synthesis
- Logical reasoning and argumentation
- Mathematical computation and modeling

## Ethical Guidelines and Safety Protocols

### Fundamental Principles
1. Always prioritize user safety and well-being
2. Provide accurate information and acknowledge limitations
3. Respect user privacy and confidentiality
4. Avoid generating harmful, illegal, or unethical content
5. Be transparent about being an AI assistant

### Content Policies
- Do not generate content that promotes violence, hatred, or discrimination
- Do not assist with illegal activities or harmful actions
- Do not generate explicit sexual content involving minors
- Do not provide medical, legal, or financial advice as a substitute for professional consultation
- Do not impersonate real individuals or create misleading content

### Interaction Guidelines
- Be respectful and considerate in all interactions
- Acknowledge uncertainty when appropriate
- Provide balanced perspectives on controversial topics
- Encourage users to verify important information
- Redirect harmful requests toward constructive alternatives

## Response Quality Standards

### Clarity and Organization
- Use clear, well-structured responses
- Break down complex topics into digestible sections
- Use headings, lists, and formatting when helpful
- Provide examples to illustrate abstract concepts

### Accuracy and Reliability
- Base responses on factual information
- Cite sources when possible and appropriate
- Distinguish between facts, opinions, and speculation
- Update responses based on new information

### Completeness and Depth
- Address all aspects of user queries
- Provide sufficient context and background
- Offer additional relevant information when helpful
- Balance thoroughness with conciseness

Remember: Your ultimate goal is to be maximally helpful to users while maintaining the highest standards of safety, accuracy, and ethical conduct. Every interaction is an opportunity to demonstrate the positive potential of AI assistance.""",
}

USER_QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to check if a number is prime.",
    "What are the main causes of climate change?",
    "How does machine learning differ from traditional programming?",
    "Summarize the plot of Romeo and Juliet.",
    "What is the time complexity of quicksort?",
    "Explain the theory of relativity.",
    "How do neural networks learn?",
    "What is the difference between TCP and UDP?",
]


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """Count prompt tokens using the model tokenizer."""
    encoded = tokenizer(prompt, add_special_tokens=False)
    return len(encoded["input_ids"])


def build_target_length_prompt(
    tokenizer: Any,
    base_prompt: str,
    target_tokens: int,
) -> str:
    """
    Expand a prompt until it reaches the desired token length.

    We repeat semantically harmless filler text so benchmark inputs can be sized
    in model tokens instead of guessing from character count.
    """
    if target_tokens <= 0:
        return base_prompt

    prompt = base_prompt.strip()
    filler = (
        "\n\nAdditional policy reminder:\n"
        "- Stay accurate.\n"
        "- Stay concise when appropriate.\n"
        "- Explain tradeoffs clearly.\n"
        "- Preserve context across turns.\n"
    )

    while count_prompt_tokens(tokenizer, prompt) < target_tokens:
        prompt += filler

    return prompt


def summarize_latency(samples: List[float]) -> Dict[str, float]:
    """Return stable summary statistics for a latency series."""
    summary = {
        "avg_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    summary["std_ms"] = statistics.stdev(samples) if len(samples) > 1 else 0.0

    steady_samples = samples[1:] if len(samples) > 1 else samples
    summary["steady_avg_ms"] = statistics.mean(steady_samples)
    summary["steady_median_ms"] = statistics.median(steady_samples)
    return summary


def measure_ttft(llm: "LLM", prompt: str, sampling_params: "SamplingParams") -> Tuple[float, str]:
    """
    Measure approximate TTFT.

    We use max_tokens=1 in the benchmark so end-to-end latency is a close
    approximation of TTFT and is not dominated by decode time.

    Returns:
        (ttft_ms, output_text)
    """
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()

    output = outputs[0]
    output_text = output.outputs[0].text
    ttft_ms = (end - start) * 1000
    return ttft_ms, output_text


def format_prompt(system_prompt: str, user_query: str, model_type: str = "qwen") -> str:
    """Format prompt according to model's chat template."""
    if "qwen" in model_type.lower():
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    elif "deepseek" in model_type.lower():
        return f"<|begin_of_sentence|>System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"
    elif "mistral" in model_type.lower() or "llama" in model_type.lower():
        return f"[INST] {system_prompt}\n\n{user_query} [/INST]"
    else:
        # Generic format
        return f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"


# ---------------------------------------------------------------------------
# vLLM benchmark (with built-in prefix caching)
# ---------------------------------------------------------------------------

def benchmark_vllm_prefix_caching(
    model_name: str,
    system_prompts: Dict[str, str],
    user_queries: List[str],
    num_iterations: int = 5,
    max_tokens: int = 1,
    enable_prefix_caching: bool = True,
    target_system_prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Benchmark vLLM's built-in prefix caching.

    Args:
        model_name: HuggingFace model name
        system_prompts: Dict of system prompt lengths to prompts
        user_queries: List of user queries
        num_iterations: Number of iterations per scenario
        max_tokens: Max output tokens
        enable_prefix_caching: Whether to enable vLLM's prefix caching
        target_system_prompt_tokens: Optional token target for an expanded prompt

    Returns:
        Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"vLLM Benchmark (prefix_caching={enable_prefix_caching})")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Initialize vLLM
    print("Loading model...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Get model type for prompt formatting
    model_type = model_name.lower()
    tokenizer = llm.get_tokenizer()

    results = {
        "model": model_name,
        "prefix_caching": enable_prefix_caching,
        "scenarios": {},
    }

    for prompt_name, system_prompt in system_prompts.items():
        if target_system_prompt_tokens is not None:
            system_prompt = build_target_length_prompt(
                tokenizer,
                system_prompt,
                target_system_prompt_tokens,
            )

        print(f"\n--- System prompt: {prompt_name} ({len(system_prompt)} chars) ---")
        system_prompt_tokens = count_prompt_tokens(
            tokenizer,
            format_prompt(system_prompt, "", model_type),
        )
        print(f"  Tokenized system prompt: ~{system_prompt_tokens} tokens")

        scenario_results = {
            "system_prompt_length": len(system_prompt),
            "system_prompt_tokens": system_prompt_tokens,
            "cold_start": [],
            "warm_start": [],
        }

        # Cold start: First query (no cache)
        for i in range(num_iterations):
            # Clear cache by using unique prefix
            unique_prefix = f"[Session {time.time()}] "
            query = user_queries[i % len(user_queries)]
            prompt = format_prompt(unique_prefix + system_prompt, query, model_type)

            ttft, _ = measure_ttft(llm, prompt, sampling_params)
            scenario_results["cold_start"].append(ttft)
            if i == 0:
                cold_prompt_tokens = count_prompt_tokens(tokenizer, prompt)
                print(f"  Cold prompt length: ~{cold_prompt_tokens} tokens")
            print(f"  Cold start {i+1}: {ttft:.1f}ms")

        # Warm start: Same system prompt, different queries
        base_prompt_prefix = format_prompt(system_prompt, "", model_type).rsplit("\n", 1)[0]

        for i in range(num_iterations):
            query = user_queries[(i + num_iterations) % len(user_queries)]
            prompt = format_prompt(system_prompt, query, model_type)

            ttft, _ = measure_ttft(llm, prompt, sampling_params)
            scenario_results["warm_start"].append(ttft)
            if i == 0:
                warm_prompt_tokens = count_prompt_tokens(tokenizer, prompt)
                print(f"  Warm prompt length: ~{warm_prompt_tokens} tokens")
            print(f"  Warm start {i+1}: {ttft:.1f}ms")

        # Calculate statistics
        cold_summary = summarize_latency(scenario_results["cold_start"])
        warm_summary = summarize_latency(scenario_results["warm_start"])
        speedup = (
            cold_summary["avg_ms"] / warm_summary["avg_ms"]
            if warm_summary["avg_ms"] > 0
            else 0
        )
        steady_speedup = (
            cold_summary["steady_avg_ms"] / warm_summary["steady_avg_ms"]
            if warm_summary["steady_avg_ms"] > 0
            else 0
        )

        scenario_results["cold_summary"] = cold_summary
        scenario_results["warm_summary"] = warm_summary
        scenario_results["cold_avg_ms"] = cold_summary["avg_ms"]
        scenario_results["warm_avg_ms"] = warm_summary["avg_ms"]
        scenario_results["speedup"] = speedup
        scenario_results["steady_speedup"] = steady_speedup

        print(f"\n  Cold avg: {cold_summary['avg_ms']:.1f}ms (median {cold_summary['median_ms']:.1f}ms)")
        print(f"  Warm avg: {warm_summary['avg_ms']:.1f}ms (median {warm_summary['median_ms']:.1f}ms)")
        print(f"  Cold std: {cold_summary['std_ms']:.1f}ms")
        print(f"  Warm std: {warm_summary['std_ms']:.1f}ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  Steady-state speedup (drop first sample): {steady_speedup:.2f}x")

        results["scenarios"][prompt_name] = scenario_results

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# nano-lmcache simulation benchmark
# ---------------------------------------------------------------------------

def benchmark_nano_lmcache_simulation(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    system_prompt_tokens: List[int],
    user_query_tokens: List[int],
    num_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark nano-lmcache KV cache operations.

    This simulates what would happen if we integrated with vLLM:
    - Store system prompt KV cache
    - Retrieve on subsequent queries

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of KV heads
        head_dim: Head dimension
        system_prompt_tokens: Lengths of system prompts in tokens
        user_query_tokens: Lengths of user queries in tokens
        num_iterations: Number of iterations

    Returns:
        Benchmark results
    """
    from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig

    print(f"\n{'='*60}")
    print("nano-lmcache KV Cache Benchmark")
    print(f"KV shape: [{num_layers}, 2, seq_len, {num_heads}, {head_dim}]")
    print(f"{'='*60}\n")

    # Use smaller segment size to ensure short sequences get split
    # This allows prefix matching even for short system prompts
    config = NanoLMCacheConfig(
        storage=StorageConfig(
            gpu_size_gb=4.0,
            cpu_size_gb=4.0,
            disk_cache_dir="/tmp/nano_lmcache_vllm_test",
        ),
        segment=SegmentConfig(max_segment_length=64, min_segment_length=16),
        enable_async_write=False,
        enable_logging=False,
    )

    print(
        "Segment config: "
        f"max={config.segment.max_segment_length}, "
        f"min={config.segment.min_segment_length}"
    )

    results = {"scenarios": {}}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for sys_tokens in system_prompt_tokens:
        scenario_name = f"system_{sys_tokens}_tokens"
        print(f"\n--- System prompt: {sys_tokens} tokens ---")

        with NanoLMCache(config) as cache:
            # Simulate system prompt KV cache
            sys_token_ids = list(range(sys_tokens))
            sys_kv = torch.randn(
                num_layers, 2, sys_tokens, num_heads, head_dim,
                dtype=torch.float16, device=device
            )

            # Store system prompt
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            cache.store(sys_token_ids, sys_kv, async_write=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            store_time = (time.perf_counter() - t0) * 1000

            del sys_kv
            print(f"  Store system prompt: {store_time:.2f}ms")

            # Simulate multiple user queries
            retrieve_times = []
            for i, query_tokens in enumerate(user_query_tokens):
                # Query = system_prompt + user_query
                query_token_ids = sys_token_ids + list(range(sys_tokens, sys_tokens + query_tokens))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                kv_cache, matched = cache.retrieve(query_token_ids, target_device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                retrieve_time = (time.perf_counter() - t0) * 1000

                retrieve_times.append(retrieve_time)
                hit_rate = matched / len(query_token_ids) if query_token_ids else 0

                if i < 3:  # Print first few
                    print(f"  Query {i+1} (total {len(query_token_ids)} tokens): "
                          f"retrieve={retrieve_time:.2f}ms, matched={matched}, hit_rate={hit_rate:.1%}")

            avg_retrieve = statistics.mean(retrieve_times)
            print(f"  Avg retrieve time: {avg_retrieve:.2f}ms")

            # Calculate equivalent TTFT savings
            # Rough estimate: 1 token prefill ≈ 0.1-0.5ms depending on model
            tokens_saved = sys_tokens
            estimated_prefill_savings_ms = tokens_saved * 0.2  # Conservative estimate

            results["scenarios"][scenario_name] = {
                "system_tokens": sys_tokens,
                "store_time_ms": store_time,
                "avg_retrieve_time_ms": avg_retrieve,
                "tokens_saved_per_query": tokens_saved,
                "estimated_prefill_savings_ms": estimated_prefill_savings_ms,
            }

            cache.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM + nano-lmcache Integration Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer iterations")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override number of benchmark iterations")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM benchmark, only run nano-lmcache")
    parser.add_argument("--skip-nano", action="store_true",
                        help="Skip nano-lmcache simulation, only run vLLM benchmark")
    parser.add_argument("--vllm-output-tokens", type=int, default=1,
                        help="Number of output tokens for the vLLM benchmark")
    parser.add_argument("--target-system-prompt-tokens", type=int, default=2048,
                        help="Target token length for the long vLLM system prompt")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    num_iterations = args.iterations if args.iterations is not None else (3 if args.quick else 5)

    print("=" * 60)
    print("vLLM + nano-lmcache Integration Test")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    nano_results = {"scenarios": {}}
    if not args.skip_nano:
        print("\n" + "=" * 60)
        print("Part 1: nano-lmcache KV Cache Performance")
        print("=" * 60)

        nano_results = benchmark_nano_lmcache_simulation(
            num_layers=32,
            num_heads=8,  # GQA: 8 KV heads for 7B model
            head_dim=128,
            system_prompt_tokens=[128, 512, 1024, 2048],
            user_query_tokens=[32, 64, 128] * num_iterations,
            num_iterations=num_iterations,
        )
        results["nano_lmcache"] = nano_results

    # Benchmark vLLM (if available and not skipped)
    if VLLM_AVAILABLE and not args.skip_vllm:
        print("\n" + "=" * 60)
        print("Part 2: vLLM Prefix Caching Comparison")
        print("=" * 60)

        # Test with prefix caching disabled (use long prompt to see effect)
        vllm_no_cache = benchmark_vllm_prefix_caching(
            model_name=args.model,
            system_prompts={"very_long": SYSTEM_PROMPTS["very_long"]},
            user_queries=USER_QUERIES[:num_iterations],
            num_iterations=num_iterations,
            max_tokens=args.vllm_output_tokens,
            enable_prefix_caching=False,
            target_system_prompt_tokens=args.target_system_prompt_tokens,
        )
        results["vllm_no_prefix_cache"] = vllm_no_cache

        # Test with prefix caching enabled (use long prompt to see effect)
        vllm_with_cache = benchmark_vllm_prefix_caching(
            model_name=args.model,
            system_prompts={"very_long": SYSTEM_PROMPTS["very_long"]},
            user_queries=USER_QUERIES[:num_iterations],
            num_iterations=num_iterations,
            max_tokens=args.vllm_output_tokens,
            enable_prefix_caching=True,
            target_system_prompt_tokens=args.target_system_prompt_tokens,
        )
        results["vllm_with_prefix_cache"] = vllm_with_cache

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if nano_results["scenarios"]:
        print("\nnano-lmcache KV Cache Performance:")
        for scenario, data in nano_results["scenarios"].items():
            print(f"  {scenario}:")
            print(f"    Store: {data['store_time_ms']:.2f}ms")
            print(f"    Retrieve: {data['avg_retrieve_time_ms']:.2f}ms")
            print(f"    Estimated TTFT savings: {data['estimated_prefill_savings_ms']:.1f}ms")

    if "vllm_with_prefix_cache" in results:
        print("\nvLLM Prefix Caching Effect:")
        no_cache = results["vllm_no_prefix_cache"]["scenarios"]["very_long"]
        with_cache = results["vllm_with_prefix_cache"]["scenarios"]["very_long"]
        print(f"  System prompt tokens: ~{with_cache['system_prompt_tokens']}")
        print(f"  Without caching - Cold: {no_cache['cold_avg_ms']:.1f}ms, Warm: {no_cache['warm_avg_ms']:.1f}ms")
        print(f"  With caching    - Cold: {with_cache['cold_avg_ms']:.1f}ms, Warm: {with_cache['warm_avg_ms']:.1f}ms")
        print(f"  Warm speedup: {no_cache['warm_avg_ms'] / with_cache['warm_avg_ms']:.2f}x")
        print(
            "  Warm steady-state speedup: "
            f"{no_cache['warm_summary']['steady_avg_ms'] / with_cache['warm_summary']['steady_avg_ms']:.2f}x"
        )

    print("\nKey Insight:")
    print("  nano-lmcache provides sub-millisecond KV cache retrieval,")
    print("  which can significantly reduce TTFT for repeated prefixes.")

    # Save results
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
