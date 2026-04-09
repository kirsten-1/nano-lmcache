# vLLM Real Integration Plan

This document describes the path from the current benchmark-only setup to a real nano-lmcache + vLLM runtime integration.

## Current status

- `examples/vllm_integration_test.py` now measures vLLM prefix-caching with prompt lengths sized by tokenizer tokens instead of raw character count.
- `nano_lmcache.integration.VLLMConnector` supports direct tensor KV cache storage and retrieval.
- Native vLLM block-level KV cache conversion is not implemented yet.

## Why the current connector is not fully integrated

vLLM does not hand user code a simple `[layers, 2, seq, heads, head_dim]` KV tensor during normal inference. Its runtime stores KV in block-oriented layouts that depend on:

- vLLM version
- attention backend
- block size
- model architecture
- paged attention / scheduler internals

That means a real integration needs runtime hooks at the point where vLLM:

1. Knows the tokenized prefix
2. Can decide whether a cached prefix exists
3. Can inject matched KV blocks into the active request
4. Can export new KV blocks after prefill

## Practical implementation path

### Phase 1: request-level hook

Goal: add a small patch in vLLM request handling that:

- computes a stable cache key from prefix tokens
- queries nano-lmcache before prefill
- records matched prefix length in request metadata

Suggested touchpoints in vLLM 0.19:

- request preprocessing / prompt tokenization path
- scheduler path where prefix caching decisions are already made
- KV cache manager / block table path that knows slot mappings

### Phase 2: block export/import adapter

Goal: implement a version-specific adapter layer:

- `extract_blocks(request) -> canonical_tensor`
- `inject_blocks(request, canonical_tensor, matched_len)`

Adapter responsibilities:

- map vLLM slot indices back to logical token positions
- convert K/V block layout to canonical tensor layout
- copy canonical tensor back into vLLM-owned block buffers

### Phase 3: hybrid cache policy

Goal: decide when to use:

- vLLM native prefix caching only
- nano-lmcache only
- both together

Recommended policy:

- first try vLLM native in-GPU prefix caching for hot prefixes
- use nano-lmcache as a second-level cache for prefixes that were evicted from vLLM GPU memory
- promote hot hits from CPU/disk back into GPU-tier nano-lmcache storage

## Engineering tasks for the next iteration

1. Add a small vLLM-side probe patch that logs:
   - prompt token ids
   - native prefix cache matched length
   - slot mapping / block table metadata
2. Freeze against one exact runtime:
   - vLLM `0.19.0`
   - one model family
   - one attention backend
3. Implement one adapter for that exact configuration only.
4. Add an end-to-end benchmark comparing:
   - no cache
   - vLLM native prefix caching
   - nano-lmcache second-level cache

## Acceptance criteria

A real integration is only considered done when all of these are true:

- matched prefix KV is injected into vLLM without recomputing those prefix tokens
- outputs match the no-cache baseline
- latency improvement remains after excluding model load / compile / graph-capture overhead
- cache hit / miss behavior is observable in logs and tests
