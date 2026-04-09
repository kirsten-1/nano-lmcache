# vLLM Real Integration Plan

This document describes the path from the current benchmark-only setup to a real nano-lmcache + vLLM runtime integration.

## Current status

- `examples/vllm_integration_test.py` now measures vLLM prefix-caching with prompt lengths sized by tokenizer tokens instead of raw character count.
- `nano_lmcache.integration.VLLMConnector` supports direct tensor KV cache storage and retrieval.
- Native vLLM block-level KV cache conversion is not implemented yet.
- Runtime probes against vLLM `0.19.0` confirmed that current wins come from vLLM's native scheduler prefix cache path, not from any external connector path.
- `nano_lmcache.integration.NanoLMCacheVLLMAdapter` now mirrors the `KVConnectorBase_V1` scheduler lifecycle (`matched_tokens -> alloc -> connector_meta -> worker output`) without claiming KV block injection support.
- `nano_lmcache.integration.NanoLMCacheConnectorV1` now provides a scheduler-side connector core that can be loaded through vLLM's `kv_connector_module_path`.

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

## What the probe already proved

Probe output from the live runtime showed lines like:

- `local-prefix-hit ... prompt_tokens=2085 hit_tokens=2064`
- `schedule ... scheduled=21 computed=2085 cached=2064 external=0`

This means:

- native hits happen in `KVCacheManager.get_computed_blocks()`
- the scheduler only reuses full blocks
- the final token still gets recomputed for logits
- the external connector path is currently unused (`external=0`)

That evidence lets us stop guessing about scheduler semantics and build the
external connector against the real lifecycle vLLM expects.

### Phase 1: request-level hook

Goal: add a small patch in vLLM request handling that:

- computes a stable cache key from prefix tokens
- queries nano-lmcache before prefill
- records matched prefix length in request metadata

Suggested touchpoints in vLLM 0.19:

- request preprocessing / prompt tokenization path
- scheduler path where prefix caching decisions are already made:
  `vllm/v1/core/sched/scheduler.py`
- KV cache manager / block table path that knows slot mappings
  `vllm/v1/core/kv_cache_manager.py`
  `vllm/v1/worker/gpu/block_table.py`

### Current scheduler-side connector shape

The current integration path does not require editing vLLM's connector factory.
vLLM `0.19.0` can dynamically import a connector class when both of these are
set in `KVTransferConfig`:

- `kv_connector="NanoLMCacheConnectorV1"`
- `kv_connector_module_path="nano_lmcache.integration.vllm_v1_connector"`

The remaining scheduler-side state is passed by key, not by embedding a live
engine object in the config. The current helper API is:

```python
from nano_lmcache.integration import (
    build_vllm_kv_transfer_config,
    register_nano_lmcache_engine,
)

register_nano_lmcache_engine("demo-engine", nano_cache)
kv_transfer_config = build_vllm_kv_transfer_config(
    registry_key="demo-engine",
)
```

This keeps the config serializable for spawned workers while still letting the
scheduler-side connector resolve the local `NanoLMCache` engine.

By default, this helper keeps the connector in a dry-run-safe mode:

- candidate external matches are still computed from nano-lmcache
- worker-side connector stats can report those candidate hits
- but the scheduler does **not** yet skip token computation based on them

That default exists because worker-side KV block injection is not implemented
yet. To let the scheduler actually consume external matched tokens, the config
must explicitly opt in with:

```python
kv_transfer_config = build_vllm_kv_transfer_config(
    registry_key="demo-engine",
    enable_external_matching=True,
)
```

Do not enable that in production until worker-side KV load/store is wired up.

## Dry-run runtime validation

The repo now includes a dry-run runtime example:

- [vllm_connector_dry_run.py](/Users/apple/Documents/AI/nano-lmcache/examples/vllm_connector_dry_run.py)

It mounts `NanoLMCacheConnectorV1` into a real vLLM run, seeds nano-lmcache
with a prompt prefix, and keeps `enable_external_matching=False`.

That lets you validate three things safely:

1. vLLM can import and construct the connector through
   `kv_connector_module_path`
2. scheduler-side nano-lmcache candidate hits are computed for real requests
3. connector stats are emitted without changing inference behavior

Expected log signals include:

- `nano-lmcache connector match ... candidate_external_tokens=...`
- `nano-lmcache connector stats={...}`
- `KV Transfer metrics: nano_lmcache_candidate_tokens=...`

In this mode, `nano_lmcache_actual_external_tokens` should remain `0`.

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

1. Freeze against one exact runtime:
   - vLLM `0.19.0`
   - one model family
   - one attention backend
2. Replace the current adapter scaffold with a real `KVConnectorBase_V1`
   implementation for that exact configuration only.
3. Extend the runtime probe to hook the worker-side attention preparation path
   for the active runner class only, instead of assuming a fixed method exists.
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
