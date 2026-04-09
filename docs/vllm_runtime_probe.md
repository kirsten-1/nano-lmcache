# vLLM Runtime Probe

This probe logs the real runtime data path for prefix caching in vLLM 0.19 without modifying `site-packages`.

## What it hooks

It monkeypatches these runtime points:

- `vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks`
- `vllm.v1.core.sched.scheduler.Scheduler.schedule`
- `vllm.v1.worker.gpu.block_table.BlockTables.compute_slot_mappings`
- worker `GPUModelRunner.prepare_attn` if the active runner class exposes it

These correspond to:

- local prefix-cache hit detection
- scheduler accounting of cached / external tokens
- token-position to KV-slot mapping
- actual worker-side block table and slot mapping preparation

## How to run on the server

From the `nano-lmcache` repo:

```bash
cd ~/nano-lmcache
PYTHONPATH=/root/nano-lmcache:$PYTHONPATH \
NANO_LMCACHE_VLLM_PROBE=1 \
NANO_LMCACHE_VLLM_PROBE_LOG=/tmp/vllm_probe.log \
NANO_LMCACHE_VLLM_PROBE_MAX_LINES=200 \
python examples/vllm_integration_test.py \
  --model /root/autodl-tmp \
  --skip-nano \
  --iterations 2 \
  --target-system-prompt-tokens 2048 \
  --vllm-output-tokens 1
```

Then inspect:

```bash
tail -n 200 /tmp/vllm_probe.log
```

## What to look for

### 1. Local prefix hit

Lines like:

```text
[nano-vllm-probe] local-prefix-hit req=... hit_tokens=...
```

This tells you how many tokens the native vLLM prefix cache matched before scheduling.

For the current 2048-token prompt experiment, a typical warm hit looked like:

```text
[nano-vllm-probe] local-prefix-hit req=... prompt_tokens=2085 hit_tokens=2064
```

That is consistent with block-size `16` alignment and vLLM's rule that the
last token still needs recomputation for logits.

### 2. Scheduler accounting

Lines like:

```text
[nano-vllm-probe] schedule ... reqs=[{..., "cached": ..., "external": ...}]
```

Important fields:

- `computed`: total already-available tokens at that point
- `cached`: tokens counted as prefix-cached by the scheduler
- `external`: tokens that would come from an external KV connector

For the current probe, `external=0` while `cached=2064`, which shows the speedup
is coming from vLLM's native local prefix cache path rather than an external
connector.

### 3. Worker-side slot mapping

Lines like:

```text
[nano-vllm-probe] slot-mappings ...
[nano-vllm-probe] prepare-attn ...
```

These tell you:

- block-table tensor shape
- slot-mapping tensor shape
- sample slot ids
- request batch shape at the worker forward boundary

## Why this matters

For nano-lmcache real integration, we need a stable place to:

1. read prompt-token prefix hit decisions
2. map logical token positions to vLLM KV slots
3. inject external KV before forward
4. store new KV after prefill

This probe gives the concrete runtime evidence for step 1 and step 2.

## Current best candidate hook points

- Scheduler-side read path:
  - `vllm/v1/core/sched/scheduler.py`
  - around `KVCacheManager.get_computed_blocks(...)`
  - around `self.connector.get_num_new_matched_tokens(...)`

- Worker-side write / inject path:
  - `vllm/v1/worker/gpu/model_runner.py`
  - around `prepare_attn(...)`
  - around `self.kv_connector.pre_forward(scheduler_output)`

- Slot/block translation:
  - `vllm/v1/worker/gpu/block_table.py`
  - `compute_slot_mappings(...)`
