"""Runtime probe hooks for vLLM internals.

Enable by adding this repo root to PYTHONPATH and setting:

    NANO_LMCACHE_VLLM_PROBE=1

This file is imported automatically by Python as ``sitecustomize``. That
allows the probe hooks to execute in child processes created with ``spawn``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any


_ENABLED = os.getenv("NANO_LMCACHE_VLLM_PROBE", "").lower() in {"1", "true", "yes"}
_MAX_LINES = int(os.getenv("NANO_LMCACHE_VLLM_PROBE_MAX_LINES", "200"))
_LOG_PATH = os.getenv("NANO_LMCACHE_VLLM_PROBE_LOG", "").strip()
_LOCK = threading.Lock()
_LINE_COUNT = 0


def _emit(message: str) -> None:
    global _LINE_COUNT
    if not _ENABLED:
        return
    with _LOCK:
        if _LINE_COUNT >= _MAX_LINES:
            return
        _LINE_COUNT += 1
        line = f"[nano-vllm-probe] {message}\n"
        if _LOG_PATH:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        else:
            sys.stderr.write(line)
            sys.stderr.flush()


def _short_list(values: Any, limit: int = 8) -> str:
    try:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, (list, tuple)):
            truncated = list(values[:limit])
            suffix = "..." if len(values) > limit else ""
            return f"{truncated}{suffix}"
    except Exception:
        pass
    return repr(values)


def _patch_vllm() -> None:
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.worker.gpu.block_table import BlockTables
    except Exception as exc:
        _emit(f"probe init failed to import vllm internals: {exc!r}")
        return

    gpu_model_runner_classes = []
    for module_name in (
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
    ):
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
            cls = getattr(module, "GPUModelRunner", None)
            if cls is not None:
                gpu_model_runner_classes.append(cls)
        except Exception:
            continue

    if getattr(Scheduler.schedule, "_nano_probe_wrapped", False):
        return

    original_get_computed_blocks = KVCacheManager.get_computed_blocks
    original_schedule = Scheduler.schedule
    original_compute_slot_mappings = BlockTables.compute_slot_mappings

    def wrapped_get_computed_blocks(self, request):
        blocks, num_tokens = original_get_computed_blocks(self, request)
        if num_tokens > 0:
            _emit(
                "local-prefix-hit "
                f"req={request.request_id} "
                f"prompt_tokens={request.num_tokens} "
                f"hit_tokens={num_tokens} "
                f"block_groups={len(blocks.blocks)} "
                f"block_ids={blocks.get_block_ids(allow_none=True)}"
            )
        return blocks, num_tokens

    def wrapped_schedule(self):
        started = time.perf_counter()
        output = original_schedule(self)
        elapsed_ms = (time.perf_counter() - started) * 1000

        scheduled_req_ids = list(output.num_scheduled_tokens.keys())
        if scheduled_req_ids:
            req_summaries = []
            for req_id in scheduled_req_ids[:6]:
                req = self.requests.get(req_id)
                if req is None:
                    continue
                req_summaries.append(
                    {
                        "req_id": req_id,
                        "scheduled": output.num_scheduled_tokens.get(req_id),
                        "computed": req.num_computed_tokens,
                        "cached": req.num_cached_tokens,
                        "external": req.num_external_computed_tokens,
                        "tokens": req.num_tokens,
                        "prompt_tokens": req.num_prompt_tokens,
                        "status": str(req.status),
                    }
                )
            _emit(
                "schedule "
                f"elapsed_ms={elapsed_ms:.2f} "
                f"scheduled_reqs={len(scheduled_req_ids)} "
                f"total_scheduled_tokens={output.total_num_scheduled_tokens} "
                f"common_prefix_blocks={output.num_common_prefix_blocks} "
                f"reqs={req_summaries}"
            )
        return output

    def wrapped_compute_slot_mappings(
        self, idx_mapping, query_start_loc, positions, num_tokens_padded
    ):
        slot_mappings = original_compute_slot_mappings(
            self,
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded,
        )
        preview = None
        try:
            preview = _short_list(
                slot_mappings[0, : min(12, slot_mappings.shape[1])].cpu()
            )
        except Exception:
            preview = "<unavailable>"
        _emit(
            "slot-mappings "
            f"shape={tuple(slot_mappings.shape)} "
            f"num_tokens_padded={num_tokens_padded} "
            f"idx_mapping={_short_list(idx_mapping.cpu())} "
            f"query_start_loc={_short_list(query_start_loc.cpu())} "
            f"positions={_short_list(positions[: min(12, positions.shape[0])].cpu())} "
            f"preview={preview}"
        )
        return slot_mappings

    wrapped_get_computed_blocks._nano_probe_wrapped = True
    wrapped_schedule._nano_probe_wrapped = True
    wrapped_compute_slot_mappings._nano_probe_wrapped = True

    KVCacheManager.get_computed_blocks = wrapped_get_computed_blocks
    Scheduler.schedule = wrapped_schedule
    BlockTables.compute_slot_mappings = wrapped_compute_slot_mappings

    for runner_cls in gpu_model_runner_classes:
        if getattr(runner_cls.prepare_attn, "_nano_probe_wrapped", False):
            continue

        original_prepare_attn = runner_cls.prepare_attn

        def wrapped_prepare_attn(self, input_batch, _orig=original_prepare_attn):
            block_tables, slot_mappings = _orig(self, input_batch)
            try:
                block_preview = []
                for table in block_tables[:1]:
                    block_preview.append(
                        _short_list(table[0, : min(8, table.shape[1])].cpu())
                    )
            except Exception:
                block_preview = ["<unavailable>"]
            try:
                slot_preview = _short_list(
                    slot_mappings[0, : min(12, slot_mappings.shape[1])].cpu()
                )
            except Exception:
                slot_preview = "<unavailable>"
            _emit(
                "prepare-attn "
                f"runner={runner_cls.__module__}.{runner_cls.__name__} "
                f"num_reqs={input_batch.num_reqs} "
                f"num_tokens={input_batch.num_tokens} "
                f"query_start_loc={_short_list(input_batch.query_start_loc.cpu())} "
                f"seq_lens={_short_list(input_batch.seq_lens.cpu())} "
                f"block_table_shapes={[tuple(t.shape) for t in block_tables]} "
                f"block_preview={block_preview} "
                f"slot_shape={tuple(slot_mappings.shape)} "
                f"slot_preview={slot_preview}"
            )
            return block_tables, slot_mappings

        wrapped_prepare_attn._nano_probe_wrapped = True
        runner_cls.prepare_attn = wrapped_prepare_attn

    _emit("vllm probe hooks installed")


if _ENABLED:
    _patch_vllm()
