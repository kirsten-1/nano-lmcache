"""
nano-lmcache: A minimal, educational implementation of LMCache.

The package root uses lazy imports so lightweight modules such as the radix
tree index and scheduler-side integration helpers remain usable even when the
runtime environment cannot import torch.
"""

from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"
__all__ = [
    "EvictionPolicy",
    "MatchResult",
    "NanoLMCache",
    "NanoLMCacheConfig",
    "Segment",
    "SegmentConfig",
    "SegmentRadixTree",
    "SegmentSplitter",
    "StorageConfig",
    "StorageTier",
]


_SYMBOL_TO_MODULE = {
    "EvictionPolicy": ".config",
    "MatchResult": ".index",
    "NanoLMCache": ".engine",
    "NanoLMCacheConfig": ".config",
    "Segment": ".segment",
    "SegmentConfig": ".config",
    "SegmentRadixTree": ".index",
    "SegmentSplitter": ".segment",
    "StorageConfig": ".config",
    "StorageTier": ".index",
}


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
