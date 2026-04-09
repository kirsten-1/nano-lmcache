"""Integration modules for nano-lmcache.

This package uses lazy imports so scheduler-side integration helpers can be
imported without eagerly importing torch-dependent runtime connectors.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "ConnectorMeta",
    "ConnectorRequestMetadata",
    "ExternalMatchResult",
    "NanoLMCacheConnectorMetadata",
    "NanoLMCacheConnectorRequestMetadata",
    "NanoLMCacheConnectorStats",
    "NanoLMCacheConnectorV1",
    "NanoLMCacheIndexMatcher",
    "NanoLMCacheVLLMAdapter",
    "NanoLMCacheVLLMConfig",
    "NanoLMCacheVLLMConnectorCore",
    "build_vllm_kv_transfer_config",
    "register_nano_lmcache_engine",
    "SchedulerPendingRequest",
    "unregister_nano_lmcache_engine",
    "VLLMConnector",
    "WorkerConnectorOutput",
]


_SYMBOL_TO_MODULE = {
    "ConnectorMeta": ".vllm_v1_adapter",
    "ConnectorRequestMetadata": ".vllm_v1_adapter",
    "ExternalMatchResult": ".vllm_v1_adapter",
    "NanoLMCacheConnectorMetadata": ".vllm_v1_connector",
    "NanoLMCacheConnectorRequestMetadata": ".vllm_v1_connector",
    "NanoLMCacheConnectorStats": ".vllm_v1_connector",
    "NanoLMCacheConnectorV1": ".vllm_v1_connector",
    "NanoLMCacheIndexMatcher": ".vllm_v1_connector",
    "NanoLMCacheVLLMAdapter": ".vllm_v1_adapter",
    "NanoLMCacheVLLMConfig": ".vllm_connector",
    "NanoLMCacheVLLMConnectorCore": ".vllm_v1_connector",
    "build_vllm_kv_transfer_config": ".vllm_v1_connector",
    "register_nano_lmcache_engine": ".vllm_v1_connector",
    "SchedulerPendingRequest": ".vllm_v1_adapter",
    "unregister_nano_lmcache_engine": ".vllm_v1_connector",
    "VLLMConnector": ".vllm_connector",
    "WorkerConnectorOutput": ".vllm_v1_adapter",
}


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
