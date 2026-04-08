"""Storage backends for nano-lmcache."""

from .base import StorageBackend
from .cpu import CPUStorageBackend
from .disk import DiskStorageBackend
from .manager import TieredStorageManager

# GPU backend is optional (requires CUDA)
try:
    from .gpu import GPUStorageBackend
    _HAS_GPU = True
except (ImportError, RuntimeError):
    GPUStorageBackend = None
    _HAS_GPU = False

__all__ = [
    "StorageBackend",
    "CPUStorageBackend",
    "DiskStorageBackend",
    "GPUStorageBackend",
    "TieredStorageManager",
]
