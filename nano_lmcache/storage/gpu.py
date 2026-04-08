"""GPU memory storage backend."""

from typing import Dict, Optional, List
import torch
from .base import StorageBackend


class GPUStorageBackend(StorageBackend):
    """
    GPU memory storage backend.

    Features:
    - Fastest access (GPU memory bandwidth)
    - Limited capacity (GPU VRAM)
    - Supports CUDA streams for async operations
    """

    def __init__(
        self,
        max_size_bytes: int,
        device: str = "cuda:0",
        use_streams: bool = True,
    ):
        """
        Initialize GPU storage.

        Args:
            max_size_bytes: Maximum storage capacity in bytes
            device: CUDA device string (e.g., "cuda:0", "cuda:1")
            use_streams: Use CUDA streams for async copy
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot use GPUStorageBackend.")

        self.max_size = max_size_bytes
        self.device = torch.device(device)
        self.use_streams = use_streams

        self._cache: Dict[str, torch.Tensor] = {}
        self._sizes: Dict[str, int] = {}
        self._current_size = 0

        # CUDA stream for async operations
        if self.use_streams:
            self._stream = torch.cuda.Stream(device=self.device)
        else:
            self._stream = None

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Store a tensor in GPU memory."""
        tensor_size = tensor.numel() * tensor.element_size()

        # Check space
        if self._current_size + tensor_size > self.max_size:
            return False

        # Remove existing if present
        if key in self._cache:
            self.delete(key)

        # Copy to GPU
        if self.use_streams and self._stream is not None:
            with torch.cuda.stream(self._stream):
                gpu_tensor = tensor.to(self.device, non_blocking=True)
                self._stream.synchronize()
        else:
            gpu_tensor = tensor.to(self.device)

        self._cache[key] = gpu_tensor
        self._sizes[key] = tensor_size
        self._current_size += tensor_size

        return True

    def put_async(self, key: str, tensor: torch.Tensor) -> bool:
        """
        Store a tensor asynchronously (non-blocking).

        Caller should call synchronize() before using the data.
        """
        if not self.use_streams or self._stream is None:
            return self.put(key, tensor)

        tensor_size = tensor.numel() * tensor.element_size()

        if self._current_size + tensor_size > self.max_size:
            return False

        if key in self._cache:
            self.delete(key)

        with torch.cuda.stream(self._stream):
            gpu_tensor = tensor.to(self.device, non_blocking=True)

        self._cache[key] = gpu_tensor
        self._sizes[key] = tensor_size
        self._current_size += tensor_size

        return True

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a tensor from GPU memory."""
        return self._cache.get(key)

    def get_to_cpu(self, key: str, non_blocking: bool = False) -> Optional[torch.Tensor]:
        """Retrieve a tensor and copy to CPU."""
        tensor = self._cache.get(key)
        if tensor is None:
            return None

        if self.use_streams and self._stream is not None:
            with torch.cuda.stream(self._stream):
                cpu_tensor = tensor.cpu()
            if not non_blocking:
                self._stream.synchronize()
        else:
            cpu_tensor = tensor.cpu()

        return cpu_tensor

    def delete(self, key: str) -> bool:
        """Delete a tensor from GPU memory."""
        if key in self._cache:
            tensor = self._cache.pop(key)
            size = self._sizes.pop(key)
            self._current_size -= size
            del tensor
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache

    def size_bytes(self) -> int:
        """Get current used space."""
        return self._current_size

    def capacity_bytes(self) -> int:
        """Get total capacity."""
        return self.max_size

    def keys(self) -> List[str]:
        """Get all stored keys."""
        return list(self._cache.keys())

    def synchronize(self):
        """Synchronize CUDA stream."""
        if self._stream is not None:
            self._stream.synchronize()

    def clear(self):
        """Clear all cached tensors."""
        self._cache.clear()
        self._sizes.clear()
        self._current_size = 0
        torch.cuda.empty_cache()

    def memory_stats(self) -> Dict:
        """Get detailed GPU memory statistics."""
        return {
            "cache_size_bytes": self._current_size,
            "cache_capacity_bytes": self.max_size,
            "cache_utilization": self.utilization(),
            "num_entries": len(self._cache),
            "device": str(self.device),
            "cuda_allocated": torch.cuda.memory_allocated(self.device),
            "cuda_reserved": torch.cuda.memory_reserved(self.device),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(self.device),
        }
