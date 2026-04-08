"""CPU memory storage backend."""

from typing import Dict, Optional, List
import torch
from .base import StorageBackend


class CPUStorageBackend(StorageBackend):
    """
    CPU memory storage backend.

    Features:
    - Fast access (RAM speed)
    - Optional pinned memory for faster GPU transfer
    - LRU-friendly design
    """

    def __init__(
        self,
        max_size_bytes: int,
        use_pinned: bool = True,
    ):
        """
        Initialize CPU storage.

        Args:
            max_size_bytes: Maximum storage capacity in bytes
            use_pinned: Use pinned (page-locked) memory for faster GPU transfer
        """
        self.max_size = max_size_bytes
        self.use_pinned = use_pinned

        self._cache: Dict[str, torch.Tensor] = {}
        self._sizes: Dict[str, int] = {}
        self._current_size = 0

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Store a tensor in CPU memory."""
        tensor_size = tensor.numel() * tensor.element_size()

        # Check space
        if self._current_size + tensor_size > self.max_size:
            return False

        # Remove existing if present
        if key in self._cache:
            self.delete(key)

        # Copy to CPU with optional pinned memory
        if self.use_pinned and tensor.is_cuda:
            try:
                cpu_tensor = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_tensor.copy_(tensor)
            except RuntimeError:
                # Fall back to regular copy if pinned memory fails
                cpu_tensor = tensor.cpu().clone()
        else:
            cpu_tensor = tensor.cpu().clone()

        self._cache[key] = cpu_tensor
        self._sizes[key] = tensor_size
        self._current_size += tensor_size

        return True

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a tensor from CPU memory."""
        return self._cache.get(key)

    def delete(self, key: str) -> bool:
        """Delete a tensor from CPU memory."""
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

    def clear(self):
        """Clear all cached tensors."""
        self._cache.clear()
        self._sizes.clear()
        self._current_size = 0
