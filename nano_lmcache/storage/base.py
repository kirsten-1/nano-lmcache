"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class StorageBackend(ABC):
    """
    Abstract interface for KV cache storage backends.

    All storage backends must implement these methods.
    """

    @abstractmethod
    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """
        Store a tensor.

        Args:
            key: Unique identifier for the tensor
            tensor: The tensor to store

        Returns:
            True if stored successfully, False if out of space
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor.

        Args:
            key: Identifier of the tensor

        Returns:
            The tensor if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a tensor.

        Args:
            key: Identifier of the tensor

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass

    @abstractmethod
    def size_bytes(self) -> int:
        """Get current used space in bytes."""
        pass

    @abstractmethod
    def capacity_bytes(self) -> int:
        """Get total capacity in bytes."""
        pass

    def available_bytes(self) -> int:
        """Get available space in bytes."""
        return self.capacity_bytes() - self.size_bytes()

    def utilization(self) -> float:
        """Get storage utilization (0.0 to 1.0)."""
        cap = self.capacity_bytes()
        return self.size_bytes() / cap if cap > 0 else 0.0

    def keys(self) -> list:
        """Get all stored keys."""
        return []

    def clear(self):
        """Clear all stored data."""
        for key in self.keys():
            self.delete(key)

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "size_bytes": self.size_bytes(),
            "capacity_bytes": self.capacity_bytes(),
            "available_bytes": self.available_bytes(),
            "utilization": self.utilization(),
            "num_entries": len(self.keys()),
        }
