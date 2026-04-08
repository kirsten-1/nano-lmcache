"""Disk storage backend."""

from typing import Dict, Optional, List
from pathlib import Path
import torch
import shutil
from .base import StorageBackend


class DiskStorageBackend(StorageBackend):
    """
    Disk storage backend.

    Features:
    - Large capacity (limited by disk space)
    - Persistent across restarts
    - Slower than CPU memory but much larger
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int,
    ):
        """
        Initialize disk storage.

        Args:
            cache_dir: Directory for storing cache files
            max_size_bytes: Maximum storage capacity in bytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_bytes

        # Index: key -> file_size
        self._index: Dict[str, int] = {}
        self._load_existing()

    def _load_existing(self):
        """Load index of existing cache files."""
        for f in self.cache_dir.glob("*.pt"):
            try:
                self._index[f.stem] = f.stat().st_size
            except OSError:
                pass

    def _get_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.pt"

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Store a tensor to disk."""
        # Estimate size
        estimated_size = tensor.numel() * tensor.element_size()

        # Check space
        if self.size_bytes() + estimated_size > self.max_size:
            return False

        # Remove existing if present
        if key in self._index:
            self.delete(key)

        path = self._get_path(key)

        try:
            # Save tensor (always save as CPU tensor)
            torch.save(tensor.cpu(), path)
            self._index[key] = path.stat().st_size
            return True
        except (OSError, RuntimeError) as e:
            # Handle disk errors
            print(f"Disk write error for {key}: {e}")
            return False

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Load a tensor from disk."""
        if key not in self._index:
            return None

        path = self._get_path(key)

        if not path.exists():
            # File was deleted externally
            self._index.pop(key, None)
            return None

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as e:
            print(f"Disk read error for {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a tensor from disk."""
        if key not in self._index:
            return False

        path = self._get_path(key)

        try:
            if path.exists():
                path.unlink()
            self._index.pop(key, None)
            return True
        except OSError as e:
            print(f"Disk delete error for {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._index and self._get_path(key).exists()

    def size_bytes(self) -> int:
        """Get current used space."""
        return sum(self._index.values())

    def capacity_bytes(self) -> int:
        """Get total capacity."""
        return self.max_size

    def keys(self) -> List[str]:
        """Get all stored keys."""
        return list(self._index.keys())

    def clear(self):
        """Clear all cached files."""
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fall back to individual deletes
            for key in list(self._index.keys()):
                self.delete(key)

        self._index.clear()
