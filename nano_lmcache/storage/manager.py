"""Tiered storage manager."""

from typing import Dict, List, Optional, Tuple
import torch
from ..index import StorageTier
from .cpu import CPUStorageBackend
from .disk import DiskStorageBackend


class TieredStorageManager:
    """
    Multi-tier storage manager.

    Manages data across CPU and Disk tiers with automatic
    promotion/demotion based on access patterns.

    Tier hierarchy:
    - CPU (L1): Fast, limited capacity
    - Disk (L2): Slow, large capacity

    Strategy:
    - New data goes to CPU
    - When CPU is full, demote LRU to Disk
    - On access, promote from Disk to CPU if space available
    """

    def __init__(
        self,
        cpu_size_gb: float = 4.0,
        disk_cache_dir: str = "/tmp/nano_lmcache",
        disk_size_gb: float = 50.0,
        use_pinned: bool = True,
    ):
        """
        Initialize tiered storage.

        Args:
            cpu_size_gb: CPU storage capacity in GB
            disk_cache_dir: Directory for disk storage
            disk_size_gb: Disk storage capacity in GB
            use_pinned: Use pinned memory for CPU storage
        """
        self.backends = {
            StorageTier.CPU: CPUStorageBackend(
                int(cpu_size_gb * 1e9),
                use_pinned=use_pinned,
            ),
            StorageTier.DISK: DiskStorageBackend(
                disk_cache_dir,
                int(disk_size_gb * 1e9),
            ),
        }

        # Track location of each key
        self._locations: Dict[str, StorageTier] = {}

        # LRU order for each tier
        self._access_order: Dict[StorageTier, List[str]] = {
            tier: [] for tier in [StorageTier.CPU, StorageTier.DISK]
        }

    def put(
        self,
        key: str,
        tensor: torch.Tensor,
        tier: StorageTier = StorageTier.CPU,
    ) -> Tuple[bool, StorageTier]:
        """
        Store a tensor.

        Args:
            key: Unique identifier
            tensor: Tensor to store
            tier: Preferred storage tier

        Returns:
            (success, actual_tier)
        """
        backend = self.backends.get(tier)
        if backend is None:
            # Fall back to CPU if tier not available
            tier = StorageTier.CPU
            backend = self.backends[tier]

        # Try to store in preferred tier
        if backend.put(key, tensor):
            self._locations[key] = tier
            self._update_access(tier, key)
            return True, tier

        # Try eviction
        if self._evict_one(tier):
            if backend.put(key, tensor):
                self._locations[key] = tier
                self._update_access(tier, key)
                return True, tier

        # Fall back to lower tier
        lower = self._lower_tier(tier)
        if lower:
            return self.put(key, tensor, lower)

        return False, tier

    def get(
        self,
        key: str,
        target_device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor.

        Args:
            key: Identifier
            target_device: Device to move tensor to

        Returns:
            Tensor on target device, or None if not found
        """
        if key not in self._locations:
            return None

        tier = self._locations[key]
        tensor = self.backends[tier].get(key)

        if tensor is None:
            # Data lost, clean up
            self._locations.pop(key, None)
            return None

        # Update access order
        self._update_access(tier, key)

        # Move to target device
        if target_device != "cpu":
            return tensor.to(target_device)
        return tensor

    def promote(self, key: str, target_tier: StorageTier = StorageTier.CPU) -> bool:
        """
        Promote a key to a higher tier.

        Args:
            key: Key to promote
            target_tier: Target tier (must be higher than current)

        Returns:
            True if promoted successfully
        """
        if key not in self._locations:
            return False

        current_tier = self._locations[key]

        # Already at target or higher
        if self._tier_level(current_tier) <= self._tier_level(target_tier):
            return True

        # Get data
        tensor = self.backends[current_tier].get(key)
        if tensor is None:
            return False

        # Try to store in target tier
        target_backend = self.backends.get(target_tier)
        if target_backend is None:
            return False

        # Make room if needed
        while not target_backend.put(key, tensor):
            if not self._evict_one(target_tier):
                return False

        # Remove from old tier
        self.backends[current_tier].delete(key)
        self._locations[key] = target_tier

        # Update access order
        if key in self._access_order[current_tier]:
            self._access_order[current_tier].remove(key)
        self._update_access(target_tier, key)

        return True

    def demote(self, key: str) -> bool:
        """
        Demote a key to a lower tier.

        Returns:
            True if demoted successfully
        """
        if key not in self._locations:
            return False

        current_tier = self._locations[key]
        lower = self._lower_tier(current_tier)

        if lower is None:
            return False  # Already at lowest tier

        tensor = self.backends[current_tier].get(key)
        if tensor is None:
            return False

        success, actual_tier = self.put(key, tensor, lower)
        if success:
            self.backends[current_tier].delete(key)
            if key in self._access_order[current_tier]:
                self._access_order[current_tier].remove(key)

        return success

    def delete(self, key: str) -> bool:
        """Delete a key from all tiers."""
        if key not in self._locations:
            return False

        tier = self._locations[key]
        success = self.backends[tier].delete(key)

        if success:
            self._locations.pop(key, None)
            if key in self._access_order[tier]:
                self._access_order[tier].remove(key)

        return success

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._locations

    def get_location(self, key: str) -> Optional[StorageTier]:
        """Get the storage tier of a key."""
        return self._locations.get(key)

    def _evict_one(self, tier: StorageTier) -> bool:
        """Evict one entry from a tier (LRU)."""
        if not self._access_order[tier]:
            return False

        # Get LRU key
        lru_key = self._access_order[tier][0]

        # Try to demote
        if self.demote(lru_key):
            return True

        # Can't demote, just delete
        return self.delete(lru_key)

    def _update_access(self, tier: StorageTier, key: str):
        """Update LRU access order."""
        order = self._access_order[tier]
        if key in order:
            order.remove(key)
        order.append(key)

    def _lower_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """Get the next lower tier."""
        levels = [StorageTier.CPU, StorageTier.DISK]
        try:
            idx = levels.index(tier)
            return levels[idx + 1] if idx + 1 < len(levels) else None
        except ValueError:
            return None

    def _tier_level(self, tier: StorageTier) -> int:
        """Get numeric level of a tier (lower = faster)."""
        levels = {StorageTier.CPU: 0, StorageTier.DISK: 1}
        return levels.get(tier, 999)

    def stats(self) -> Dict:
        """Get storage statistics."""
        return {
            tier.value: {
                **self.backends[tier].stats(),
                "access_order_len": len(self._access_order[tier]),
            }
            for tier in [StorageTier.CPU, StorageTier.DISK]
            if tier in self.backends
        }

    def clear(self):
        """Clear all storage."""
        for tier in [StorageTier.CPU, StorageTier.DISK]:
            if tier in self.backends:
                self.backends[tier].clear()
        self._locations.clear()
        for tier in self._access_order:
            self._access_order[tier].clear()
