"""Tiered storage manager."""

from typing import Dict, List, Optional, Tuple
import torch
from ..index import StorageTier
from .cpu import CPUStorageBackend
from .disk import DiskStorageBackend


class TieredStorageManager:
    """
    Multi-tier storage manager.

    Manages data across GPU, CPU, and Disk tiers with automatic
    promotion/demotion based on access patterns.

    Tier hierarchy:
    - GPU (L0): Fastest, most limited (optional)
    - CPU (L1): Fast, limited capacity
    - Disk (L2): Slow, large capacity

    Strategy:
    - New data goes to highest available tier
    - When tier is full, demote LRU to lower tier
    - On access, optionally promote to higher tier
    """

    # Tier ordering from fastest to slowest
    TIER_ORDER = [StorageTier.GPU, StorageTier.CPU, StorageTier.DISK]

    def __init__(
        self,
        gpu_size_gb: float = 0.0,
        cpu_size_gb: float = 4.0,
        disk_cache_dir: str = "/tmp/nano_lmcache",
        disk_size_gb: float = 50.0,
        use_pinned: bool = True,
        gpu_device: str = "cuda:0",
    ):
        """
        Initialize tiered storage.

        Args:
            gpu_size_gb: GPU storage capacity in GB (0 to disable)
            cpu_size_gb: CPU storage capacity in GB
            disk_cache_dir: Directory for disk storage
            disk_size_gb: Disk storage capacity in GB
            use_pinned: Use pinned memory for CPU storage
            gpu_device: CUDA device for GPU storage
        """
        self.backends: Dict[StorageTier, any] = {}
        self._active_tiers: List[StorageTier] = []

        # Initialize GPU backend if requested and available
        if gpu_size_gb > 0:
            try:
                from .gpu import GPUStorageBackend
                self.backends[StorageTier.GPU] = GPUStorageBackend(
                    int(gpu_size_gb * 1e9),
                    device=gpu_device,
                )
                self._active_tiers.append(StorageTier.GPU)
            except (ImportError, RuntimeError) as e:
                print(f"GPU storage disabled: {e}")

        # CPU backend
        self.backends[StorageTier.CPU] = CPUStorageBackend(
            int(cpu_size_gb * 1e9),
            use_pinned=use_pinned,
        )
        self._active_tiers.append(StorageTier.CPU)

        # Disk backend
        self.backends[StorageTier.DISK] = DiskStorageBackend(
            disk_cache_dir,
            int(disk_size_gb * 1e9),
        )
        self._active_tiers.append(StorageTier.DISK)

        # Track location of each key
        self._locations: Dict[str, StorageTier] = {}

        # LRU order for each tier
        self._access_order: Dict[StorageTier, List[str]] = {
            tier: [] for tier in self._active_tiers
        }

    @property
    def default_tier(self) -> StorageTier:
        """Get the default (fastest available) tier."""
        return self._active_tiers[0]

    def has_gpu(self) -> bool:
        """Check if GPU tier is available."""
        return StorageTier.GPU in self.backends

    def put(
        self,
        key: str,
        tensor: torch.Tensor,
        tier: Optional[StorageTier] = None,
    ) -> Tuple[bool, StorageTier]:
        """
        Store a tensor.

        Args:
            key: Unique identifier
            tensor: Tensor to store
            tier: Preferred storage tier (default: fastest available)

        Returns:
            (success, actual_tier)
        """
        if tier is None:
            tier = self.default_tier

        # Ensure tier is available
        if tier not in self.backends:
            tier = self.default_tier

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
            target_device: Device to move tensor to ("cpu", "cuda:0", etc.)

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

        # Move to target device if needed
        if target_device == "cpu":
            if tensor.is_cuda:
                return tensor.cpu()
            return tensor
        else:
            return tensor.to(target_device)

    def promote(self, key: str, target_tier: Optional[StorageTier] = None) -> bool:
        """
        Promote a key to a higher tier.

        Args:
            key: Key to promote
            target_tier: Target tier (default: fastest available)

        Returns:
            True if promoted successfully
        """
        if key not in self._locations:
            return False

        if target_tier is None:
            target_tier = self.default_tier

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
        attempts = 0
        max_attempts = 10
        while not target_backend.put(key, tensor):
            if not self._evict_one(target_tier) or attempts >= max_attempts:
                return False
            attempts += 1

        # Remove from old tier
        self.backends[current_tier].delete(key)
        self._locations[key] = target_tier

        # Update access order
        if key in self._access_order.get(current_tier, []):
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
            if key in self._access_order.get(current_tier, []):
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
            if key in self._access_order.get(tier, []):
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
        if tier not in self._access_order or not self._access_order[tier]:
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
        if tier not in self._access_order:
            self._access_order[tier] = []
        order = self._access_order[tier]
        if key in order:
            order.remove(key)
        order.append(key)

    def _lower_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """Get the next lower tier."""
        try:
            idx = self._active_tiers.index(tier)
            return self._active_tiers[idx + 1] if idx + 1 < len(self._active_tiers) else None
        except ValueError:
            return None

    def _higher_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """Get the next higher tier."""
        try:
            idx = self._active_tiers.index(tier)
            return self._active_tiers[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def _tier_level(self, tier: StorageTier) -> int:
        """Get numeric level of a tier (lower = faster)."""
        try:
            return self._active_tiers.index(tier)
        except ValueError:
            return 999

    def stats(self) -> Dict:
        """Get storage statistics."""
        result = {}
        for tier in self._active_tiers:
            if tier in self.backends:
                backend_stats = self.backends[tier].stats()
                backend_stats["access_order_len"] = len(self._access_order.get(tier, []))
                result[tier.value] = backend_stats
        return result

    def clear(self):
        """Clear all storage."""
        for tier in self._active_tiers:
            if tier in self.backends:
                self.backends[tier].clear()
        self._locations.clear()
        for tier in self._access_order:
            self._access_order[tier].clear()

    def synchronize(self):
        """Synchronize all GPU operations."""
        if StorageTier.GPU in self.backends:
            self.backends[StorageTier.GPU].synchronize()
