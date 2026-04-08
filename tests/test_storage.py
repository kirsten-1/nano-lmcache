"""Tests for storage backends."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from nano_lmcache.storage.cpu import CPUStorageBackend
from nano_lmcache.storage.disk import DiskStorageBackend
from nano_lmcache.storage.manager import TieredStorageManager
from nano_lmcache.index import StorageTier


def make_tensor(size: int = 100) -> torch.Tensor:
    """Create a test tensor."""
    return torch.randn(size, dtype=torch.float32)


class TestCPUStorageBackend:
    """Tests for CPU storage backend."""

    def test_put_get(self):
        """Test basic put and get."""
        backend = CPUStorageBackend(max_size_bytes=1024 * 1024)

        tensor = make_tensor(100)
        assert backend.put("key1", tensor)

        retrieved = backend.get("key1")
        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)

    def test_capacity_limit(self):
        """Test capacity limit enforcement."""
        # Very small capacity
        backend = CPUStorageBackend(max_size_bytes=100)

        # This should fail (tensor is too large)
        tensor = make_tensor(1000)
        assert not backend.put("key1", tensor)

    def test_delete(self):
        """Test deletion."""
        backend = CPUStorageBackend(max_size_bytes=1024 * 1024)

        tensor = make_tensor(100)
        backend.put("key1", tensor)

        assert backend.exists("key1")
        assert backend.delete("key1")
        assert not backend.exists("key1")
        assert backend.get("key1") is None

    def test_overwrite(self):
        """Test overwriting existing key."""
        backend = CPUStorageBackend(max_size_bytes=1024 * 1024)

        tensor1 = make_tensor(100)
        tensor2 = make_tensor(100)

        backend.put("key1", tensor1)
        backend.put("key1", tensor2)

        retrieved = backend.get("key1")
        assert torch.allclose(tensor2, retrieved)

    def test_size_tracking(self):
        """Test size tracking."""
        backend = CPUStorageBackend(max_size_bytes=1024 * 1024)

        assert backend.size_bytes() == 0

        tensor = make_tensor(100)
        backend.put("key1", tensor)

        assert backend.size_bytes() > 0

        backend.delete("key1")
        assert backend.size_bytes() == 0

    def test_clear(self):
        """Test clearing all data."""
        backend = CPUStorageBackend(max_size_bytes=1024 * 1024)

        for i in range(5):
            backend.put(f"key{i}", make_tensor(100))

        assert len(backend.keys()) == 5

        backend.clear()

        assert len(backend.keys()) == 0
        assert backend.size_bytes() == 0


class TestDiskStorageBackend:
    """Tests for disk storage backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_put_get(self, temp_dir):
        """Test basic put and get."""
        backend = DiskStorageBackend(temp_dir, max_size_bytes=10 * 1024 * 1024)

        tensor = make_tensor(100)
        assert backend.put("key1", tensor)

        retrieved = backend.get("key1")
        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)

    def test_persistence(self, temp_dir):
        """Test that data persists across instances."""
        tensor = make_tensor(100)

        # First instance: write
        backend1 = DiskStorageBackend(temp_dir, max_size_bytes=10 * 1024 * 1024)
        backend1.put("key1", tensor)

        # Second instance: read
        backend2 = DiskStorageBackend(temp_dir, max_size_bytes=10 * 1024 * 1024)
        retrieved = backend2.get("key1")

        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)

    def test_delete(self, temp_dir):
        """Test deletion."""
        backend = DiskStorageBackend(temp_dir, max_size_bytes=10 * 1024 * 1024)

        tensor = make_tensor(100)
        backend.put("key1", tensor)

        assert backend.exists("key1")
        assert backend.delete("key1")
        assert not backend.exists("key1")

    def test_clear(self, temp_dir):
        """Test clearing all data."""
        backend = DiskStorageBackend(temp_dir, max_size_bytes=10 * 1024 * 1024)

        for i in range(5):
            backend.put(f"key{i}", make_tensor(100))

        assert len(backend.keys()) == 5

        backend.clear()

        assert len(backend.keys()) == 0


class TestTieredStorageManager:
    """Tests for tiered storage manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_put_get(self, temp_dir):
        """Test basic put and get."""
        manager = TieredStorageManager(
            cpu_size_gb=0.001,  # 1 MB
            disk_cache_dir=temp_dir,
            disk_size_gb=0.01,  # 10 MB
        )

        tensor = make_tensor(100)
        success, tier = manager.put("key1", tensor)

        assert success
        assert tier == StorageTier.CPU

        retrieved = manager.get("key1")
        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)

    def test_demotion(self, temp_dir):
        """Test automatic demotion when CPU is full."""
        manager = TieredStorageManager(
            cpu_size_gb=0.000001,  # ~1 KB
            disk_cache_dir=temp_dir,
            disk_size_gb=0.01,
        )

        # Fill CPU with a small tensor first
        tensor1 = make_tensor(100)
        manager.put("key1", tensor1)

        # This should exceed CPU capacity and force demotion/fallback to disk
        tensor2 = make_tensor(1000)
        success, tier = manager.put("key2", tensor2)

        # key2 might go to disk, or key1 might be demoted
        assert success
        # At least one should be on disk
        assert (
            manager.get_location("key1") == StorageTier.DISK
            or manager.get_location("key2") == StorageTier.DISK
        )

    def test_promote(self, temp_dir):
        """Test manual promotion."""
        manager = TieredStorageManager(
            cpu_size_gb=0.001,
            disk_cache_dir=temp_dir,
            disk_size_gb=0.01,
        )

        tensor = make_tensor(100)

        # Store directly to disk
        manager.put("key1", tensor, tier=StorageTier.DISK)
        assert manager.get_location("key1") == StorageTier.DISK

        # Promote to CPU
        success = manager.promote("key1", StorageTier.CPU)
        assert success
        assert manager.get_location("key1") == StorageTier.CPU

    def test_delete(self, temp_dir):
        """Test deletion."""
        manager = TieredStorageManager(
            cpu_size_gb=0.001,
            disk_cache_dir=temp_dir,
            disk_size_gb=0.01,
        )

        tensor = make_tensor(100)
        manager.put("key1", tensor)

        assert manager.exists("key1")
        assert manager.delete("key1")
        assert not manager.exists("key1")

    def test_stats(self, temp_dir):
        """Test stats method."""
        manager = TieredStorageManager(
            cpu_size_gb=0.001,
            disk_cache_dir=temp_dir,
            disk_size_gb=0.01,
        )

        tensor = make_tensor(100)
        manager.put("key1", tensor)

        stats = manager.stats()

        assert "cpu" in stats
        assert "disk" in stats
        assert stats["cpu"]["num_entries"] >= 0
