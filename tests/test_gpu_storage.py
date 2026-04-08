"""Tests for GPU storage backend."""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def gpu_backend():
    """Create a GPU storage backend."""
    from nano_lmcache.storage.gpu import GPUStorageBackend
    backend = GPUStorageBackend(max_size_bytes=100 * 1024 * 1024)  # 100 MB
    yield backend
    backend.clear()


def make_tensor(size: int = 100, device: str = "cuda:0") -> torch.Tensor:
    """Create a test tensor."""
    return torch.randn(size, dtype=torch.float16, device=device)


class TestGPUStorageBackend:
    """Tests for GPU storage backend."""

    def test_put_get(self, gpu_backend):
        """Test basic put and get."""
        tensor = make_tensor(1000)
        assert gpu_backend.put("key1", tensor)

        retrieved = gpu_backend.get("key1")
        assert retrieved is not None
        assert retrieved.device.type == "cuda"
        assert torch.allclose(tensor, retrieved)

    def test_put_from_cpu(self, gpu_backend):
        """Test putting a CPU tensor."""
        cpu_tensor = torch.randn(1000, dtype=torch.float16)
        assert gpu_backend.put("key1", cpu_tensor)

        retrieved = gpu_backend.get("key1")
        assert retrieved is not None
        assert retrieved.device.type == "cuda"

    def test_capacity_limit(self, gpu_backend):
        """Test capacity limit enforcement."""
        # Try to store more than capacity
        huge_tensor = torch.randn(100 * 1024 * 1024, dtype=torch.float16, device="cuda:0")
        assert not gpu_backend.put("key1", huge_tensor)

    def test_delete(self, gpu_backend):
        """Test deletion."""
        tensor = make_tensor(1000)
        gpu_backend.put("key1", tensor)

        assert gpu_backend.exists("key1")
        assert gpu_backend.delete("key1")
        assert not gpu_backend.exists("key1")
        assert gpu_backend.get("key1") is None

    def test_get_to_cpu(self, gpu_backend):
        """Test getting tensor to CPU."""
        tensor = make_tensor(1000)
        gpu_backend.put("key1", tensor)

        cpu_tensor = gpu_backend.get_to_cpu("key1")
        assert cpu_tensor is not None
        assert cpu_tensor.device.type == "cpu"

    def test_size_tracking(self, gpu_backend):
        """Test size tracking."""
        assert gpu_backend.size_bytes() == 0

        tensor = make_tensor(1000)
        gpu_backend.put("key1", tensor)

        assert gpu_backend.size_bytes() > 0

        gpu_backend.delete("key1")
        assert gpu_backend.size_bytes() == 0

    def test_put_async(self, gpu_backend):
        """Test async put."""
        tensor = make_tensor(1000, device="cpu")
        assert gpu_backend.put_async("key1", tensor)

        gpu_backend.synchronize()

        retrieved = gpu_backend.get("key1")
        assert retrieved is not None

    def test_memory_stats(self, gpu_backend):
        """Test memory statistics."""
        tensor = make_tensor(1000)
        gpu_backend.put("key1", tensor)

        stats = gpu_backend.memory_stats()

        assert "cache_size_bytes" in stats
        assert "cuda_allocated" in stats
        assert stats["num_entries"] == 1

    def test_clear(self, gpu_backend):
        """Test clearing all data."""
        for i in range(5):
            gpu_backend.put(f"key{i}", make_tensor(100))

        assert len(gpu_backend.keys()) == 5

        gpu_backend.clear()

        assert len(gpu_backend.keys()) == 0
        assert gpu_backend.size_bytes() == 0


class TestGPUTieredStorage:
    """Test tiered storage with GPU."""

    def test_gpu_tier_available(self):
        """Test that GPU tier is available."""
        from nano_lmcache.storage.manager import TieredStorageManager
        from nano_lmcache.index import StorageTier

        manager = TieredStorageManager(
            gpu_size_gb=0.1,
            cpu_size_gb=0.1,
        )

        assert manager.has_gpu()
        assert manager.default_tier == StorageTier.GPU

    def test_put_to_gpu(self):
        """Test storing to GPU tier."""
        from nano_lmcache.storage.manager import TieredStorageManager
        from nano_lmcache.index import StorageTier

        manager = TieredStorageManager(gpu_size_gb=0.1)

        tensor = make_tensor(1000)
        success, tier = manager.put("key1", tensor, StorageTier.GPU)

        assert success
        assert tier == StorageTier.GPU
        assert manager.get_location("key1") == StorageTier.GPU

    def test_gpu_to_cpu_demotion(self):
        """Test demotion from GPU to CPU."""
        from nano_lmcache.storage.manager import TieredStorageManager
        from nano_lmcache.index import StorageTier

        manager = TieredStorageManager(
            gpu_size_gb=0.001,  # Very small
            cpu_size_gb=1.0,
        )

        # Fill GPU
        tensor = make_tensor(1000)
        manager.put("key1", tensor, StorageTier.GPU)

        # Demote
        assert manager.demote("key1")
        assert manager.get_location("key1") == StorageTier.CPU

    def test_cpu_to_gpu_promotion(self):
        """Test promotion from CPU to GPU."""
        from nano_lmcache.storage.manager import TieredStorageManager
        from nano_lmcache.index import StorageTier

        manager = TieredStorageManager(
            gpu_size_gb=1.0,
            cpu_size_gb=1.0,
        )

        # Store in CPU
        tensor = make_tensor(1000)
        manager.put("key1", tensor, StorageTier.CPU)
        assert manager.get_location("key1") == StorageTier.CPU

        # Promote to GPU
        assert manager.promote("key1", StorageTier.GPU)
        assert manager.get_location("key1") == StorageTier.GPU

    def test_get_with_device(self):
        """Test getting tensor to specific device."""
        from nano_lmcache.storage.manager import TieredStorageManager

        manager = TieredStorageManager(gpu_size_gb=0.1)

        tensor = make_tensor(1000)
        manager.put("key1", tensor)

        # Get to CPU
        cpu_tensor = manager.get("key1", target_device="cpu")
        assert cpu_tensor.device.type == "cpu"

        # Get to GPU
        gpu_tensor = manager.get("key1", target_device="cuda:0")
        assert gpu_tensor.device.type == "cuda"
