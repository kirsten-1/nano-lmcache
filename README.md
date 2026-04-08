# nano-lmcache

一个精简的、教学用途的 [LMCache](https://github.com/LMCache/LMCache) 实现 - 用于 LLM 推理加速的 KV Cache 管理系统。

## 项目目标

从零开始实现 LMCache 的核心概念，深入理解：
- KV Cache 复用如何降低 TTFT (Time To First Token)
- 多层存储架构 (GPU → CPU → Disk)
- 基于 Token 的 Cache 索引与检索
- 异步 Cache 操作

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     NanoLMCacheEngine                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ TokenIndex  │  │ CacheManager │  │ MemoryAllocator  │   │
│  │ (hash→key)  │  │ (get/put)    │  │ (alloc/free)     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Storage Backends                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │   CPU    │ ←→ │   Disk   │ ←→ │  Remote  │             │
│  │  Memory  │    │  (File)  │    │ (Redis)  │             │
│  └──────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Token Index (`nano_lmcache/index/`)
将 token 序列映射到 cache key，使用 rolling hash 算法。
- `TokenHasher`: 计算 token 序列的 hash 值
- `TokenDatabase`: 维护 token → cache_key 的映射关系

### 2. Cache Manager (`nano_lmcache/cache/`)
Cache 操作的中央协调器。
- `CacheEngine`: 主入口，协调 store/retrieve 操作
- `CachePolicy`: 淘汰策略 (LRU, LFU)

### 3. Storage Backends (`nano_lmcache/storage/`)
可插拔的存储层，统一接口设计。
- `BaseBackend`: 抽象接口
- `CPUBackend`: 内存存储，使用 numpy/torch tensors
- `DiskBackend`: 基于文件的存储，支持 memory-mapping
- `RedisBackend`: (可选) 远程存储，用于分布式场景

### 4. Memory Management (`nano_lmcache/memory/`)
KV Tensor 的高效内存分配。
- `MemoryPool`: 预分配的 tensor pool
- `MemoryObj`: KV cache 数据的封装，包含 metadata

### 5. Serialization (`nano_lmcache/serde/`)
KV Cache 的高效序列化/反序列化。
- `Serializer`: Tensor → bytes
- `Deserializer`: bytes → Tensor

## 项目结构

```
nano-lmcache/
├── README.md
├── pyproject.toml
├── nano_lmcache/
│   ├── __init__.py
│   ├── config.py              # 配置 dataclasses
│   ├── engine.py              # 主 CacheEngine
│   │
│   ├── index/
│   │   ├── __init__.py
│   │   ├── hasher.py          # Token 序列 hashing
│   │   └── database.py        # Token → key 映射
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── manager.py         # Cache 协调
│   │   └── policy.py          # 淘汰策略 (LRU)
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py            # 抽象 backend 接口
│   │   ├── cpu.py             # CPU memory backend
│   │   ├── disk.py            # Disk backend
│   │   └── redis.py           # Redis backend (可选)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── allocator.py       # Memory pool 管理
│   │   └── obj.py             # MemoryObj 封装
│   │
│   └── serde/
│       ├── __init__.py
│       └── tensor.py          # Tensor 序列化
│
├── examples/
│   ├── basic_usage.py         # 简单的 store/retrieve demo
│   ├── multi_tier.py          # CPU + Disk 分层存储
│   └── benchmark.py           # 性能对比测试
│
└── tests/
    ├── test_hasher.py
    ├── test_storage.py
    ├── test_cache.py
    └── test_engine.py
```

## 实现计划

### Phase 1: 核心基础
- [ ] 项目初始化 (pyproject.toml, 基础结构)
- [ ] 配置系统 (`config.py`)
- [ ] Token hasher (`index/hasher.py`)
- [ ] 基础 token database (`index/database.py`)

### Phase 2: 存储层
- [ ] 抽象 backend 接口 (`storage/base.py`)
- [ ] CPU memory backend (`storage/cpu.py`)
- [ ] Disk backend (`storage/disk.py`)
- [ ] 基础序列化 (`serde/tensor.py`)

### Phase 3: Cache 管理
- [ ] Memory object 封装 (`memory/obj.py`)
- [ ] LRU 淘汰策略 (`cache/policy.py`)
- [ ] Cache manager (`cache/manager.py`)

### Phase 4: Engine 集成
- [ ] 主 CacheEngine (`engine.py`)
- [ ] Store/Retrieve API
- [ ] 异步操作支持

### Phase 5: 进阶特性
- [ ] 多层存储 (CPU → Disk 的 promotion/demotion)
- [ ] Redis backend 用于分布式 caching
- [ ] Prefix matching 实现部分 cache hit
- [ ] Metrics 和 observability

## 与 LMCache 的主要差异

| 方面 | LMCache | nano-lmcache |
|------|---------|--------------|
| 定位 | 生产级别 | 教学用途 |
| GPU 支持 | CUDA kernels, GDS | 仅 CPU (更简单) |
| Backends | NIXL, P2P, PD 等 | CPU, Disk, Redis |
| 集成 | vLLM, SGLang | 独立运行 |
| 多进程 | 复杂 IPC | 单进程 |
| 可观测性 | 完整 metrics | 基础 logging |

## 快速开始

```python
from nano_lmcache import NanoLMCache, Config

# 初始化 cache
config = Config(
    chunk_size=256,
    max_cache_size_gb=4.0,
    storage_backend="cpu",  # 或 "disk", "redis"
)
cache = NanoLMCache(config)

# 存储 KV cache
tokens = [1, 2, 3, 4, 5]  # Token IDs
kv_tensor = torch.randn(32, 2, 256, 128)  # [layers, 2, seq_len, head_dim]
cache.store(tokens, kv_tensor)

# 检索 KV cache
cached_kv, hit_length = cache.retrieve(tokens)
if cached_kv is not None:
    print(f"Cache hit! 检索到 {hit_length} 个 tokens")
```

## 核心概念解释

### 为什么需要 KV Cache？
在 Transformer 的自回归推理中，每生成一个新 token 都需要重新计算之前所有 token 的 Key 和 Value。KV Cache 将这些中间结果缓存起来，避免重复计算。

### 为什么需要 LMCache？
当多个请求有相同的 prefix（如系统提示词、RAG 检索的文档），这些 prefix 的 KV Cache 可以被复用。LMCache 将 KV Cache 存储到 CPU/Disk/远程存储，实现跨请求复用。

### Chunking 策略
将长序列分成固定大小的 chunks，每个 chunk 独立计算 hash 和存储。这样可以实现更细粒度的 cache 复用。

## 学习资源

- [LMCache Technical Report](https://lmcache.ai/tech_report.pdf)
- [CacheGen Paper (SIGCOMM 2024)](https://dl.acm.org/doi/10.1145/3651890.3672274)
- [CacheBlend Paper (EuroSys 2025)](https://doi.org/10.1145/3689031.3696098)

## 致谢

本项目灵感来源于 [LMCache](https://github.com/LMCache/LMCache)，感谢 LMCache 团队的原创设计。

## License

Apache-2.0
