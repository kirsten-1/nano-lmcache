# nano-lmcache

一个精简的、教学用途的 KV Cache 管理系统 - 独立设计，不依赖 LMCache 实现。

## 设计亮点

相比 LMCache 的固定 chunk 设计，我们采用了：

| 特性 | LMCache | nano-lmcache |
|------|---------|--------------|
| 切分策略 | 固定大小 chunk | **变长 Segment** (语义切分) |
| 索引结构 | Hash Table | **Radix Tree** (高效前缀匹配) |
| 复杂度 | 生产级 (多进程/CUDA) | 教学级 (简洁清晰) |
| 存储层 | GPU/CPU/Disk/Remote | CPU/Disk (可扩展) |

## 安装

```bash
cd nano-lmcache
pip install -e .
```

## 快速开始

```python
import torch
from nano_lmcache import NanoLMCache, NanoLMCacheConfig

# 初始化 cache
config = NanoLMCacheConfig()
cache = NanoLMCache(config)

# 存储 KV cache
tokens = [1, 2, 3, 4, 5]  # Token IDs
# Shape: [num_layers, 2, seq_len, num_heads, head_dim]
kv_tensor = torch.randn(32, 2, 5, 8, 128, dtype=torch.float16)
cache.store(tokens, kv_tensor)

# 检索 KV cache
cached_kv, hit_length = cache.retrieve(tokens)
if cached_kv is not None:
    print(f"Cache hit! 检索到 {hit_length} 个 tokens")

# 前缀匹配
new_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 前 5 个相同
cached_kv, hit_length = cache.retrieve(new_tokens)
print(f"Prefix match: {hit_length} tokens")  # 输出: 5 tokens
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       NanoLMCache                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌────────────────────────────┐    │
│  │ SegmentSplitter │────▶│   Segment-based RadixTree  │    │
│  │  (变长切分)      │     │   (高效前缀匹配索引)        │    │
│  └─────────────────┘     └────────────────────────────┘    │
│                                      │                      │
│                                      ▼                      │
│                    ┌─────────────────────────────────┐     │
│                    │    TieredStorageManager         │     │
│                    │  (分层存储 + LRU 淘汰)           │     │
│                    └─────────────────────────────────┘     │
│                           │                  │              │
│                           ▼                  ▼              │
│                    ┌───────────┐      ┌───────────┐        │
│                    │  L1: CPU  │ ───▶ │ L2: Disk  │        │
│                    │  (热数据)  │      │  (冷数据)  │        │
│                    └───────────┘      └───────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
nano-lmcache/
├── pyproject.toml
├── nano_lmcache/
│   ├── __init__.py
│   ├── config.py          # 配置 dataclasses
│   ├── engine.py          # 主引擎 NanoLMCache
│   ├── segment.py         # Segment 切分 + Rolling Hash
│   ├── index.py           # Radix Tree 索引
│   └── storage/
│       ├── base.py        # 存储抽象接口
│       ├── cpu.py         # CPU 内存存储
│       ├── disk.py        # 磁盘存储
│       └── manager.py     # 分层存储管理器
├── examples/
│   ├── basic_usage.py     # 基础用法示例
│   └── benchmark.py       # 性能测试
└── tests/
    ├── test_segment.py
    ├── test_index.py
    ├── test_storage.py
    └── test_engine.py
```

## 核心组件

### 1. SegmentSplitter (`segment.py`)

**变长 Segment 切分** - 不同于 LMCache 的固定 chunk：

```python
# LMCache: 固定 256 tokens 一个 chunk
# nano-lmcache: 根据语义边界自动切分

splitter = SegmentSplitter(
    max_length=512,
    min_length=32,
    boundary_tokens=[13, 198],  # 换行符等
)
segments = splitter.split(tokens)
```

### 2. SegmentRadixTree (`index.py`)

**Radix Tree 索引** - 高效前缀匹配：

```python
# O(n) 前缀匹配，n 是 segment 数量
tree = SegmentRadixTree()
tree.insert(segments, StorageTier.CPU)

result = tree.match_prefix(query_segments)
# result.matched_tokens: 匹配的 token 数
# result.matched_segments: 匹配的 segment 元数据
```

### 3. TieredStorageManager (`storage/manager.py`)

**分层存储** - 自动 promotion/demotion：

```python
manager = TieredStorageManager(
    cpu_size_gb=4.0,
    disk_size_gb=50.0,
)

# 存储 (优先 CPU，满了自动降级到 Disk)
manager.put(key, tensor)

# 读取 (自动从正确的层级加载)
tensor = manager.get(key)

# 手动提升到更快的层级
manager.promote(key, StorageTier.CPU)
```

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 运行示例

```bash
# 基础用法
python examples/basic_usage.py

# CPU 性能测试
python examples/benchmark.py

# GPU 性能测试 (需要 CUDA)
python examples/gpu_benchmark.py

# vLLM 集成示例
python examples/vllm_integration_example.py
```

## 配置选项

```python
from nano_lmcache import NanoLMCacheConfig, StorageConfig, SegmentConfig

config = NanoLMCacheConfig(
    storage=StorageConfig(
        cpu_size_gb=4.0,           # CPU 缓存大小
        disk_size_gb=50.0,         # 磁盘缓存大小
        disk_cache_dir="/tmp/cache",
        use_pinned_memory=True,    # 使用 pinned memory 加速传输
    ),
    segment=SegmentConfig(
        max_segment_length=512,    # 最大 segment 长度
        min_segment_length=32,     # 最小 segment 长度
    ),
    enable_async_write=True,       # 异步写入
    enable_prefetch=True,          # 预取支持
)
```

## 性能对比

后续我们会与 LMCache 进行对比测试，验证：
- 前缀匹配效率
- 变长 Segment 的复用率
- 存储层切换延迟

## 学习资源

- [教程目录](/Users/apple/Documents/AI/面试题收集/最新最全高质量面试题/My_LMCache/) - 从原理到实现的完整教程
- [LMCache Technical Report](https://lmcache.ai/tech_report.pdf)
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104)

## License

Apache-2.0
