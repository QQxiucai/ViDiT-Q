# ViDiT-Q + TeaCache 后续工作计划

**日期:** 2026-06-12
**作者:** 基于对 ViDiT-Q 和 TeaCache 的全面代码审查与文档阅读

---

## 一、当前项目状态总结

### 1.1 已完成的里程碑

| 阶段 | 内容 | 状态 |
|:-----|:-----|:----:|
| 环境搭建 | WSL2 + RTX 5080 + CUDA 12.8 + torch 2.9.1 + xformers 0.0.33 | ✅ |
| FP16 推理 | OpenSora v1.2 144p 120f 5s 端到端推理 (9.3s) | ✅ |
| W8A8 软件量化 | calibration → PTQ → 量化推理 (96.8s) | ✅ |
| W4A8 软件量化 | mixed precision 量化推理 (86.5s) | ✅ |
| W8A8 CUDA Kernel | FWHT+hadK+quant 融合 kernel，5.6s 端到端 | ✅ |
| W4A8 CUDA Kernel | ViDiT-Q 预处理 + QServe W4A8 GEMM (部分层) | ✅ |
| Benchmark 框架 | 三模式自动化性能对比 (W8A8-HW 比 FP16 快 19%) | ✅ |
| Timestep 分析 | t_mlp 连续步余弦相似度 0.98，验证 TeaCache 可行性 | ✅ |

### 1.2 当前性能基线 (144p, 120f, 10 steps, RTX 5080)

| 模式 | Per-step | vs FP16 | 峰值显存 |
|:-----|:--------:|:-------:|:--------:|
| FP16 | 531ms | 1.00× | 2,604 MB |
| W8A8 软件仿真 | 963ms | 0.57× | 7,423 MB |
| **W8A8 CUDA Kernel** | **447ms** | **1.19×** | 3,665 MB |

### 1.3 关键技术债

| 项目 | 严重度 | 说明 |
|:-----|:------:|:-----|
| `viditq_fused.cu` 仅支持 K=1152 | 中 | hadK 和 FWHT stage 数硬编码 |
| W4A8 attn 层不可用 | 中 | G=128 + half2 打包限制 |
| xformers sm_120 兼容性 | 低 | 已通过 cutlass 后端修复 |
| 无自动化测试 | 高 | 所有验证均为手动 |

---

## 二、TeaCache 技术原理深度解析

### 2.1 论文核心思想

TeaCache (CVPR 2025 Highlight) 的核心假设：**在扩散模型的去噪过程中，timestep embedding 的变化是缓慢且可预测的，相邻 timestep 之间的模型输出高度相似，可以被缓存复用。**

关键洞察：
1. Timestep embedding 经过 `t_block` (SiLU + Linear) 映射后得到 `t_mlp`，它是所有 transformer block 的调制输入
2. `t_mlp` 控制了每个 block 的 shift/scale/gate 参数，因此 `t_mlp` 的相似度直接决定 block 输出的相似度
3. 通过监测 `t_mlp` 的变化量，可以判断当前 step 是否可以跳过完整计算

### 2.2 TeaCache 的两种缓存策略

**策略 A: 全局残差缓存 (teacache_forward)**

来自 `eval/teacache/experiments/opensora.py`，是 TeaCache 最简洁的实现：

```python
# 1. 计算当前 modulated_input
modulated_inp = t2i_modulate(norm1(x), shift_msa, scale_msa)

# 2. 比较与上一次的差异
if timestep == first or timestep == last:
    should_calc = True  # 首尾步必须计算
else:
    delta = rescale_func(|modulated_inp - prev_modulated_inp| / |prev_modulated_inp|)
    accumulated_distance += delta
    if accumulated_distance < threshold:
        should_calc = False  # 缓存命中，跳过计算
    else:
        should_calc = True
        accumulated_distance = 0

# 3. 执行或跳过
if not should_calc:
    x += previous_residual  # 复用上次的残差
else:
    # 完整前向传播
    origin_x = x.clone()
    for spatial_block, temporal_block in zip(...):
        x = spatial_block(x, y, t_mlp, ...)
        x = temporal_block(x, y, t_mlp, ...)
    previous_residual = x - origin_x  # 保存残差供后续复用
```

**策略 B: 金字塔注意力广播 (PAB - Pyramid Attention Broadcast)**

来自 `videosys/core/pab_mgr.py` 和 `videosys/models/transformers/open_sora_transformer_3d.py`，是更细粒度的缓存：

- **Spatial Broadcast**: 在 spatial attention 层缓存 attention 输出
- **Temporal Broadcast**: 在 temporal attention 层缓存 attention 输出
- **Cross Broadcast**: 在 cross-attention 层缓存输出
- **MLP Broadcast**: 在 MLP 层缓存输出（基于预定义的 skip 配置）

每个 broadcast 策略独立控制：
- `threshold`: timestep 范围（如 `[450, 930]`）
- `range`: 每隔 N 步完整计算一次
- MLP 还有 `skip_count` 和 per-block 配置

### 2.3 两种策略的对比

| 维度 | 全局残差缓存 | PAB 细粒度缓存 |
|:-----|:-----------|:-------------|
| 实现复杂度 | 低 (~50 行) | 高 (~300+ 行) |
| 缓存粒度 | 整个 transformer 输出 | per-block per-layer |
| 加速比 | 中等 (1.5-2×) | 高 (2-3×) |
| 质量风险 | 低 (首尾步保证完整) | 中 (需仔细调参) |
| 适合场景 | 快速验证 | 最终部署 |

### 2.4 已在 ViDiT-Q 上的验证 (Phase 4a)

来自 `tools/profiling/analyze_timestep.py` 的分析结果：

- **t_embed 连续步余弦相似度**: 0.925
- **t_mlp 连续步余弦相似度**: **0.980** ← 非常有利于缓存
- **Spatial blocks 比 temporal blocks 更稳定** — 前 20 个最可缓存的全部是 spatial
- **阈值 0.10 时**: 33/56 blocks (59%) 可缓存
- **推荐阈值区间**: 0.08~0.10

---

## 三、TeaCache 工程实现分析

### 3.1 TeaCache 官方代码架构

```
TeaCache-main/
├── videosys/                          # 核心库
│   ├── core/
│   │   ├── pab_mgr.py                 # PAB 管理器 (广播策略核心)
│   │   ├── engine.py                  # 多 GPU 引擎
│   │   └── pipeline.py                # Pipeline 基类
│   ├── models/transformers/
│   │   └── open_sora_transformer_3d.py # 带 PAB hooks 的 STDiT3
│   ├── pipelines/open_sora/
│   │   └── pipeline_open_sora.py      # OpenSora Pipeline
│   └── schedulers/
│       └── scheduling_rflow_open_sora.py # RFLOW Scheduler
├── eval/teacache/experiments/
│   └── opensora.py                    # TeaCache 全局残差缓存实现
└── TeaCache4*/                        # 各模型独立实现
```

### 3.2 STDiT3Block 中的 PAB hook 点

在 TeaCache 的 `STDiT3Block.forward()` 中，PAB 在三个位置插入缓存逻辑：

1. **Self-Attention 缓存** (行 177-213):
   ```python
   if enable_pab() and broadcast_attn:
       x_m_s = self.last_attn  # 复用缓存
   else:
       # 正常 attention 计算
       if enable_pab():
           self.last_attn = x_m_s  # 更新缓存
   ```

2. **Cross-Attention 缓存** (行 219-228):
   ```python
   if enable_pab() and broadcast_cross:
       x = x + self.last_cross  # 复用缓存
   else:
       # 正常 cross-attention 计算
   ```

3. **MLP 缓存** (行 230-268):
   ```python
   if enable_pab() and broadcast_mlp:
       x_m_s = get_mlp_output(...)  # 从全局 dict 获取缓存
   else:
       # 正常 MLP 计算
       if enable_pab() and broadcast_next:
           save_mlp_output(...)  # 保存到全局 dict
   ```

### 3.3 与 ViDiT-Q 的集成点分析

ViDiT-Q 的 `STDiT3BlockWithCudaKernel` (在 `quant_opensora_cuda.py`) 与原始 `STDiT3Block` 的关键差异：

| 维度 | 原始 STDiT3Block | ViDiT-Q STDiT3BlockWithCudaKernel |
|:-----|:----------------|:----------------------------------|
| Self-Attention | `Attention` | `AttentionWithCudaKernel` (W8A8 GEMM) |
| Cross-Attention | `MultiHeadCrossAttention` | `MultiHeadCrossAttentionWithCudaKernel` |
| MLP | `Mlp` (timm) | `MlpWithCudaKernel` (W8A8 GEMM) |
| LayerNorm | 标准 `LayerNorm` | `LayerNormGeneral` (含量化融合) |
| Gate+Residual | `gate * x + residual` | `fused_kernels.gate_residual_fuse` (CUDA 融合) |
| Forward 签名 | `(x, y, t, mask, x_mask, t0, T, S)` | 相同 |
| **新增参数** | — | `quant_params` (QuantParams) |
| **输出** | FP16/FP32 | FP16 (W8A8 GEMM 输出) |

**关键发现**: ViDiT-Q 的 block forward 签名与原始 block 几乎完全一致（多了 `quant_params`），且输出仍是 FP16，这意味着 TeaCache 的缓存逻辑可以直接作用在 ViDiT-Q block 上。

---

## 四、后续工作计划

### Phase 4b: FP16 路径 TeaCache 实现（预计 2-3 天）

#### 目标
在 ViDiT-Q 的 FP16 推理路径上实现 TeaCache 全局残差缓存，验证加速效果。

#### 4b.1 创建 TeaCache Wrapper（1 天）

**新建文件**: `examples/opensora1.2/teacache/teacache_wrapper.py`

```python
@dataclass
class TeaCacheConfig:
    rel_l1_thresh: float = 0.10       # L1 距离阈值
    coeff_rescale: List[float] = None  # 多项式 rescale 系数
    cache_start_step: int = 0
    cache_end_step: int = -1          # -1 = 最后一个步之前
    enable_mlp_cache: bool = False    # 是否启用 per-block MLP 缓存

class TeaCacheWrapper:
    """包装 STDiT3.forward()，实现全局残差缓存"""
    def __init__(self, model, config: TeaCacheConfig):
        self.model = model
        self.config = config
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0

    def forward(self, x, timestep, all_timesteps, y, mask=None, ...):
        # 与 teacache_forward 相同的逻辑
        pass
```

**关键设计决策**:
- 使用 Phase 4a 的分析数据确定初始阈值 (推荐 0.10)
- 复用 `eval/teacache/experiments/opensora.py` 中的 rescale 系数拟合方法
- 对 ViDiT-Q 的所有 block 都走原始 forward 路径（FP16 模式不涉及量化）

#### 4b.2 集成到推理脚本（0.5 天）

修改 `examples/opensora1.2/fp_inference.py`（或新建 `fp_inference_teacache.py`）：

```python
# 新增参数
parser.add_argument("--teacache", action="store_true")
parser.add_argument("--teacache-thresh", type=float, default=0.10)

# 在 model 构建后 wrap
if args.teacache:
    from teacache.teacache_wrapper import TeaCacheWrapper, TeaCacheConfig
    config = TeaCacheConfig(rel_l1_thresh=args.teacache_thresh)
    model = TeaCacheWrapper(model, config)
```

#### 4b.3 验证与调试（0.5 天）

- 用 `local_144p_120f_5s.py` 配置验证 FP16 + TeaCache 推理
- 对比 FP16 和 FP16+TeaCache 的输出视频质量
- 测量加速比
- 预期结果: 1.3-1.8× 加速，质量损失 < 1dB PSNR

#### 4b.4 Rescale 系数校准（可选，0.5 天）

如果默认系数不适用于当前模型，编写校准脚本：
- 运行一次完整推理并记录每步的 modulated_input
- 用 numpy 拟合 5 阶多项式
- 更新 `TeaCacheConfig.coeff_rescale`

---

### Phase 4c: W8A8 量化路径 TeaCache 适配（预计 2-3 天）

#### 目标
将 TeaCache 缓存机制集成到 ViDiT-Q 的 W8A8 CUDA Kernel 推理路径。

#### 4c.1 分析 W8A8 Block 的缓存兼容性（0.5 天）

**关键问题**: `STDiT3BlockWithCudaKernel.forward()` 的输出是否适合作为缓存？

- **Self-Attention 输出**: 经过 `gate_residual_fuse` 融合，是 FP16 张量 → 可直接缓存
- **Cross-Attention 输出**: 经过 `quant_sum` 量化预处理 + `W8A8OF16LinearDynamicInputScale` → 输出 FP16 → 可直接缓存
- **MLP 输出**: 经过 `LayerNormGeneral` + `MlpWithCudaKernel` + `gate_residual_fuse` → 输出 FP16 → 可直接缓存

**结论**: 所有 W8A8 block 输出都是 FP16，可以直接缓存复用。

**但需注意**: W8A8 路径中 `quant_params` 的 `scale_input` 和 `sum_input` buffer 在缓存命中时不会被消耗（无需更新），这简化了实现。

#### 4c.2 修改 STDiT3BlockWithCudaKernel 添加缓存 hook（1 天）

在 `quant_opensora_cuda.py` 的 `STDiT3BlockWithCudaKernel` 中添加缓存逻辑：

```python
class STDiT3BlockWithCudaKernel(nn.Module):
    def __init__(self, ..., enable_teacache=False, teacache_config=None):
        ...
        self.enable_teacache = enable_teacache
        self.teacache_config = teacache_config
        # 缓存状态
        self.cached_attn_output = None
        self.cached_cross_attn_output = None
        self.cached_mlp_output = None
        self.attn_counter = 0
        self.cross_counter = 0
        self.mlp_counter = 0

    def forward(self, x, y, t, mask, x_mask, t0, T, S, 
                timestep=None, all_timesteps=None):  # 新增参数
        # 添加 broadcast 判断逻辑
        ...
```

这种方案将 PAB 风格的细粒度缓存直接嵌入到每个 block 中，与 TeaCache 官方实现一致。

#### 4c.3 修改 quant_inference.py 支持 TeaCache 开关（0.5 天）

```python
# 在 quant_inference.py 中添加
if cfg.get("enable_teacache", False):
    model.enable_teacache(cfg.teacache_config)
```

#### 4c.4 验证与调试（1 天）

- 在 W8A8 CUDA Kernel 模式下启用 TeaCache
- 确保缓存命中时跳过 CUDA kernel 调用
- 对比 W8A8 和 W8A8+TeaCache 的输出质量和速度

---

### Phase 4d: Pareto 分析与参数调优（预计 1-2 天）

#### 目标
在速度-质量 trade-off 曲线上找到最优配置。

#### 4d.1 建立评估框架（0.5 天）

**新建文件**: `tools/profiling/teacache_sweep.py`

```python
# 扫描阈值范围
thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
for thresh in thresholds:
    # 运行 FP16+TeaCache 推理
    # 记录: 总耗时, per-step 耗时, 缓存命中率
    # 保存输出视频用于质量评估
```

#### 4d.2 质量评估（0.5 天）

对每个阈值的输出视频：
- 计算与 FP16 baseline 的 PSNR/SSIM/LPIPS
- 肉眼检查是否有闪烁/伪影
- 记录帧间一致性

#### 4d.3 Pareto 曲线绘制与推荐配置（0.5 天）

输出：
- 速度加速比 vs PSNR 曲线
- 不同阈值下的缓存命中率统计
- 推荐配置：速度优先 / 质量优先 / 平衡模式

---

### Phase 5a: 高分辨率扩展（预计 2-3 天）

#### 目标
在 240p 和 360p 分辨率下测试 TeaCache 效果。

#### 5a.1 创建高分辨率配置（0.5 天）

- `configs/local_240p_120f_5s.py`
- `configs/local_360p_120f_5s.py`

注意：360p 下 token 数量约为 144p 的 6.25 倍（~3,888K vs ~622K），TeaCache 的收益会更大。

#### 5a.2 高分辨率 FP16 + TeaCache 验证（1 天）

- 确保 FP16 baseline 在高分辨率下不 OOM（16GB VRAM）
- 测试 TeaCache 在高分辨率下的加速比

#### 5a.3 高分辨率 W8A8 + TeaCache 验证（1 天）

- 重新生成 calibration 数据（不同分辨率需要不同的 calibration）
- 测试 W8A8 CUDA Kernel + TeaCache 的组合效果

---

### Phase 5b: 细粒度 Block 级 PAB 策略（预计 3-4 天）

#### 目标
实现 TeaCache 官方的 PAB (Pyramid Attention Broadcast) 细粒度缓存策略，进一步提升加速比。

#### 5b.1 移植 PABManager（1 天）

从 TeaCache 的 `videosys/core/pab_mgr.py` 移植 PABManager 到 ViDiT-Q：
- 简化 PABConfig（移除多 GPU 相关逻辑）
- 适配 ViDiT-Q 的 block 命名约定

#### 5b.2 在 STDiT3BlockWithCudaKernel 中实现 PAB hooks（1.5 天）

基于 Phase 4a 的 per-block 分析结果：
- Spatial blocks S00-S27：相对 delta 0.066~0.133 → 较稳定，适合缓存
- Temporal blocks T00-T27：相对 delta 0.101~0.133 → 较敏感，谨慎缓存

推荐策略：
- Spatial blocks: broadcast_range=2 (每 2 步完整计算一次)
- Temporal blocks: broadcast_range=3 (每 3 步完整计算一次)
- Cross-attention: broadcast_range=4
- MLP: 仅在特定 timestep 范围 [676, 864] 缓存

#### 5b.3 验证与调优（1 天）

- 对比全局残差缓存 vs PAB 细粒度缓存的加速比
- 调整 per-block 参数

---

### Phase 6: 系统级优化（预计 2-3 天）

| 优化项 | 内容 | 预期收益 |
|:-------|:-----|:--------:|
| 6a. INT8 KV Cache | 量化 attention 的 K/V 到 INT8 | 显存节省 ~20% |
| 6b. VAE CPU Offload | VAE decode 时移到 CPU | 显存节省 ~500MB |
| 6c. Batch 推理 | 利用 16GB 显存做 batch>1 | 吞吐量提升 ~1.5× |
| 6d. CUDA Graph | 消除 kernel launch overhead | 延迟降低 ~10% |

---

### Phase 7: 工程质量（预计 3-5 天）

| 任务 | 内容 |
|:-----|:-----|
| 7a. 自动化测试 | 建立 smoke test: FP16/W8A8/W8A8+TeaCache 端到端推理 |
| 7b. 代码清理 | 移除调试代码、提取公共函数、统一命名 |
| 7c. 配置文件整理 | 统一管理所有 config，建立配置模板 |
| 7d. 文档完善 | 编写用户使用指南、API 文档 |

---

## 五、时间线排期

```
Week 1 (6/13-6/19):
  Day 1-2: Phase 4b.1-4b.2 - TeaCache wrapper + 集成
  Day 3:   Phase 4b.3-4b.4 - 验证调试 + rescale 校准
  Day 4-5: Phase 4c.1-4c.3 - W8A8 适配分析 + 修改 + 集成

Week 2 (6/20-6/26):
  Day 1-2: Phase 4c.4 - W8A8+TeaCache 验证调试
  Day 3-4: Phase 4d   - Pareto 分析与参数调优
  Day 5:   Phase 5a.1 - 高分辨率配置

Week 3 (6/27-7/3):
  Day 1-3: Phase 5a.2-5a.3 - 高分辨率验证
  Day 4-5: Phase 5b (开始) - PAB 细粒度缓存

Week 4 (7/4-7/10):
  Day 1-3: Phase 5b (完成) - PAB 验证调优
  Day 4-5: Phase 6   - 系统级优化

Week 5 (7/11-7/17):
  Day 1-5: Phase 7   - 工程质量与文档
```

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|:-----|:----:|:----:|:---------|
| TeaCache 在量化 block 上缓存质量差 | 中 | 高 | 先用 FP16 验证，再逐步适配 W8A8 |
| W8A8 block 的 gate_residual_fuse 与缓存冲突 | 低 | 中 | 缓存完整的 block 输出（含 residual），而不是 residual 增量 |
| 高分辨率 OOM | 中 | 中 | 优先测试 240p；360p 可能需要 VAE CPU offload |
| rescale 系数对 ViDiT-Q 不适用 | 低 | 低 | 使用校准脚本重新拟合 |
| xformers 兼容性问题复现 | 低 | 中 | 保持 `VIDITQ_XFORMERS_OP=cutlass` 环境变量 |

## 七、预期最终成果

| 模式 | 当前 Per-step | 目标 Per-step | 加速比 |
|:-----|:------------:|:------------:|:------:|
| FP16 | 531ms | 531ms | 1.00× (baseline) |
| **FP16 + TeaCache** | — | **~280ms** | **~1.9×** |
| W8A8 CUDA Kernel | 447ms | 447ms | 1.19× |
| **W8A8 + TeaCache** | — | **~220ms** | **~2.4×** |
| **W8A8 + PAB** | — | **~180ms** | **~2.9×** |

目标: 在 144p 120f 5s 视频生成中，将推理时间从 FP16 的 5.3s 降低到 ~2.2s（W8A8+TeaCache）或 ~1.8s（W8A8+PAB）。
