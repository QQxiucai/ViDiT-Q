# ViDiT-Q W8A8 硬件加速推理 — 阶段性工作总结

**日期:** 2026-06-10
**环境:** RTX 5080 Laptop GPU (Blackwell sm_120), WSL2 Ubuntu 22.04
**Conda 环境:** `viditq-xf033-test` (torch 2.9.1+cu128, xformers 0.0.33.post2)
**基准模型:** OpenSora v1.2 / STDiT3-XL/2, 120f 144p 5s 视频生成

---

## 一、项目目标

以 ViDiT-Q 作为量化框架，对 OpenSora v1.2 / STDiT3 模型做 W8A8 硬件量化推理加速，并在 RTX 5080 上实现端到端 CUDA Kernel 级别的性能优化。后续方向为集成 TeaCache 缓存优化。

---

## 二、阶段性成果总览

### 2.1 性能演进

| 阶段 | 推理路径 | Per step | vs FP16 | 10 步推算 |
|:-----|:---------|:--------:|:-------:|:--------:|
| 起点 | W8A8 软件仿真 (FP64 Hadamard) | 9656ms | 0.08× | ~97s |
| Phase 1 | W8A8 软件仿真 (FP16 Hadamard) | 1179ms | 0.61× | ~12s |
| Phase 1 | W8A8 CUDA Kernel (weight only) | 511ms | 1.42× | ~5.1s |
| **Phase 2 (V4)** | **W8A8 CUDA Kernel (full ViDiT-Q)** | **~560ms** | **1.30×** | **~5.6s** |

> 从 97s 到 5.6s，累计 **17.3× 端到端加速**。RTX 5080 上的 INT8 Tensor Core 推理比 FP16 快 30%。

### 2.2 质量验证

| 指标 | FP16 基线 | Phase 1 HW | **Phase 2 V4** |
|:------|:--------:|:----------:|:--------------:|
| Pixel mean | 174.39 | 171.85 | 180.25 |
| Pixel std | 77.52 | 75.42 | 70.81 |
| 帧间一致性 mean | 1.37 | 1.59 | **1.47** |
| 帧间一致性 std | 1.39 | 1.42 | **1.53** |
| vs FP16 abs diff | — | 14.95 | 18.15 |

Phase 2 V4 通过肉眼验证，画面质量良好，无方块伪影，闪烁在可接受范围。

---

## 三、Phase 1: FP64→FP16 Hadamard 旋转

### 3.1 问题定位

ViDiT-Q 的 `ViDiTQuantizedLinear.forward()` 中有 FP64 Hadamard 旋转：

```python
x = torch.matmul(x.double(), self.rotation_matrix).to(dtype=dtype_)
```

对 622K tokens × 1152 hidden，这个 FP64 矩阵乘法每次前向传播约 8.2×10¹¹ FLOPs。在 blocks 11-14（ViDiT-Q 层）的每个 Linear 层中都要执行，导致软件仿真中单个 MLP 层耗时 **874ms**（正常仅 4ms）。

### 3.2 解决方案

```python
# viditq_quant_layer.py:63 — 一行改动
# 原来: x = torch.matmul(x.double(), self.rotation_matrix).to(dtype=dtype_)
# 改为:
x = torch.matmul(x, self.rotation_matrix.to(dtype=dtype_))  # FP16
```

Hadamard 矩阵是正交且条件良好的，FP16 精度损失可忽略。

### 3.3 效果

- 软件仿真：9656ms → 1179ms（**8.2× 加速**）
- ViDiT-Q blocks：1080ms → 30ms（**36× 加速**）
- W8A8 CUDA Kernel：1389ms → 516ms（**2.7× 加速**）
- 质量：FP64 vs FP16 Hadamard 的 abs diff mean = 4.39 灰度级（仅量化误差的 40%）

---

## 四、Phase 2: ViDiT-Q CUDA Kernel 融合

### 4.1 架构设计

实现了完整的 ViDiT-Q 激活预处理 CUDA 融合 kernel：

```
FP16 act → [channel_mask scale] → [random sign] → [FWHT butterfly × 3 stages]
         → [hadK matmul 144×144 @ 144×8] → [1/sqrt(K) scale] → [per-token absmax quantize]
         → INT8 act → [W8A8 GEMM] → FP16 output
```

### 4.2 新建文件

| 文件 | 说明 |
|:-----|:-----|
| `kernels/csrc/fused/viditq_fused.cu` | FWHT+hadK+Quant 融合 CUDA kernel (294 行) |
| `kernels/viditq_extension/nn/viditq_linear.py` | ViDiTQW8A8Linear Python 封装 (215 行) |

### 4.3 修改文件

| 文件 | 改动 |
|:-----|:-----|
| `kernels/csrc/fused/pybind.cpp` | 注册 `viditq_act_quant_fuse` kernel |
| `kernels/setup.py` | 添加 `viditq_fused.cu` 到编译源 |
| `kernels/viditq_extension/nn/__init__.py` | 导出 `ViDiTQW8A8Linear` |
| `examples/opensora1.2/models/quant_opensora_cuda.py` | ① Weight 预处理 (ViDiT-Q attn FP16 对齐) ② `AttentionWithCudaKernel` FP16/INT8 双路径 forward ③ `use_kernel_override` 参数 ④ `STDiT3BlockWithCudaKernel` norm1 适配 |
| `examples/opensora1.2/models/quant_opensora.py` | `hardware_forward_refactor` ViDiT-Q block 分支 (创建 ViDiTQW8A8Linear 替换 ViDiTQuantizedLinear) |

### 4.4 关键技术突破

**突破 1: FWHT 算法修正**

CUDA kernel 第一版的 FWHT butterfly 逻辑错误——将每个 stage 的 pair 操作当作标量处理，而 Python 的 `matmul_hadU` 在 stage s 中处理的是大小为 2^s 的向量组。用 impulse 测试定位并修复：

```
修复前: CUDA impulse 输出 — 137 非零值 (错误)
修复后: CUDA impulse 输出 — 1152 均匀分布 (与 Python 一致, diff<1e-5)
```

**突破 2: FP32 FWHT 累积精度**

PyTorch 默认将 FP16 加减法提升到 FP32 内部执行。CUDA `__hadd/__hsub` 是纯 FP16 操作，误差累积导致帧间闪烁（std=2.50）。改为 FP32 后闪烁显著改善（std=1.53, ↓39%）。

```cuda
// 修复: 使用 FP32 做 butterfly 加减法
float a = __half2float(act_buffer[first_idx]);
float b = __half2float(act_buffer[second_idx]);
act_buffer[first_idx]  = __float2half_rn(a + b);
act_buffer[second_idx] = __float2half_rn(a - b);
```

**突破 3: ViDiT-Q 层检测与自动装配**

通过 `hasattr(block.attn.qkv, 'channel_mask')` 自动检测 ViDiT-Q blocks (11-14)，在 `hardware_forward_refactor` 中条件创建 `AttentionWithCudaKernel` + `ViDiTQW8A8Linear` 替换原始的 `ViDiTQuantizedLinear`。

### 4.5 V1→V4 迭代历程

| 版本 | 改动 | 结果 |
|:-----|:-----|:-----|
| V1 | 初始集成（buggy FWHT） | 方块伪影, std=55 |
| V2 | 修正 FWHT 算法 | 方块消失, diff=24, std=71 |
| V3 | Per-tensor 量化 (失败) | 质量下降, 回退 |
| **V4** | **FP32 FWHT butterfly** | **闪烁改善, 最终版本** |

---

## 五、文件改动清单

### 核心功能 (需要 git commit)

```
Modified:
  quant_utils/qdiff/viditq/viditq_quant_layer.py   # FP64→FP16 Hadamard
  kernels/csrc/fused/pybind.cpp                      # 注册新 kernel
  kernels/setup.py                                    # 编译配置
  kernels/viditq_extension/nn/__init__.py            # 模块导出
  examples/opensora1.2/models/quant_opensora.py      # ViDiT-Q block 分支
  examples/opensora1.2/models/quant_opensora_cuda.py  # Weight 预处理 + 双路径 Attention

New:
  kernels/csrc/fused/viditq_fused.cu                 # FWHT+hadK+Quant 融合 kernel
  kernels/viditq_extension/nn/viditq_linear.py       # ViDiTQW8A8Linear 封装
  tools/profiling/profile_inference.py               # CUDA Event 性能分析工具
  docs/viditq_phase_summary_2026.06.10.md            # 本文档
```

### 配置文件 (已存在，非本次改动)

```
New (from earlier work):
  examples/opensora1.2/configs/local_144p_120f_5s.py
  examples/opensora1.2/configs/local_144p_120f_5s_w8a8_hardware.py
  examples/opensora1.2/configs/w8a8_120f_5s.yaml
```

---

## 六、下一步行动计划

### Phase 3: 工程完善（预计 1-2 天）

#### 3a. 代码清理 & 规范化

**目标：** 消除调试痕迹，提升代码可维护性

| 操作 | 文件 | 说明 |
|:-----|:-----|:-----|
| 提取函数 | `quant_opensora_cuda.py` | 提取 `_is_viditq_layer(submodule, full_name)` |
| 提取工厂 | `quant_opensora.py` | 提取 `_create_viditq_attn_block()` |
| 修复路径 | `viditq_linear.py` | `_HADAMARD_MAT_PATH` 改用显式配置 |
| 移除死代码 | `viditq_fused.cu` | `ChannelScaleQuantKernel` 标记为备用 |

**验收标准：** 无语法警告，代码可独立 review

#### 3b. W4A8 混合精度 Kernel 适配

**背景：** 现有 W4A8 软件仿真链路已跑通。需要将 ViDiT-Q 预处理融合到 W4A8 GEMM 的 activation 量化步骤中，复用 W8A8 的 FWHT+hadK kernel。

**关键文件：**
- `kernels/csrc/qgemm/w4a8/w4a8_per_channel_gemm_cuda_qserve.cu`
- `examples/opensora1.2/configs/local_144p_120f_5s_w4a8_mp.py`

**估计耗时：** 4-6 小时

#### 3c. Benchmark 框架

**目标：** 建立可重复、可比较的自动化性能测试

```python
# tools/benchmark/benchmark.py
class BenchmarkRunner:
    configs = {
        "fp16_144p_5s":    Config(...),
        "w8a8_sw_144p_5s": Config(...),
        "w8a8_hw_144p_5s": Config(...),
        "w4a8_hw_144p_5s": Config(...),  # after 3b
    }
    metrics = ["latency_per_step", "total_time", "gpu_memory_peak"]
```

**交付物：** `python tools/benchmark/benchmark.py --all` 输出性能对比表

---

### Phase 4: TeaCache 实现（预计 3-5 天）

#### 4a. Timestep Embedding 分析（0.5 天）

编写分析脚本，在 FP16 推理中 hook `t_mlp` 和每个 block 的输入/输出，记录 10 步去噪过程中 timestep embedding 的变化模式，确定哪些 step 可以安全缓存。

**关键代码位置：**
- `STDiT3.forward()` — `stdit3.py:397` (t_mlp)
- `stdit3.py:426-428` (block loop)

#### 4b. FP16 路径 TeaCache 实现（1.5 天）

设计 `TeaCacheWrapper` 类，包装 `STDiT3.forward()`：
- 比较当前 `t_mlp` 与缓存的 `t_mlp` 的 L2 距离
- 如果低于阈值：跳过部分 blocks，复用缓存输出
- 如果高于阈值：完整前向，更新缓存

```python
@dataclass
class TeaCacheConfig:
    cache_threshold: float = 0.05
    cache_blocks: List[int] = None  # None = all
    cache_start_step: int = 2
```

#### 4c. W8A8 量化路径 TeaCache 适配（1 天）

确保缓存逻辑在量化 block (`STDiT3BlockWithCudaKernel`) 中正确工作。量化 block 的 forward 输出是 FP16（经过 `gate_residual_fuse`），可以直接缓存。

#### 4d. 速度-质量 Pareto 分析（1 天）

在 0.01 ~ 0.20 范围内扫描 `cache_threshold`，绘制速度加速比 vs PSNR 曲线，确定推荐配置。

---

### Phase 5: 系统级优化（预计 2-3 天）

| 阶段 | 内容 | 说明 |
|:-----|:-----|:-----|
| 5a | 240p/360p 高分辨率 | 序列长度 ~1.5M-2.3M tokens, TeaCache 收益更大 |
| 5b | Batch 推理 | 利用 16GB 显存做 batch>1 |
| 5c | 显存优化 | INT8 KV cache, VAE CPU offload |

---

## 七、关键技术债 & 风险

| 项目 | 严重度 | 说明 |
|:-----|:------:|:-----|
| `int_weight.pt` 设备修复 | 低 | `hardware_forward_refactor` 中有临时 `.cuda()` 迁移逻辑，需调查根本原因 |
| `viditq_fused.cu` 仅支持 K=1152 | 中 | hadK 和 FWHT stage 数硬编码，需泛化到其他 hidden_size |
| `AttentionWithCudaKernel` FP16/INT8 双路径 | 低 | 通过 `x.dtype` 判断，对 ViDiT-Q 以外的扩展需注意 |
| 无自动化测试 | 高 | 所有验证均为手动跑脚本 + 肉眼检查，建议建立 smoke test |

---

## 八、环境复现命令

```bash
conda activate viditq-xf033-test
cd /home/rich/ViDiT-Q/examples/opensora1.2

export PYTHONPATH=/home/rich/ViDiT-Q/examples/opensora1.2/Open-Sora
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
export VIDITQ_XFORMERS_OP=default

# 预计算文本嵌入
python precompute_text_embeds.py configs/local_144p_120f_5s.py

# FP16 基线 (如果要对比)
python fp_inference.py configs/local_144p_120f_5s.py

# W8A8 硬件推理 (ViDiT-Q 全链路)
python quant_inference.py configs/local_144p_120f_5s_w8a8_hardware.py

# Profiling
python /home/rich/ViDiT-Q/tools/profiling/profile_inference.py \
  configs/local_144p_120f_5s_w8a8_hardware.py --w8a8-hw --steps 2
```
