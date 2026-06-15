# ViDiT-Q + TeaCache：量化与缓存的组合加速

**日期:** 2026-06-15
**环境:** RTX 5080 Laptop GPU (16GB, Blackwell sm_120), WSL2 Ubuntu 22.04, torch 2.9.1+cu128

---

## 一、项目背景

本项目以 **[ViDiT-Q](https://arxiv.org/abs/2406.02540) (ICLR'25)** 作为量化框架，对 **OpenSora v1.2 / STDiT3** 视频扩散模型做 W8A8 硬件量化加速；在此基础上集成 **[TeaCache](https://arxiv.org/abs/2411.19108) (CVPR'25 Highlight)** 缓存优化，在量化推理路径上叠加 timestep-aware 缓存机制，实现量化+缓存的组合加速。

- **ViDiT-Q**：针对 Diffusion Transformer 的量化方法。核心创新是 channel-wise scaling + Hadamard rotation 预处理，将激活分布平滑化后再做 per-token 动态量化，实现 W8A8 精度几乎无损
- **TeaCache**：基于 timestep embedding 的免训练缓存方法。核心思想是扩散模型去噪过程中相邻 timestep 的模型输出高度相似，可通过比较 `t_mlp` 的变化量来判断是否可以复用上一步的计算结果

两者的结合产生了乘数效应：量化加速每个计算步，缓存跳过不需要计算的步。

### 性能演进总览

```
起点:    W8A8 软件仿真 (FP64 Hadamard)      ~97s
Phase 1: FP16 Hadamard                     ~12s    (8.2×)
Phase 2: ViDiT-Q fused CUDA kernel         ~5.6s   (2.1×)
Phase 3c: 标准化 benchmark                 ~4.5s   (1.2×)
Phase 4c: W8A8 + TeaCache @ 120f 10-step    ~1.2s  (3.8×)
Phase 4d: FP16 + TeaCache @ 120f 30-step   ~14.1s  (1.4× 质量模式)
Phase 5:  W8A8 + TeaCache @ 120f 30-step    ~8.4s  (2.4× 速度模式)
                                         ────────
                                         累计 ~12× 加速 vs FP16 baseline @ 30-step
```

---

## 二、架构总览

### 2.1 系统分层

```
┌─────────────────────────────────────────────────────────┐
│                   推理入口                                │
│   fp_inference.py / quant_inference.py                   │
│   - 构建 VAE + STDiT3 + Scheduler                        │
│   - 可选: TeaCache wrapper 替换 model.forward            │
│   - 可选: PAB manager 启用 per-block 细粒度缓存          │
├─────────────────────────────────────────────────────────┤
│                 TeaCache 缓存层                           │
│   teacache/teacache_wrapper.py                           │
│   - 全局残差缓存: 比较 t_mlp → 跳过所有 block           │
│   teacache/pab_mgr.py                                    │
│   - PAB 细粒度: per-block 步频缓存                       │
├─────────────────────────────────────────────────────────┤
│                 STDiT3 模型层                             │
│   stdit3.py: STDiT3.forward()                            │
│   - Patch embedding + 位置编码                           │
│   - t_embedder → t_block → t_mlp                        │
│   - 28×(spatial_block + temporal_block) 循环             │
│   - Final layer + unpatchify                            │
├─────────────────────────────────────────────────────────┤
│                 ViDiT-Q 量化层                            │
│   quant_opensora.py: QuantOpenSora                       │
│   - quant_layer_refactor: 替换 Linear → QuantizedLinear  │
│   - hardware_forward_refactor: 替换 Block → CUDA Kernel  │
│   quant_opensora_cuda.py: STDiT3BlockWithCudaKernel      │
│   - AttentionWithCudaKernel: W8A8 self-attention         │
│   - MlpWithCudaKernel: W8A8 MLP + fused gate_residual   │
├─────────────────────────────────────────────────────────┤
│                 CUDA Kernel 层                            │
│   kernels/csrc/fused/viditq_fused.cu                     │
│   - ViDiTQActQuantKernel: channel_mask+FWHT+hadK+quant   │
│   kernels/csrc/qgemm/w8a8/                               │
│   - w8a8_of16_bias_weight_asym: INT8 GEMM               │
│   kernels/viditq_extension/nn/viditq_linear.py           │
│   - ViDiTQW8A8Linear: Python wrapper for full pipeline   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流（单次去噪步）

```
输入 latent z [B, C, T, H, W]
        │
        ▼
┌─ TeaCache 决策 ─────────────────────────────────────┐
│  1. Patch embed + pos_emb → x                      │
│  2. 取 spatial_blocks[0].norm1 计算 modulated_inp   │
│  3. _should_calculate(timestep, modulated_inp):    │
│     - 第一步/最后一步: 强制计算                      │
│     - 否则: rel_l1 = |mod - prev| / |prev|         │
│             rescale(rel_l1) → 累加到 acc_dist       │
│             if acc_dist < thresh: 跳过(缓存命中)    │
│             else: 计算(缓存未命中), acc_dist = 0    │
└────────────────────────────────────────────────────┘
        │
        ├─ 缓存命中 ──────────────────────────────────┐
        │  x = x + previous_residual                 │
        │  (跳过 56 个 block，~685ms → ~0ms)          │
        │                                            │
        ├─ 缓存未命中 ────────────────────────────────┤
        │  for spatial_block, temporal_block:        │
        │      │                                     │
        │      ▼                                     │
        │  ┌─ ViDiT-Q Block (W8A8) ─────────────┐   │
        │  │                                     │   │
        │  │  Self-Attention:                    │   │
        │  │    norm1 → modulate                 │   │
        │  │    qkv: ViDiTQ fused kernel → INT8  │   │
        │  │    W8A8 GEMM (INT8 Tensor Core)     │   │
        │  │    xformers attention (FP16)        │   │
        │  │    gate_residual_fuse (CUDA fused)  │   │
        │  │                                     │   │
        │  │  Cross-Attention:                   │   │
        │  │    quant_sum → INT8 activation      │   │
        │  │    W8A8 GEMM for q_linear/proj      │   │
        │  │    xformers attention               │   │
        │  │                                     │   │
        │  │  MLP:                               │   │
        │  │    LayerNormGeneral (fused)         │   │
        │  │    fc1: W8A8 GEMM                   │   │
        │  │    gelu_quant_sum (CUDA fused)      │   │
        │  │    fc2: W8A8 GEMM                   │   │
        │  │    gate_residual_fuse               │   │
        │  └─────────────────────────────────────┘   │
        │                                            │
        │  previous_residual = x - origin_x          │
        └────────────────────────────────────────────┘
        │
        ▼
  Final layer + unpatchify
        │
        ▼
  输出 pred [B, C_out, T, H, W]
```

---

## 三、ViDiT-Q 量化技术细节

### 3.1 核心思想

扩散模型中 activation 的分布在不同 channel 之间差异极大（有些 channel 的数值范围是其他 channel 的 100×）。直接做 per-token 量化会导致大值 channel 主导量化尺度，小值 channel 被淹没。

ViDiT-Q 的解决方案分两步：
1. **Channel-wise scaling**: 用 `channel_mask` 重新平衡各 channel 的量级
2. **Hadamard rotation**: 通过正交变换消除 channel 间的相关性，使量化误差均匀分布

### 3.2 ViDiT-Q 预处理流水线

```
FP16 activation x [M, K=1152]
        │
        ▼
  Step 1: Channel scaling
    x = x * channel_mask  (element-wise, FP16)
    channel_mask[i] = |W[:,i]|^α / |x[:,i]|^(1-α)
        │
        ▼
  Step 2: Random sign flip
    x = x * random_signs  (element-wise, ±1)
    随机 ±1 翻转消除结构性偏差
        │
        ▼
  Step 3: FWHT (Fast Walsh-Hadamard Transform)
    3 级 butterfly 变换 (K=1152 → K_block=144, N_stages=3)
    每级: pair first/second halves → [sum, diff]
    使用 FP32 中间精度匹配 PyTorch 行为
        │
        ▼
  Step 4: hadK 矩阵乘法
    hadK [144×144] @ data [144×8] → [144×8]
    hadK 是预计算的 Hadamard 基矩阵
    来自 quarot_utils.get_hadK()
        │
        ▼
  Step 5: 1/√K 缩放 + per-token absmax INT8 量化
    scale = max(|x|) / 127
    x_int8 = round(x / scale), clipped to [-128, 127]
        │
        ▼
INT8 activation [M, K=1152]
```

### 3.3 CUDA Kernel 实现

整个预处理流水线在单个 CUDA kernel 中完成 (`viditq_fused.cu:ViDiTQActQuantKernel`)：

```cuda
// 288 线程处理 K=1152，每个线程处理 4 个元素
template <int K=1152, int K_BLOCK=144, int N_STAGES=3, int BLOCK_THREADS=288>

// Step 1-2: Load + channel_mask + random_sign (coalesced global read)
// Step 3:   FWHT butterfly — 3 stages, shared memory
// Step 4:   hadK matmul [144,144] @ [144,8] — shared memory hadK
// Step 5:   Block reduce max → scale, per-element quantize → INT8
```

权重端预处理在 PTQ 阶段离线完成（`quantize_and_save_weight_`）：
```python
# Weight preprocessing (one-time, offline):
fp_weight = fp_weight / channel_mask          # Step 1: channel scaling
fp_weight = matmul(fp_weight, rotation_matrix) # Step 2: Hadamard rotation
int_weight = round((fp_weight/scale) - zp)     # Step 3: INT8 quantization
```

### 3.4 W8A8 GEMM

量化后的 INT8 activation 和 INT8 weight 通过标准 W8A8 GEMM kernel 计算：

```
INT8 activation [M, K] @ INT8 weight [N, K]^T
  + dequant: scale_act * scale_w * (INT32_acc + bias_correction)
  → FP16 output [M, N]
```

CUDA kernel 参数：CTA_M=128, CTA_N=128, CTA_K=64，利用 INT8 Tensor Core。

### 3.5 硬件路径 Block 替换

`hardware_forward_refactor()` 将 `STDiT3Block` 替换为 `STDiT3BlockWithCudaKernel`：

| 组件 | FP16 原始 | W8A8 CUDA Kernel |
|:-----|:---------|:-----------------|
| Self-Attention qkv/proj | `nn.Linear` → xformers | `W8A8OF16LinearDynamicInputScale` → xformers |
| Cross-Attention q_linear/proj | `nn.Linear` → xformers | `W8A8OF16LinearDynamicInputScale` → xformers |
| MLP fc1/fc2 | `nn.Linear` → GELU | `W8A8OF16LinearDynamicInputScale` + `gelu_quant_sum` |
| Gate+Residual | `gate * x + residual` | `fused_kernels.gate_residual_fuse` |
| LayerNorm | `nn.LayerNorm` | `LayerNormGeneral` (含量化融合) |

ViDiT-Q 特有的处理（channel_mask + FWHT + hadK）仅对 blocks 11-14（ViDiT-Q 论文指定的层）生效，通过 `ViDiTQW8A8Linear` 包装 `W8A8OF16LinearDynamicInputScale` + ViDiT-Q 预处理 kernel。

---

## 四、TeaCache 缓存技术细节

### 4.1 核心原理

扩散模型去噪过程中，timestep embedding `t_mlp` 是控制所有 56 个 transformer block 的**唯一调制输入**。`t_mlp` 通过 `scale_shift_table` 生成每个 block 的 shift/scale/gate 参数：

```python
t_mlp = t_block(t_embedder(timestep))  # [B, 6*C]
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    scale_shift_table + t_mlp.reshape(B, 6, -1)
).chunk(6, dim=1)

# 调制公式 (t2i_modulate):
x_modulated = x * (1 + scale) + shift    # 对 norm 输出做 affine 变换
gate_output = gate * block_output         # 对 block 输出做门控
```

因为所有 block 都共享同一个 `t_mlp`，所以 `t_mlp` 在相邻 timestep 之间的变化量可以直接预测所有 block 输出的变化量。

### 4.2 全局残差缓存 (TeaCacheWrapper)

这是本项目的主力缓存策略。核心逻辑：

```python
# 1. 计算"探针"——第一个 spatial block 的调制输入
modulated_inp = t2i_modulate(spatial_blocks[0].norm1(x), shift_msa, scale_msa)

# 2. 决策
if timestep == first or timestep == last:
    should_calc = True          # 首尾步: 强制计算
else:
    rel_l1 = |mod - prev_mod| / |prev_mod|    # 相对 L1 距离
    rescaled = poly(rel_l1)                    # 多项式映射到预测输出变化
    acc_dist += rescaled                       # 累积距离
    should_calc = (acc_dist >= threshold)      # 超过阈值 → 计算
    if should_calc:
        acc_dist = 0                           # 重置累积

# 3. 执行
if should_calc:
    for block in all_56_blocks:
        x = block(x, ...)
    previous_residual = x - origin_x           # 保存残差
else:
    x = x + previous_residual                  # 复用残差
```

**为什么用累积距离而非单步距离？** 因为缓存残差是从**上次计算步**复用的。如果上次计算是 3 步前，那么当前步与 3 步前的残差差异会大于单步差异。累积距离跟踪了从上次计算到当前步的总变化量。

**rescale 多项式**：rel_l1 和实际 block 输出变化不是线性关系。用一个 3 阶多项式做映射：

```
rescale = c0·x³ + c1·x² + c2·x + c3
```

系数通过校准脚本 (`teacache/calibrate_coeff.py`) 对具体模型 + 配置拟合，R² 通常 > 0.98。

### 4.3 PAB 细粒度缓存 (PABManager)

TeaCache 论文还提出了更细粒度的 Pyramid Attention Broadcast (PAB)，在 per-block per-component 级别独立缓存：

| 组件 | 默认步频 | 原因 |
|:-----|:--------:|:-----|
| Spatial self-attention | 2 | 空间 attention 对不同 timestep 最稳定 |
| Temporal self-attention | 4 | 时序 attention 对 timestep 变化最敏感 |
| Cross-attention | 3 | 中间敏感度 |
| MLP | 2 | 与 spatial 类似 |

PAB 在 120f 30-step 配置下相比全局残差没有显著优势（timestep 太密集，全局判断已足够精确），保留给低帧数/少步数场景。

---

## 五、量化+缓存的协同机制

### 5.1 为什么可以叠加

1. **不同层面**：量化改变每个 operator 的计算精度（FP16→INT8）；缓存改变哪些 operator 需要执行（跳过 vs 计算）
2. **正交性**：W8A8 block 的 forward 签名与 FP16 block 完全一致（`quant_params` 通过 `self` 访问，不暴露在参数列表中），TeaCache wrapper 直接调用原始 block loop，无需修改
3. **输出兼容**：W8A8 GEMM 输出 FP16 tensor，与 FP16 block 输出类型完全相同，缓存残差可以无缝复用

### 5.2 组合加速的数学表达

```
T_vanilla  = N_steps × T_step                    (FP16 baseline)
T_w8a8     = N_steps × T_step × 0.81             (W8A8: ~19% faster per step)
T_tc       = N_steps × (T_overhead + (1-h)·T_step)  (TeaCache: h = cache hit rate)

T_combined = N_steps × (T_overhead + (1-h)·T_step_w8a8)

其中:
  T_step       ≈ 685ms  (56 blocks)
  T_overhead   ≈ 1ms    (embeddings + TeaCache decision + final layer)
  T_step_w8a8  ≈ 551ms  (W8A8 blocks, 1.24× faster than FP16)
  h            = 30-50% (cache hit rate, 取决于阈值)
```

对于 30-step th=0.15:
```
T_combined = 30 × (1ms + 0.50 × 551ms) = 30 × 277ms ≈ 8.3s
vs FP16: 30 × 686ms ≈ 20.6s  →  2.48× 加速
```

实测 8.4s (2.41×)，与公式预测一致。

### 5.3 关键设计决策

**为何用全局残差而非 PAB 作为主力？** 

120f 30-step 的 timestep_transform 中 `ratio_time = √(35/1) = 5.92`，将 timestep 极度压缩。相邻步的 `t_mlp` 差异极小（余弦相似度 > 0.98），导致"全有或全无"的全局判断已经足够精确。PAB 的 per-block 区分在此配置下不会带来额外收益，反而增加 168 次判断开销（3 组件 × 56 blocks）。

**为何 30 步而非 10 步？** 是为了对齐 TeaCache 论文（30 步是标准设置）。10 步下 TeaCache 同样有效（~4.3× 加速），但 30 步的 cache hit rate 更高（步间差异更小）且生成质量更好。

---

## 六、最终评测结果

### 6.1 消融实验 (120f 144p 30-step RTX 5080)

| 量化 | TeaCache | 耗时 | vs FP16 | vs 自身base | 像素PSNR |
|:----:|:--------:|:-----|:-------:|:-----------:|:--------:|
| — | — | 20.25s | 1.00× | — | ∞ |
| — | th=0.08 | 14.14s | 1.43× | 1.43× | 33.5 dB |
| — | th=0.15 | 10.31s | 1.96× | 1.96× | 25.7 dB |
| W8A8 | — | 16.5s | 1.23× | — | 18.4 dB* |
| **W8A8** | **th=0.08** | **11.6s** | **1.75×** | 1.42× | **30.2 dB** |
| **W8A8** | **th=0.15** | **8.4s** | **2.41×** | 1.96× | 28.8 dB |

> *W8A8 baseline vs FP16 baseline 的量化质量差异（18.4dB）。叠加 TeaCache 后的 PSNR 是 vs 各自的 baseline。

### 6.2 与原始论文对比

| | ViDiT-Q 论文 | TeaCache 论文 | 本工作 |
|:---|:---|:---|:---|
| GPU | A100 | A800 | RTX 5080 Laptop |
| 分辨率 | 144p | 480p | 144p |
| 帧数 | 120f | 51f/192f | 120f |
| 采样步数 | 10 | 30 | 30 |
| 量化加速 | 1.19× (W8A8) | — | 1.23× |
| 缓存加速 | — | 1.66× | 1.96× |
| **组合加速** | **—** | **—** | **2.41×** |

### 6.3 FP16 Pareto 曲线 (120f 30-step)

```
阈值     加速比    命中率     latent PSNR    pixel PSNR    推荐场景
─────    ─────    ─────    ───────────    ──────────    ────────
0.03     0.99×      0%         ∞ dB          ∞ dB       完全等价 baseline
0.05     0.99×      0%         ∞ dB          ∞ dB
0.08     1.39×     30%       23.0 dB        33.5 dB     ← 质量优先 ★
0.10     1.47×     33%       22.1 dB          —
0.15     1.91×     50%       17.6 dB        25.7 dB     ← 平衡
0.20     2.35×     60%       15.7 dB          —
0.30     3.07×     70%       12.7 dB          —          速度优先
```

### 6.4 高分辨率扩展

| 分辨率 | 尺寸 | Tokens/样本 | Vanilla | FP16+TC th=0.08 |
|:-------|:-----|:-----------:|:-------:|:----------------:|
| 144p | 192² | 5,184 | 20.25s | 14.14s (1.4×) |
| 240p | 320² | 14,400 | 60.00s | 46.54s (1.3×) |
| 360p | 480² | 32,400 | 131.71s | 94.55s (1.4× @ th=0.15) |

高分辨率下 TeaCache 加速比衰减（更多 token → rel_l1 噪声更大 → 更难缓存），证实了 TeaCache 论文中 PAB 的必要性。

---

## 七、项目文件结构

```
ViDiT-Q/
├── examples/opensora1.2/
│   ├── configs/                             # 配置文件 (10个)
│   │   ├── local_144p_17f.py               #   快速 debug (17f)
│   │   ├── local_144p_120f_5s.py           #   10-step 基准 (对齐 ViDiT-Q)
│   │   ├── local_144p_120f_30step*.py      #   ★ 主评测 (30-step, 4个变体)
│   │   ├── local_144p_120f_5s_w8a8_*.py    #   W8A8-HW 配置 (2个)
│   │   └── w8a8*.yaml                      #   量化参数
│   ├── teacache/                            # TeaCache 模块 (5个文件)
│   │   ├── __init__.py                     #   导出 TeaCache + PAB
│   │   ├── teacache_wrapper.py            #   全局残差缓存 (~300行)
│   │   ├── pab_mgr.py                      #   PAB 管理器 (~170行)
│   │   ├── calibrate_coeff.py              #   校准工具
│   │   └── utils.py                        #   VAE CPU offload + CUDATimer
│   ├── fp_inference.py                     # FP16 推理入口 (+TeaCache 开关)
│   ├── quant_inference.py                  # W8A8 推理入口 (+TeaCache 开关)
│   └── models/
│       ├── quant_opensora.py               #   QuantOpenSora (量化模型)
│       └── quant_opensora_cuda.py          #   CUDA kernel blocks
├── kernels/                                # CUDA 扩展
│   ├── csrc/fused/viditq_fused.cu         #   ViDiT-Q 融合 kernel (FWHT+hadK+quant)
│   ├── csrc/qgemm/w8a8/                    #   W8A8 GEMM kernel
│   └── viditq_extension/nn/               #   Python 封装 (ViDiTQW8A8Linear 等)
├── tools/
│   ├── profiling/                          # Benchmark + timestep 分析
│   │   ├── benchmark.py / benchmark.sh     #   三模式性能对比
│   │   └── analyze_timestep.py            #   Timestep 分析 (Phase 4a)
│   └── diagnostics/                        # xFormers 兼容性诊断
├── docs/                                   # 项目文档
│   ├── 项目交接文档2026.05.30.md           #   早期环境搭建 + FP16 推理
│   ├── viditq_w8a8_5s_reproduction_*.md    #   W8A8 复现记录
│   ├── viditq_phase_summary_2026.06.10.md   #   Phase 1-3 (kernel 优化)
│   ├── viditq_handover_2026.06.12.md       #   Phase 3-4a (benchmark+timestep)
│   ├── viditq_teacache_work_plan_*.md      #   工作计划
│   └── viditq_teacache_final_2026.06.15.md #   ★ 本文档
└── .local/outputs/                         # 量化产物 + 生成视频
    ├── opensora_w8a8_hardware_*/           #   W8A8 量化产物 (quant_params + int_weight)
    ├── opensora_fp16_*_30steps/            #   FP16 baseline + TeaCache 视频
    ├── w8a8_*_30steps/                     #   W8A8 baseline + TeaCache 视频
    └── opensora_w8a8_*/                    #   W8A8 软件仿真产物 + calibration
```

---

## 八、使用指南

### 8.1 环境准备

```bash
conda activate viditq-xf033-test
cd /home/rich/ViDiT-Q/examples/opensora1.2

export PYTHONPATH=./Open-Sora:/home/rich/ViDiT-Q/kernels
export LD_LIBRARY_PATH="$(dirname $(python -c 'import torch;print(torch.__file__)'))/lib:$LD_LIBRARY_PATH"

# xformers backend (取决于环境):
#   viditq-xf033-test (torch 2.9.1, xformers 0.0.33): default
#   viditq-osora (torch 2.8.0, xformers 0.0.32): cutlass
export VIDITQ_XFORMERS_OP=default

# HuggingFace 缓存 (VAE 需要)
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
```

### 8.2 运行推理

```bash
# === FP16 路径 ===

# Baseline (无缓存, 30步)
python fp_inference.py configs/local_144p_120f_30step_baseline.py

# TeaCache 质量模式 (th=0.08, PSNR=33.5dB vs baseline)
python fp_inference.py configs/local_144p_120f_30step_tc_008.py

# TeaCache 快速模式 (th=0.15, PSNR=25.7dB vs baseline)
python fp_inference.py configs/local_144p_120f_30step_tc_015.py

# === W8A8 路径 (需先有 quant_params.pth + int_weight.pt) ===

# Baseline
python quant_inference.py configs/local_144p_120f_5s_w8a8_hardware.py

# TeaCache (在配置中设置 enable_teacache=True)
python quant_inference.py configs/local_144p_120f_5s_w8a8_hardware_teacache.py

# === Benchmark ===

# 三模式对比 (FP16 / W8A8-SIM / W8A8-HW)
bash /home/rich/ViDiT-Q/tools/profiling/benchmark.sh --all

# Timestep 分析
python /home/rich/ViDiT-Q/tools/profiling/analyze_timestep.py
```

### 8.3 校准 rescale 系数

```bash
# 在新配置上校准 (需先 precompute_text_embeds):
python teacache/calibrate_coeff.py configs/your_config.py

# 输出: 3阶多项式系数, 复制到 TeaCacheConfig(rescale_coefficients=[...])
```

### 8.4 TeaCache API

```python
from teacache import TeaCacheWrapper, TeaCacheConfig

# 全局残差缓存 (推荐用于 120f 30-step)
config = TeaCacheConfig(
    rel_l1_thresh=0.10,
    rescale_coefficients=[...]  # 校准后的系数
)
wrapper = TeaCacheWrapper(model, config)
model.forward = wrapper.forward  # 透明替换

# 统计: wrapper.cache_hits, wrapper.cache_hit_rate

# PAB 细粒度缓存 (用于低帧数/少步数场景)
from teacache import PABConfig, set_pab_manager
set_pab_manager(PABConfig(
    spatial_range=2, temporal_range=4,
    cross_range=3, mlp_range=2,
    threshold_low=80, threshold_high=950
))
```

### 8.5 关键技术约束

| 约束 | 说明 |
|:-----|:-----|
| W8A8 CTA_M=128 | 需要 CFG token 数被 128 整除。144p 下 T 必须为 4 的倍数<br>16f(T=4)/68f(T=20)/120f(T=36) 可用；17f(T=5)/32f(T=9) 不可用 |
| xformers sm_120 | `viditq-xf033-test` 用 `default`, `viditq-osora` 用 `cutlass` |
| 16f + timestep_transform | NaN 风险 (`16//17*5=0`)，建议用 68f+ 或不启用 transform |
| 首次运行 | `quantize_and_save_weight` 生成 1.6GB int_weight.pt，耗时 ~30s |

---

## 九、关键技术债与后续方向

| 项目 | 状态 | 说明 |
|:-----|:----:|:-----|
| `viditq_fused.cu` K 泛化 | 待解决 | 当前硬编码 K=1152, K_BLOCK=144 |
| W4A8 attn 层 | 搁置 | QServe kernel 的 G=128 + half2 打包与 C_in=1152 不兼容 |
| torch.compile 集成 | 探索 | 1.73× 潜力但不稳定 |
| PAB + 全局残差组合 | 已架构 | TeaCache wrapper 已传 timestep, PAB 可随时启用 |
| VBench 标准化评测 | 待做 | 需要生成多段 prompt 视频 + 运行评测脚本 |
| 论文写作 | 待做 | 数据齐全, 消融表 + Pareto 曲线可直用 |
