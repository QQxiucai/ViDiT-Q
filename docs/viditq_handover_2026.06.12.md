# ViDiT-Q 项目交接文档

**日期:** 2026-06-12
**环境:** RTX 5080 Laptop GPU (Blackwell sm_120, 16GB VRAM), WSL2 Ubuntu 22.04
**Conda 环境:** `viditq-osora` (torch 2.8.0+cu128, Python 3.10), 内核扩展在 `viditq-xf033-test` 编译
**基准模型:** OpenSora v1.2 / STDiT3-XL/2, 120f 144p 5s 视频生成

---

## 一、前期回顾（截至 2026.06.10）

详见 `docs/viditq_phase_summary_2026.06.10.md`。核心成果：

- **Phase 1**: FP64→FP16 Hadamard 旋转，软件仿真 8.2× 加速
- **Phase 2**: ViDiT-Q CUDA fused kernel（FWHT+hadK+quant），端到端 17.3× 加速（97s → 5.6s）
- 最终 V4 版本通过肉眼验证，画面质量良好

---

## 二、Phase 3b: W4A8 混合精度 CUDA Kernel 适配

### 2.1 目标

将 QServe W4A8 GEMM kernel（G=128 per-group 权重量化）集成到 ViDiT-Q 硬件推理路径，对符合条件的层使用 4-bit 权重以降低显存和提升速度。

### 2.2 完成的工作

**新增模块** (`kernels/viditq_extension/nn/`):

- **`W4A8OF16LinearDynamicInputScale`** (qlinear.py): W4A8 GEMM 封装，权重格式为 `[C_out, C_in/2]` INT8（2×4-bit 打包），per-group (G=128) 的 `wscales`/`w_szs` 以 half2 打包存储
- **`ViDiTQW4A8Linear`** (viditq_linear.py): 继承 ViDiT-Q 预处理（FWHT+hadK+quant）→ W4A8 packed GEMM 的完整 pipeline

**权重量化逻辑** (`quant_opensora_cuda.py`):

- `quantize_and_save_weight_()` 新增 W4A8 分支：per-group (G=128) 4-bit 量化，half2-packed scales/szs，权重打包为 2×4-bit/byte
- 奇数 group 检测：当 `C_in / 128` 为奇数时回退到 W8A8

**Block 替换逻辑** (`quant_opensora.py`):

- `hardware_forward_refactor()` 新增 W4A8 检测：`int(w_quantizer.n_bits) == 4` 且 `(C_in // 128) % 2 == 0` 时创建 `ViDiTQW4A8Linear`
- `_replace_block_with_cuda()` 中同步奇数 group 检测逻辑

### 2.3 核心限制：G=128 与 half2 打包

QServe kernel 要求 half2-packed scales（group 数必须为偶数），且 G=128 不可配置：

| 层 | C_in | Groups (G=128) | W4A8 可用? |
|---|---|---|---|
| attn.qkv (ViDiT-Q blocks 11-14) | 1152 | 9 (奇数) | ❌ 回退 W8A8 |
| attn.proj | 1152 | 9 (奇数) | ❌ 回退 W8A8 |
| mlp.fc1 | 4608 | 36 (偶数) | ✅ |
| mlp.fc2 | 4608 | 36 (偶数) | ✅ |

**结论：ViDiT-Q attn 层全部回退 W8A8，只有 mlp.fc1/fc2 能使用 W4A8。**

### 2.4 与原始 ViDiT-Q 论文的关系

原始 ViDiT-Q 论文声称 W4A8 "without notable visual quality degradation"，但他们的 W4A8 是**纯软件模拟**（无 CUDA kernel）。QServe W4A8 kernel 来自另一个团队（Lin, Tang, Yang et al., 2024），专门针对 LLM 设计（hidden_size 为 2 的幂），从未在 ViDiT-Q 的架构（hidden_size=1152，非 2 的幂）上集成测试过。我们面对的是一个前人未解决的跨项目集成问题。

### 2.5 W4A8 后续路径

| 方案 | 难度 | 说明 |
|------|------|------|
| A. 改 G=64 | 中等 | `1152/64=18` groups（偶数），需改 kernel 模板参数并重新编译 |
| B. 改 kernel 支持奇数 groups | 较高 | 修改 QServe kernel 的 half2 索引逻辑 |
| C. 接受现状 | 无 | mlp 层已经能用 W4A8；attn 层用 W8A8 |

**建议：** 当前不做 W4A8 attn 层适配。W4A8 边际收益有限（ViDiT-Q 瓶颈在激活预处理，不在 GEMM），TeaCache 的端到端加速收益远大于 W4A8。

---

## 三、Phase 3c: Benchmark 框架

### 3.1 目标

建立可重复、可比较的自动化性能测试，一键对比 FP16 / W8A8 软件仿真 / W8A8 硬件推理。

### 3.2 新建文件

| 文件 | 说明 |
|:-----|:-----|
| `tools/profiling/benchmark.py` | 主 benchmark 脚本 |
| `tools/profiling/benchmark.sh` | 便捷启动脚本（自动设置 LD_LIBRARY_PATH 等） |

### 3.3 使用方式

```bash
# 对比全部三种模式（默认 10 steps, 2 warmup）
bash tools/profiling/benchmark.sh --all

# 单模式
bash tools/profiling/benchmark.sh --fp16
bash tools/profiling/benchmark.sh --w8a8-sim
bash tools/profiling/benchmark.sh --w8a8-hw

# 自定义参数
bash tools/profiling/benchmark.sh --all --steps 20 --warmup 3
```

### 3.4 Benchmark 结果（144p, 120f, 10 steps）

| Metric | FP16 | W8A8-SIM | W8A8-HW |
|--------|------|----------|---------|
| **Per-step latency** | 0.531s | 0.963s | **0.447s** |
| **Speedup vs FP16** | 1.00× | 0.57× | **1.19×** |
| **Peak GPU memory** | 2,604 MB | 7,423 MB | 3,665 MB |
| **Static GPU memory** | 2,328 MB | 6,738 MB | 3,323 MB |

**关键发现：**

1. **W8A8-HW 最快** — 比 FP16 快 19%，比软件仿真快 2.15×
2. **W8A8-SIM 比 FP16 还慢** — Python 量化开销压倒 INT8 GEMM 收益；仅适用于 PTQ 标定
3. **W8A8-HW 显存比 FP16 多 ~1GB** — INT8 权重副本 + scale/zp/sum buffer 开销；比 SIM 省 ~3.4GB

### 3.5 Benchmark 指标说明

- **Peak GPU memory**: `torch.cuda.max_memory_allocated()` — 推理过程中峰值，含模型权重 + 中间激活
- **Static GPU memory**: `torch.cuda.memory_allocated()` — 推理结束后稳态值，主要是模型常驻张量
- 差值 ≈ 激活张量开销：FP16 ~276MB / W8A8-HW ~342MB / W8A8-SIM ~684MB

### 3.6 沿途修复的 Bug

**xformers sm_120 兼容性:** `MultiHeadCrossAttentionWithCudaKernel.forward()` 调用 `xformers.ops.memory_efficient_attention` 时未传 `op` 参数，默认 dispatch 选中了不稳定的 Hopper flash-attention kernel（CUDA `invalid argument`）。

**修复** (`quant_opensora_cuda.py`):
- 添加 `_get_xformers_op()` 函数（读取 `VIDITQ_XFORMERS_OP` 环境变量）
- 在 `MultiHeadCrossAttentionWithCudaKernel.forward()` 中传入 `op=_get_xformers_op()`
- 启动时设置 `VIDITQ_XFORMERS_OP=cutlass`，强制使用稳定的 CUTLASS 后端

**VAE 离线加载:** Benchmark 不构建完整 VAE（需要网络下载 PixArt 子模型），改为硬编码 VAE 架构常量计算 latent_size：
- Spatial VAE: patch_size=(1,8,8), out_channels=4
- Temporal VAE: time_downsample_factor=4, patch_size=(4,1,1)
- 144p 120f → latent_size = [36, 24, 24]

---

## 四、Phase 4a: Timestep Embedding 分析（TeaCache 预备）

### 4.1 目标

分析 STDiT3 模型中 timestep embedding 在去噪轨迹上的变化规律，为 TeaCache 缓存策略提供数据支撑。

### 4.2 分析脚本

**文件:** `tools/profiling/analyze_timestep.py`

**方法:**
1. 在 FP16 模型上跑完整 10 步去噪轨迹
2. 通过 hook 捕获每步的 `t_embed`（原始 sinusoidal 嵌入）、`t_mlp`（经 t_block 映射后的调制嵌入）、以及每个 block 的调制参数（shift/scale/gate × msa/mlp）
3. 计算连续步之间的余弦相似度和调制参数变化量

### 4.3 分析结果

#### t-embedding 全局相似度

| 指标 | 值 |
|------|-----|
| t_embed 连续步余弦相似度 | **0.925** |
| t_mlp 连续步余弦相似度 | **0.980** |
| t_mlp 全局平均相似度 | 0.904 |
| t_mlp 首尾步相似度（最大差异） | 0.487 |

**t_mlp 连续步相似度 0.98** 意味着相邻 timestep 的 block 调制输入几乎一样，TeaCache 可以大量复用 block 输出。

#### 各 Block 调制参数变化量

按相对 delta（L2 差 / 平均 norm）排序：

```
最稳定（适合缓存）                     最敏感（需保留计算）
──────────────────────────────────────────────────────────
S27: 0.066 ██                          T08: 0.115 █████
S00: 0.072 ██                          T13: 0.116 █████
S26: 0.075 ██                          T00: 0.118 █████
S23: 0.075 ██                          T06: 0.119 █████
S19: 0.077 ██                          T04: 0.124 █████
S18: 0.079 ██                          T01: 0.127 ██████
S24: 0.079 ██                          T05: 0.130 ██████
S25: 0.080 ██                          T02: 0.132 ██████
S17: 0.080 ██                          T03: 0.133 ██████
S20: 0.081 ██
S16: 0.081 ██
...（中间全部为 Spatial blocks）
T27: 0.101 ████
```

#### 不同阈值下的可缓存比例

| 阈值 | 可缓存 blocks | 比例 |
|------|:----------:|------|
| 0.05 | 0/56 | 0% |
| 0.07 | ~4/56 | ~7% |
| 0.08 | ~22/56 | ~39% |
| **0.10** | **33/56** | **59%** |

### 4.4 关键发现

1. **Spatial blocks 比 temporal blocks 稳定** — 前 20 个最可缓存的全部是 spatial（S00-S27），后 10 个最敏感的全部是 temporal（T00-T09）。直观上合理：时序维度的动态特征对 timestep 更敏感。

2. **连续步高度相似** — t_mlp 余弦相似度 0.98，意味着 TeaCache 可以在大多数连续步之间复用缓存。

3. **推荐阈值区间** — 0.08~0.10，可缓存 40-60% blocks。具体值需通过 Phase 4d Pareto 分析确定。

4. **与原始 TeaCache 论文一致** — 论文的核心假设（timestep embedding 缓慢变化，可用距离阈值控制缓存）在 ViDiT-Q/STDiT3 架构上成立。

---

## 五、Phase 3a: 代码清理（部分完成）

Phase 3a 规划的完整代码规范化未作为独立阶段执行，但在 Phase 3b/3c 中同步进行了以下清理：

- `quant_opensora_cuda.py`: 添加 `_get_xformers_op()` 函数，消除 xformers 硬编码依赖
- `benchmark.py`: 提取 `_compute_latent_size()` 独立函数，去 VAE 网络依赖
- `quant_opensora.py`: W4A8/W8A8 检测逻辑在 `hardware_forward_refactor` 和 `quantize_and_save_weight_` 之间同步

---

## 六、文件改动总览（6.10 → 6.12）

### 新建文件

```
tools/profiling/benchmark.py              # Phase 3c: Benchmark 主脚本
tools/profiling/benchmark.sh              # Phase 3c: 便捷启动器
tools/profiling/analyze_timestep.py       # Phase 4a: Timestep 分析脚本
docs/viditq_handover_2026.06.12.md        # 本文档
```

### 修改文件

```
kernels/viditq_extension/nn/viditq_linear.py    # Phase 3b: ViDiTQW4A8Linear 类
kernels/viditq_extension/nn/qlinear.py          # Phase 3b: W4A8OF16LinearDynamicInputScale
examples/opensora1.2/models/quant_opensora.py   # Phase 3b: W4A8 检测 + block 工厂逻辑
examples/opensora1.2/models/quant_opensora_cuda.py  # Phase 3b: W4A8 weight 打包
                                                     # Phase 3c: _get_xformers_op + sm120 fix
examples/opensora1.2/quant_inference.py         # Phase 3b: bitwidth_refactor 顺序修复
```

### 配置文件（Phase 3b 新建 / 保持）

```
examples/opensora1.2/configs/w4a8_mp_120f_5s.yaml
examples/opensora1.2/configs/local_144p_120f_5s_w4a8_mp.py
examples/opensora1.2/configs/local_144p_120f_5s_w4a8_mp_hardware.py
```

---

## 七、关键技术债 & 风险（更新）

| 项目 | 严重度 | 说明 |
|:-----|:------:|:-----|
| W4A8 attn 层不可用 | 中 | G=128 硬编码 + half2 打包导致 C_in=1152 层回退 W8A8 |
| `viditq_fused.cu` 仅支持 K=1152 | 中 | hadK 和 FWHT stage 数硬编码 |
| xformers sm_120 兼容性 | 低 | 已修复（cutlass 后端），但需保持 `VIDITQ_XFORMERS_OP=cutlass` |
| VAE 加载需网络 | 低 | Benchmark 已绕过，但完整推理仍需预先下载 PixArt VAE |
| Benchmark OOM 风险 | 中 | 三模式串联可能因显存碎片 OOM；已改为独立进程运行 |
| 无自动化测试 | 高 | 所有验证仍为手动 |

---

## 八、下一步：Phase 4b-4d TeaCache 实现

### Phase 4b: FP16 TeaCache（预计 1-2 天）

基于 Phase 4a 分析结果实现 `TeaCacheWrapper`：
- 比较当前 `t_mlp` 与缓存 `t_mlp` 的差异
- 差异 < 阈值：跳过 blocks，复用缓存输出
- 差异 ≥ 阈值：完整前向，更新缓存
- 支持 `cache_threshold` 和 `cache_blocks` 配置

### Phase 4c: W8A8 TeaCache 适配（预计 1 天）

确保缓存逻辑在 `STDiT3BlockWithCudaKernel` 中正确工作。

### Phase 4d: Pareto 分析（预计 1 天）

扫描 0.05~0.15 阈值范围，绘制速度 vs PSNR 曲线，确定推荐配置。

---

## 九、环境复现命令（更新）

```bash
conda activate viditq-osora
cd /home/rich/ViDiT-Q/examples/opensora1.2

export LD_LIBRARY_PATH="/home/rich/miniconda3/envs/viditq-osora/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=".:./Open-Sora:/home/rich/ViDiT-Q/kernels"
export HF_HUB_OFFLINE=1
export VIDITQ_XFORMERS_OP=cutlass

# === Benchmark (Phase 3c) ===
python /home/rich/ViDiT-Q/tools/profiling/benchmark.py --all --steps 10

# === Timestep Analysis (Phase 4a) ===
python /home/rich/ViDiT-Q/tools/profiling/analyze_timestep.py

# === W8A8 硬件推理 ===
python quant_inference.py configs/local_144p_120f_5s_w8a8_hardware.py

# === 便捷方式 ===
bash /home/rich/ViDiT-Q/tools/profiling/benchmark.sh --all
```

---

## 十、性能演进总览（截至 6.12）

| 阶段 | 推理路径 | Per step | vs FP16 |
|:-----|:---------|:--------:|:-------:|
| 起点 | W8A8 软件仿真 (FP64 Hadamard) | 9656ms | 0.05× |
| Phase 1 | W8A8 软件仿真 (FP16 Hadamard) | 1179ms | 0.61× |
| Phase 1 | W8A8 CUDA Kernel (weight only) | 511ms | 1.42× |
| Phase 2 V4 | W8A8 CUDA Kernel (full ViDiT-Q) | ~560ms | ~1.30× |
| **Phase 3c** | **W8A8 HW（标准化 benchmark）** | **447ms** | **1.19×** |
| Phase 4b (预计) | W8A8 HW + TeaCache | ~280ms (估) | ~1.9× (估) |

> Phase 3c 的 447ms 比 Phase 2 的 ~560ms 更快，主要因为 benchmark 使用无 hook 纯计时，且环境 torch/cuda 版本略有差异。
