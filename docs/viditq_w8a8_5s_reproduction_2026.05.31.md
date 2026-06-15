# ViDiT-Q W8A8 5s 复现记录 2026.05.31

## 结论

- 后续 OpenSora / ViDiT-Q 推理实验统一以 5s 视频作为默认工作时长。
- 当前 5s 基准配置为：
  - `num_frames = 120`
  - `fps = 24`
  - `save_fps = 24`
  - `resolution = "144p"`
  - `num_sampling_steps = 10`
  - `dtype = "fp16"`
  - `precompute_text_embeds = True`
- 已在长期环境 `viditq-xf033-test` 中跑通 W8A8 软件量化链路：
  - calibration 数据生成：成功
  - PTQ 量化参数生成：成功
  - W8A8 软件量化推理：成功

## 使用环境

```bash
conda activate viditq-xf033-test
cd /home/rich/ViDiT-Q/examples/opensora1.2

export PYTHONPATH=/home/rich/ViDiT-Q/examples/opensora1.2/Open-Sora
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
export VIDITQ_XFORMERS_OP=default
```

关键环境：

```text
torch: 2.9.1+cu128
xformers: 0.0.33.post2
GPU: NVIDIA GeForce RTX 5080 Laptop GPU
capability: (12, 0)
```

## 新增配置

- `/home/rich/ViDiT-Q/examples/opensora1.2/configs/local_144p_120f_5s.py`
- `/home/rich/ViDiT-Q/examples/opensora1.2/configs/w8a8_120f_5s.yaml`

输出目录：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

## 运行命令

### 1. 生成 calibration 数据

```bash
python get_calib_data.py configs/local_144p_120f_5s.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

产物：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps/calib_data.pth
```

已验证：

```text
calib_data.pth: 13M
hooked layers: 400
example tensor: t_embedder.mlp.0 (10, 256) torch.float16
```

### 2. 运行 PTQ

```bash
python ptq.py configs/local_144p_120f_5s.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

产物：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps/quant_params.pth
```

已验证：

```text
quant_params.pth: 4.2M
STDiT3 checkpoint path resolved to:
/home/rich/ViDiT-Q/.local/models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors
```

### 3. 运行 W8A8 软件量化推理

```bash
python quant_inference.py configs/local_144p_120f_5s.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

产物：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps/sample_0000.mp4
```

推理进度条耗时：

```text
96.78s
```

视频检查：

```text
shape: (120, 192, 192, 3)
fps: 24.0
duration_sec: 5.0
pixel range: 0 ~ 255
mean: 171.1961
```

## 已做的必要代码适配

- `examples/opensora1.2/models/quant_opensora.py`
  - 支持本地 HuggingFace `model.safetensors` 目录解析。
  - 将 `viditq_extension` 相关 CUDA kernel 导入改为 lazy import，避免 `hardware=False` 软件模拟量化被可选 kernel 包阻塞。
- `examples/opensora1.2/ptq.py`
  - `STDiT3Config` 中 `qk_norm`、`enable_flash_attn`、`enable_layernorm_kernel` 改为读取配置。
- `examples/opensora1.2/quant_inference.py`
  - 同步读取配置中的 `qk_norm`、`enable_flash_attn`、`enable_layernorm_kernel`。

## 注意事项

- 后续实验不要再默认使用 17 帧配置，除非只是做快速 smoke test。
- 后续有效推理、量化、TeaCache 对比实验默认使用：
  ```text
  configs/local_144p_120f_5s.py
  ```
- 如果修改 `num_frames`、`resolution`、`num_sampling_steps` 等关键推理条件，建议重新生成 calibration 数据和 PTQ 参数。
- `ptq.py` 当前会额外生成调试文件：
  ```text
  /home/rich/ViDiT-Q/examples/opensora1.2/spatial_blocks.11.pth
  ```
  该文件不是 W8A8 推理必须产物，后续可考虑把保存逻辑改为可选。

## FP16 vs W8A8 5s 对比

对比配置：

```text
resolution: 144p
num_frames: 120
save_fps: 24
duration: 5.0s
num_sampling_steps: 10
seed: 42
prompt_path: ./prompts.txt
```

FP16 输出目录：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_fp16_144p_120f_5s_10steps
```

W8A8 输出目录：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

### FP16

运行命令：

```bash
python fp_inference.py configs/local_144p_120f_5s.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_fp16_144p_120f_5s_10steps
```

推理进度条耗时：

```text
9.29s
```

视频检查：

```text
file size: 90100 bytes
shape: (120, 192, 192, 3)
fps: 24.0
duration_sec: 5.0
pixel range: 0 ~ 255
mean: 174.3903
std: 77.5248
```

### W8A8

推理命令：

```bash
python quant_inference.py configs/local_144p_120f_5s.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_144p_120f_5s_10steps
```

推理进度条耗时：

```text
96.78s
```

视频检查：

```text
file size: 96465 bytes
shape: (120, 192, 192, 3)
fps: 24.0
duration_sec: 5.0
pixel range: 0 ~ 255
mean: 171.1961
std: 78.0733
```

### 像素级 sanity check

直接对两个压缩后 mp4 解码结果做逐像素差异：

```text
abs diff mean: 11.2706
abs diff max: 214.0
RMSE: 20.8983
```

说明：

- W8A8 当前为软件模拟量化，不是 CUDA kernel 实际 INT 推理，因此速度显著慢于 FP16 是预期结果。
- 当前对比只用于确认 5s W8A8 输出可用、非黑帧、基本统计正常。
- 后续质量评估需要使用更正式的视频指标或人工检查，不能只依赖逐像素差异。

## W4A8 Mixed Precision 5s 复现

新增配置：

```text
examples/opensora1.2/configs/local_144p_120f_5s_w4a8_mp.py
examples/opensora1.2/configs/w4a8_mp_120f_5s.yaml
```

输出目录：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w4a8_mp_144p_120f_5s_10steps
```

产物：

```text
calib_data.pth: 13M
quant_params.pth: 8.1M
sample_0000.mp4: 90K
```

运行命令：

```bash
python ptq.py configs/local_144p_120f_5s_w4a8_mp.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w4a8_mp_144p_120f_5s_10steps

python quant_inference.py configs/local_144p_120f_5s_w4a8_mp.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w4a8_mp_144p_120f_5s_10steps
```

耗时记录：

```text
calibration progress: 10.20s
quant inference progress: 86.52s
```

视频检查：

```text
file size: 91150 bytes
shape: (120, 192, 192, 3)
fps: 24.0
duration_sec: 5.0
pixel range: 0 ~ 255
mean: 164.7605
std: 80.9579
```

### FP16 vs W4A8 Mixed Precision

直接对两个压缩后 mp4 解码结果做逐像素差异：

```text
abs diff mean: 13.0979
abs diff max: 217.0
RMSE: 23.9006
```

说明：

- W4A8 mixed precision 已完成 5s 复现，输出视频长度、帧率、分辨率均与 FP16/W8A8 对齐。
- 当前 W4A8 mixed precision 仍是软件模拟量化路径，耗时不代表真实 INT kernel 性能。
- 从 mp4 解码后的 sanity check 看，W4A8 mixed precision 相比 FP16 的逐像素差异略高于 W8A8，符合更低权重量化 bitwidth 的预期。

## W8A8 CUDA Kernel 5s 初步复现

日期：2026-06-01

环境：

```text
conda env: viditq-xf033-test
GPU: RTX 5080 Laptop, capability sm_120
torch: 2.9.1+cu128
nvcc: 12.8.93
```

新增配置：

```text
examples/opensora1.2/configs/local_144p_120f_5s_w8a8_hardware.py
```

输出目录：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_hardware_144p_120f_5s_10steps
```

产物：

```text
quant_params.pth: 4.2M
int_weight.pt: 1.7G
sample_0000.mp4: 99K
```

运行命令：

```bash
python quant_inference.py configs/local_144p_120f_5s_w8a8_hardware.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_w8a8_hardware_144p_120f_5s_10steps
```

推理进度条耗时：

```text
121.08s
```

视频检查：

```text
file size: 101024 bytes
shape: (120, 192, 192, 3)
fps: 24.0
duration_sec: 5.0
pixel range: 0 ~ 255
mean: 171.8518
std: 75.4243
```

### 已做的硬件路径适配

- 在 `viditq-xf033-test` 内安装 conda 版 CUDA 编译组件：
  ```text
  cuda-nvcc 12.8.93
  cuda-cudart-dev 12.8.90
  ```
- `kernels/setup.py`
  - 增加 `12.0` / `sm_120` 支持。
  - 修正 architecture 字符串生成逻辑，避免 `12.0` 被错误拼成 `sm_10`。
  - 自动加入 PyTorch wheel 自带的 `nvidia/*/include`，解决 `cusparse.h` 头文件查找问题。
- `examples/opensora1.2/models/quant_opensora_cuda.py`
  - 硬件权重导出时跳过仍走普通 `F.linear` 的层：
    ```text
    *.attn.qkv
    *.attn.proj
    *.cross_attn.kv_linear
    ```
  - 这些层保留 FP16 / 软件量化权重；真正接入 W8A8 CUDA Linear 的 `cross_attn.q_linear`、`cross_attn.proj`、`mlp.fc1`、`mlp.fc2` 导出为 int8。
- `examples/opensora1.2/models/quant_opensora.py`
  - 硬件 block refactor 时复用原软件模拟中的 `cross_attn.kv_linear`。
  - 原因：硬件版 `MultiHeadCrossAttentionWithCudaKernel` 默认创建的是普通 `nn.Linear`，会丢失软件模拟中 `QuantizedLinear` 对文本条件 K/V 的 activation quant 行为，导致输出严重偏亮、画面退化成彩色亮块。

### FP16 / W8A8 软件模拟 / W8A8 CUDA Kernel 对比

```text
fp16:
  size: 90100 bytes
  mean: 174.3903
  std: 77.5248

w8a8 software simulation:
  size: 96465 bytes
  mean: 171.1961
  std: 78.0733

w8a8 cuda kernel:
  size: 101024 bytes
  mean: 171.8518
  std: 75.4243
```

像素级 sanity check：

```text
fp16_vs_w8a8_sim:
  abs diff mean: 11.2706
  abs diff max: 214.0
  RMSE: 20.8983

fp16_vs_w8a8_hw:
  abs diff mean: 14.9535
  abs diff max: 213.0
  RMSE: 26.0349

w8a8_sim_vs_hw:
  abs diff mean: 9.2771
  abs diff max: 206.0
  RMSE: 16.0924
```

说明：

- W8A8 CUDA kernel 路径已在 RTX 5080 / `sm_120` 上完成 5s 推理，证明编译、导入、权重导出、block 替换、采样和视频保存链路可运行。
- 修复 `cross_attn.kv_linear` activation quant 缺失后，硬件输出统计已回到 W8A8 软件模拟附近，不再出现 mean 约 234 的明显偏亮异常。
- 当前硬件结果和软件模拟仍不是逐像素完全一致，后续如需继续收敛，应做 block 级 hook 对齐，优先比较 `cross_attn` 与 `mlp` 的 CUDA kernel 输出和软件模拟输出。
