# 长期候选环境记录：RTX 5080 + xFormers 0.0.33

本文档记录用于长期验证 OpenSora v1.2 / ViDiT-Q 的 RTX 5080 候选环境。该环境的目标是验证 Blackwell / `sm_120` 上较新的 PyTorch 与 xFormers 组合，尽量回到 xFormers `default` dispatch，而不是长期依赖旧环境中的强制 `cutlass` 补丁。

## 环境定位

- 短期可用主环境：`viditq-osora`
- 长期候选环境：`viditq-xf033-test`

两者不要混用。当前结论是：

```text
viditq-osora:
  torch 2.8.0 + xformers 0.0.32.post2
  需要 VIDITQ_XFORMERS_OP=cutlass

viditq-xf033-test:
  torch 2.9.1 + xformers 0.0.33.post2
  使用 VIDITQ_XFORMERS_OP=default
```

## 主机与 GPU

- 主机系统：Windows 11
- 运行环境：WSL2 中的 Ubuntu 22.04.5 LTS
- GPU：NVIDIA GeForce RTX 5080 Laptop GPU
- GPU 计算能力：`(12, 0)`
- Windows NVIDIA 驱动：596.36
- WSL 中 `nvidia-smi` 显示 CUDA Driver API：13.2

注意：WSL2 中不要安装 Linux NVIDIA Driver。PyTorch wheel 安装的 `nvidia-cudnn-cu12`、`nvidia-cublas-cu12` 等是 CUDA 用户态运行库，不是显卡驱动。

## Conda 环境

```bash
conda create -n viditq-xf033-test python=3.10 pip -y
conda activate viditq-xf033-test
```

环境路径：

```text
/home/rich/miniconda3/envs/viditq-xf033-test
```

## 核心 GPU 软件栈

目标版本：

```text
torch==2.9.1+cu128
torchvision==0.24.1+cu128
torchaudio==2.9.1+cu128
xformers==0.0.33.post2
triton==3.5.1
```

安装命令：

```bash
pip install \
  --timeout 1200 --retries 10 --resume-retries 10 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  --timeout 1200 --retries 10 --resume-retries 10 \
  xformers==0.0.33.post2 \
  --index-url https://download.pytorch.org/whl/cu128
```

核验命令：

```bash
python - <<'PY'
import torch, torchvision, torchaudio, xformers
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("xformers:", xformers.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

期望结果：

```text
torch: 2.9.1+cu128
torchvision: 0.24.1+cu128
torchaudio: 2.9.1+cu128
xformers: 0.0.33.post2
cuda: 12.8
cuda available: True
gpu: NVIDIA GeForce RTX 5080 Laptop GPU
capability: (12, 0)
```

## OpenSora / ViDiT-Q 关键包

当前已验证的关键版本：

```text
diffusers==0.27.2
transformers==4.39.3
huggingface_hub==0.21.4
mmengine==0.10.7
colossalai==0.4.0
qdiff==0.0.0 editable
```

不要直接安装 OpenSora 的完整 `requirements/requirements.txt`。即使使用 `--no-deps`，requirements 中的直接条目仍可能安装 `torch==2.2.2`，源码包构建过程也可能触发 build isolation。

建议手动安装依赖，并且谨慎使用 `--no-deps`：

```bash
pip install --no-deps \
  numpy==1.26.4 \
  packaging \
  filelock \
  fsspec \
  typing_extensions \
  pyyaml \
  requests \
  tqdm \
  regex \
  safetensors \
  sentencepiece \
  tokenizers==0.15.2 \
  protobuf \
  einops \
  ftfy \
  pillow \
  opencv-python==4.11.0.86 \
  av \
  imageio \
  imageio-ffmpeg

pip install --no-deps \
  diffusers==0.27.2 \
  transformers==4.39.3 \
  huggingface_hub==0.21.4 \
  accelerate==0.29.2 \
  timm==0.9.16 \
  mmengine==0.10.7 \
  rotary_embedding_torch==0.5.3

pip install --no-deps --no-build-isolation colossalai==0.4.0

pip install --no-deps \
  addict \
  yapf \
  rich \
  termcolor \
  matplotlib \
  pandas \
  beautifulsoup4 \
  bs4 \
  psutil \
  tensorboard \
  wandb \
  peft==0.13.2 \
  bitsandbytes==0.47.0 \
  ninja

cd /home/rich/ViDiT-Q
pip install -e ./quant_utils --no-deps
```

## 禁止覆盖的包

后续安装任何依赖时，如果 pip 输出中出现以下包即将被安装、卸载或降级，应立即停止：

```text
torch
torchvision
torchaudio
xformers
triton
nvidia-cuda-*
nvidia-cudnn-*
nvidia-cublas-*
nvidia-cufft-*
nvidia-curand-*
nvidia-cusolver-*
nvidia-cusparse-*
nvidia-nccl-*
```

尤其要警惕：

```text
Collecting torch==2.2.2
Attempting uninstall: torch
Successfully installed torch-2.2.2 torchvision-0.17.2 triton-2.2.0
```

这表示环境已被 OpenSora 原始依赖污染，需要重新安装核心 GPU 软件栈。

## HuggingFace 缓存

继续使用项目内缓存目录：

```bash
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
```

运行 OpenSora 脚本时建议同时设置：

```bash
export PYTHONPATH=/home/rich/ViDiT-Q/examples/opensora1.2/Open-Sora
```

## xFormers 验证结论

在 `torch==2.9.1+cu128` 与 `xformers==0.0.33.post2` 下，RTX 5080 的 xFormers `default` dispatch 已通过测试。

最小 attention 测试：

```bash
cd /home/rich/ViDiT-Q
CUDA_LAUNCH_BLOCKING=1 python tools/diagnostics/test_xformers_attention.py
```

结果：

```text
fp16 tiny/small/medium: OK
bf16 tiny/small/medium: OK
fp32: 不支持，这是 xFormers fused attention 的正常限制
```

backend 测试：

```bash
CUDA_LAUNCH_BLOCKING=1 python tools/diagnostics/test_xformers_ops.py --op default --dtype fp16 --seq-len 1024
CUDA_LAUNCH_BLOCKING=1 python tools/diagnostics/test_xformers_ops.py --op flash --dtype fp16 --seq-len 1024
CUDA_LAUNCH_BLOCKING=1 python tools/diagnostics/test_xformers_ops.py --op cutlass --dtype fp16 --seq-len 1024
```

结果：

```text
default: OK
flash: OK
cutlass: FAILED
```

因此该环境必须使用：

```bash
export VIDITQ_XFORMERS_OP=default
```

不要在该环境中强制 `cutlass`。

OpenSora STDiT3-XL/2 真实形状验证：

```text
self-attention:
  q/k/v shape = [1, 576, 16, 72]
  default OK, finite=True

cross-attention + BlockDiagonalMask:
  q shape = [1, 1152, 16, 72]
  k/v text length = [300, 260]
  default OK, finite=True
```

## OpenSora 17 帧整链验证

注意：不要使用 `num_frames=16` 配合 `use_timestep_transform=True`。OpenSora 的 `timestep_transform()` 中存在：

```python
num_frames = model_kwargs["num_frames"] // 17 * 5
```

当 `num_frames=16` 时会得到 0，并在第一个 timestep 处产生 `0 / 0`，导致 model pred 全 NaN。当前使用 `local_144p_17f.py`。

运行命令：

```bash
conda activate viditq-xf033-test
cd /home/rich/ViDiT-Q/examples/opensora1.2

export PYTHONPATH=/home/rich/ViDiT-Q/examples/opensora1.2/Open-Sora
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
export VIDITQ_XFORMERS_OP=default

python precompute_text_embeds.py configs/local_144p_17f.py

python fp_inference.py configs/local_144p_17f.py \
  --save-dir /home/rich/ViDiT-Q/.local/outputs/opensora_xf033_fp16_144p_17f_10steps_default
```

已验证输出：

```text
/home/rich/ViDiT-Q/.local/outputs/opensora_xf033_fp16_144p_17f_10steps_default/sample_0000.mp4
```

视频检查结果：

```text
size: 11121 bytes
fps: 24
frame_count: 17
resolution: 192 x 192
frames_read: 17
pixel mean: 108.68 ~ 109.17
pixel range: 0 ~ 255
```

首帧为海岸悬崖图像，非黑帧。

## 当前结论

`viditq-xf033-test` 是当前更适合 RTX 5080 / Blackwell 的长期候选环境：

```text
torch 2.9.1+cu128
xformers 0.0.33.post2
VIDITQ_XFORMERS_OP=default
num_frames=17
```

该环境已通过：

1. PyTorch CUDA 可用性验证。
2. xFormers default / flash backend 验证。
3. OpenSora 真实 attention shape 验证。
4. OpenSora 17 帧 FP16 推理整链验证。

后续进行 TeaCache 或 ViDiT-Q 量化实验时，建议优先在该环境上继续推进；短期主环境仅作为对照和回退方案。
