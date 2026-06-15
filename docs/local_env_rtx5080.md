# 本地环境记录：RTX 5080 + WSL2

本文档记录当前工作站用于运行 ViDiT-Q + OpenSora v1.2 的本地环境。由于本机使用的是 RTX 5080 Laptop GPU，计算能力为 `12.0`，因此环境配置有意区别于 OpenSora 原始 CUDA 12.1 环境。

## 主机与 GPU

- 主机系统：Windows 11
- 运行环境：WSL2 中的 Ubuntu 22.04.5 LTS
- WSL 内核：`6.6.114.1-microsoft-standard-WSL2`
- GPU：NVIDIA GeForce RTX 5080 Laptop GPU
- 显存：16 GB
- Windows NVIDIA 驱动：596.36
- WSL 中 `nvidia-smi` 显示的 CUDA Driver API：13.2
- PyTorch 可见的 GPU 计算能力：`(12, 0)`

WSL2 的 GPU 透传需要能正常看到：

```bash
nvidia-smi
ls -l /dev/dxg
```

## Conda 环境

- 环境名称：`viditq-osora`
- Python 版本：3.10
- 环境路径：`/home/rich/miniconda3/envs/viditq-osora`

创建和激活环境：

```bash
conda create -n viditq-osora python=3.10 -y
conda activate viditq-osora
```

## 核心 GPU 软件栈

OpenSora 原始依赖使用 `torch==2.2.2+cu121`，但这套环境不适合 RTX 5080 / Blackwell 架构。本地环境改用 PyTorch CUDA 12.8 wheel：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu128
```

当前记录的核心版本：

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
xformers==0.0.32.post2
triton==3.4.0
numpy==1.26.4
```

## OpenSora / ViDiT-Q 软件栈

安装 OpenSora 时要避免 pip 把 PyTorch 降级：

```bash
cd /home/rich/ViDiT-Q/examples/opensora1.2/Open-Sora
pip install -r requirements/requirements.txt

# 如果依赖解析把 PyTorch 降级了，先重新安装上面的 GPU 软件栈，
# 然后用 --no-deps 只安装 OpenSora 本体：
pip install -v -e . --no-deps
```

安装 ViDiT-Q 的量化工具包：

```bash
cd /home/rich/ViDiT-Q/quant_utils
pip install -e .
```

当前记录的关键包版本：

```text
opensora==1.2.0 (editable)
qdiff==0.0.0 (editable)
colossalai==0.4.0
mmengine==0.10.7
timm==0.9.16
rotary_embedding_torch==0.5.3
diffusers==0.27.2
accelerate==0.29.2
transformers==4.39.3
tokenizers==0.15.2
safetensors==0.7.0
```

## HuggingFace Hub 版本固定

固定使用：

```text
huggingface-hub==0.21.4
```

原因：

- `diffusers==0.27.2` 会导入 `huggingface_hub.cached_download`。
- 较新的 `huggingface_hub`，例如 `0.36.2`，已经移除了该符号。
- `0.21.4` 与 OpenSora 较早期依赖族更一致，同时仍满足 `transformers==4.39.3` 的最低版本要求。

安装命令：

```bash
pip install --force-reinstall "huggingface_hub==0.21.4"
```

验证命令：

```bash
python - <<'PY'
from huggingface_hub import cached_download
import diffusers, transformers, huggingface_hub
print(huggingface_hub.__version__, diffusers.__version__, transformers.__version__)
print("cached_download ok")
PY
```

当前存在但暂不阻塞主线任务的依赖警告：

```text
colossalai 0.4.0 requires torch<2.3.0,>=2.1.0
gradio 6.14.0 requires huggingface-hub>=0.33.5
peft 0.19.1 requires huggingface_hub>=0.25.0
```

说明：

- 当前命令行推理和 PTQ 主线不使用 `gradio`。
- 当前命令行推理和 PTQ 主线不使用 `peft`。
- `colossalai` 会被 OpenSora 导入，但单卡推理不应进入其分布式训练路径。
- 因此当前优先保证 RTX 5080 可用的 PyTorch 栈和 OpenSora 推理依赖可导入。

## 模型与缓存目录

本项目使用项目内本地资产目录，而不是把模型散放到系统路径：

```text
/home/rich/ViDiT-Q/.local/
  models/
    hpcai-tech/
      OpenSora-STDiT-v3/
      OpenSora-VAE-v1.2/
    DeepFloyd/
      t5-v1_1-xxl/
  cache/
    huggingface/
  outputs/
```

建议设置环境变量：

```bash
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
```

当前环境中观察到默认缓存路径 `/home/rich/.cache/huggingface/hub` 不可写，因此建议优先使用项目内 `.local/cache/huggingface`。

运行 OpenSora/ViDiT-Q 脚本前，建议在当前 shell 中显式设置这两个变量：

```bash
cd /home/rich/ViDiT-Q/examples/opensora1.2
export HF_HOME=/home/rich/ViDiT-Q/.local/cache/huggingface
export HF_HUB_CACHE=/home/rich/ViDiT-Q/.local/cache/huggingface/hub
```

这样 `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers` 等间接依赖模型会缓存到项目内目录，而不是写入系统默认缓存目录。

## CUDA Kernel 状态

当前尚未启用 ViDiT-Q CUDA 扩展。

原因：

- 初始检查时未发现 `nvcc`。
- `kernels/setup.py` 当前只支持以下架构：`8.0`、`8.6`、`8.7`、`8.9`、`9.0`。
- RTX 5080 的计算能力是 `12.0`，因此在编译硬件 kernel 前，需要先更新 kernel 构建脚本对架构的支持。

当前阶段先使用软件仿真路径：

```text
hardware = False
```

## 当前工作计划

1. 先运行低分辨率、低帧数的 FP16 最小冒烟推理。
2. 生成 calibration data。
3. 跑 W8A8 软件仿真 PTQ。
4. 跑 W4A8 mixed precision 软件仿真 PTQ。
5. 在软件路径上加入 TeaCache。
6. 等算法路径稳定后，再回头处理 RTX 5080 的 CUDA kernel 支持。
