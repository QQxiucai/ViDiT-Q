#!/bin/bash
# Quick benchmark launcher for ViDiT-Q Phase 3c.
# Usage:
#   ./tools/profiling/benchmark.sh              # run all 3 modes
#   ./tools/profiling/benchmark.sh --fp16       # FP16 only
#   ./tools/profiling/benchmark.sh --w8a8-hw    # W8A8 hardware only
#   ./tools/profiling/benchmark.sh --steps 20   # 20 steps per mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/rich/miniconda3/envs/viditq-osora/bin/python"
TORCH_LIB="/home/rich/miniconda3/envs/viditq-osora/lib/python3.10/site-packages/torch/lib"
WORKDIR="$PROJECT_ROOT/examples/opensora1.2"

export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORKDIR}:${WORKDIR}/Open-Sora:${PROJECT_ROOT}/kernels:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export VIDITQ_XFORMERS_OP=cutlass  # RTX 5080 / sm120 fix: avoid xformers Hopper kernel crash

cd "$WORKDIR"
exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/profiling/benchmark.py" "${@:---all}"
