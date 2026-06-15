import argparse

import torch
import xformers
import xformers.ops as xops
from xformers.ops import fmha


def print_env():
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("xformers:", xformers.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))


def try_op(name, op, q, k, v):
    print(f"\n=== op: {name} ===")
    try:
        torch.cuda.synchronize()
        out = xops.memory_efficient_attention(q, k, v, op=op)
        torch.cuda.synchronize()
        print("OK:", tuple(out.shape), out.dtype)
        print("finite:", torch.isfinite(out).all().item())
        print("max_abs:", out.abs().max().item())
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print("FAILED:", repr(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["default", "cutlass", "flash", "all"], default="all")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    print_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this diagnostic must run on GPU.")

    torch.manual_seed(0)
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    q = torch.randn(1, args.seq_len, 16, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, args.seq_len, 16, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, args.seq_len, 16, 64, device="cuda", dtype=dtype)

    ops_to_try = [
        ("default", None),
        ("cutlass", (fmha.cutlass.FwOp, fmha.cutlass.BwOp)),
        ("flash", (fmha.flash.FwOp, fmha.flash.BwOp)),
    ]

    for name, op in ops_to_try:
        if args.op != "all" and args.op != name:
            continue
        try_op(name, op, q, k, v)


if __name__ == "__main__":
    main()
