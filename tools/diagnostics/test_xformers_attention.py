import torch
import xformers
import xformers.ops as xops


def print_env():
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("xformers:", xformers.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))


def test_case(name, batch, seq_len, heads, head_dim, dtype):
    print(f"\n=== {name} | B={batch}, S={seq_len}, H={heads}, D={head_dim}, dtype={dtype} ===")
    try:
        q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)

        torch.cuda.synchronize()
        out = xops.memory_efficient_attention(q, k, v)
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
    print_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this diagnostic must run on GPU.")

    torch.manual_seed(0)
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        test_case("tiny", batch=1, seq_len=256, heads=16, head_dim=64, dtype=dtype)
        test_case("small", batch=1, seq_len=1024, heads=16, head_dim=64, dtype=dtype)
        test_case("medium", batch=1, seq_len=4096, heads=16, head_dim=64, dtype=dtype)


if __name__ == "__main__":
    main()
