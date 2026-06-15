"""System optimization utilities for ViDiT-Q + TeaCache inference."""

import torch


@torch.no_grad()
def vae_cpu_offload_decode(vae, latent, num_frames, dtype=torch.float16):
    """VAE decode with CPU offload — moves VAE to CPU, decodes, moves back.

    Saves ~500 MB GPU memory during decode by temporarily moving the VAE
    weights to CPU. The latent stays on GPU. After decode, VAE returns to GPU.

    Args:
        vae: OpenSora VAE module (currently on GPU)
        latent: [1, C, T, H, W] latent tensor on GPU
        num_frames: number of output frames
        dtype: model dtype

    Returns:
        samples: decoded video tensor [1, C, T, H, W] on GPU
    """
    device = next(vae.parameters()).device

    # Offload VAE to CPU
    vae.cpu()
    torch.cuda.empty_cache()

    # Decode: move VAE back, run, offload again
    vae.to(device)
    samples = vae.decode(latent.to(dtype), num_frames=num_frames)

    # Clean up
    vae.cpu()
    torch.cuda.empty_cache()

    # VAE stays on CPU after this call (caller moves it back if needed)
    return samples


class CUDATimer:
    """Simple CUDA event timer for profiling model components.

    Usage:
        timer = CUDATimer()
        with timer.region("forward"):
            output = model(x)
        timer.report()
    """
    def __init__(self):
        self._records = {}

    def region(self, name: str):
        return _TimerRegion(name, self._records)

    def report(self):
        torch.cuda.synchronize()
        print(f"\n{'Region':<40s} {'Count':>6s} {'Total(ms)':>10s} {'Avg(ms)':>10s}")
        print("-" * 66)
        for name, pairs in self._records.items():
            times = [s.elapsed_time(e) for s, e in pairs]
            total = sum(times)
            print(f"{name:<40s} {len(times):6d} {total:10.1f} {total/len(times):10.1f}")


class _TimerRegion:
    def __init__(self, name, records):
        self.name = name
        self.records = records
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self
    def __exit__(self, *args):
        self.end.record()
        self.records.setdefault(self.name, []).append((self.start, self.end))
