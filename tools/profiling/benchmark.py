#!/usr/bin/env python3
"""
Phase 3c: ViDiT-Q Inference Benchmark Framework.

Compares FP16, W8A8 software simulation, and W8A8 CUDA kernel (hardware)
inference using CUDA event timing and GPU memory tracking.

Usage:
  cd examples/opensora1.2
  python ../../tools/profiling/benchmark.py --all

  # Individual modes:
  python ../../tools/profiling/benchmark.py --fp16
  python ../../tools/profiling/benchmark.py --w8a8-sim
  python ../../tools/profiling/benchmark.py --w8a8-hw
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from contextlib import contextmanager

import torch

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
OPEN_SORA_DIR = os.path.join(PROJECT_ROOT, "examples", "opensora1.2", "Open-Sora")
WORKDIR = os.path.join(PROJECT_ROOT, "examples", "opensora1.2")

# Ensure torch libs are findable (needed for viditq_extension .so files)
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = _torch_lib
elif _torch_lib not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{os.environ['LD_LIBRARY_PATH']}"

# Fix-up sys.path
for _p in [WORKDIR, OPEN_SORA_DIR, KERNELS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Environment fixes
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VIDITQ_XFORMERS_OP", "cutlass")  # RTX 5080 / sm120 stable xformers backend

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
BASE_CONFIG = os.path.join(WORKDIR, "configs", "local_144p_120f_5s.py")
HW_CONFIG = os.path.join(WORKDIR, "configs", "local_144p_120f_5s_w8a8_hardware.py")
PRECOMPUTED_EMBEDS = os.path.join(WORKDIR, "precomputed_text_embeds.pth")

# ---------------------------------------------------------------------------
# CUDA timer
# ---------------------------------------------------------------------------
class CUDATimer:
    """Collects CUDA event pairs per named region."""

    def __init__(self):
        self._records: dict = defaultdict(list)

    @contextmanager
    def region(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._records[name].append((start, end))

    def report(self, sort_by="total_ms"):
        torch.cuda.synchronize()
        rows = []
        for name, pairs in self._records.items():
            times_ms = [s.elapsed_time(e) for s, e in pairs]
            total = sum(times_ms)
            count = len(times_ms)
            avg = total / count if count else 0.0
            rows.append((name, count, total, avg, min(times_ms), max(times_ms)))

        if sort_by == "total_ms":
            rows.sort(key=lambda r: -r[2])

        print("\n" + "=" * 100)
        print(f"{'Region':55s} {'Count':>6s} {'Total(ms)':>12s} {'Avg(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s}")
        print("-" * 100)
        for name, count, total, avg, vmin, vmax in rows:
            print(f"{name:55s} {count:6d} {total:12.3f} {avg:10.3f} {vmin:10.3f} {vmax:10.3f}")
        print("=" * 100)

        # Category rollup
        cat_totals = defaultdict(float)
        for name, _, total, _, _, _ in rows:
            cat = name.split("/")[0] if "/" in name else name
            cat_totals[cat] += total
        print("\n--- Category totals ---")
        for cat, t in sorted(cat_totals.items(), key=lambda x: -x[1]):
            print(f"  {cat:50s} {t:10.3f} ms")
        print()

        return rows

    def reset(self):
        self._records.clear()


_timer = CUDATimer()


# ---------------------------------------------------------------------------
# Profiling hooks (block-level)
# ---------------------------------------------------------------------------
def install_block_hooks(model, prefix="model"):
    """Register forward pre/post hooks on spatial/temporal blocks."""
    handles = []
    for group_name in ["spatial_blocks", "temporal_blocks"]:
        blocks = getattr(model, group_name, None)
        if blocks is None:
            continue
        for i, block in enumerate(blocks):
            label = f"{prefix}/{group_name}.{i:02d}/full"
            h_pre = block.register_forward_pre_hook(_make_pre_hook(label))
            h_post = block.register_forward_hook(_make_post_hook(label))
            handles.extend([h_pre, h_post])

            for sub in ["attn", "cross_attn", "mlp"]:
                sub_mod = getattr(block, sub, None)
                if sub_mod is None:
                    continue
                sub_label = f"{prefix}/{group_name}.{i:02d}/{sub}"
                h_pre = sub_mod.register_forward_pre_hook(_make_pre_hook(sub_label))
                h_post = sub_mod.register_forward_hook(_make_post_hook(sub_label))
                handles.extend([h_pre, h_post])

    if hasattr(model, "final_layer"):
        h_pre = model.final_layer.register_forward_pre_hook(
            _make_pre_hook(f"{prefix}/final_layer"))
        h_post = model.final_layer.register_forward_hook(
            _make_post_hook(f"{prefix}/final_layer"))
        handles.extend([h_pre, h_post])

    return handles


def _make_pre_hook(name):
    def pre_hook(module, input):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module._profile_start = start
        module._profile_end = end
    return pre_hook


def _make_post_hook(name):
    def post_hook(module, input, output):
        module._profile_end.record()
        _timer._records[name].append((module._profile_start, module._profile_end))
    return post_hook


# ---------------------------------------------------------------------------
# Latent size computation (avoids network-requiring VAE build)
# ---------------------------------------------------------------------------
def _compute_latent_size(image_size: tuple, num_frames: int, micro_frame_size: int = 17):
    """Compute VAE latent size for OpenSoraVAE_V1_2 without building the VAE.

    VAE architecture constants (hard-coded for OpenSora VAE v1.2):
      - Spatial VAE (PixArt AutoencoderKL):   patch_size = (1, 8, 8)
      - Temporal VAE (VAE_Temporal_SD):       time_downsample_factor = 4
                                               patch_size = (4, 1, 1)
                                               temporal_downsample = (True, True, False)
    """
    SPATIAL_PATCH = (1, 8, 8)
    TEMPORAL_TS_DOWNSAMPLE = 4
    TEMPORAL_PATCH_T = 4

    def _spatial(tsz):
        return [tsz[0] // SPATIAL_PATCH[0],
                tsz[1] // SPATIAL_PATCH[1],
                tsz[2] // SPATIAL_PATCH[2]]

    def _temporal(tsz):
        res = []
        for i in range(3):
            if tsz[i] is None:
                res.append(None)
            elif i == 0:
                pad = 0 if (tsz[0] % TEMPORAL_TS_DOWNSAMPLE == 0) \
                      else TEMPORAL_TS_DOWNSAMPLE - tsz[0] % TEMPORAL_TS_DOWNSAMPLE
                res.append((tsz[0] + pad) // TEMPORAL_PATCH_T)
            else:
                res.append(tsz[i] // 1)
        return res

    if micro_frame_size is None or num_frames is None:
        return _temporal(_spatial([num_frames, image_size[0], image_size[1]]))

    sub_in = [micro_frame_size, image_size[0], image_size[1]]
    sub_lat = _temporal(_spatial(sub_in))
    sub_lat[0] = sub_lat[0] * (num_frames // micro_frame_size)

    remain_t = num_frames % micro_frame_size
    if remain_t > 0:
        remain_lat = _temporal([remain_t, None, None])
        sub_lat[0] += remain_lat[0]

    return sub_lat


# OpenSoraVAE_V1_2  out_channels (fixed for this model)
_VAE_OUT_CHANNELS = 4


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(cfg, device, dtype, mode: str):
    """Build the model for a given profiling mode.

    Args:
        cfg: mmengine Config object
        device: torch device
        dtype: torch dtype
        mode: "fp16", "w8a8_sim", or "w8a8_hw"
    """
    from opensora.models.stdit.stdit3 import STDiT3Config
    from opensora.datasets.aspect import get_image_size, get_num_frames
    from omegaconf import OmegaConf, ListConfig

    # Compute latent size without building the VAE (avoids network access)
    image_size = get_image_size(cfg.get("resolution"), cfg.get("aspect_ratio"))
    num_frames = get_num_frames(cfg.num_frames)
    latent_size = _compute_latent_size(image_size, num_frames, micro_frame_size=17)

    # STDiT3 config
    model_config = STDiT3Config(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        qk_norm=cfg.model.get("qk_norm", True),
        enable_flash_attn=cfg.model.get("enable_flash_attn", False),
        enable_layernorm_kernel=cfg.model.get("enable_layernorm_kernel", False),
        input_size=latent_size,
        in_channels=_VAE_OUT_CHANNELS,
        caption_channels=4096,  # precomputed embed dim
        model_max_length=300,
        enable_sequence_parallelism=False,
    )

    model_path = cfg.get("model_path", os.path.join(PROJECT_ROOT, ".local", "models"))
    from_pretrained = os.path.join(model_path, "hpcai-tech/OpenSora-STDiT-v3")

    # Resolve checkpoint path
    safetensors_path = os.path.join(from_pretrained, "model.safetensors")
    checkpoint_path = safetensors_path if os.path.isfile(safetensors_path) else from_pretrained

    if mode == "fp16":
        # Pure FP16 baseline — no quantization
        from opensora.models.stdit.stdit3 import STDiT3
        from opensora.utils.ckpt_utils import load_checkpoint
        model = STDiT3(model_config)
        load_checkpoint(model, checkpoint_path)
        model = model.to(device, dtype).eval()

    else:
        # Quantized model (W8A8 SIM or HW)
        ptq_config_file = cfg.get("ptq_config", None)
        assert ptq_config_file is not None, f"ptq_config required for {mode}"
        quant_config = OmegaConf.load(ptq_config_file) if isinstance(ptq_config_file, str) else ptq_config_file

        from models.quant_opensora import QuantOpenSora
        model = QuantOpenSora(quant_config, model_config, checkpoint_path).to(device, dtype).eval()
        model.config = model_config

        # Load quant params
        if mode == "w8a8_hw":
            save_path = os.path.join(cfg.save_dir, "int_weight.pt")
            quant_ckpt = torch.load(
                os.path.join(cfg.save_dir, "quant_params.pth"),
                weights_only=True, map_location="cuda")
            model.load_quant_param_dict(quant_ckpt)

            # bitwidth_refactor AFTER load_quant_param_dict (load resets n_bits)
            qc = model.quant_config
            if_mixed = (isinstance(qc.weight.n_bits, ListConfig) or
                       isinstance(qc.act.n_bits, ListConfig))
            if if_mixed:
                model.bitwidth_refactor()

            model.quantize_and_save_weight(save_path=save_path)
            model.hardware_forward_refactor(load_path=save_path)
        else:
            # Software simulation
            quant_ckpt = torch.load(
                os.path.join(cfg.save_dir, "quant_params.pth"), weights_only=True)
            model.load_quant_param_dict(quant_ckpt)

        model.set_init_done()

    return model, latent_size, num_frames


# ---------------------------------------------------------------------------
# Profiling runner
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_profile(model, latent_size, num_frames, cfg, device, dtype,
                num_steps: int = 10, warmup_steps: int = 2):
    """Run inference and collect timings + memory.

    Returns:
        dict with keys: wall_time_s, cuda_event_ms, peak_mem_mb, static_mem_mb
    """
    from opensora.registry import SCHEDULERS, build_module
    from qdiff.utils import seed_everything

    # Load precomputed text embeds
    save_d = torch.load(PRECOMPUTED_EMBEDS, map_location=device, weights_only=True)
    model_args = save_d["model_args"]

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    seed_everything(cfg.get("seed", 1024))
    z = torch.randn(1, _VAE_OUT_CHANNELS, *latent_size, device=device, dtype=dtype)

    # Build timesteps
    timesteps = [(1.0 - i / scheduler.num_sampling_steps) * scheduler.num_timesteps
                 for i in range(scheduler.num_sampling_steps)]
    if scheduler.use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    if scheduler.use_timestep_transform:
        from opensora.schedulers.rf.rectified_flow import timestep_transform
        timesteps = [timestep_transform(t, model_args, num_timesteps=scheduler.num_timesteps)
                     for t in timesteps]

    guidance_scale = cfg.scheduler.get("cfg_scale", 7.0)
    total_steps = min(num_steps + warmup_steps, len(timesteps))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print(f"  Warmup: {warmup_steps} steps, Profile: {num_steps} steps")
    wall_start = time.time()

    for i in range(total_steps):
        t = timesteps[i]
        if i == warmup_steps:
            # End warmup — reset timers and memory stats
            torch.cuda.synchronize()
            wall_start = time.time()
            _timer.reset()

        with _timer.region(f"step_{i:02d}/total"):
            # CFG: batch doubler
            z_in = torch.cat([z, z], 0)
            t_in = torch.cat([t, t], 0)

            with _timer.region(f"step_{i:02d}/model_forward"):
                pred = model(z_in, t_in, **model_args)

            pred = pred.chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if i < len(timesteps) - 1:
                dt = timesteps[i] - timesteps[i + 1]
            else:
                dt = timesteps[i]
            dt = dt / scheduler.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

    torch.cuda.synchronize()
    wall_time = time.time() - wall_start

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    static_mem = torch.cuda.memory_allocated() / (1024 ** 2)    # MB

    # Collect CUDA event totals
    event_total_ms = sum(
        sum(s.elapsed_time(e) for s, e in pairs)
        for pairs in _timer._records.values()
    )

    return {
        "wall_time_s": wall_time,
        "cuda_event_total_ms": event_total_ms,
        "peak_mem_mb": peak_mem,
        "static_mem_mb": static_mem,
        "num_steps": num_steps,
    }


# ---------------------------------------------------------------------------
# Cross-mode summary formatter
# ---------------------------------------------------------------------------
def print_summary(results: dict):
    """Print a clean comparison table across all profiled modes."""
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " ViDiT-Q Inference Benchmark — Comparison Summary".center(96) + "║")
    print("╠" + "═" * 98 + "╣")

    # Header
    print("║ {:22s} │ {:14s} │ {:14s} │ {:14s} │ {:22s} ║".format(
        "Metric", "FP16", "W8A8-SIM", "W8A8-HW", "HW vs FP16 Speedup"))
    print("╠" + "═" * 98 + "╣")

    fp16 = results.get("fp16", {})
    sim = results.get("w8a8_sim", {})
    hw = results.get("w8a8_hw", {})

    def _val(d, key, fmt=".2f"):
        v = d.get(key)
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    def _step_time(d):
        n = d.get("num_steps", 1)
        return d.get("wall_time_s", 0) / n if n else float("nan")

    # Per-step latency
    fp16_step = _step_time(fp16)
    hw_step = _step_time(hw)
    sim_step = _step_time(sim)
    speedup = fp16_step / hw_step if hw_step and fp16_step else 0

    for label, fp16_key, sim_key, hw_key, fmt in [
        ("Wall time (total, s)", "wall_time_s", "wall_time_s", "wall_time_s", ".2f"),
        ("Per-step latency (s)", None, None, None, None),  # handled separately
        ("Per-step speedup", None, None, None, None),
        ("CUDA event total (ms)", "cuda_event_total_ms", "cuda_event_total_ms", "cuda_event_total_ms", ".1f"),
        ("Peak GPU memory (MB)", "peak_mem_mb", "peak_mem_mb", "peak_mem_mb", ".0f"),
        ("Static GPU memory (MB)", "static_mem_mb", "static_mem_mb", "static_mem_mb", ".0f"),
    ]:
        if label == "Per-step latency (s)":
            print("║ {:22s} │ {:14.3f} │ {:14.3f} │ {:14.3f} │ {:22s} ║".format(
                label, fp16_step, sim_step, hw_step, ""))
        elif label == "Per-step speedup":
            sim_speedup = fp16_step / sim_step if sim_step and fp16_step else 0
            print("║ {:22s} │ {:14s} │ {:13.2f}x │ {:13.2f}x │ {:21.2f}x ║".format(
                label, "1.00x (base)", sim_speedup, speedup, speedup))
        else:
            print("║ {:22s} │ {:>14s} │ {:>14s} │ {:>14s} │ {:22s} ║".format(
                label,
                _val(fp16, fp16_key, fmt),
                _val(sim, sim_key, fmt),
                _val(hw, hw_key, fmt),
                ""))
    print("╚" + "═" * 98 + "╝")

    # Memory breakdown comment
    if fp16 and hw:
        mem_saved = fp16.get("static_mem_mb", 0) - hw.get("static_mem_mb", 0)
        if mem_saved > 0:
            print(f"  → W8A8-HW saves {mem_saved:.0f} MB GPU memory vs FP16 ({mem_saved/fp16['static_mem_mb']*100:.1f}%)")
        else:
            print(f"  → Memory delta: {mem_saved:.0f} MB (INT8 weights partially offset by quant buffers)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ViDiT-Q Inference Benchmark (Phase 3c)")
    parser.add_argument("--fp16", action="store_true", help="Profile FP16 baseline")
    parser.add_argument("--w8a8-sim", action="store_true", help="Profile W8A8 software simulation")
    parser.add_argument("--w8a8-hw", action="store_true", help="Profile W8A8 CUDA kernel (hardware)")
    parser.add_argument("--all", action="store_true", help="Profile all three and compare")
    parser.add_argument("--steps", type=int, default=10, help="Sampling steps AFTER warmup (default: 10)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps (default: 2)")
    parser.add_argument("--no-hooks", action="store_true", help="Disable per-block profiling hooks (less overhead)")
    args = parser.parse_args()

    # Determine modes
    to_profile = []
    if args.all:
        to_profile = ["fp16", "w8a8_sim", "w8a8_hw"]
    else:
        if args.fp16:
            to_profile.append("fp16")
        if args.w8a8_sim:
            to_profile.append("w8a8_sim")
        if args.w8a8_hw:
            to_profile.append("w8a8_hw")

    if not to_profile:
        print("ERROR: specify at least one of --fp16, --w8a8-sim, --w8a8-hw, or --all")
        sys.exit(1)

    os.chdir(WORKDIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    from mmengine.config import Config

    results = {}

    for mode in to_profile:
        _timer.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if mode == "fp16":
            print("\n" + "=" * 100)
            print("PROFILING: FP16 Baseline")
            print("=" * 100)
            cfg = Config.fromfile(BASE_CONFIG)
            cfg.ptq_config = None  # force no quantization
            model, latent_size, num_frames = build_model(cfg, device, dtype, mode)

        elif mode == "w8a8_sim":
            print("\n" + "=" * 100)
            print("PROFILING: W8A8 Software Simulation")
            print("=" * 100)
            cfg = Config.fromfile(BASE_CONFIG)
            cfg.hardware = False
            model, latent_size, num_frames = build_model(cfg, device, dtype, mode)

        elif mode == "w8a8_hw":
            print("\n" + "=" * 100)
            print("PROFILING: W8A8 CUDA Kernel (Hardware)")
            print("=" * 100)
            cfg = Config.fromfile(HW_CONFIG)
            cfg.hardware = True
            model, latent_size, num_frames = build_model(cfg, device, dtype, mode)

        # Install hooks (unless disabled)
        handles = []
        if not args.no_hooks:
            handles = install_block_hooks(model)
            print(f"  Installed {len(handles)} profiling hooks")

        # Run
        result = run_profile(model, latent_size, num_frames, cfg, device, dtype,
                            num_steps=args.steps, warmup_steps=args.warmup)
        results[mode] = result

        # Per-mode report
        print(f"\n  Wall time: {result['wall_time_s']:.2f}s for {result['num_steps']} steps "
              f"({result['wall_time_s']/result['num_steps']:.3f}s/step)")
        print(f"  Peak GPU memory: {result['peak_mem_mb']:.0f} MB")
        print(f"  Static GPU memory: {result['static_mem_mb']:.0f} MB")

        if not args.no_hooks:
            _timer.report()

        # Cleanup
        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()

    # Cross-mode comparison
    if len(to_profile) > 1:
        print_summary(results)

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
