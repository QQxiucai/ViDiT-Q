"""
CUDA Event-based profiling for OpenSora FP16 / W8A8 hardware inference.

Usage:
  # Profile FP16 baseline
  python tools/profiling/profile_inference.py configs/local_144p_120f_5s.py --fp16

  # Profile W8A8 software simulation
  python tools/profiling/profile_inference.py configs/local_144p_120f_5s.py --w8a8-sim

  # Profile W8A8 CUDA kernel (hardware)
  python tools/profiling/profile_inference.py configs/local_144p_120f_5s_w8a8_hardware.py --w8a8-hw

  # Profile all three and compare
  python tools/profiling/profile_inference.py configs/local_144p_120f_5s.py --all
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from contextlib import contextmanager
from pprint import pformat

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from omegaconf import OmegaConf, ListConfig

# ---------------------------------------------------------------------------
# Profiling infrastructure
# ---------------------------------------------------------------------------

class CUDATimer:
    """Collects CUDA event pairs per named region."""

    def __init__(self):
        self._records: dict = defaultdict(list)  # name -> [(start_event, end_event)]

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

        # category rollup
        cat_totals: dict = defaultdict(float)
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


# Global timer instance
_timer = CUDATimer()


def install_block_hooks(model, prefix="model"):
    """Register forward pre/post hooks on all blocks and key sub-modules.

    Hooks are installed on:
      - spatial_blocks.N / temporal_blocks.N   (full block forward)
      - spatial_blocks.N.attn                   (self-attention)
      - spatial_blocks.N.cross_attn             (cross-attention)
      - spatial_blocks.N.mlp                    (mlp)
      - spatial_blocks.N.norm1 / norm2          (layernorm + modulate)

    For hardware blocks (STDiT3BlockWithCudaKernel), we also time the fused
    kernel sections by wrapping the sub-module forwards.
    """
    handles = []

    # -- block-level hooks --
    for group_name in ["spatial_blocks", "temporal_blocks"]:
        blocks = getattr(model, group_name, None)
        if blocks is None:
            continue
        for i, block in enumerate(blocks):
            block_label = f"{group_name}.{i:02d}"
            # Full block forward
            h_pre = block.register_forward_pre_hook(
                _make_pre_hook(f"{prefix}/{block_label}/full"))
            h_post = block.register_forward_hook(
                _make_post_hook(f"{prefix}/{block_label}/full"))
            handles.extend([h_pre, h_post])

            # Sub-modules within block
            for sub_name in ["attn", "cross_attn", "mlp", "norm1", "norm2"]:
                sub = getattr(block, sub_name, None)
                if sub is None:
                    continue
                h_pre = sub.register_forward_pre_hook(
                    _make_pre_hook(f"{prefix}/{block_label}/{sub_name}"))
                h_post = sub.register_forward_hook(
                    _make_post_hook(f"{prefix}/{block_label}/{sub_name}"))
                handles.extend([h_pre, h_post])

    # -- final layer --
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
# Model loader (reuses quant_inference.py logic, trimmed for profiling)
# ---------------------------------------------------------------------------

def build_model(cfg, device, dtype):
    """Build the model for profiling. Mirrors quant_inference.py setup."""
    from opensora.models.stdit.stdit3 import STDiT3Config
    from opensora.registry import MODELS, build_module
    from opensora.datasets.aspect import get_image_size, get_num_frames

    # VAE for latent size
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    image_size = get_image_size(cfg.get("resolution"), cfg.get("aspect_ratio"))
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)

    ptq_config_file = cfg.get("ptq_config", None)
    quant_config = OmegaConf.load(ptq_config_file) if ptq_config_file else None

    text_encoder_output_dim = 4096  # precomputed embed dim
    text_encoder_max_length = 300

    config = STDiT3Config(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        qk_norm=cfg.model.get("qk_norm", True),
        enable_flash_attn=cfg.model.get("enable_flash_attn", False),
        enable_layernorm_kernel=cfg.model.get("enable_layernorm_kernel", False),
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder_output_dim,
        model_max_length=text_encoder_max_length,
        enable_sequence_parallelism=False,
    )

    model_path = cfg.get("model_path", "/home/rich/ViDiT-Q/.local/models")
    from_pretrained = os.path.join(model_path, "hpcai-tech/OpenSora-STDiT-v3")

    # Resolve HuggingFace safetensors path (same logic as quant_opensora.py)
    safetensors_path = os.path.join(from_pretrained, "model.safetensors")
    if os.path.isfile(safetensors_path):
        checkpoint_path = safetensors_path
    else:
        checkpoint_path = from_pretrained

    if quant_config is not None:
        from models.quant_opensora import QuantOpenSora
        model = QuantOpenSora(quant_config, config, checkpoint_path).to(device, dtype).eval()
        model.config = config  # needed by hardware_forward_refactor
    else:
        from opensora.models.stdit.stdit3 import STDiT3
        from opensora.utils.ckpt_utils import load_checkpoint
        model = STDiT3(config)
        load_checkpoint(model, checkpoint_path)
        model = model.to(device, dtype).eval()

    # Handle mixed precision bitwidth
    if quant_config is not None:
        if_mixed = (isinstance(quant_config.weight.n_bits, ListConfig) or
                    isinstance(quant_config.act.n_bits, ListConfig))
        if if_mixed:
            model.bitwidth_refactor()

    return model, config, vae, latent_size, num_frames, image_size


def apply_quant_params(cfg, model):
    """Load quant params and (optionally) refactor for hardware."""
    # Apply mixed precision bitwidth refactor (must be done before hardware refactor)
    if hasattr(model, 'quant_config'):
        qc = model.quant_config
        if hasattr(qc, 'weight') and hasattr(qc.weight, 'n_bits'):
            if_mixed = isinstance(qc.weight.n_bits, ListConfig) or \
                       (hasattr(qc, 'act') and hasattr(qc.act, 'n_bits') and isinstance(qc.act.n_bits, ListConfig))
            if if_mixed:
                model.bitwidth_refactor()
    """Load quant params and (optionally) refactor for hardware."""
    if_hardware = cfg.get("hardware", False)
    quant_weight_ckpt = cfg.get("quant_weight_ckpt", None)

    if if_hardware:
        save_path = os.path.join(cfg.save_dir, "int_weight.pt")
        quant_param_ckpt = torch.load(
            os.path.join(cfg.save_dir, "quant_params.pth"),
            weights_only=True, map_location="cuda")
        model.load_quant_param_dict(quant_param_ckpt)
        model.quantize_and_save_weight(save_path=save_path)
        model.hardware_forward_refactor(load_path=save_path)
    else:
        quant_param_ckpt = torch.load(
            os.path.join(cfg.save_dir, "quant_params.pth"), weights_only=True)
        model.load_quant_param_dict(quant_param_ckpt)

    model.set_init_done()


# ---------------------------------------------------------------------------
# Profiling run
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_profile(cfg, model, vae, latent_size, num_frames, image_size, device, dtype,
                num_steps=2):
    """Run inference for `num_steps` timesteps and collect CUDA timings."""
    from opensora.registry import SCHEDULERS, build_module
    from qdiff.utils import seed_everything

    # Load precomputed text embeds (must already exist)
    save_d = torch.load("./precomputed_text_embeds.pth", map_location=device, weights_only=True)
    model_args = save_d["model_args"]

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    fps = cfg.fps if hasattr(cfg, "fps") else 24

    seed_everything(cfg.get("seed", 1024))
    z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)

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

    print(f"\nRunning {min(num_steps, len(timesteps))} sampling steps for profiling...")
    t0 = time.time()

    for i in range(min(num_steps, len(timesteps))):
        t = timesteps[i]
        with _timer.region(f"step_{i:02d}/total"):
            # CFG: batch = 2
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
    elapsed = time.time() - t0
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load config using mmengine, with minimal sys.argv override for parse_configs."""
    from mmengine.config import Config
    # Use mmengine directly — simpler and avoids argparse interference
    cfg = Config.fromfile(config_path)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Profile OpenSora inference paths")
    parser.add_argument("config", help="Path to base config file (e.g. configs/local_144p_120f_5s.py)")
    parser.add_argument("--fp16", action="store_true", help="Profile FP16 baseline")
    parser.add_argument("--w8a8-sim", action="store_true", help="Profile W8A8 software simulation")
    parser.add_argument("--w8a8-hw", action="store_true", help="Profile W8A8 CUDA kernel (hardware)")
    parser.add_argument("--all", action="store_true", help="Profile all three paths and compare")
    parser.add_argument("--steps", type=int, default=2, help="Number of sampling steps (default: 2)")
    parser.add_argument("--hook-depth", choices=["blocks", "full"], default="full",
                        help="Hook granularity: 'blocks' = block-level only, 'full' = sub-modules too")
    parser.add_argument("--workdir", default=None,
                        help="Working directory (default: examples/opensora1.2 under project root)")
    args = parser.parse_args()

    # Resolve working directory
    if args.workdir:
        workdir = args.workdir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        workdir = os.path.join(project_root, "examples", "opensora1.2")
    workdir = os.path.abspath(workdir)
    print(f"Working directory: {workdir}")
    os.chdir(workdir)
    # Ensure workdir is in sys.path (needed for "from models.xxx" imports)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(workdir, config_path)
    config_path = os.path.abspath(config_path)
    print(f"Config path: {config_path}")

    # Determine what to profile
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = "fp16"
    dtype = torch.float16

    results = {}

    for mode in to_profile:
        _timer.reset()

        if mode == "fp16":
            print("\n" + "=" * 100)
            print("PROFILING: FP16 Baseline")
            print("=" * 100)
            cfg = load_config(config_path)
            cfg.ptq_config = None  # force no quantization
            model, config, vae, latent_size, num_frames, image_size = build_model(cfg, device, dtype)

        elif mode == "w8a8_sim":
            print("\n" + "=" * 100)
            print("PROFILING: W8A8 Software Simulation")
            print("=" * 100)
            cfg = load_config(config_path)
            cfg.hardware = False
            model, config, vae, latent_size, num_frames, image_size = build_model(cfg, device, dtype)
            apply_quant_params(cfg, model)

        elif mode == "w8a8_hw":
            print("\n" + "=" * 100)
            print("PROFILING: W8A8 CUDA Kernel (Hardware)")
            print("=" * 100)
            cfg = load_config(config_path)
            cfg.hardware = True
            model, config, vae, latent_size, num_frames, image_size = build_model(cfg, device, dtype)
            apply_quant_params(cfg, model)

        # Install profiling hooks
        handles = install_block_hooks(model)
        print(f"Installed {len(handles)} profiling hooks ({args.hook_depth} depth)")

        # Run
        elapsed = run_profile(cfg, model, vae, latent_size, num_frames, image_size,
                              device, dtype, num_steps=args.steps)
        print(f"\nWall-clock time ({args.steps} steps): {elapsed:.2f}s")

        # Report
        _timer.report()

        # Remove hooks for next run
        for h in handles:
            h.remove()

        results[mode] = elapsed

    # Cross-mode summary
    if len(results) > 1:
        print("\n" + "=" * 100)
        print("CROSS-MODE COMPARISON")
        print("-" * 100)
        baseline = results.get("fp16", None)
        for mode, t in results.items():
            ratio = f"{t / baseline:.1f}x vs FP16" if baseline else ""
            print(f"  {mode:20s} {t:8.2f}s  {ratio}")
        print("=" * 100)


if __name__ == "__main__":
    main()
