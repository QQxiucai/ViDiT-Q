"""
Calibrate TeaCache rescale coefficients for a specific model checkpoint.

TeaCache uses a 5th-degree polynomial to rescale the per-step L1 distance
between modulated_inputs. The polynomial maps "relative L1 change of
modulated_input" -> "expected relative L1 change of full block output".

This script runs one full FP16 sampling trajectory WITH hooks to capture
per-step modulated_input and block output, then fits the polynomial.

Usage:
  conda activate viditq-xf033-test
  cd examples/opensora1.2

  export PYTHONPATH=./Open-Sora
  export VIDITQ_XFORMERS_OP=default

  python teacache/calibrate_coeff.py configs/local_144p_17f.py
"""

import os
import sys
import argparse

import torch
import numpy as np

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN_SORA = os.path.join(WORKDIR, "Open-Sora")
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)
if OPEN_SORA not in sys.path:
    sys.path.insert(0, OPEN_SORA)


def _compute_latent_size(image_size, num_frames, micro_frame_size=17):
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

    sub_in = [micro_frame_size, image_size[0], image_size[1]]
    sub_lat = _temporal(_spatial(sub_in))
    sub_lat[0] = sub_lat[0] * (num_frames // micro_frame_size)
    remain_t = num_frames % micro_frame_size
    if remain_t > 0:
        remain_lat = _temporal([remain_t, None, None])
        sub_lat[0] += remain_lat[0]
    return sub_lat


def calibrate(config_path: str):
    from mmengine.config import Config
    from opensora.models.stdit.stdit3 import STDiT3Config, STDiT3
    from opensora.utils.ckpt_utils import load_checkpoint
    from opensora.registry import SCHEDULERS, build_module
    from opensora.datasets.aspect import get_image_size, get_num_frames
    from opensora.models.layers.blocks import t2i_modulate
    from qdiff.utils import seed_everything

    device = torch.device("cuda")
    dtype = torch.float16

    cfg = Config.fromfile(config_path)
    seed_everything(cfg.get("seed", 42))

    image_size = get_image_size(cfg.get("resolution", "144p"),
                                cfg.get("aspect_ratio", "1:1"))
    num_frames = get_num_frames(cfg.num_frames)
    latent_size = _compute_latent_size(image_size, num_frames)

    print(f"Config: {num_frames}f, {image_size}, latent={latent_size}")

    # Build STDiT3 model (FP16, no quantization)
    print("Building STDiT3 model ...")
    model_config = STDiT3Config(
        depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16,
        qk_norm=cfg.model.get("qk_norm", True),
        enable_flash_attn=cfg.model.get("enable_flash_attn", False),
        enable_layernorm_kernel=cfg.model.get("enable_layernorm_kernel", False),
        input_size=latent_size, in_channels=4, caption_channels=4096,
        model_max_length=300, enable_sequence_parallelism=False,
    )
    model = STDiT3(model_config).to(device, dtype).eval()

    model_path = cfg.get("model_path",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(WORKDIR))),
                                      ".local", "models"))
    from_pretrained = os.path.join(model_path, "hpcai-tech/OpenSora-STDiT-v3")
    safetensors_path = os.path.join(from_pretrained, "model.safetensors")
    ckpt = safetensors_path if os.path.isfile(safetensors_path) else from_pretrained
    load_checkpoint(model, ckpt)
    print(f"  Model loaded: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # Load precomputed text embeds
    embed_path = os.path.join(WORKDIR, "precomputed_text_embeds.pth")
    if not os.path.isfile(embed_path):
        print(f"ERROR: precomputed_text_embeds.pth not found. Run precompute_text_embeds.py first.")
        sys.exit(1)
    save_d = torch.load(embed_path, map_location=device, weights_only=True)
    model_args = save_d["model_args"]
    print(f"  Precomputed embeds loaded (y shape={model_args['y'].shape})")

    # Scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # Timesteps
    timesteps = [(1.0 - i / scheduler.num_sampling_steps) * scheduler.num_timesteps
                 for i in range(scheduler.num_sampling_steps)]
    if scheduler.use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    ts_tensors = [torch.tensor([t] * 1, device=device) for t in timesteps]
    if scheduler.use_timestep_transform:
        from opensora.schedulers.rf.rectified_flow import timestep_transform
        ts_tensors = [timestep_transform(t, model_args, num_timesteps=scheduler.num_timesteps)
                      for t in ts_tensors]

    print(f"Timesteps: {[int(t[0].item()) for t in ts_tensors]}")
    guidance_scale = cfg.scheduler.get("cfg_scale", 7.0)

    # ---- Install hooks on the model ----
    # Hook 1: capture modulated_input from first spatial block
    modulated_inputs = []
    def pre_hook_spatial0(module, args):
        x_arg, y_arg, t_mlp_arg = args[0], args[1], args[2]
        B = x_arg.shape[0]
        shift_msa, scale_msa, _, _, _, _ = (
            module.scale_shift_table[None] + t_mlp_arg.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        mod_inp = t2i_modulate(module.norm1(x_arg), shift_msa, scale_msa)
        modulated_inputs.append(mod_inp.detach().cpu())

    # Hook 2: capture full block output from last temporal block
    block_outputs = []
    def post_hook_temporal_last(module, args, output):
        block_outputs.append(output.detach().cpu())

    h_pre = model.spatial_blocks[0].register_forward_pre_hook(pre_hook_spatial0)
    h_post = model.temporal_blocks[-1].register_forward_hook(post_hook_temporal_last)

    # ---- Run one full sampling trajectory ----
    print(f"Running {len(ts_tensors)}-step trajectory with hooks ...")
    z = torch.randn(1, 4, *latent_size, device=device, dtype=dtype)

    for i, t_val in enumerate(ts_tensors):
        z_in = torch.cat([z, z], 0)          # CFG doubled batch
        t_in = torch.cat([t_val, t_val], 0)

        with torch.no_grad():
            pred = model(z_in, t_in, **model_args)

        # CFG update
        pred = pred.chunk(2, dim=1)[0]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        dt = ((ts_tensors[i] - ts_tensors[i + 1]) if i < len(ts_tensors) - 1
              else ts_tensors[i]) / scheduler.num_timesteps
        z = z + v_pred * dt[:, None, None, None, None]

    h_pre.remove()
    h_post.remove()

    # Note: CFG is handled via doubled batch in a single forward call, so
    # each hook fires once per step (not twice). No need for [::2] slicing.
    print(f"\nCollected {len(modulated_inputs)} steps (hooks fire once per forward)")

    # ---- Compute rel_l1 and output_diff between consecutive steps ----
    rel_l1_list = []
    output_diff_list = []

    print(f"\n{'Step':>5s}  {'rel_l1':>10s}  {'output_diff':>12s}")
    print("-" * 32)
    for i in range(1, len(modulated_inputs)):
        rel_l1 = ((modulated_inputs[i] - modulated_inputs[i - 1]).abs().mean()
                  / (modulated_inputs[i - 1].abs().mean() + 1e-8)).item()
        out_diff = ((block_outputs[i] - block_outputs[i - 1]).abs().mean()
                    / (block_outputs[i - 1].abs().mean() + 1e-8)).item()
        rel_l1_list.append(rel_l1)
        output_diff_list.append(out_diff)
        print(f"  {i:3d}   {rel_l1:10.6f}  {out_diff:12.6f}")

    # Use degree=3 (9 data points -> cubic is numerically stable)
    poly_degree = min(3, len(rel_l1_list) - 1)
    coeffs = np.polyfit(rel_l1_list, output_diff_list, poly_degree)
    poly = np.poly1d(coeffs)

    # R-squared
    preds = poly(rel_l1_list)
    ss_res = np.sum((np.array(output_diff_list) - preds) ** 2)
    ss_tot = np.sum((np.array(output_diff_list) - np.mean(output_diff_list)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    print(f"\n{'='*70}")
    print(f"Fitted degree-{poly_degree} rescale coefficients (R\xc2\xb2 = {r2:.4f}):")
    coeff_str = ", ".join(f"{c:.8e}" for c in coeffs)
    print(f"  [{coeff_str}]")
    print(f"\nUse in TeaCacheConfig:")
    print(f"  TeaCacheConfig(rescale_coefficients=[{coeff_str}])")
    print(f"{'='*70}")

    return coeffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate TeaCache rescale coefficients")
    parser.add_argument("config", default="configs/local_144p_17f.py", nargs="?",
                        help="Config file (default: configs/local_144p_17f.py)")
    args = parser.parse_args()
    calibrate(args.config)
