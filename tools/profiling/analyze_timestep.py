#!/usr/bin/env python3
"""
Phase 4a: Timestep Embedding Analysis for TeaCache.

Captures the timestep embedding (t) and per-block modulation parameters across
a full sampling trajectory. Outputs:
  1. t-embedding evolution across timesteps (cosine similarity matrix)
  2. Per-block modulation variance across the trajectory
  3. Pairwise similarity between consecutive timesteps → cache-able block ratio

Usage:
  cd examples/opensora1.2
  python ../../tools/profiling/analyze_timestep.py
"""

import os
import sys

# --- path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
OPEN_SORA_DIR = os.path.join(PROJECT_ROOT, "examples", "opensora1.2", "Open-Sora")
WORKDIR = os.path.join(PROJECT_ROOT, "examples", "opensora1.2")

_torch_lib = os.path.join(os.path.dirname(__import__("torch").__file__), "lib")
if _torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

for _p in [WORKDIR, OPEN_SORA_DIR, KERNELS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VIDITQ_XFORMERS_OP", "cutlass")

import torch
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_latent_size(image_size, num_frames, micro_frame_size=17):
    SPATIAL_PATCH = (1, 8, 8)
    TEMPORAL_TS_DOWNSAMPLE = 4
    TEMPORAL_PATCH_T = 4
    def _spatial(tsz):
        return [tsz[0] // SPATIAL_PATCH[0], tsz[1] // SPATIAL_PATCH[1], tsz[2] // SPATIAL_PATCH[2]]
    def _temporal(tsz):
        res = []
        for i in range(3):
            if tsz[i] is None: res.append(None)
            elif i == 0:
                pad = 0 if (tsz[0] % TEMPORAL_TS_DOWNSAMPLE == 0) else TEMPORAL_TS_DOWNSAMPLE - tsz[0] % TEMPORAL_TS_DOWNSAMPLE
                res.append((tsz[0] + pad) // TEMPORAL_PATCH_T)
            else: res.append(tsz[i] // 1)
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

# ---------------------------------------------------------------------------
# Hook-based modulation capture
# ---------------------------------------------------------------------------
def install_modulation_hooks(model):
    """Capture t_mlp input to each block's modulate step.

    Hooks into STDiT3Block.forward to record the shift/scale/gate parameters
    derived from the timestep embedding.
    """
    records = defaultdict(list)  # step_idx -> list of (block_name, modulation_dict)

    def _make_block_hook(block_name, step_capture):
        def pre_hook(module, args):
            # args: (self, x, y, t, mask, x_mask, t0, T, S)
            x, y, t_mlp = args[0], args[1], args[2]
            B = x.shape[0]
            # t_mlp: [B, 6*C] — the post-t_block timestep embedding
            # scale_shift_table: [6, C]
            modulation = (module.scale_shift_table[None] + t_mlp.reshape(B, 6, -1))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
            step_capture.append({
                "block": block_name,
                "t_mlp": t_mlp.detach().cpu(),
                "shift_msa": shift_msa.detach().cpu(),
                "scale_msa": scale_msa.detach().cpu(),
                "gate_msa": gate_msa.detach().cpu(),
                "shift_mlp": shift_mlp.detach().cpu(),
                "scale_mlp": scale_mlp.detach().cpu(),
                "gate_mlp": gate_mlp.detach().cpu(),
            })
        return pre_hook

    handles = []
    step_captures = []  # list of lists, one per step

    # Hook spatial blocks
    for i, block in enumerate(model.spatial_blocks):
        cap = []
        step_captures.append(cap)
        h = block.register_forward_pre_hook(_make_block_hook(f"spatial.{i:02d}", cap))
        handles.append(h)

    # Hook temporal blocks
    for i, block in enumerate(model.temporal_blocks):
        cap = []
        step_captures.append(cap)
        h = block.register_forward_pre_hook(_make_block_hook(f"temporal.{i:02d}", cap))
        handles.append(h)

    return handles, step_captures


def install_t_embed_hook(model):
    """Capture the raw t-embedding before t_block."""
    t_embeds = []

    def hook(module, input, output):
        # output is t_mlp = t_block(t_embed)
        # We also want t_embed — capture from t_embedder output
        pass

    # Hook t_embedder
    t_list = []
    t_mlp_list = []

    def t_emb_hook(module, input, output):
        t_list.append(output.detach().cpu())  # t_embed before t_block

    def t_mlp_hook(module, input, output):
        t_mlp_list.append(output.detach().cpu())  # t_mlp after t_block

    h1 = model.t_embedder.register_forward_hook(t_emb_hook)
    h2 = model.t_block.register_forward_hook(t_mlp_hook)

    return [h1, h2], t_list, t_mlp_list


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze(config_path: str, output_dir: str = "/tmp/viditq_teacache_analysis"):
    from mmengine.config import Config
    from opensora.models.stdit.stdit3 import STDiT3Config, STDiT3
    from opensora.utils.ckpt_utils import load_checkpoint
    from opensora.registry import SCHEDULERS, build_module
    from opensora.datasets.aspect import get_image_size, get_num_frames
    from qdiff.utils import seed_everything

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(WORKDIR)

    device = "cuda"
    dtype = torch.float16
    cfg = Config.fromfile(config_path)

    # Build model (FP16 baseline)
    image_size = get_image_size(cfg.get("resolution"), cfg.get("aspect_ratio"))
    num_frames = get_num_frames(cfg.num_frames)
    latent_size = _compute_latent_size(image_size, num_frames, micro_frame_size=17)

    model_config = STDiT3Config(
        depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16,
        qk_norm=cfg.model.get("qk_norm", True),
        enable_flash_attn=cfg.model.get("enable_flash_attn", False),
        enable_layernorm_kernel=cfg.model.get("enable_layernorm_kernel", False),
        input_size=latent_size, in_channels=4, caption_channels=4096,
        model_max_length=300, enable_sequence_parallelism=False,
    )

    model_path = cfg.get("model_path", os.path.join(PROJECT_ROOT, ".local", "models"))
    from_pretrained = os.path.join(model_path, "hpcai-tech/OpenSora-STDiT-v3")
    safetensors_path = os.path.join(from_pretrained, "model.safetensors")
    checkpoint_path = safetensors_path if os.path.isfile(safetensors_path) else from_pretrained

    model = STDiT3(model_config)
    load_checkpoint(model, checkpoint_path)
    model = model.to(device, dtype).eval()

    # Load precomputed embeds
    save_d = torch.load(os.path.join(WORKDIR, "precomputed_text_embeds.pth"),
                        map_location=device, weights_only=True)
    model_args = save_d["model_args"]

    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    seed_everything(cfg.get("seed", 1024))
    z = torch.randn(1, 4, *latent_size, device=device, dtype=dtype)

    # Build timesteps
    timesteps = [(1.0 - i / scheduler.num_sampling_steps) * scheduler.num_timesteps
                 for i in range(scheduler.num_sampling_steps)]
    if scheduler.use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    ts_tensors = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    if scheduler.use_timestep_transform:
        from opensora.schedulers.rf.rectified_flow import timestep_transform
        ts_tensors = [timestep_transform(t, model_args, num_timesteps=scheduler.num_timesteps)
                      for t in ts_tensors]

    guidance_scale = cfg.scheduler.get("cfg_scale", 7.0)

    # Install hooks
    t_hooks, t_list, t_mlp_list = install_t_embed_hook(model)
    mod_hooks, step_captures = install_modulation_hooks(model)

    print(f"Running {len(ts_tensors)} sampling steps with hooks...")

    for i, t_val in enumerate(ts_tensors):
        z_in = torch.cat([z, z], 0)
        t_in = torch.cat([t_val, t_val], 0)
        pred = model(z_in, t_in, **model_args)
        pred = pred.chunk(2, dim=1)[0]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        if i < len(ts_tensors) - 1:
            dt = ts_tensors[i] - ts_tensors[i + 1]
        else:
            dt = ts_tensors[i]
        dt = dt / scheduler.num_timesteps
        z = z + v_pred * dt[:, None, None, None, None]

    # Cleanup hooks
    for h in t_hooks + mod_hooks:
        h.remove()

    # --- Analysis ---
    t_raw = torch.stack(t_list)              # [num_steps, B, C]
    t_mlp_raw = torch.stack(t_mlp_list)      # [num_steps, B, 6*C]

    num_steps = t_raw.shape[0]
    B = t_raw.shape[1]
    C = t_raw.shape[2]

    # Use the first (cond) in the batch for CFG
    t_raw = t_raw[:, 0, :]
    t_mlp_raw = t_mlp_raw[:, 0, :]

    print(f"\n{'='*80}")
    print(f"Timestep Embedding Analysis Results")
    print(f"{'='*80}")
    print(f"  Steps: {num_steps}, hidden_size: {C}")
    print(f"  t_raw shape: {t_raw.shape}, t_mlp shape: {t_mlp_raw.shape}")

    # --- 1. Cosine similarity of t_embed across timesteps ---
    t_norm = t_raw / t_raw.norm(dim=-1, keepdim=True)
    cos_sim_t = (t_norm @ t_norm.T).numpy()

    print(f"\n--- t_embed cosine similarity matrix ---")
    print(f"  Mean similarity: {cos_sim_t.mean():.4f}")
    print(f"  Min similarity:  {cos_sim_t.min():.4f}")
    print(f"  Consecutive steps mean similarity: {np.diag(cos_sim_t, 1).mean():.4f}")

    # Save similarity matrix
    np.save(os.path.join(output_dir, "t_embed_cosine.npy"), cos_sim_t)

    # --- 2. t_mlp similarity across timesteps ---
    t_mlp_norm = t_mlp_raw / t_mlp_raw.norm(dim=-1, keepdim=True)
    cos_sim_mlp = (t_mlp_norm @ t_mlp_norm.T).numpy()

    print(f"\n--- t_mlp cosine similarity matrix ---")
    print(f"  Mean similarity: {cos_sim_mlp.mean():.4f}")
    print(f"  Min similarity:  {cos_sim_mlp.min():.4f}")
    print(f"  Consecutive steps mean similarity: {np.diag(cos_sim_mlp, 1).mean():.4f}")

    np.save(os.path.join(output_dir, "t_mlp_cosine.npy"), cos_sim_mlp)

    # --- 3. Per-block modulation variance ---
    # step_captures is a list of 56 lists (one per hook/block).
    # Each inner list has one capture dict per timestep.
    # Index: step_captures[block_idx][step_idx] = {"block": ..., ...}

    block_modulations = defaultdict(lambda: defaultdict(list))  # block_name -> param_name -> [steps]

    for block_idx, caps in enumerate(step_captures):
        if not caps:
            continue
        for step_idx, d in enumerate(caps):
            bn = d["block"]
            for key in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
                block_modulations[bn][key].append(d[key])

    # Compute per-block modulation change between consecutive steps
    block_deltas = {}
    for bn, params in block_modulations.items():
        deltas = {}
        for key, vals in params.items():
            if len(vals) < 2:
                continue
            stacked = torch.stack(vals)  # [steps, 1, C]
            # L2 delta between consecutive steps, normalized by norm
            diff = (stacked[1:] - stacked[:-1]).norm(dim=-1)  # [steps-1, 1]
            total = stacked.norm(dim=-1).mean()
            deltas[key] = diff.mean().item() / (total.item() + 1e-8)
        block_deltas[bn] = deltas

    # Print top-10 most stable blocks (lowest modulation change)
    avg_delta = {bn: np.mean(list(d.values())) for bn, d in block_deltas.items()}
    sorted_blocks = sorted(avg_delta.items(), key=lambda x: x[1])

    print(f"\n--- Top-20 most cache-able blocks (lowest modulation delta) ---")
    for bn, delta in sorted_blocks[:20]:
        spatial = "S" if bn.startswith("spatial") else "T"
        idx = bn.split(".")[1]
        print(f"  {spatial}{idx}: avg_delta={delta:.6f}")

    print(f"\n--- Bottom-10 least cache-able blocks (highest modulation delta) ---")
    for bn, delta in sorted_blocks[-10:]:
        spatial = "S" if bn.startswith("spatial") else "T"
        idx = bn.split(".")[1]
        print(f"  {spatial}{idx}: avg_delta={delta:.6f}")

    # --- 4. Cache-able ratio at different thresholds ---
    print(f"\n--- Cache-able block ratio vs similarity threshold ---")
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for thresh in thresholds:
        cacheable = sum(1 for d in avg_delta.values() if d < thresh)
        ratio = cacheable / len(avg_delta) * 100
        print(f"  threshold={thresh:.3f}: {cacheable}/{len(avg_delta)} blocks ({ratio:.1f}%) cache-able")

    # --- 5. Save all data ---
    torch.save({
        "t_raw": t_raw,
        "t_mlp_raw": t_mlp_raw,
        "cos_sim_t": cos_sim_t,
        "cos_sim_mlp": cos_sim_mlp,
        "block_deltas": block_deltas,
        "sorted_blocks": sorted_blocks,
    }, os.path.join(output_dir, "teacache_analysis.pt"))

    print(f"\nResults saved to {output_dir}/")
    return t_raw, t_mlp_raw, block_deltas, sorted_blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4a: Timestep embedding analysis")
    parser.add_argument("--config", default="configs/local_144p_120f_5s.py",
                        help="Base config file (default: configs/local_144p_120f_5s.py)")
    parser.add_argument("--output", default="/tmp/viditq_teacache_analysis",
                        help="Output directory for results")
    args = parser.parse_args()

    analyze(args.config, args.output)
