"""
TeaCache wrapper for STDiT3 — global residual caching strategy.

Replaces STDiT3.forward() with a version that compares modulated_input
between consecutive timesteps. When the accumulated difference is below a
threshold, the full block loop is skipped and the cached residual is reused.

Reference: TeaCache (CVPR 2025 Highlight)
  - eval/teacache/experiments/opensora.py (teacache_forward)
  - Paper: https://arxiv.org/abs/2411.19108

Key insight:
  The timestep embedding t_mlp is the sole modulation input to every
  transformer block. By monitoring how the first block's modulated input
  changes between steps, we can predict whether the output of ALL blocks
  will change enough to warrant re-computation.
"""

import torch
import numpy as np
from einops import rearrange

from opensora.models.layers.blocks import t2i_modulate
from opensora.acceleration.checkpoint import auto_grad_checkpoint


class TeaCacheConfig:
    """Configuration for TeaCache global residual caching.

    Attributes:
        rel_l1_thresh:
            Accumulated relative L1 distance threshold.
            Lower = better quality but less caching.
            Typical values: 0.10 (slow/quality), 0.20 (fast).
        rescale_coefficients:
            5th-degree polynomial coefficients for rescaling the per-step
            L1 distance. Model-specific — should be calibrated if the
            default values produce poor results.
        start_step:
            First sampling step (0-indexed) to apply caching.
        end_step:
            Last sampling step to apply caching.
            -1 = exclude the final step (always computed).
    """

    def __init__(
        self,
        rel_l1_thresh: float = 0.10,
        rescale_coefficients: list = None,
        start_step: int = 0,
        end_step: int = -1,
    ):
        self.rel_l1_thresh = rel_l1_thresh
        # Default coefficients calibrated on TeaCache official OpenSora v1.2
        self.rescale_coefficients = rescale_coefficients or [
            2.17546007e02, -1.18329252e02,  2.68662585e+01,
            -4.59364272e-02,  4.84426240e-02,
        ]
        self.start_step = start_step
        self.end_step = end_step


class TeaCacheWrapper:
    """Wraps an STDiT3 model with TeaCache global residual caching.

    The wrapper replaces model.forward with a caching-aware version.
    When consecutive timesteps are similar enough (based on the modulated
    input to the first spatial block), the entire block loop is skipped
    and the previous residual is reused.

    Usage:
        model = STDiT3(config)
        load_checkpoint(model, path)
        wrapper = TeaCacheWrapper(model, TeaCacheConfig(rel_l1_thresh=0.10))

        # Bind to model — all callers transparently get TeaCache behavior:
        model.forward = wrapper.forward

        # Or call wrapper directly:
        output = wrapper(x, timestep, y, ...)

    The wrapper expects `all_timesteps` to be present in **kwargs
    (as passed by the modified RFLOW scheduler).
    """

    def __init__(self, model, config: TeaCacheConfig):
        self.model = model
        self.config = config

        # --- Cache state ---
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0

        # --- Statistics ---
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_hit_rate(self):
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None,
                height=None, width=None, **kwargs):
        """TeaCache-enabled forward pass.

        Args:
            x:          Input latent    [B, C, T, H, W]
            timestep:   Timestep tensor [B] (float, e.g. [987.0, 987.0] for CFG)
            y:          Text embeddings [1, N_text, C] or [B, N_text, C]
            mask:       Text attention mask (list of ints)
            x_mask:     Temporal mask for I2V (optional)
            fps, height, width: Metadata tensors
            **kwargs:   Must contain 'all_timesteps' — list of int timestep values
                        for the full sampling trajectory.

        Returns:
            Model output [B, C_out, T, H, W]
        """
        # ================================================================
        # Part 1: Embeddings — identical to original STDiT3.forward()
        # ================================================================
        dtype = self.model.x_embedder.proj.weight.dtype
        B_orig = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # Dynamic size
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.model.get_dynamic_size(x)
        S = H * W

        # Position embedding
        base_size = round(S ** 0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.model.input_sq_size
        pos_emb = self.model.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # Timestep embedding
        t = self.model.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps_emb = self.model.fps_embedder(fps.unsqueeze(1), B_orig)
        t = t + fps_emb
        t_mlp = self.model.t_block(t)  # [B, 6*C]

        # t0 for masked regions
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.model.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps_emb
            t0_mlp = self.model.t_block(t0)

        # Text embedding
        if self.model.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.model.encode_text(y, mask)

        # ================================================================
        # Part 2: Patch embedding + TeaCache decision
        # ================================================================
        # Patch embedding
        x = self.model.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # ---- TeaCache: compute modulated input from first spatial block ----
        # We use the FIRST spatial block's norm1 + scale_shift_table as a
        # "probe" to estimate how much the timestep modulation has changed.
        # This is cheap (~1 layernorm + element-wise ops) compared to running
        # all 56 blocks.
        inp_flat = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        B_flat, N_tokens, C_hidden = inp_flat.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.model.spatial_blocks[0].scale_shift_table[None]
            + t_mlp.reshape(B_flat, 6, -1)
        ).chunk(6, dim=1)

        modulated_inp = t2i_modulate(
            self.model.spatial_blocks[0].norm1(inp_flat),
            shift_msa, scale_msa
        )

        # ---- TeaCache decision logic ----
        should_calc = self._should_calculate(
            timestep, modulated_inp, kwargs.get("all_timesteps")
        )
        self.previous_modulated_input = modulated_inp

        # ================================================================
        # Part 3: Block loop — execute or skip
        # ================================================================
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        # PAB: pass scalar timestep to blocks for per-component caching
        ts_scalar = int(timestep[0].item()) if timestep is not None else None

        if not should_calc:
            # Cache hit: add previous residual to the input
            self.cache_hits += 1
            x = x + self.previous_residual
        else:
            # Cache miss: full forward through all blocks
            # (PAB can still cache individual components within blocks)
            self.cache_misses += 1
            origin_x = x.clone().detach()

            for spatial_block, temporal_block in zip(
                self.model.spatial_blocks, self.model.temporal_blocks
            ):
                x = auto_grad_checkpoint(
                    spatial_block, x, y, t_mlp, y_lens,
                    x_mask, t0_mlp, T, S, timestep=ts_scalar
                )
                x = auto_grad_checkpoint(
                    temporal_block, x, y, t_mlp, y_lens,
                    x_mask, t0_mlp, T, S, timestep=ts_scalar
                )

            self.previous_residual = x - origin_x

        # ================================================================
        # Part 4: Final layer + unpatchify — identical to original
        # ================================================================
        x = self.model.final_layer(x, t, x_mask, t0, T, S)
        x = self.model.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = x.to(torch.float32)

        return x

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_calculate(self, timestep, modulated_inp, all_timesteps):
        """Determine whether to run the full block loop.

        Returns False (skip = cache hit) when the accumulated L1 distance
        between consecutive modulated inputs is below the threshold.

        The first and last steps are always fully computed.
        """
        # Extract scalar timestep value (use the first in the batch)
        current_t = int(timestep[0].item())

        # Determine first/last timesteps
        if all_timesteps is not None:
            if isinstance(all_timesteps, list):
                first_t = int(all_timesteps[0])
                last_t = int(all_timesteps[-1])
            elif isinstance(all_timesteps, torch.Tensor):
                first_t = int(all_timesteps[0].item())
                last_t = int(all_timesteps[-1].item())
            else:
                first_t = last_t = None
        else:
            first_t = last_t = None

        # Always compute on first and last step
        if first_t is not None and (current_t == first_t or current_t == last_t):
            self.accumulated_rel_l1_distance = 0.0
            return True

        # First call — no previous to compare against
        if self.previous_modulated_input is None:
            return True

        # Compute relative L1 distance between current and previous
        # modulated inputs
        rel_l1 = (
            (modulated_inp - self.previous_modulated_input).abs().mean()
            / self.previous_modulated_input.abs().mean()
        ).cpu().item()

        # Apply rescale polynomial (fitted to map L1 distance to
        # expected output change)
        rescale_func = np.poly1d(self.config.rescale_coefficients)
        self.accumulated_rel_l1_distance += float(rescale_func(rel_l1))

        if self.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  # skip = cache hit
        else:
            self.accumulated_rel_l1_distance = 0.0
            return True  # compute = cache miss

    def reset_cache(self):
        """Reset internal cache state (call between independent generations)."""
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
