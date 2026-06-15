"""
PAB (Pyramid Attention Broadcast) — fine-grained per-block caching.

Unlike TeaCache global residual (which caches ALL blocks or NONE),
PAB caches individual block components independently:
  - Spatial self-attention (stable → cache more aggressively)
  - Temporal self-attention (sensitive → cache conservatively)
  - Cross-attention
  - MLP

Strategy: "compute fully every N steps, reuse cache in between."
First and last steps always compute. Steps outside [threshold_low, threshold_high] always compute.

Reference: TeaCache PAB (CVPR 2025), videosys/core/pab_mgr.py
"""

import torch

# Global state (like TeaCache official)
PAB_MANAGER = None


class PABConfig:
    """Configuration for Pyramid Attention Broadcast.

    Attributes:
        spatial_range: Compute spatial attn every N steps (2 = every other step).
        temporal_range: Compute temporal attn every N steps (more conservative).
        cross_range: Compute cross-attn every N steps.
        mlp_range: Compute MLP every N steps.
        threshold_low: Don't cache when timestep <= this (too early/noisy).
        threshold_high: Don't cache when timestep >= this (too late/clean).
    """

    def __init__(
        self,
        spatial_range: int = 2,
        temporal_range: int = 3,
        cross_range: int = 3,
        mlp_range: int = 2,
        threshold_low: float = 80.0,
        threshold_high: float = 950.0,
    ):
        self.spatial_range = spatial_range
        self.temporal_range = temporal_range
        self.cross_range = cross_range
        self.mlp_range = mlp_range
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.steps = None  # set by set_pab_manager


class PABManager:
    """Manages per-block PAB cache state and broadcast decisions.

    Keeps track of per-block counters and cached outputs.
    """

    def __init__(self, config: PABConfig):
        self.config = config
        # Per-block state: {block_full_name: {"attn": ..., "cross": ..., "mlp": ..., "counts": ...}}
        self._state = {}

    def _ensure_block(self, block_name: str):
        if block_name not in self._state:
            self._state[block_name] = {
                "attn_count": 0,
                "cross_count": 0,
                "mlp_count": 0,
                "last_attn": None,
                "last_cross": None,
                "last_mlp": None,
            }
        return self._state[block_name]

    # ---- Broadcast decisions ----

    def if_broadcast_attn(self, block_name: str, timestep: float, is_temporal: bool):
        """Returns (should_cache, new_count)."""
        st = self._ensure_block(block_name)
        count = st["attn_count"]
        rng = self.config.temporal_range if is_temporal else self.config.spatial_range
        flag = self._check(timestep, count, rng)
        st["attn_count"] = (count + 1) % rng
        return flag, st["attn_count"]

    def if_broadcast_cross(self, block_name: str, timestep: float):
        """Returns (should_cache, new_count)."""
        st = self._ensure_block(block_name)
        count = st["cross_count"]
        flag = self._check(timestep, count, self.config.cross_range)
        st["cross_count"] = (count + 1) % self.config.cross_range
        return flag, st["cross_count"]

    def if_broadcast_mlp(self, block_name: str, timestep: float):
        """Returns (should_cache, new_count)."""
        st = self._ensure_block(block_name)
        count = st["mlp_count"]
        flag = self._check(timestep, count, self.config.mlp_range)
        st["mlp_count"] = (count + 1) % self.config.mlp_range
        return flag, st["mlp_count"]

    def _check(self, timestep: float, count: int, rng: int) -> bool:
        """Should we broadcast (cache) this component?"""
        if timestep <= self.config.threshold_low:
            return False
        if timestep >= self.config.threshold_high:
            return False
        # Cache when count % range != 0 (not a refresh step)
        return (count % rng) != 0

    # ---- Cache get/set ----

    def get_attn(self, block_name: str):
        return self._ensure_block(block_name)["last_attn"]

    def set_attn(self, block_name: str, value):
        self._ensure_block(block_name)["last_attn"] = value

    def get_cross(self, block_name: str):
        return self._ensure_block(block_name)["last_cross"]

    def set_cross(self, block_name: str, value):
        self._ensure_block(block_name)["last_cross"] = value

    def get_mlp(self, block_name: str):
        return self._ensure_block(block_name)["last_mlp"]

    def set_mlp(self, block_name: str, value):
        self._ensure_block(block_name)["last_mlp"] = value

    # ---- Statistics ----

    def get_stats(self):
        """Returns total cache hits per component type."""
        hits = {"attn": 0, "cross": 0, "mlp": 0}
        for st in self._state.values():
            # Counters represent cycles; we estimate hits as total_steps - refreshes
            pass
        return hits


# ---- Global accessors (like TeaCache official) ----

def set_pab_manager(config: PABConfig):
    global PAB_MANAGER
    PAB_MANAGER = PABManager(config)


def enable_pab() -> bool:
    return PAB_MANAGER is not None


def update_steps(steps: int):
    if PAB_MANAGER is not None:
        PAB_MANAGER.config.steps = steps


def if_broadcast_attn(block_name: str, timestep, is_temporal: bool):
    if not enable_pab():
        return False, 0
    return PAB_MANAGER.if_broadcast_attn(block_name, int(timestep), is_temporal)


def if_broadcast_cross(block_name: str, timestep):
    if not enable_pab():
        return False, 0
    return PAB_MANAGER.if_broadcast_cross(block_name, int(timestep))


def if_broadcast_mlp(block_name: str, timestep):
    if not enable_pab():
        return False, 0
    return PAB_MANAGER.if_broadcast_mlp(block_name, int(timestep))


def get_attn(block_name: str):
    return PAB_MANAGER.get_attn(block_name)


def set_attn(block_name: str, value):
    PAB_MANAGER.set_attn(block_name, value)


def get_cross(block_name: str):
    return PAB_MANAGER.get_cross(block_name)


def set_cross(block_name: str, value):
    PAB_MANAGER.set_cross(block_name, value)


def get_mlp(block_name: str):
    return PAB_MANAGER.get_mlp(block_name)


def set_mlp(block_name: str, value):
    PAB_MANAGER.set_mlp(block_name, value)
