"""
TeaCache: Timestep Embedding Aware Cache for Video Diffusion Models.

Reference: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"
           (Liu et al., CVPR 2025 Highlight)
           https://arxiv.org/abs/2411.19108

Two caching strategies:
  - Global residual (TeaCacheWrapper): caches ALL blocks when t_mlp is similar
  - PAB (Pyramid Attention Broadcast): per-block per-component fine-grained caching
"""

from .teacache_wrapper import TeaCacheConfig, TeaCacheWrapper
from .pab_mgr import PABConfig, set_pab_manager, enable_pab, update_steps

__all__ = ["TeaCacheConfig", "TeaCacheWrapper",
           "PABConfig", "set_pab_manager", "enable_pab", "update_steps"]
