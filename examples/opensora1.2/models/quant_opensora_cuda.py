from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
import logging

from timm.models.vision_transformer import Mlp
import xformers.ops
from xformers.ops import fmha
from viditq_extension.nn.base import QuantParams

# RTX 5080 / sm120: force cutlass backend (default Hopper kernel crashes)
XFORMERS_CUTLASS_OP = (fmha.cutlass.FwOp, fmha.cutlass.BwOp)

import os as _os
def _get_xformers_op():
    op_name = _os.environ.get("VIDITQ_XFORMERS_OP", "default").lower()
    if op_name == "default":
        return None
    if op_name == "cutlass":
        return XFORMERS_CUTLASS_OP
    raise ValueError(f"Unsupported VIDITQ_XFORMERS_OP={op_name!r}; expected 'default' or 'cutlass'.")

from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

logger = logging.getLogger(__name__)

# From PyTorch internals
from functools import partial
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
    
def _is_viditq_layer(module):
    """Check if a module is a ViDiTQuantizedLinear (has channel_mask + rotation_matrix)."""
    return (
        hasattr(module, 'channel_mask') and module.channel_mask is not None
        and hasattr(module, 'rotation_matrix') and module.rotation_matrix is not None
    )


def _is_viditq_attn_layer(full_name):
    """Check if a layer name corresponds to a self-attention qkv or proj layer."""
    return ".attn.qkv" in full_name or ".attn.proj" in full_name


def quantize_and_save_weight_(submodule, full_name):
    is_viditq = _is_viditq_layer(submodule)
    is_viditq_attn = is_viditq and _is_viditq_attn_layer(full_name)

    if not is_viditq:
        # Standard layers: skip attn.qkv, attn.proj, cross_attn.kv_linear
        # These stay as dequantized FP weights (used by software attention path)
        if (
            ".attn.qkv" in full_name
            or ".attn.proj" in full_name
            or ".cross_attn.kv_linear" in full_name
        ):
            # These layers are not replaced by W8A8 CUDA Linear modules below, so
            # keep their weights in dequantized FP form for regular F.linear.
            return
    else:
        # ViDiT-Q layers:
        # - cross_attn.kv_linear: still uses software F.linear, skip
        # - cross_attn.q_linear, cross_attn.proj: already replaced by CUDA kernel
        #   (W8A8OF16LinearDynamicInputScale), process as INT8
        # - mlp.fc1, mlp.fc2: already replaced by CUDA kernel, process as INT8
        # - attn.qkv, attn.proj: software path for now (ViDiTQuantizedLinear),
        #   apply FP16 precision alignment (channel_mask + rotation) and keep as
        #   dequantized FP16 for F.linear compatibility
        if ".cross_attn.kv_linear" in full_name:
            return

    fp_weight = submodule.fp_module.weight.to(torch.float16)

    # ---- ViDiT-Q weight preprocessing (offline, one-time) ----
    if is_viditq and not is_viditq_attn:
        # CUDA kernel layers (cross_attn.q_linear/proj, mlp.fc1/fc2):
        # Apply channel_mask + rotation so the INT8 weight is consistent
        # with the future ViDiT-Q fused activation kernel.
        # NOTE: currently the CUDA kernel path does NOT apply ViDiT-Q
        # preprocessing to activations, so weight preprocessing is disabled
        # to maintain consistency. Enable when activation kernel is wired in.
        pass  # reserved for future: fp_weight = apply_viditq_preprocessing(fp_weight)

    if is_viditq_attn:
        # Self-attention in ViDiT-Q blocks: apply preprocessing + export as INT8
        # for ViDiTQW8A8Linear (hardware CUDA kernel path).
        channel_mask = submodule.channel_mask.to(device=fp_weight.device, dtype=torch.float16)
        rotation_matrix = submodule.rotation_matrix.to(device=fp_weight.device, dtype=torch.float16)
        # Step 1: channel-wise scaling
        fp_weight = fp_weight / channel_mask.reshape(1, -1)
        # Step 2: Hadamard rotation in FP16 (matches forward pass precision)
        fp_weight = torch.matmul(fp_weight, rotation_matrix)
        # Fall through to standard INT8 quantization below
        logger.debug("ViDiT-Q attn layer %s: pre-processed + exporting INT8 for ViDiTQW8A8Linear", full_name)

    # the viditq_extension.nn.qlinear use [C] as the scale shape, but the qdiff simulation code use [C, 1]

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point  # the cuda kernel code uses 128+zero_point

    # Determine bitwidth: check if this layer should use W4A8
    use_w4a8 = False
    if hasattr(submodule, 'w_quantizer') and hasattr(submodule.w_quantizer, 'n_bits'):
        use_w4a8 = (int(submodule.w_quantizer.n_bits) == 4)

    if use_w4a8:
        C_out, C_in = fp_weight.shape
        G = 128  # QServe W4A8 kernel group size

        # The QServe kernel packs wscales/w_szs as half2 pairs, which requires
        # an even number of groups (C_in/G must be even).
        # For hidden_size=1152: 1152/128=9 groups (odd) → skip W4A8, fall back to W8A8.
        # For hidden_size=4608: 4608/128=36 groups (even) → OK for W4A8.
        if (C_in // G) % 2 != 0:
            logger.debug("W4A8 skipped for %s: odd groups (%d), falling back to W8A8",
                         full_name, C_in // G)
            use_w4a8 = False

    if use_w4a8:
        # W4A8: pack 2×4-bit weights per INT8 byte, per-group scales
        C_out, C_in = fp_weight.shape
        G = 128
        num_groups = C_in // G

        fp_groups = fp_weight.view(C_out, num_groups, G)
        w_max = fp_groups.max(dim=-1).values
        w_min = fp_groups.min(dim=-1).values
        group_scale = torch.clamp((w_max - w_min) / 15.0, min=1e-8)
        group_zp = torch.round(-w_min / group_scale).clamp(0, 15).to(torch.float16)

        w_int4 = torch.clamp(
            torch.round(fp_groups / group_scale.unsqueeze(-1)) + group_zp.unsqueeze(-1),
            0, 15).to(torch.int8)

        w_low = w_int4[:, :, 0::2]
        w_high = w_int4[:, :, 1::2]
        w_packed = ((w_high << 4) | (w_low & 0x0F)).view(C_out, C_in // 2).to(torch.int8)

        submodule.weight.data = w_packed
        group_scale = group_scale.reshape(C_out, num_groups).to(torch.float16)
        group_zp = group_zp.reshape(C_out, num_groups).to(torch.float16)
        submodule.register_buffer('wscales', group_scale.reshape(C_out, -1))
        submodule.register_buffer('w_szs', group_zp.reshape(C_out, -1))
    else:
        # W8A8: standard per-channel INT8 quantization
        int_weight = torch.clamp(
                torch.round(fp_weight / scale.view(-1,1)) - zero_point.view(-1,1),
                -128, 127).to(torch.int8)
        submodule.weight.data = int_weight

    
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
    LlamaRMSNorm,
)
from timm.models.layers import DropPath 
from einops import rearrange

class STDiT3BlockWithCudaKernel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        quant_params=None,
        use_kernel_override=None,  # [attn, cross_attn, mlp] — override for ViDiT-Q blocks
    ):
        super().__init__()

        self.quant_params = quant_params

        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if use_kernel_override is not None:
            self.use_kernel = list(use_kernel_override)
        else:
            self.use_kernel = [False, True, True] 

        if self.enable_sequence_parallelism and not temporal:
            raise AssertionError
            attn_cls = SeqParallelAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = AttentionWithCudaKernel if self.use_kernel[0] else Attention
            mha_cls = MultiHeadCrossAttentionWithCudaKernel if self.use_kernel[1] else MultiHeadCrossAttention
            
        if self.use_kernel[0]:
            # When use_kernel[0]=True but attn uses ViDiTQW8A8Linear (which does its own
            # ViDiT-Q quantization), we need norm1 to output FP16 (not pre-quantized INT8).
            # Use standard layernorm without quantization fusion.
            self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.attn = attn_cls(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                rope=rope,
                enable_flash_attn=enable_flash_attn,
                quant_params=self.quant_params,
            )
        else:
            self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.attn = attn_cls(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                rope=rope,
                enable_flash_attn=enable_flash_attn,
            )
        
        if self.use_kernel[1]:
            self.cross_attn = mha_cls(hidden_size, num_heads, quant_params=self.quant_params)
        else:
            self.cross_attn = mha_cls(hidden_size, num_heads)
            
        if self.use_kernel[2]:
            self.norm2 = LayerNormGeneral(hidden_size, act_sum=True, eps=1e-6)
            self.mlp = MlpWithCudaKernel(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
                quant_params=self.quant_params,
            )
        else:
            self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
            )
            
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        timestep=None,  # PAB: current denoising timestep (unused in HW path, accepted for compatibility)
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        x = x.contiguous()
            
        # attention
        if self.use_kernel[0]:
            # CUDA kernel self-attention path.
            # norm1: standard LayerNorm (FP16 output, no pre-quantization)
            # attn: uses AttentionWithCudaKernel which internally quantizes before GEMM
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            # modulate (attention) with fused gate+residual
            residual = x
            x = fused_kernels.gate_residual_fuse(x_m.contiguous().view(-1, C), gate_msa.view(-1, C), residual.contiguous().view(-1, C)).reshape([B, N, C])
        else:
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            x_m_s = gate_msa * x_m
            x = x + self.drop_path(x_m_s)

        # cross attention
        if self.use_kernel[1]:    
            residual = x
            x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
            x = self.cross_attn(x, y, mask)
            x = residual + x
        else:
            x = x + self.cross_attn(x, y, mask)
            
        # MLP
        if self.use_kernel[2]:
            # modulate (MLP)
            residual = x
            x = self.norm2(x, shift_mlp, scale_mlp, self.quant_params)
            x = self.mlp(x)
            x = fused_kernels.gate_residual_fuse(x.contiguous().view(-1, C), gate_mlp.view(-1, C), residual.contiguous().view(-1, C)).reshape([B, N, C])
        else:
            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
            # MLP
            x_m = self.mlp(x_m)
            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            # residual
            x = x + self.drop_path(x_m_s)

        return x

class AttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        quant_params=None,
        has_bias=True,
        weight_sym=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = W8A8OF16LinearDynamicInputScale(dim, dim * 3, has_bias=has_bias, weight_sym=weight_sym)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False
        self.quant_params = quant_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Detect input type: FP16 → ViDiT-Q path (internal quantization);
        # INT8 → standard path (pre-quantized by caller).
        input_is_fp16 = (x.dtype == torch.float16)

        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x, self.quant_params)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)

        if input_is_fp16:
            # ViDiT-Q path: proj handles its own quantization (FP16→ViDiT-Q→INT8→GEMM)
            x = self.proj(x, self.quant_params)
        else:
            # Standard path: quantize attention output before proj
            x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
            x = self.proj(x, self.quant_params)

        x = self.proj_drop(x)
        return x
    
class MultiHeadCrossAttentionWithCudaKernel(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, \
            quant_params=None, has_bias=True, weight_sym=False):
        super(MultiHeadCrossAttentionWithCudaKernel, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = W8A8OF16LinearDynamicInputScale(d_model, d_model, has_bias=has_bias, weight_sym=weight_sym)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = W8A8OF16LinearDynamicInputScale(d_model, d_model, has_bias=has_bias, weight_sym=weight_sym)

        self.quant_params = quant_params

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x, self.quant_params).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=_get_xformers_op()).view(B, N, C)

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.proj(x, self.quant_params)

        return x

# located in timm.layers.Mlp
class MlpWithCudaKernel(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            # quant related attributes.
            weight_sym=False,
            quant_params=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        
        self.fc1 = W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=bias[0], weight_sym=weight_sym)
        self.fc2 = W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=bias[1], weight_sym=weight_sym)
        self.quant_params = quant_params

    def forward(self, x):
        x = self.fc1(x, self.quant_params)
        x = fused_kernels.gelu_quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.fc2(x, self.quant_params)
        return x
