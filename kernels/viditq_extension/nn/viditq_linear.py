"""
ViDiT-Q CUDA-accelerated Linear layer.

Replaces ViDiTQuantizedLinear in the hardware inference path.
Fuses channel_mask scaling + FWHT + hadK matmul + activation quantization
into a single CUDA kernel, then runs standard W8A8 INT8 GEMM.
"""

import os
import torch
import torch.nn as nn
import viditq_extension.fused as fused_kernels
import viditq_extension.qgemm as qgemm
from viditq_extension.nn.base import QuantParams


# ---------------------------------------------------------------------------
# Lazy-load hadK matrix for the given hidden_size.
# hadK is shared across ALL ViDiT-Q layers with the same hidden_size.
# ---------------------------------------------------------------------------
_HADK_CACHE = {}  # hidden_size -> (K_block, hadK_tensor)

# Path to pre-computed Hadamard matrices.
# Resolved relative to project root (kernels/viditq_extension/nn/viditq_linear.py → 4 dirs up).
_HADAMARD_MAT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "quant_utils", "qdiff", "quarot", "hadamard_utils", "hadamard_mat.pth"
)
# Fallback: try loading via quarot_utils if direct path fails
if not os.path.isfile(_HADAMARD_MAT_PATH):
    _HADAMARD_MAT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "quant_utils", "qdiff", "quarot", "hadamard_utils", "hadamard_mat.pth"
    )
    _HADAMARD_MAT_PATH = os.path.normpath(_HADAMARD_MAT_PATH)


def _get_hadK(hidden_size: int, device: torch.device):
    """Get the Hadamard basis matrix (hadK) for a given hidden_size.

    For hidden_size=1152: K_block=144, hadK shape [144, 144].
    The matrix is loaded once and cached.
    """
    if hidden_size in _HADK_CACHE:
        K_block, hadK = _HADK_CACHE[hidden_size]
        return K_block, hadK.to(device)

    # Map from hidden_size to hadamard matrix key and K_block
    # Based on quarot_utils.get_hadK() logic:
    #   hidden_size=1152 -> 1152 % 144 == 0 -> K=144, hadK = had144
    #   hidden_size=4608 -> 4608 % 144 == 0 -> K=144, hadK = had144  (same K!)
    #   hidden_size=3456 -> 3456 % 144 == 0 -> K=144, hadK = had144
    hadamard_dict = torch.load(_HADAMARD_MAT_PATH, weights_only=True)

    # Determine K_block using quarot_utils logic
    K_block = 1
    for k_val, transpose in [(172, False), (156, False), (144, False), (140, False),
                              (108, False), (60, False), (52, False), (36, False),
                              (28, False), (40, False), (20, False), (12, False)]:
        if hidden_size % k_val == 0:
            K_block = k_val
            break

    if K_block == 1 and (hidden_size & (hidden_size - 1)) == 0:
        # is_pow2: use identity
        hadK = torch.eye(1)
    elif K_block > 1:
        key_map = {
            172: "had.172.tpal",
            156: "had.156.tpal",
            144: "had.144.tpal",
            140: "had.140.tpal",
            108: "had.108.tpal",
            60: "had.60.tpal",
            52: "had.52.tpal",
            36: "had.36.pal2",
            28: "had.28.tpal",
            40: "had.40.tpal",
            20: "had.20.tpal",
            12: "had.12.tpal",
        }
        key = key_map.get(K_block)
        # Fallback: try variants
        if key not in hadamard_dict:
            for k in hadamard_dict:
                if str(K_block) in k:
                    key = k
                    break
        hadK = hadamard_dict[key].clone().to(torch.float16).contiguous()
    else:
        # Fallback: identity
        hadK = torch.eye(1)

    _HADK_CACHE[hidden_size] = (K_block, hadK)
    return K_block, hadK.to(device)


class ViDiTQW8A8Linear(nn.Module):
    """W8A8 Linear with ViDiT-Q activation preprocessing in CUDA.

    Forward pipeline:
      1. ViDiT-Q fused kernel (CUDA):
         FP16 act -> channel_mask + random_sign + FWHT + hadK -> INT8 act + scale + sum
      2. Standard W8A8 INT8 GEMM (CUDA):
         INT8 act @ INT8 weight -> FP16 output (with dequant + bias correction)

    The weight is pre-processed during PTQ (channel_mask scaling + Hadamard rotation
    are absorbed into the INT8 weight offline).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_mask: torch.Tensor,    # [in_features] FP16
        random_signs: torch.Tensor,    # [in_features] INT8 (+/-1), or None for all +1
        has_bias: bool = True,
        weight_sym: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # --- Buffers for ViDiT-Q preprocessing ---
        self.register_buffer("channel_mask", channel_mask.to(torch.float16).contiguous())
        if random_signs is not None:
            self.register_buffer("random_signs", random_signs.to(torch.int8).contiguous())
        else:
            self.register_buffer("random_signs", torch.ones(in_features, dtype=torch.int8))

        # --- Buffers for W8A8 GEMM (populated during quantize_and_save_weight) ---
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "bias",
            torch.zeros(out_features, dtype=torch.float16) if has_bias else None,
        )
        self.register_buffer(
            "scale_weight",
            torch.empty(out_features, dtype=torch.float16),
        )
        self.register_buffer(
            "zp_weight",
            None if weight_sym else torch.zeros(out_features, dtype=torch.int16),
        )

    def forward(self, input: torch.Tensor, quant_params: QuantParams):
        """
        Args:
            input: [B*N, C_in] FP16
            quant_params: QuantParams with pre-allocated scale_input and sum_input
        Returns:
            output: [B*N, C_out] FP16
        """
        shape = input.shape
        hidden_size = shape[-1]
        M = input.numel() // hidden_size
        input_2d = input.view(M, hidden_size)

        # Step 1: ViDiT-Q fused preprocessing + quantization -> INT8
        int_act = fused_kernels.viditq_act_quant_fuse(
            input_2d,
            self.channel_mask,
            self.random_signs,
            _get_hadK_cached(hidden_size, input.device),
            quant_params.scale_input[:M],
            quant_params.sum_input[:M] if quant_params.has_sum_input else torch.empty(0, device=input.device),
        )

        # Step 2: W8A8 INT8 GEMM
        output = qgemm.w8a8_of16_bias_weight_asym(
            int_act,
            self.weight,
            self.bias if self.has_bias else torch.empty(0, device=input.device),
            quant_params.scale_input[:M],
            self.scale_weight,
            quant_params.sum_input[:M] if quant_params.has_sum_input else torch.empty(0, device=input.device),
            self.zp_weight if self.zp_weight is not None else torch.empty(0, device=input.device),
        )

        return output.view(*shape[:-1], self.out_features)


class ViDiTQW4A8Linear(nn.Module):
    """W4A8 Linear with ViDiT-Q activation preprocessing in CUDA.

    Same activation preprocessing (FWHT+hadK+quant) as ViDiTQW8A8Linear,
    but uses W4A8 packed weight GEMM (QServe kernel) for the matmul.

    Weight format: 2×4-bit values packed per INT8 byte, per-group (G=128) scales.
    No bias support (W4A8 kernel limitation).
    """

    G = 128  # group size for per-group weight quantization

    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_mask: torch.Tensor,
        random_signs: torch.Tensor,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ViDiT-Q preprocessing buffers
        self.register_buffer("channel_mask", channel_mask.to(torch.float16).contiguous())
        if random_signs is not None:
            self.register_buffer("random_signs", random_signs.to(torch.int8).contiguous())
        else:
            self.register_buffer("random_signs", torch.ones(in_features, dtype=torch.int8))

        # W4A8 packed weight: [out_features, in_features // 2] INT8 (2×4-bit)
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8),
        )
        # Per-group weight scales: [out_features, in_features // G] FP16 (half2-packed)
        self.register_buffer(
            "wscales",
            torch.empty(out_features, in_features // self.G, dtype=torch.float16),
        )
        # Per-group weight zero-points
        self.register_buffer(
            "w_szs",
            torch.empty(out_features, in_features // self.G, dtype=torch.float16),
        )

    def forward(self, input: torch.Tensor, quant_params: QuantParams):
        shape = input.shape
        hidden_size = shape[-1]
        M = input.numel() // hidden_size
        input_2d = input.view(M, hidden_size)

        # Step 1: ViDiT-Q fused preprocessing + quantization -> INT8 (same as W8A8)
        int_act = fused_kernels.viditq_act_quant_fuse(
            input_2d,
            self.channel_mask,
            self.random_signs,
            _get_hadK_cached(hidden_size, input.device),
            quant_params.scale_input[:M],
            quant_params.sum_input[:M] if quant_params.has_sum_input else torch.empty(0, device=input.device),
        )

        # Step 2: W4A8 packed INT4 GEMM
        out_feats = torch.empty(M, self.out_features, dtype=torch.float16, device=input.device)
        qgemm.w4a8_of16_nobias_weight_asym_qserve(
            int_act.contiguous(),
            self.weight,
            self.wscales,
            quant_params.scale_input[:M],
            self.w_szs,
            quant_params.sum_input[:M] if quant_params.has_sum_input else torch.zeros(M, dtype=torch.float16, device=input.device),
            out_feats,
        )
        return out_feats.view(*shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Global hadK cache for CUDA kernel usage
# ---------------------------------------------------------------------------
_HADK_TENSOR_CACHE = {}  # hidden_size -> hadK_cuda_tensor


def _get_hadK_cached(hidden_size: int, device: torch.device):
    """Cached version for CUDA kernel path."""
    if hidden_size not in _HADK_TENSOR_CACHE:
        _, hadK = _get_hadK(hidden_size, device)
        _HADK_TENSOR_CACHE[hidden_size] = hadK.to(device)
    return _HADK_TENSOR_CACHE[hidden_size].to(device)


# ---------------------------------------------------------------------------
# Factory: create ViDiTQW8A8Linear from a ViDiTQuantizedLinear
# ---------------------------------------------------------------------------
def create_viditq_cuda_linear(
    viditq_linear: nn.Module,  # ViDiTQuantizedLinear
    weight_sym: bool = False,
) -> ViDiTQW8A8Linear:
    """Create a CUDA-accelerated ViDiT-Q linear layer from a software one.

    Extracts channel_mask and optionally random_signs from the ViDiTQuantizedLinear.
    The weight will be populated separately during quantize_and_save_weight.
    """
    # Extract channel_mask
    if hasattr(viditq_linear, 'channel_mask') and viditq_linear.channel_mask is not None:
        channel_mask = viditq_linear.channel_mask.detach().clone()
    else:
        # Fallback: use ones (no channel scaling)
        channel_mask = torch.ones(viditq_linear.in_features, dtype=torch.float16)

    # Extract random signs from rotation_matrix
    # rotation_matrix = matmul_hadU(diag(signs))
    # We extract signs by applying inverse matmul_hadU and reading the diagonal
    if hasattr(viditq_linear, 'rotation_matrix') and viditq_linear.rotation_matrix is not None:
        from qdiff.quarot.quarot_utils import matmul_hadU as _matmul_hadU
        rot = viditq_linear.rotation_matrix
        K = rot.shape[0]
        # matmul_hadU is its own inverse up to scaling:
        # matmul_hadU(rotation_matrix) = matmul_hadU(matmul_hadU(diag(signs))) = diag(signs) / K
        # So: signs = K * diagonal(matmul_hadU(rotation_matrix))
        recon = _matmul_hadU(rot.to(torch.float64))
        # The diagonal elements should be exactly signs[j] / K
        signs = torch.sign(recon.diag().real).to(torch.int8)
        assert torch.all(signs.abs() == 1), f"Failed to extract signs: got values outside ±1"
        random_signs = signs
    else:
        random_signs = torch.ones(viditq_linear.in_features, dtype=torch.int8)

    return ViDiTQW8A8Linear(
        in_features=viditq_linear.in_features,
        out_features=viditq_linear.out_features,
        channel_mask=channel_mask,
        random_signs=random_signs,
        has_bias=viditq_linear.bias is not None,
        weight_sym=weight_sym,
    )
