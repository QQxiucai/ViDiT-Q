import torch
import torch.nn as nn
import viditq_extension.qgemm as qgemm
from viditq_extension.nn.base import QuantParams

class W8A8OF16LinearDynamicInputScale(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "bias",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ) if self.has_bias else None,
        )
        
        self.register_buffer(
            "scale_weight",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "zp_weight",
            torch.empty(
                self.out_features,
                dtype=torch.int16,
                requires_grad=False,
            ) if not weight_sym else None,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_sym: bool = True,
        init_only: bool = False,
    ):
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            has_bias=linear.bias is not None,
            weight_sym=weight_sym,
        )

        assert linear.weight.dtype == torch.float16

        if init_only:
            return q_linear

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)

        ### quantize the weight ###
        fp16_weight = linear.weight.data

        if weight_sym:
            weight_scale = fp16_weight.abs().max(dim=-1).values / 127.0
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)), -128, 127)

            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()
            
        else:
            weight_max = fp16_weight.max(dim=-1).values
            weight_min = fp16_weight.min(dim=-1).values

            weight_scale = (weight_max - weight_min) / 255.0
            weight_zp = torch.round((weight_min) / weight_scale) + 128
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)) - weight_zp.view(-1, 1), -128, 127)

            q_linear.zp_weight.data[:] = weight_zp.to(torch.int16).contiguous()
            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()

        return q_linear
    
    def forward(self,
        input: torch.Tensor,
        quant_params: QuantParams,
    ):
        # TODO: implement the complete forward pass for other case
        shape = input.shape
        hidden_size = shape[-1]
        output = qgemm.w8a8_of16_bias_weight_asym(
            input.view(-1, hidden_size),
            self.weight,
            self.bias,
            quant_params.scale_input,
            self.scale_weight,
            quant_params.sum_input,
            self.zp_weight,
        )

        return output.view(*shape[:-1], self.out_features)


class W4A8OF16LinearDynamicInputScale(nn.Module):
    """W4A8 Linear with per-channel-group weight quantization.

    Uses the QServe W4A8 GEMM kernel (w4a8_of16_nobias_weight_asym_qserve).
    Weight is packed as 2×4-bit values per INT8 byte with per-group (G=128) scales.
    No bias support.
    """

    G = 128  # group size for per-group weight quantization

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Packed weight: [out_features, in_features // 2] INT8
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8),
        )

        # Per-group weight scales: [out_features, in_features // G] half2-packed
        self.register_buffer(
            "wscales",
            torch.empty(out_features, in_features // self.G, dtype=torch.float16),
        )

        # Per-group weight zero-points: [out_features, in_features // G] half2-packed
        self.register_buffer(
            "w_szs",
            torch.empty(out_features, in_features // self.G, dtype=torch.float16),
        )

    def forward(self, input: torch.Tensor, quant_params):
        """W4A8 GEMM forward.

        Args:
            input: [M, K] INT8 activation
            quant_params: QuantParams with scale_input [M] and sum_input [M]
        Returns:
            output: [M, N] FP16
        """
        import viditq_extension.qgemm as qgemm

        shape = input.shape
        K = shape[-1]
        M = input.numel() // K
        out_feats = torch.empty(M, self.out_features, dtype=torch.float16, device=input.device)

        qgemm.w4a8_of16_nobias_weight_asym_qserve(
            input.view(M, K).contiguous(),
            self.weight,
            self.wscales,
            quant_params.scale_input[:M],
            self.w_szs,
            quant_params.sum_input[:M] if quant_params.has_sum_input else torch.zeros(M, dtype=torch.float16, device=input.device),
            out_feats,
        )
        return out_feats.view(*shape[:-1], self.out_features)

