"""Core innovation of RGCA + foreground-mask-guided GFB.

This file is intentionally self-contained.  It extracts the active, no-prior
innovation used by the C0 detector from the much larger ``models/common.py``:

1. shared-latent bidirectional cross-covariance attention;
2. directional spatial reliability gates (IR -> RGB and RGB -> IR);
3. global/local complementary-value mixing and bounded residual injection;
4. foreground-aware spatial Mask;
5. Mask-guided original GFB modal fusion;
6. optional detached reliability supervision (RS-RGCA).

The module/parameter names match the active ``HAFFormerRGCAMaskGFB`` path, so
the corresponding submodule tensors in trained no-prior RGCA checkpoints can
be loaded directly.  It has no dependency on YOLO-specific code.

Example
-------
>>> block = HAFFormerRGCAMaskGFB(dim=128)
>>> rgb = torch.randn(2, 128, 64, 64)
>>> ir = torch.randn(2, 128, 64, 64)
>>> fused = block((rgb, ir))
>>> fused.shape
torch.Size([2, 128, 64, 64])
"""

from __future__ import annotations

import math
import numbers
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ReliabilityGuidedBidirectionalCrossAttention",
    "ForegroundSparseFusionMask",
    "GatedModalFusion",
    "HAFFormerRGCAMaskGFB",
    "enable_rs_rgca_reliability_supervision",
    "disable_rs_rgca_reliability_supervision",
    "build_directional_reliability_targets",
    "compute_rs_rgca_reliability_loss",
]


# ---------------------------------------------------------------------------
# Small self-contained utilities
# ---------------------------------------------------------------------------


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """B,C,H,W -> B,H*W,C for channel LayerNorm."""
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """B,H*W,C -> B,C,H,W."""
    return x.reshape(x.shape[0], height, width, x.shape[-1]).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("BiasFree_LayerNorm expects one channel dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBias_LayerNorm expects one channel dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """LayerNorm over channels at every spatial position."""

    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBias_LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported LayerNorm type: {layer_norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class Concat(nn.Module):
    """Parameter-free compatibility wrapper used by the original GFB."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = int(dimension)

    def forward(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(tensors), dim=self.d)


def inverse_softplus(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = torch.as_tensor(value).clamp_min(eps)
    return value + torch.log(-torch.expm1(-value))


def init_positive_temperature(num_heads: int, value: float = 1.0) -> torch.Tensor:
    return inverse_softplus(torch.full((num_heads, 1, 1), float(value)))


def positive_temperature(raw_temperature: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(raw_temperature) + eps


def inverse_bounded_sigmoid(
    value: float, lower: float, upper: float, eps: float = 1e-4
) -> torch.Tensor:
    if not lower < upper:
        raise ValueError("lower must be smaller than upper")
    value_tensor = torch.as_tensor(value, dtype=torch.float32)
    probability = ((value_tensor - lower) / (upper - lower)).clamp(
        eps, 1.0 - eps
    )
    return torch.log(probability) - torch.log1p(-probability)


# ---------------------------------------------------------------------------
# Innovation 1: reliability-guided bidirectional cross attention (RGCA)
# ---------------------------------------------------------------------------


class ReliabilityGuidedBidirectionalCrossAttention(nn.Module):
    """Frequency-free, reliability-guided RGB/IR cross attention.

    RGB queries interact with IR keys/values, while IR queries interact with
    RGB keys/values.  QKV weights are shared so both modalities live in the
    same latent coordinates.  Channel cross-covariance attention avoids the
    quadratic ``(H*W)^2`` token-attention matrix.

    Direction convention
    --------------------
    ``reliability_rgb`` / ``r_ir_to_rgb`` controls IR information injected
    into RGB. ``reliability_ir`` / ``r_rgb_to_ir`` controls RGB information
    injected into IR.

    ``prior_mode`` is kept in the signature for YAML/checkpoint compatibility,
    but this consolidated innovation file deliberately contains only the
    experimentally selected no-prior path.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 2,
        num_heads: int = 4,
        bias: bool = False,
        local_kernel_size: int = 5,
        residual_init: float = 0.1,
        channel_prior_init: float = 1.0,
        channel_prior_min: float = 0.05,
        channel_prior_max: float = 1.5,
        residual_min: float = 0.02,
        residual_max: float = 0.15,
        prior_mode: str = "none",
        becp_max_margin: float = 0.5,
        dacp_rho_max: float = 0.1,
        dacp_rho_init: float = 0.01,
        residual_mode: str = "legacy_channel",
        use_reliability_gate: bool = True,
    ):
        super().__init__()
        # Compatibility arguments are intentionally named even though the
        # selected no-prior implementation does not consume them.
        del channel_prior_init, channel_prior_min, channel_prior_max
        del becp_max_margin, dacp_rho_max, dacp_rho_init

        if dim <= 0 or reduction <= 0:
            raise ValueError("dim and reduction must be positive")
        if local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd")
        if not residual_min < residual_max:
            raise ValueError("residual_min must be smaller than residual_max")
        if not residual_min <= residual_init <= residual_max:
            raise ValueError("residual_init must be inside residual bounds")
        if prior_mode != "none":
            raise ValueError(
                "This consolidated file implements the selected prior_mode='none' "
                "path only; historical prior ablations remain in models/common.py"
            )
        if residual_mode not in ("legacy_channel", "bounded_scalar"):
            raise ValueError(f"Unsupported residual mode: {residual_mode}")

        reduced_dim = min(max(dim // reduction, 16), dim)
        heads = min(int(num_heads), reduced_dim)
        while heads > 1 and reduced_dim % heads:
            heads -= 1
        reduced_dim = max(heads, (reduced_dim // heads) * heads)

        self.dim = int(dim)
        self.reduced_dim = int(reduced_dim)
        self.num_heads = int(heads)
        self.head_dim = self.reduced_dim // self.num_heads
        self.residual_min = float(residual_min)
        self.residual_max = float(residual_max)
        self.residual_mode = str(residual_mode)
        self.prior_mode = "none"
        self.channel_prior = None

        self.rgb_norm = LayerNorm(dim, "WithBias")
        self.ir_norm = LayerNorm(dim, "WithBias")

        # Shared projections establish one comparable RGB/IR latent space.
        self.qkv = nn.Conv2d(dim, 3 * self.reduced_dim, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            3 * self.reduced_dim,
            3 * self.reduced_dim,
            3,
            stride=1,
            padding=1,
            groups=3 * self.reduced_dim,
            bias=bias,
        )
        self.local_value = nn.Conv2d(
            self.reduced_dim,
            self.reduced_dim,
            local_kernel_size,
            stride=1,
            padding=local_kernel_size // 2,
            groups=self.reduced_dim,
            bias=bias,
        )

        self.raw_temperature = nn.Parameter(
            init_positive_temperature(
                self.num_heads, value=math.sqrt(self.head_dim)
            )
        )
        self.cross_mix_logit = nn.Parameter(torch.zeros(self.num_heads))

        self.use_reliability_gate = bool(use_reliability_gate)
        gate_hidden = max(self.reduced_dim // 2, 8)
        reliability_gate = nn.Sequential(
            nn.Conv2d(4 * self.reduced_dim, gate_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                gate_hidden,
                gate_hidden,
                3,
                stride=1,
                padding=1,
                groups=gate_hidden,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.kaiming_normal_(
            reliability_gate[-2].weight, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(reliability_gate[-2].bias)
        # Discarding after initialization preserves RNG equivalence for the
        # strict no-reliability ablation while leaving no hidden parameters.
        self.reliability_gate = reliability_gate if self.use_reliability_gate else None

        self.out_proj = nn.Conv2d(self.reduced_dim, dim, 1, bias=bias)
        if self.residual_mode == "legacy_channel":
            self.residual_scale = nn.Parameter(
                torch.full((2, dim, 1, 1), float(residual_init))
            )
        else:
            raw = inverse_bounded_sigmoid(
                residual_init, self.residual_min, self.residual_max
            )
            self.raw_residual_scale = nn.Parameter(raw.repeat(2).view(2, 1, 1, 1))

        # Transient hooks: no checkpoint parameter or persistent buffer is added.
        self.track_attention_stats = False
        self.last_attention_stats = None
        self.enable_rs_reliability_supervision = False
        self.last_rs_reliability_predictions: Optional[Dict[str, torch.Tensor]] = None
        self.export_reliability_maps = False
        self.last_reliability_maps: Optional[Dict[str, torch.Tensor]] = None
        self.export_cross_modal_response_maps = False
        self.last_cross_modal_response_maps: Optional[Dict[str, torch.Tensor]] = None

    def get_temperature(self) -> torch.Tensor:
        return positive_temperature(self.raw_temperature)

    def get_residual_scale(self) -> torch.Tensor:
        if hasattr(self, "residual_scale"):
            return self.residual_scale.float()
        span = self.residual_max - self.residual_min
        return self.residual_min + span * torch.sigmoid(self.raw_residual_scale.float())

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        return x.reshape(batch, self.num_heads, self.head_dim, height * width)

    def _from_heads(
        self, x: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        return x.reshape(x.shape[0], self.reduced_dim, height, width)

    def _content_logits(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.get_temperature().to(dtype=query.dtype).unsqueeze(0)
        return (query @ key.transpose(-2, -1)) * temperature

    @staticmethod
    def _descriptor(primary: torch.Tensor, complementary: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                primary,
                complementary,
                torch.abs(primary - complementary),
                primary * complementary,
            ],
            dim=1,
        )

    def _reliability_with_auxiliary(
        self, primary: torch.Tensor, complementary: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        descriptor = self._descriptor(primary, complementary)
        reliability = self.reliability_gate(descriptor)
        auxiliary = None
        if self.training and self.enable_rs_reliability_supervision:
            # Stop-gradient is the key separation: auxiliary reliability labels
            # train the gate, not a shortcut in the backbone/QKV features.
            auxiliary = self.reliability_gate(descriptor.detach())
        return reliability, auxiliary

    def _record_stats(
        self,
        rgb_attention: torch.Tensor,
        ir_attention: torch.Tensor,
        rgb_reliability: torch.Tensor,
        ir_reliability: torch.Tensor,
    ) -> None:
        if not self.track_attention_stats:
            return

        def normalized_entropy(attention: torch.Tensor) -> float:
            probability = attention.detach().float().clamp_min(1e-8)
            denominator = math.log(max(self.head_dim, 2))
            value = -(probability * probability.log()).sum(-1) / denominator
            return float(value.mean())

        residual_scale = self.get_residual_scale().detach().float()
        self.last_attention_stats = (
            normalized_entropy(rgb_attention),
            normalized_entropy(ir_attention),
            float(rgb_attention.detach().float().std()),
            float(ir_attention.detach().float().std()),
            float(rgb_reliability.detach().float().mean()),
            float(rgb_reliability.detach().float().std()),
            float(ir_reliability.detach().float().mean()),
            float(ir_reliability.detach().float().std()),
            float(torch.sigmoid(self.cross_mix_logit.detach().float()).mean()),
            float(residual_scale.mean()),
            float(residual_scale.min()),
            0.0,  # no-prior margin
            1.0 / float(self.head_dim),
        )
        self.track_attention_stats = False

    def forward(
        self, rgb_fea: torch.Tensor, ir_fea: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb_fea.shape != ir_fea.shape:
            raise ValueError(
                "RGB and IR features must have identical shapes, got "
                f"{tuple(rgb_fea.shape)} and {tuple(ir_fea.shape)}"
            )
        _, channels, height, width = rgb_fea.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}")

        normalized = torch.cat(
            [self.rgb_norm(rgb_fea), self.ir_norm(ir_fea)], dim=0
        )
        projected = self.qkv_dwconv(self.qkv(normalized))
        rgb_projected, ir_projected = projected.chunk(2, dim=0)
        q_rgb, k_rgb, v_rgb = rgb_projected.chunk(3, dim=1)
        q_ir, k_ir, v_ir = ir_projected.chunk(3, dim=1)

        q_rgb_heads = F.normalize(self._to_heads(q_rgb), dim=-1, eps=1e-6)
        k_rgb_heads = F.normalize(self._to_heads(k_rgb), dim=-1, eps=1e-6)
        q_ir_heads = F.normalize(self._to_heads(q_ir), dim=-1, eps=1e-6)
        k_ir_heads = F.normalize(self._to_heads(k_ir), dim=-1, eps=1e-6)
        v_rgb_heads = self._to_heads(v_rgb)
        v_ir_heads = self._to_heads(v_ir)

        # True bidirectional cross-modal content logits.
        content_rgb = self._content_logits(q_rgb_heads, k_ir_heads)
        content_ir = self._content_logits(q_ir_heads, k_rgb_heads)
        attention_rgb = content_rgb.float().softmax(dim=-1).to(v_ir_heads.dtype)
        attention_ir = content_ir.float().softmax(dim=-1).to(v_rgb_heads.dtype)
        cross_rgb = attention_rgb @ v_ir_heads
        cross_ir = attention_ir @ v_rgb_heads

        # Complementary local values are mixed with the global channel path.
        local_values = self.local_value(torch.cat([v_rgb, v_ir], dim=0))
        local_rgb_source, local_ir_source = local_values.chunk(2, dim=0)
        local_rgb = self._to_heads(local_ir_source)
        local_ir = self._to_heads(local_rgb_source)
        cross_mix = torch.sigmoid(self.cross_mix_logit).to(cross_rgb.dtype).view(
            1, self.num_heads, 1, 1
        )
        mixed_rgb = cross_mix * cross_rgb + (1.0 - cross_mix) * local_rgb
        mixed_ir = cross_mix * cross_ir + (1.0 - cross_mix) * local_ir
        mixed_rgb = self._from_heads(mixed_rgb, height, width)
        mixed_ir = self._from_heads(mixed_ir, height, width)

        if self.use_reliability_gate:
            reliability_rgb, auxiliary_rgb = self._reliability_with_auxiliary(
                v_rgb, v_ir
            )
            reliability_ir, auxiliary_ir = self._reliability_with_auxiliary(
                v_ir, v_rgb
            )
        else:
            reliability_rgb = mixed_rgb.new_ones((mixed_rgb.shape[0], 1, height, width))
            reliability_ir = mixed_ir.new_ones((mixed_ir.shape[0], 1, height, width))
            auxiliary_rgb = auxiliary_ir = None

        if auxiliary_rgb is not None and auxiliary_ir is not None:
            # Historical gates are neutral around 0.5, hence 2*GAP(gate)
            # represents normalized reliability with a clean neutral value 1.
            self.last_rs_reliability_predictions = {
                "r_ir_to_rgb": 2.0 * auxiliary_rgb.mean(dim=(2, 3)),
                "r_rgb_to_ir": 2.0 * auxiliary_ir.mean(dim=(2, 3)),
            }
        else:
            self.last_rs_reliability_predictions = None

        if self.export_reliability_maps:
            self.last_reliability_maps = {
                "r_ir_to_rgb": reliability_rgb.detach().float().cpu(),
                "r_rgb_to_ir": reliability_ir.detach().float().cpu(),
            }

        mixed = torch.cat(
            [mixed_rgb * reliability_rgb, mixed_ir * reliability_ir], dim=0
        )
        delta_rgb, delta_ir = self.out_proj(mixed).chunk(2, dim=0)
        residual_scale = self.get_residual_scale().to(
            device=delta_rgb.device, dtype=delta_rgb.dtype
        )
        delta_rgb = delta_rgb * residual_scale[0]
        delta_ir = delta_ir * residual_scale[1]

        if self.export_cross_modal_response_maps:
            self.last_cross_modal_response_maps = {
                "rgb_update": delta_rgb.detach().float().cpu(),
                "ir_update": delta_ir.detach().float().cpu(),
            }

        self._record_stats(
            attention_rgb,
            attention_ir,
            reliability_rgb,
            reliability_ir,
        )
        return delta_rgb, delta_ir


# ---------------------------------------------------------------------------
# Innovation 2: foreground Mask + original GFB logit fusion
# ---------------------------------------------------------------------------


class ForegroundSparseFusionMask(nn.Module):
    """Predict a one-channel foreground mask from RGB/IR relations."""

    def __init__(self, channel: int, bias: bool = False, output_bias: float = 0.0):
        super().__init__()
        hidden = max(channel // 4, 16)
        self.mask_head = nn.Sequential(
            nn.Conv2d(3 * channel, hidden, 1, bias=bias),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden, hidden, 3, padding=1, groups=hidden, bias=bias
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.mask_head[-2].bias, float(output_bias))

    def forward(self, rgb_fea: torch.Tensor, ir_fea: torch.Tensor) -> torch.Tensor:
        if rgb_fea.shape != ir_fea.shape:
            raise ValueError("Foreground mask expects aligned RGB/IR feature shapes")
        mask_input = torch.cat(
            [rgb_fea, ir_fea, torch.abs(rgb_fea - ir_fea)], dim=1
        )
        return self.mask_head(mask_input)


class GatedModalFusion(nn.Module):
    """Original LCAFNet GFB, exposed as logits for Mask guidance."""

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=bias),
            nn.GELU(),
        )
        self.dwconv = nn.Conv2d(
            dim, dim, 3, padding=1, groups=dim, bias=bias
        )

    def compute_logits(
        self, rgb_fea: torch.Tensor, ir_fea: torch.Tensor
    ) -> torch.Tensor:
        seed = self.conv(self.concat([rgb_fea, ir_fea]))
        return self.dwconv(seed)

    def forward(self, rgb_fea: torch.Tensor, ir_fea: torch.Tensor) -> torch.Tensor:
        rgb_weight = self.compute_logits(rgb_fea, ir_fea).sigmoid()
        return rgb_weight * rgb_fea + (1.0 - rgb_weight) * ir_fea


class HAFFormerRGCAMaskGFB(nn.Module):
    """Complete innovation block used independently at P2/P3/P4/P5.

    The outer RGB/IR residuals preserve unimodal features.  The foreground mask
    corrects the original GFB logit instead of replacing it, retaining the
    stable baseline fusion rule while making foreground selection explicit.
    """

    def __init__(
        self,
        dim: int,
        mask_logit_gain: float = 1.0,
        attention_reduction: int = 2,
        channel_prior_init: float = 1.0,
        prior_mode: str = "none",
        becp_max_margin: float = 0.5,
        dacp_rho_max: float = 0.1,
        dacp_rho_init: float = 0.01,
        residual_mode: str = "legacy_channel",
        use_reliability_gate: bool = True,
    ):
        super().__init__()
        bias = False
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0
        )
        self.cross_modal_attention = ReliabilityGuidedBidirectionalCrossAttention(
            dim,
            reduction=attention_reduction,
            num_heads=4,
            bias=bias,
            local_kernel_size=5,
            residual_init=0.1,
            channel_prior_init=channel_prior_init,
            prior_mode=prior_mode,
            becp_max_margin=becp_max_margin,
            dacp_rho_max=dacp_rho_max,
            dacp_rho_init=dacp_rho_init,
            residual_mode=residual_mode,
            use_reliability_gate=use_reliability_gate,
        )
        self.register_buffer(
            "mask_logit_gain",
            torch.tensor(float(mask_logit_gain)),
            persistent=True,
        )
        self.track_fusion_stats = False
        self.last_fusion_stats = None

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise TypeError("HAFFormerRGCAMaskGFB expects (rgb_feature, ir_feature)")
        rgb_fea, ir_fea = x
        delta_rgb, delta_ir = self.cross_modal_attention(rgb_fea, ir_fea)
        out_rgb = rgb_fea + delta_rgb
        out_ir = ir_fea + delta_ir

        foreground_mask = self.foreground_mask(out_rgb, out_ir)
        foreground_mask = torch.nan_to_num(
            foreground_mask, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(gfb_logit.dtype) * (
            2.0 * foreground_mask - 1.0
        )
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if self.track_fusion_stats:
            mask_stats = foreground_mask.detach().float()
            weight_stats = rgb_weight.detach().float()
            self.last_fusion_stats = (
                float(mask_stats.mean()),
                float(mask_stats.std()),
                float(mask_stats.min()),
                float(mask_stats.max()),
                float(weight_stats.mean()),
                float(weight_stats.std()),
            )
            self.track_fusion_stats = False
        return output


# ---------------------------------------------------------------------------
# Innovation 3: optional detached directional reliability supervision
# ---------------------------------------------------------------------------


def _iter_rgca_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, ReliabilityGuidedBidirectionalCrossAttention):
            yield module


def enable_rs_rgca_reliability_supervision(model: nn.Module) -> int:
    """Enable transient gate-only auxiliary predictions; return gate count."""
    count = 0
    for module in _iter_rgca_modules(model):
        if not module.use_reliability_gate:
            continue
        module.enable_rs_reliability_supervision = True
        module.last_rs_reliability_predictions = None
        count += 1
    if count == 0:
        raise RuntimeError("No reliability-enabled RGCA module was found")
    return count


def disable_rs_rgca_reliability_supervision(model: nn.Module) -> None:
    for module in _iter_rgca_modules(model):
        module.enable_rs_reliability_supervision = False
        module.last_rs_reliability_predictions = None


def build_directional_reliability_targets(
    quality_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[q_rgb, q_ir, correspondence]`` to directional targets.

    Source reliability, rather than destination reliability, controls transfer:
    IR -> RGB uses ``q_ir * correspondence`` and RGB -> IR uses
    ``q_rgb * correspondence``.
    """
    if quality_target.ndim != 2 or quality_target.shape[1] != 3:
        raise ValueError("quality_target must have shape [batch, 3]")
    target_rgb, target_ir, target_pair = quality_target.float().split(1, dim=1)
    return target_ir * target_pair, target_rgb * target_pair


def compute_rs_rgca_reliability_loss(
    model: nn.Module,
    quality_target: torch.Tensor,
    ranking_gain: float = 0.25,
    clear_predictions: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate existing directional gates without gradients into features.

    Returns ``(total, detached_calibration, detached_ranking)``.  Multiply
    ``total`` by the desired auxiliary gain (the experiments use a small gain,
    e.g. 0.02) before adding it to the detector loss.
    """
    if ranking_gain < 0:
        raise ValueError("ranking_gain must be non-negative")
    target_ir_to_rgb, target_rgb_to_ir = build_directional_reliability_targets(
        quality_target
    )
    calibration_losses = []
    ranking_losses = []
    for module in _iter_rgca_modules(model):
        prediction = module.last_rs_reliability_predictions
        if prediction is None:
            continue
        pred_ir_to_rgb = prediction["r_ir_to_rgb"].float()
        pred_rgb_to_ir = prediction["r_rgb_to_ir"].float()
        calibration_losses.append(
            (
                F.smooth_l1_loss(pred_ir_to_rgb, target_ir_to_rgb)
                + F.smooth_l1_loss(pred_rgb_to_ir, target_rgb_to_ir)
            )
            / 2.0
        )
        ranking_losses.append(
            F.smooth_l1_loss(
                pred_ir_to_rgb - pred_rgb_to_ir,
                target_ir_to_rgb - target_rgb_to_ir,
            )
        )
        if clear_predictions:
            module.last_rs_reliability_predictions = None

    if not calibration_losses:
        raise RuntimeError(
            "No RS-RGCA predictions are available; enable supervision and run "
            "a training-mode forward pass before computing this loss"
        )
    calibration = torch.stack(calibration_losses).mean()
    ranking = torch.stack(ranking_losses).mean()
    total = calibration + float(ranking_gain) * ranking
    return total, calibration.detach(), ranking.detach()


if __name__ == "__main__":
    # Lightweight executable smoke test for the standalone file.
    torch.manual_seed(7)
    demo = HAFFormerRGCAMaskGFB(dim=64).train()
    enable_rs_rgca_reliability_supervision(demo)
    demo_rgb = torch.randn(2, 64, 24, 32, requires_grad=True)
    demo_ir = torch.randn(2, 64, 24, 32, requires_grad=True)
    demo_output = demo((demo_rgb, demo_ir))
    demo_quality = torch.tensor([[1.0, 0.4, 0.9], [0.7, 1.0, 1.0]])
    demo_aux, _, _ = compute_rs_rgca_reliability_loss(demo, demo_quality)
    (demo_output.mean() + 0.02 * demo_aux).backward()
    print(
        "standalone RGCA+Mask+GFB OK:",
        tuple(demo_output.shape),
        "aux=", float(demo_aux.detach()),
    )
