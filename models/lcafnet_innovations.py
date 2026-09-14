"""Standalone implementation of the active LCAFNet C0 innovations.

This file collects only the modules used by the current
``c0_full_dual_rgca`` detector:

1. RGCA: reliability-guided bidirectional RGB/IR cross-modal attention.
2. Foreground mask: a detector-loss-driven spatial routing correction.
3. GFB: channel-wise gated modal fusion.
4. P2--P5 fusion: the same fusion block is instantiated at four scales.

Historical experimental branches such as frequency attention, BECP, DACP,
SOD-ASF and C3CrossConv are deliberately excluded.  The active constructor,
module hierarchy and state-dict names match ``models.common`` so a trained C0
fusion block can be loaded here with ``strict=True``.

The production training path still imports the implementation in
``models.common``.  Keeping this file separate makes it a readable reference
without changing the behavior of an already validated model.
"""

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# The C0 detector applies the proposed fusion at P2, P3, P4 and P5.
C0_FUSION_CHANNELS = (128, 256, 512, 1024)


def to_3d(x):
    """Convert BCHW features to B(HW)C for channel normalization."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, height, width):
    """Restore B(HW)C features to BCHW."""
    return rearrange(
        x, 'b (h w) c -> b c h w', h=height, w=width)


class BiasFree_LayerNorm(nn.Module):
    """Bias-free channel LayerNorm retained for checkpoint compatibility."""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError('LayerNorm expects a one-dimensional shape')
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Channel LayerNorm with modality-specific affine parameters."""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError('LayerNorm expects a one-dimensional shape')
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return ((x - mean) / torch.sqrt(variance + 1e-5)
                * self.weight + self.bias)


class LayerNorm(nn.Module):
    """Apply LayerNorm over channels while preserving the feature map."""

    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class Concat(nn.Module):
    """Small compatibility wrapper used by the original GFB."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, tensors):
        return torch.cat(tensors, self.d)


def inverse_softplus(value, eps=1e-6):
    value = torch.as_tensor(value).clamp_min(eps)
    return value + torch.log(-torch.expm1(-value))


def init_positive_temperature(num_heads, value=1.0):
    return inverse_softplus(
        torch.full((num_heads, 1, 1), float(value)))


def positive_temperature(raw_temperature, eps=1e-6):
    return F.softplus(raw_temperature) + eps


def inverse_bounded_sigmoid(value, lower, upper, eps=1e-4):
    """Map a bounded residual scale to an unconstrained logit."""
    if not lower < upper:
        raise ValueError('lower must be smaller than upper')
    value = torch.as_tensor(value, dtype=torch.float32)
    probability = ((value - lower) / (upper - lower)).clamp(
        eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class GatedModalFusion(nn.Module):
    """GFB: learn a per-channel RGB/IR fusion logit efficiently.

    A 1x1 projection first compresses concatenated RGB/IR features from 2C to
    C channels.  A depthwise 3x3 convolution then predicts one modal-routing
    logit per channel and spatial location.
    """

    def __init__(self, dim, bias=False):
        super().__init__()
        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(
            nn.Conv2d(
                2 * dim, dim, kernel_size=1, stride=1,
                padding=0, bias=bias),
            nn.GELU(),
        )
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1,
            groups=dim, bias=bias)

    def compute_logits(self, rgb_fea, ir_fea):
        seed = self.conv(self.concat([rgb_fea, ir_fea]))
        return self.dwconv(seed)

    def forward(self, rgb_fea, ir_fea):
        rgb_weight = self.compute_logits(rgb_fea, ir_fea).sigmoid()
        return rgb_weight * rgb_fea + (1.0 - rgb_weight) * ir_fea


class ForegroundSparseFusionMask(nn.Module):
    """Predict a lightweight foreground-aware spatial mask.

    RGB, IR and their absolute discrepancy provide complementary appearance,
    thermal and cross-modal-relation cues.  The mask is learned through the
    detection loss and therefore requires no additional pixel-level labels.
    """

    def __init__(self, channel, bias=False, output_bias=2.0):
        super().__init__()
        hidden = max(channel // 4, 16)
        self.mask_head = nn.Sequential(
            nn.Conv2d(
                3 * channel, hidden, kernel_size=1, stride=1,
                padding=0, bias=bias),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden, hidden, kernel_size=3, stride=1, padding=1,
                groups=hidden, bias=bias),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden, 1, kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.mask_head[-2].bias, float(output_bias))

    def forward(self, rgb_fea, ir_fea):
        mask_input = torch.cat(
            [rgb_fea, ir_fea, torch.abs(rgb_fea - ir_fea)], dim=1)
        return self.mask_head(mask_input)


class ReliabilityGuidedBidirectionalCrossAttention(nn.Module):
    """RGCA for aligned RGB/IR feature maps.

    Innovation A -- true bidirectional cross-modal interaction:
    RGB queries attend to IR keys/values and IR queries attend to RGB
    keys/values.  Shared QKV projections place both modalities in a common
    latent space while separate LayerNorms retain modal statistics.

    Innovation B -- global/local complementary modeling:
    cross-covariance channel attention supplies global interaction with cost
    linear in H*W; a depthwise value path supplies an inexpensive local bias.

    Innovation C -- reliability-controlled residual transfer:
    a content gate built from the primary feature, complementary feature,
    absolute difference and agreement can suppress unreliable cross-modal
    information.  A small residual scale preserves the original feature path.

    This standalone file intentionally implements the active ``prior_mode``
    of ``none``.  Experimental BECP/DACP/fixed-prior branches are not part of
    the current C0 contribution collected here.
    """

    def __init__(self, dim, reduction=2, num_heads=4, bias=False,
                 local_kernel_size=5, residual_init=0.1,
                 channel_prior_init=1.0, channel_prior_min=0.05,
                 channel_prior_max=1.5, residual_min=0.02,
                 residual_max=0.15, prior_mode='none',
                 becp_max_margin=0.5, dacp_rho_max=0.1,
                 dacp_rho_init=0.01, residual_mode='legacy_channel'):
        super().__init__()
        del channel_prior_init, channel_prior_min, channel_prior_max
        del becp_max_margin, dacp_rho_max, dacp_rho_init
        if dim <= 0:
            raise ValueError('dim must be positive')
        if reduction <= 0:
            raise ValueError('reduction must be positive')
        if local_kernel_size % 2 == 0:
            raise ValueError('local_kernel_size must be odd')
        if not residual_min < residual_max:
            raise ValueError('residual_min must be smaller than residual_max')
        if not residual_min <= residual_init <= residual_max:
            raise ValueError(
                f'residual_init={residual_init} must be inside '
                f'[{residual_min}, {residual_max}]')
        if prior_mode != 'none':
            raise ValueError(
                "The standalone C0 file contains only prior_mode='none'; "
                'historical prior ablations remain in models.common')

        reduced_dim = max(dim // reduction, 16)
        reduced_dim = min(reduced_dim, dim)
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
        if residual_mode not in ('legacy_channel', 'bounded_scalar'):
            raise ValueError(
                f'Unsupported RGCA residual mode: {residual_mode}')
        self.residual_mode = str(residual_mode)

        self.rgb_norm = LayerNorm(dim, 'WithBias')
        self.ir_norm = LayerNorm(dim, 'WithBias')

        # RGB and IR deliberately share both projection operators.
        self.qkv = nn.Conv2d(
            dim, 3 * self.reduced_dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            3 * self.reduced_dim,
            3 * self.reduced_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=3 * self.reduced_dim,
            bias=bias,
        )

        local_padding = local_kernel_size // 2
        self.local_value = nn.Conv2d(
            self.reduced_dim,
            self.reduced_dim,
            kernel_size=local_kernel_size,
            stride=1,
            padding=local_padding,
            groups=self.reduced_dim,
            bias=bias,
        )

        self.raw_temperature = nn.Parameter(init_positive_temperature(
            self.num_heads, value=math.sqrt(self.head_dim)))
        self.cross_mix_logit = nn.Parameter(torch.zeros(self.num_heads))
        self.channel_prior = None
        self.prior_mode = 'none'

        gate_hidden = max(self.reduced_dim // 2, 8)
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(
                4 * self.reduced_dim, gate_hidden,
                kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                gate_hidden, gate_hidden, kernel_size=3,
                stride=1, padding=1, groups=gate_hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.kaiming_normal_(
            self.reliability_gate[-2].weight,
            mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.reliability_gate[-2].bias)

        self.out_proj = nn.Conv2d(
            self.reduced_dim, dim, kernel_size=1, bias=bias)
        if self.residual_mode == 'legacy_channel':
            self.residual_scale = nn.Parameter(torch.full(
                (2, dim, 1, 1), float(residual_init)))
        else:
            raw_residual = inverse_bounded_sigmoid(
                float(residual_init), self.residual_min, self.residual_max)
            self.raw_residual_scale = nn.Parameter(
                raw_residual.repeat(2).view(2, 1, 1, 1))

        self.track_attention_stats = False
        self.last_attention_stats = None

    def get_temperature(self):
        return positive_temperature(self.raw_temperature, eps=1e-6)

    def get_residual_scale(self):
        if hasattr(self, 'residual_scale'):
            return self.residual_scale.float()
        span = self.residual_max - self.residual_min
        return self.residual_min + span * torch.sigmoid(
            self.raw_residual_scale.float())

    def _to_heads(self, x):
        return rearrange(
            x, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads)

    def _from_heads(self, x, height, width):
        return rearrange(
            x, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=height, w=width)

    def _content_logits(self, query, key):
        temperature = self.get_temperature().to(
            dtype=query.dtype).unsqueeze(0)
        return (query @ key.transpose(-2, -1)) * temperature

    def _reliability(self, primary, complementary):
        difference = torch.abs(primary - complementary)
        agreement = primary * complementary
        descriptor = torch.cat(
            [primary, complementary, difference, agreement], dim=1)
        return self.reliability_gate(descriptor)

    def _record_stats(self, rgb_attention, ir_attention,
                      rgb_reliability, ir_reliability):
        if not self.track_attention_stats:
            return

        def normalized_entropy(attention):
            probability = attention.detach().float().clamp_min(1e-8)
            denominator = math.log(max(self.head_dim, 2))
            return float((-(probability * probability.log()).sum(-1)
                          / denominator).mean())

        rgb_attention = rgb_attention.detach().float()
        ir_attention = ir_attention.detach().float()
        rgb_reliability = rgb_reliability.detach().float()
        ir_reliability = ir_reliability.detach().float()
        residual_scale = self.get_residual_scale().detach().float()
        prior_margin = residual_scale.new_zeros(self.num_heads)
        prior_diagonal_probability = residual_scale.new_full(
            (self.num_heads,), 1.0 / float(self.head_dim))
        self.last_attention_stats = (
            normalized_entropy(rgb_attention),
            normalized_entropy(ir_attention),
            float(rgb_attention.std()),
            float(ir_attention.std()),
            float(rgb_reliability.mean()),
            float(rgb_reliability.std()),
            float(ir_reliability.mean()),
            float(ir_reliability.std()),
            float(torch.sigmoid(
                self.cross_mix_logit.detach().float()).mean()),
            float(residual_scale.mean()),
            float(residual_scale.min()),
            float(prior_margin.mean()),
            float(prior_diagonal_probability.mean()),
        )
        self.track_attention_stats = False

    def forward(self, rgb_fea, ir_fea):
        if rgb_fea.shape != ir_fea.shape:
            raise ValueError(
                'RGB and IR features must have identical shapes, got '
                f'{tuple(rgb_fea.shape)} and {tuple(ir_fea.shape)}')
        _, _, height, width = rgb_fea.shape

        normalized = torch.cat(
            [self.rgb_norm(rgb_fea), self.ir_norm(ir_fea)], dim=0)
        projected = self.qkv_dwconv(self.qkv(normalized))
        rgb_projected, ir_projected = projected.chunk(2, dim=0)
        q_rgb, k_rgb, v_rgb = rgb_projected.chunk(3, dim=1)
        q_ir, k_ir, v_ir = ir_projected.chunk(3, dim=1)

        # Cross-covariance is CxC per head; no quadratic (HW)x(HW) map exists.
        q_rgb_heads = F.normalize(
            self._to_heads(q_rgb), dim=-1, eps=1e-6)
        k_rgb_heads = F.normalize(
            self._to_heads(k_rgb), dim=-1, eps=1e-6)
        q_ir_heads = F.normalize(
            self._to_heads(q_ir), dim=-1, eps=1e-6)
        k_ir_heads = F.normalize(
            self._to_heads(k_ir), dim=-1, eps=1e-6)
        v_rgb_heads = self._to_heads(v_rgb)
        v_ir_heads = self._to_heads(v_ir)

        attention_rgb = self._content_logits(
            q_rgb_heads, k_ir_heads).float().softmax(dim=-1).to(
                dtype=v_ir_heads.dtype)
        attention_ir = self._content_logits(
            q_ir_heads, k_rgb_heads).float().softmax(dim=-1).to(
                dtype=v_rgb_heads.dtype)
        cross_rgb = attention_rgb @ v_ir_heads
        cross_ir = attention_ir @ v_rgb_heads

        local_values = self.local_value(torch.cat([v_rgb, v_ir], dim=0))
        local_rgb_source, local_ir_source = local_values.chunk(2, dim=0)
        local_rgb = self._to_heads(local_ir_source)
        local_ir = self._to_heads(local_rgb_source)

        cross_mix = torch.sigmoid(self.cross_mix_logit).to(
            dtype=cross_rgb.dtype).view(1, self.num_heads, 1, 1)
        mixed_rgb = cross_mix * cross_rgb + (1.0 - cross_mix) * local_rgb
        mixed_ir = cross_mix * cross_ir + (1.0 - cross_mix) * local_ir
        mixed_rgb = self._from_heads(mixed_rgb, height, width)
        mixed_ir = self._from_heads(mixed_ir, height, width)

        reliability_rgb = self._reliability(v_rgb, v_ir)
        reliability_ir = self._reliability(v_ir, v_rgb)
        mixed = torch.cat(
            [mixed_rgb * reliability_rgb,
             mixed_ir * reliability_ir], dim=0)
        delta_rgb, delta_ir = self.out_proj(mixed).chunk(2, dim=0)
        residual_scale = self.get_residual_scale().to(
            device=delta_rgb.device, dtype=delta_rgb.dtype)
        delta_rgb = delta_rgb * residual_scale[0]
        delta_ir = delta_ir * residual_scale[1]

        self._record_stats(
            attention_rgb, attention_ir,
            reliability_rgb, reliability_ir)
        return delta_rgb, delta_ir


class HAFFormerRGCAMaskGFB(nn.Module):
    """Complete C0 fusion block: RGCA -> foreground Mask -> GFB.

    RGCA produces bounded residual enhancements for both modalities.  The
    foreground mask then adds a bounded spatial logit correction to the GFB
    channel-routing logits.  A neutral mask value of 0.5 leaves GFB unchanged.
    """

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 channel_prior_init=1.0, prior_mode='none',
                 becp_max_margin=0.5, dacp_rho_max=0.1,
                 dacp_rho_init=0.01,
                 residual_mode='legacy_channel'):
        super().__init__()
        bias = False
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.cross_modal_attention = (
            ReliabilityGuidedBidirectionalCrossAttention(
                dim, reduction=attention_reduction, num_heads=4,
                bias=bias, local_kernel_size=5, residual_init=0.1,
                channel_prior_init=channel_prior_init,
                prior_mode=prior_mode,
                becp_max_margin=becp_max_margin,
                dacp_rho_max=dacp_rho_max,
                dacp_rho_init=dacp_rho_init,
                residual_mode=residual_mode))
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)

    def forward(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError('Expected [rgb_feature, ir_feature]')
        rgb_fea, ir_fea = x
        delta_rgb, delta_ir = self.cross_modal_attention(
            rgb_fea, ir_fea)
        out_rgb = rgb_fea + delta_rgb
        out_ir = ir_fea + delta_ir

        foreground_mask = self.foreground_mask(out_rgb, out_ir)
        foreground_mask = torch.nan_to_num(
            foreground_mask, nan=0.5,
            posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * foreground_mask - 1.0)
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if getattr(self, 'track_fusion_stats', False):
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


def build_c0_fusion_pyramid(channels=C0_FUSION_CHANNELS):
    """Build the four proposed fusion blocks used at P2--P5."""
    return nn.ModuleList([
        HAFFormerRGCAMaskGFB(
            dim=dim,
            mask_logit_gain=1.0,
            attention_reduction=2,
            channel_prior_init=1.0,
            prior_mode='none',
            becp_max_margin=0.5,
            dacp_rho_max=0.1,
            dacp_rho_init=0.01,
            residual_mode='legacy_channel',
        )
        for dim in channels
    ])


__all__ = [
    'C0_FUSION_CHANNELS',
    'ForegroundSparseFusionMask',
    'GatedModalFusion',
    'HAFFormerRGCAMaskGFB',
    'ReliabilityGuidedBidirectionalCrossAttention',
    'build_c0_fusion_pyramid',
]

