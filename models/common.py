# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
import warnings
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from models.gnn import GraphReasoning

from torch.nn import init, Sequential
import math
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import numbers
from einops import rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_withoutBN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion * c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=self.expansion * c2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * c2),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion * c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class _TLACrossConv(nn.Module):
    """Residual asymmetric CrossConv used by YOLO-TLA C3CrossConv.

    YOLO-TLA Figure 4 factorizes a 3x3 convolution into 1x3 followed by
    3x1, with k=3 and stride=1 inside C3CrossConv.  This local implementation
    avoids a circular import with ``models.experimental.CrossConv`` while
    retaining the same operation and checkpoint naming (cv1/cv2).
    """

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3CrossConv(nn.Module):
    """YOLO-TLA C3CrossConv: C3/CSP shell with CrossConv residual blocks.

    The outer split, concatenation and projection are deliberately identical
    to YOLOv5 C3.  Only the internal Bottleneck sequence is replaced, which
    implements the paper's CC1 controlled backbone modification.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[
            _TLACrossConv(c_, c_, k=3, s=1, g=g, e=1.0,
                          shortcut=shortcut)
            for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class D3(nn.Module):
    """YOLO-ULM-inspired three-depthwise-convolution residual block.

    The two DSConv stages follow the paper's pointwise-then-depthwise order.
    A large-kernel depthwise convolution between them expands the receptive
    field without quadratic channel cost.  D3 is used only at equal width,
    which keeps the paper's identity residual exact.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        channels, kernel_size = int(channels), int(kernel_size)
        if channels < 1:
            raise ValueError('D3 channels must be positive')
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError('D3 large kernel must be an odd integer >= 3')
        self.dsconv1 = nn.Sequential(
            Conv(channels, channels, 1, 1),
            Conv(channels, channels, 3, 1, g=channels))
        self.large_dw = Conv(
            channels, channels, kernel_size, 1, g=channels)
        self.dsconv2 = nn.Sequential(
            Conv(channels, channels, 1, 1),
            Conv(channels, channels, 3, 1, g=channels))

    def forward(self, x):
        return x + self.dsconv2(self.large_dw(self.dsconv1(x)))


class D3C3(nn.Module):
    """Adapt YOLO-ULM D3 aggregation to the YOLOv5 C3/CSP interface.

    This deliberately retains YOLOv5 C3's two-path split and final projection,
    changing only its internal full-convolution bottlenecks into D3 blocks.
    It is therefore a controlled PAN replacement rather than a migration to a
    YOLOv12 C2f neck.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 kernel_size=7):
        super().__init__()
        if int(g) != 1:
            raise ValueError('D3C3 does not use grouped pointwise projections')
        if not shortcut:
            raise ValueError('D3C3 requires the D3 residual shortcut')
        hidden = int(c2 * float(e))
        if hidden < 1:
            raise ValueError('D3C3 hidden width must be positive')
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.m = nn.Sequential(*[
            D3(hidden, kernel_size=kernel_size) for _ in range(int(n))])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class ModalityPrivateLowRankStem(nn.Module):
    """Pack independent RGB/IR C3 features into one efficient low-rank state.

    Grouped convolution computes both modality-specific projections in one
    kernel launch. ``groups=2`` prevents any RGB/IR cross-connection: the two
    groups have distinct weights even though their computation is packed.
    """

    def __init__(self, private_channels, rank=16, downsample_steps=1):
        super().__init__()
        rank, downsample_steps = int(rank), int(downsample_steps)
        if rank < 1 or rank > private_channels:
            raise ValueError('rank must be in [1, private_channels]')
        if downsample_steps < 1:
            raise ValueError('downsample_steps must be positive')
        self.rank = rank
        self.reduce = Conv(
            2 * private_channels, 2 * rank, 1, 1, g=2)
        self.transport = nn.Sequential(*[
            Conv(2 * rank, 2 * rank, 3, 2, g=2 * rank)
            for _ in range(downsample_steps)])

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError('ModalityPrivateLowRankStem expects [rgb_c3, ir_c3]')
        rgb_private, ir_private = features
        if rgb_private.shape != ir_private.shape:
            raise RuntimeError('RGB and IR private feature shapes must match')
        return self.transport(self.reduce(torch.cat(features, dim=1)))


class ModalityPrivateLowRankDownsample(nn.Module):
    """Advance the packed RGB/IR private state by one pyramid level."""

    def __init__(self, packed_channels):
        super().__init__()
        if int(packed_channels) % 2:
            raise ValueError('packed private channels must be even')
        self.downsample = Conv(
            packed_channels, packed_channels, 3, 2, g=packed_channels)

    def forward(self, packed_private):
        return self.downsample(packed_private)


class ModalityPrivateLowRankInject(nn.Module):
    """Reliability-weight and inject a packed private state into C4 or C5.

    ``foreground_supervision`` adds a two-channel (RGB/IR), group-isolated
    training head.  The head is deliberately not executed in evaluation mode,
    so C3 has exactly the same deployed feature path as C2.
    """

    def __init__(self, shared_channels, packed_channels, initial_scale=0.1,
                 foreground_supervision=False, max_scale=0.5):
        super().__init__()
        if int(packed_channels) % 2:
            raise ValueError('packed private channels must be even')
        initial_scale, max_scale = float(initial_scale), float(max_scale)
        if not 0.0 < initial_scale < max_scale:
            raise ValueError('initial_scale must satisfy 0 < initial_scale < max_scale')
        self.rank = int(packed_channels) // 2
        self.max_scale = max_scale
        # The two grouped projections are parameter-private and cross-free.
        self.expand = nn.Conv2d(
            packed_channels, 2 * shared_channels, kernel_size=1,
            groups=2, bias=False)
        self.gate_score = nn.Conv2d(
            packed_channels, 2, kernel_size=1, groups=2, bias=True)
        nn.init.zeros_(self.gate_score.weight)
        nn.init.zeros_(self.gate_score.bias)
        self.foreground_supervision = bool(foreground_supervision)
        self.foreground_head = None
        if self.foreground_supervision:
            # Isolate auxiliary-head initialization from the global RNG so all
            # shared C2/C3 layers retain identical seeded initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3100 + int(shared_channels))
                self.foreground_head = nn.Conv2d(
                    packed_channels, 2, kernel_size=1, groups=2, bias=True)
                nn.init.normal_(self.foreground_head.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.foreground_head.bias)
        initial_probability = initial_scale / max_scale
        self.private_scale_logit = nn.Parameter(torch.tensor(
            math.log(initial_probability / (1.0 - initial_probability))))
        self.track_private_bypass_stats = False
        self.last_private_bypass_stats = None
        self.last_private_foreground_logits = None

    @property
    def private_scale(self):
        return self.max_scale * torch.sigmoid(self.private_scale_logit)

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError(
                'ModalityPrivateLowRankInject expects [shared, packed_private]')
        shared, packed_private = features
        self.last_private_foreground_logits = (
            self.foreground_head(packed_private)
            if self.training and self.foreground_head is not None else None)
        rgb_delta, ir_delta = self.expand(packed_private).chunk(2, dim=1)
        if rgb_delta.shape != shared.shape or ir_delta.shape != shared.shape:
            raise RuntimeError(
                'Private bypass spatial/channel shape does not match shared feature')
        modal_weights = torch.softmax(
            self.gate_score(F.adaptive_avg_pool2d(packed_private, 1)), dim=1)
        rgb_weight, ir_weight = modal_weights.chunk(2, dim=1)
        private_delta = rgb_weight * rgb_delta + ir_weight * ir_delta
        scale = self.private_scale.to(dtype=shared.dtype)

        if self.track_private_bypass_stats:
            with torch.no_grad():
                shared_float = shared.detach().float()
                rgb_float = rgb_delta.detach().float()
                ir_float = ir_delta.detach().float()
                mixed_float = private_delta.detach().float()
                shared_rms = shared_float.square().mean().sqrt()
                mixed_rms = mixed_float.square().mean().sqrt()
                self.last_private_bypass_stats = (
                    float(rgb_weight.detach().float().mean()),
                    float(rgb_weight.detach().float().std(unbiased=False)),
                    float(ir_weight.detach().float().mean()),
                    float(ir_weight.detach().float().std(unbiased=False)),
                    float(rgb_float.square().mean().sqrt()),
                    float(ir_float.square().mean().sqrt()),
                    float(mixed_rms),
                    float(shared_rms),
                    float(scale.detach().float()),
                    float(scale.detach().float() * mixed_rms /
                          shared_rms.clamp_min(1e-8)))
            self.track_private_bypass_stats = False

        return shared + scale * private_delta


class AdaptiveKernelConv2d(nn.Module):
    """AKConv-style learned sampling with a bounded, zero-offset start.

    The reference AKCMamba project manually gathers four bilinear neighbours
    and stacks the sampled points along the height axis.  This implementation
    uses ``grid_sample`` and a 1x1 aggregation over ``C * N`` channels, which
    is mathematically equivalent but considerably easier to audit.  Offsets
    start at zero and are bounded in pixel units to avoid unstable sampling
    during the first epochs of detector training.
    """

    def __init__(self, c1, c2, num_points=5, max_offset=2.0,
                 offset_grad_scale=0.1):
        super().__init__()
        if int(num_points) < 1:
            raise ValueError('num_points must be positive')
        if float(max_offset) <= 0:
            raise ValueError('max_offset must be positive')
        self.num_points = int(num_points)
        self.max_offset = float(max_offset)
        self.offset_grad_scale = float(offset_grad_scale)
        self.offset = nn.Conv2d(
            c1, 2 * self.num_points, kernel_size=3, stride=1,
            padding=1, bias=True)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.aggregate = Conv(c1 * self.num_points, c2, 1, 1)
        self.register_buffer(
            'base_offsets', self._make_base_offsets(self.num_points),
            persistent=True)
        self.register_buffer('_cached_pixel_grid', None, persistent=False)
        self._cached_grid_shape = None
        self.track_stats = False
        self.last_offset_stats = None

    @staticmethod
    def _make_base_offsets(num_points):
        # Select the N lattice locations nearest the origin.  N=5 gives a
        # symmetric cross instead of the corner-biased layout in the supplied
        # reference code.
        radius = int(math.ceil(math.sqrt(num_points)))
        candidates = [
            (y, x) for y in range(-radius, radius + 1)
            for x in range(-radius, radius + 1)
        ]
        candidates.sort(key=lambda point: (
            point[0] ** 2 + point[1] ** 2,
            abs(point[0]) + abs(point[1]), point[0], point[1]))
        return torch.tensor(candidates[:num_points], dtype=torch.float32)

    def forward(self, x):
        batch, channels, height, width = x.shape
        raw_offset = self.offset(x)
        if self.offset_grad_scale != 1.0:
            # Identity in the forward pass, scaled derivative in backward;
            # unlike a Tensor hook this remains safe for deepcopy/checkpoints.
            raw_offset = (
                raw_offset * self.offset_grad_scale
                + raw_offset.detach() * (1.0 - self.offset_grad_scale))
        raw_offset = raw_offset.view(
            batch, self.num_points, 2, height, width)
        pixel_offset = self.max_offset * torch.tanh(
            raw_offset / self.max_offset)
        if self.track_stats:
            detached_offset = pixel_offset.detach().float()
            self.last_offset_stats = (
                float(detached_offset.abs().mean()),
                float(detached_offset.std()),
                float(detached_offset.abs().max()))
            self.track_stats = False
        pixel_offset = pixel_offset.permute(0, 3, 4, 1, 2)

        grid_shape = (height, width, x.device.type, x.device.index, x.dtype)
        if (self._cached_pixel_grid is None
                or self._cached_grid_shape != grid_shape):
            y, x_coordinate = torch.meshgrid(
                torch.arange(height, device=x.device, dtype=x.dtype),
                torch.arange(width, device=x.device, dtype=x.dtype),
                indexing='ij')
            base = torch.stack((y, x_coordinate), dim=-1).view(
                1, height, width, 1, 2)
            self._cached_pixel_grid = (
                base + self.base_offsets.to(dtype=x.dtype).view(
                    1, 1, 1, self.num_points, 2))
            self._cached_grid_shape = grid_shape
        sample = self._cached_pixel_grid + pixel_offset.to(dtype=x.dtype)

        # grid_sample expects coordinates ordered as (x, y) in [-1, 1].
        sample_y, sample_x = sample.unbind(dim=-1)
        sample_x = (2.0 * sample_x / max(width - 1, 1)) - 1.0
        sample_y = (2.0 * sample_y / max(height - 1, 1)) - 1.0
        grid = torch.stack((sample_x, sample_y), dim=-1).reshape(
            batch, height, width * self.num_points, 2)
        sampled = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='border',
            align_corners=True)
        sampled = sampled.view(
            batch, channels, height, width, self.num_points)
        sampled = sampled.permute(0, 1, 4, 2, 3).reshape(
            batch, channels * self.num_points, height, width)
        return self.aggregate(sampled)


class AKCResidualBlock(nn.Module):
    """Paper-aligned 1x1 -> AKConv -> 1x1 block with an optional shortcut."""

    def __init__(self, channels, num_points=5, shortcut=True):
        super().__init__()
        self.cv1 = Conv(channels, channels, 1, 1)
        self.akc = AdaptiveKernelConv2d(
            channels, channels, num_points=num_points)
        self.cv2 = Conv(channels, channels, 1, 1, act=False)
        self.add = bool(shortcut)
        self.act = nn.SiLU()

    def forward(self, x):
        transformed = self.cv2(self.akc(self.cv1(x)))
        return self.act(x + transformed) if self.add else self.act(transformed)


class SelectiveAxisStateSpace2D(nn.Module):
    """Portable four-direction selective state-space context mixer.

    This is a first-order diagonal selective SSM rather than a copy of the
    reference CUDA S6 kernel.  Its input-dependent retention implements
    ``h_t = a_t h_(t-1) + (1-a_t) x_t`` in parallel with cumulative products
    and sums.  Horizontal/vertical, forward/reverse scans give every feature
    access to long-range spatial context with O(HW) arithmetic and no custom
    extension dependency.
    """

    def __init__(self, channels, state_ratio=0.25,
                 retention_min=0.90, retention_max=0.99):
        super().__init__()
        if not 0.0 < retention_min < retention_max < 1.0:
            raise ValueError('retention bounds must satisfy 0 < min < max < 1')
        state_channels = max(8, int(round(channels * float(state_ratio))))
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        self.value_proj = Conv(channels, state_channels, 1, 1)
        self.local_value = nn.Sequential(
            nn.Conv2d(
                state_channels, state_channels, kernel_size=3, stride=1,
                padding=1, groups=state_channels, bias=False),
            nn.BatchNorm2d(state_channels),
            nn.SiLU())
        self.gate_proj = nn.Conv2d(
            channels, 2 * state_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.out_proj = Conv(state_channels, channels, 1, 1, act=False)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.track_stats = False
        self.last_state_stats = None

    def _retention(self, logits):
        span = self.retention_max - self.retention_min
        return self.retention_min + span * torch.sigmoid(logits)

    @staticmethod
    def _scan(value, retention, dimension, reverse=False):
        if reverse:
            value = torch.flip(value, dims=(dimension,))
            retention = torch.flip(retention, dims=(dimension,))
        # Keep cumulative arithmetic in FP32 under AMP.  With retention >=0.9
        # the products remain well-conditioned for the P3-P5 neck resolutions.
        value_float = value.float()
        retention_float = retention.float()
        product = torch.cumprod(retention_float, dim=dimension)
        tiny = torch.finfo(product.dtype).tiny
        source = ((1.0 - retention_float) * value_float
                  / product.clamp_min(tiny))
        output = product * torch.cumsum(source, dim=dimension)
        if reverse:
            output = torch.flip(output, dims=(dimension,))
        return output.to(dtype=value.dtype)

    @classmethod
    def _bidirectional_scan(cls, value, retention, dimension):
        # Merge both directions into the batch dimension so each axis needs
        # only one cumprod/cumsum kernel sequence.
        paired_value = torch.cat(
            (value, torch.flip(value, dims=(dimension,))), dim=0)
        paired_retention = torch.cat(
            (retention, torch.flip(retention, dims=(dimension,))), dim=0)
        forward, backward = cls._scan(
            paired_value, paired_retention, dimension).chunk(2, dim=0)
        return forward + torch.flip(backward, dims=(dimension,))

    def forward(self, x):
        value = self.local_value(self.value_proj(x))
        horizontal_logits, vertical_logits = self.gate_proj(x).chunk(2, 1)
        horizontal = self._retention(horizontal_logits)
        vertical = self._retention(vertical_logits)
        if self.track_stats:
            horizontal_stats = horizontal.detach().float()
            vertical_stats = vertical.detach().float()
            self.last_state_stats = (
                float(horizontal_stats.mean()),
                float(horizontal_stats.std()),
                float(vertical_stats.mean()),
                float(vertical_stats.std()),
                float(self.context_scale.detach().float()))
            self.track_stats = False
        context = (
            self._bidirectional_scan(value, horizontal, -1)
            + self._bidirectional_scan(value, vertical, -2)
        ) * 0.25
        return x + self.context_scale.to(dtype=x.dtype) * self.out_proj(context)


class AKCChannelRecalibration(nn.Module):
    """Neutral-start channel recalibration corresponding to AKCAttention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // int(reduction), 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        self.track_stats = False
        self.last_gate_stats = None

    def forward(self, x):
        gate = torch.sigmoid(self.expand(self.act(self.reduce(self.pool(x)))))
        if self.track_stats:
            detached_gate = gate.detach().float()
            self.last_gate_stats = (
                float(detached_gate.mean()), float(detached_gate.std()))
            self.track_stats = False
        # Initial gate=0.5 gives exact identity; the learned range is [0.5,1.5].
        return x * (0.5 + gate)


class C3AKCMambaLite(nn.Module):
    """YOLOv5 CSP adaptation of AKConv + selective SSM + AKCAttention.

    It is intended for the fused PAN neck.  Keeping the CSP bypass protects
    local detail while the processed branch adds adaptive geometric sampling
    and four-direction long-range context.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, num_points=5,
                 e=0.5, state_ratio=0.25):
        super().__init__()
        hidden = max(int(c2 * float(e)), 8)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.akc_blocks = nn.Sequential(*[
            AKCResidualBlock(
                hidden, num_points=int(num_points), shortcut=shortcut)
            for _ in range(max(int(n), 1))
        ])
        self.state_space = SelectiveAxisStateSpace2D(
            hidden, state_ratio=float(state_ratio))
        self.attention = AKCChannelRecalibration(hidden)
        self.track_akc_stats = False
        self.last_akc_stats = None

    def forward(self, x):
        if self.track_akc_stats:
            for block in self.akc_blocks:
                block.akc.track_stats = True
                block.akc.last_offset_stats = None
            self.state_space.track_stats = True
            self.state_space.last_state_stats = None
            self.attention.track_stats = True
            self.attention.last_gate_stats = None
        adaptive = self.akc_blocks(self.cv1(x))
        contextual = self.attention(self.state_space(adaptive))
        bypass = self.cv2(x)
        output = self.cv3(torch.cat((contextual, bypass), dim=1))
        if self.track_akc_stats:
            offset_rows = [
                block.akc.last_offset_stats for block in self.akc_blocks
                if block.akc.last_offset_stats is not None]
            state_stats = self.state_space.last_state_stats
            gate_stats = self.attention.last_gate_stats
            if offset_rows and state_stats is not None and gate_stats is not None:
                offset_stats = tuple(
                    sum(row[index] for row in offset_rows) / len(offset_rows)
                    for index in range(3))
                self.last_akc_stats = (
                    *offset_stats, *state_stats, *gate_stats)
            self.track_akc_stats = False
        return output


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class SODScaleSequenceFusion(nn.Module):
    """Fuse three adjacent feature scales at the highest input resolution.

    This is the lightweight Attentional Scale Sequence Fusion operation used
    by SOD-YOLO's ASF models.  It is kept under a project-specific name so the
    existing weighted ``Add`` layer retains its historical behaviour.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(
                'SODScaleSequenceFusion expects exactly three input scales')
        self.proj0 = (Conv(in_channels[0], out_channels, 1)
                      if in_channels[0] != out_channels else nn.Identity())
        self.proj1 = Conv(in_channels[1], out_channels, 1)
        self.proj2 = Conv(in_channels[2], out_channels, 1)
        self.scale_conv = nn.Conv3d(
            out_channels, out_channels, kernel_size=1, bias=True)
        self.scale_bn = nn.BatchNorm3d(out_channels)
        self.scale_act = nn.LeakyReLU(0.1, inplace=True)
        self.scale_pool = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        if len(x) != 3:
            raise ValueError(
                'SODScaleSequenceFusion expects exactly three input tensors')
        target_size = x[0].shape[-2:]
        p0 = self.proj0(x[0])
        p1 = F.interpolate(
            self.proj1(x[1]), target_size, mode='nearest')
        p2 = F.interpolate(
            self.proj2(x[2]), target_size, mode='nearest')
        sequence = torch.stack((p0, p1, p2), dim=2)
        sequence = self.scale_act(self.scale_bn(self.scale_conv(sequence)))
        return self.scale_pool(sequence).squeeze(2)


class SODResidualAdd(nn.Module):
    """Safely introduce the SOD-YOLO ASF residual during fine-tuning.

    A small learnable gate avoids the randomly initialized scale branch
    immediately corrupting an already trained detector.  Historical
    checkpoints created before the gate was added retain their original
    unweighted summation behaviour through the ``getattr`` fallback.
    """

    def __init__(self, initial_scale=0.05):
        super().__init__()
        self.sod_residual_scale = nn.Parameter(
            torch.tensor(float(initial_scale)))

    def forward(self, x):
        if len(x) != 2 or x[0].shape != x[1].shape:
            shapes = [tuple(t.shape) for t in x]
            raise ValueError(
                f'SODResidualAdd expects two equal-shaped tensors, got {shapes}')
        scale = getattr(self, 'sod_residual_scale', None)
        if scale is None:
            return x[0] + x[1]
        return x[0] + torch.tanh(scale) * x[1]


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class DMAF(nn.Module):
    def __init__(self, c2):
        super(DMAF, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
                input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels // reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels // reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
                               dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio,
                                                   pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


#################################################### ACFormer
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        #self.relu = nn.ReLU()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        #attn = self.relu(attn)**2

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out





# =========================================================
# Dynamic Frequency Filtering modules for CrossAttention
# =========================================================
def inverse_softplus(x, eps=1e-6):
    x = torch.as_tensor(x).clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def init_positive_temperature(num_heads, value=1.0):
    raw = inverse_softplus(torch.full((num_heads, 1, 1), float(value)))
    return raw


def positive_temperature(raw_temperature, eps=1e-6):
    return F.softplus(raw_temperature) + eps


def inverse_bounded_sigmoid(value, lower, upper, eps=1e-4):
    """Map a value inside ``[lower, upper]`` to an unconstrained logit."""
    if not lower < upper:
        raise ValueError('lower must be smaller than upper')
    value = torch.as_tensor(value, dtype=torch.float32)
    probability = ((value - lower) / (upper - lower)).clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


def resize_complex_weight(origin_weight, new_h, new_w):
    """
    Resize learnable complex frequency bases.

    Args:
        origin_weight: [H, W_freq, num_filters, 2], real/imag stored in last dim.
        new_h: target FFT height.
        new_w: target FFT width after rfft2, i.e. W // 2 + 1.

    Returns:
        new_weight: [new_h, new_w, num_filters, 2]
    """
    h, w, num_filters, _ = origin_weight.shape

    # rfft2 keeps the height axis in FFT order [0, +freq, -freq].
    # Shift before interpolation so positive/negative frequency neighbors meet
    # at the boundary only after resizing.
    origin_weight = torch.fft.fftshift(origin_weight, dim=0)

    # [H, W, F, 2] -> [1, F*2, H, W]
    origin_weight = origin_weight.permute(2, 3, 0, 1).reshape(1, num_filters * 2, h, w)

    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=True
    )

    # [1, F*2, H, W] -> [H, W, F, 2]
    new_weight = new_weight.reshape(num_filters, 2, new_h, new_w).permute(2, 3, 0, 1)
    new_weight = torch.fft.ifftshift(new_weight, dim=0)

    return new_weight


def init_complex_residual_bases(h, w_freq, num_filters, scale=0.20):
    """
    Initialize residual complex bases delta_H around zero.

    The final response is 1 + beta * delta_H, so these bases must not be
    initialized around one. Distinct low/band/high templates give the router a
    meaningful signal from the first backward pass.
    """
    fy = torch.fft.fftfreq(h, d=1.0).view(h, 1)
    full_w = max((w_freq - 1) * 2, 1)
    fx = torch.fft.rfftfreq(full_w, d=1.0).view(1, w_freq)
    radius = torch.sqrt(fx.square() + fy.square())
    radius = radius / radius.max().clamp_min(1e-6)

    templates = []
    low = torch.exp(-(radius / 0.28).square())
    mid = torch.exp(-((radius - 0.45) / 0.18).square())
    high = 1.0 - torch.exp(-(radius / 0.42).square())
    diag = torch.sin(2.3 * math.pi * fx).expand(h, w_freq) * torch.cos(1.7 * math.pi * fy).expand(h, w_freq)
    base = [low, mid, high, diag]
    for i in range(num_filters):
        real = base[i % len(base)].clone()
        real = real - real.mean()
        real = real / real.abs().max().clamp_min(1e-6)
        imag = torch.randn_like(real) * 0.05
        templates.append(torch.stack([scale * real, scale * imag], dim=-1))
    return torch.stack(templates, dim=2)

class FrequencyResponseGate(nn.Module):
    """
    Gate the residual response caused by dynamic frequency filtering.
    This keeps the module stable at the beginning of training.
    """
    def __init__(self, channel, bias=False):
        super(FrequencyResponseGate, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, diff):
        return self.gate(diff)


class ForegroundSparseFusionMask(nn.Module):
    """
    Lightweight foreground-aware spatial gate for multimodal fusion.

    It predicts a single spatial mask from RGB features, IR features, and their
    difference. No extra supervision is required; the mask is learned only from
    the detector loss.
    """
    def __init__(self, channel, bias=False, output_bias=2.0):
        super(ForegroundSparseFusionMask, self).__init__()
        hidden = max(channel // 4, 16)
        self.mask_head = nn.Sequential(
            nn.Conv2d(3 * channel, hidden, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=bias),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.mask_head[-2].bias, float(output_bias))

    def forward(self, rgb_fea, ir_fea):
        mask_input = torch.cat([rgb_fea, ir_fea, torch.abs(rgb_fea - ir_fea)], dim=1)
        return self.mask_head(mask_input)


class RelationForegroundSparseFusionMask(nn.Module):
    """
    Relation-aware refinement for the foreground sparse fusion mask.

    It estimates where the GFB result behaves closer to RGB or IR activation
    energy, then refines that relation together with the learned foreground
    mask. The output is used with a tiny residual gate in HAFFormer.
    """
    def __init__(self, channel, bias=False):
        super(RelationForegroundSparseFusionMask, self).__init__()
        hidden = max(channel // 16, 8)
        self.refine = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.refine[-2].weight)
        nn.init.constant_(self.refine[-2].bias, 2.0)

    def forward(self, rgb_fea, ir_fea, gfb_fea, fg_mask):
        eps = 1e-6
        dtype = fg_mask.dtype
        rgb_f = rgb_fea.float()
        ir_f = ir_fea.float()
        gfb_f = gfb_fea.float()

        rgb_dist = (gfb_f - rgb_f).abs().mean(dim=1, keepdim=True)
        ir_dist = (gfb_f - ir_f).abs().mean(dim=1, keepdim=True)
        relation = 0.5 + 0.5 * (ir_dist - rgb_dist) / (rgb_dist + ir_dist + eps)
        relation = torch.nan_to_num(relation, nan=0.5, posinf=1.0, neginf=0.0)
        relation = torch.clamp(relation, 0.0, 1.0)

        contrast = torch.abs(rgb_f - ir_f).mean(dim=1, keepdim=True)
        contrast = contrast / (contrast.mean(dim=(2, 3), keepdim=True).clamp_min(eps))
        contrast = torch.nan_to_num(contrast, nan=0.0, posinf=1.0, neginf=0.0)
        contrast = torch.sigmoid(contrast - 1.0)

        mask_input = torch.cat([fg_mask.float(), relation, contrast], dim=1).to(dtype=dtype)
        return torch.nan_to_num(self.refine(mask_input), nan=0.5, posinf=1.0, neginf=0.0)


class EOTChannelPrior(nn.Module):
    """
    Channel-space EOT log-prior.

    This is not a spatial GOAT prior. It is a channel log-prior:
        K(h, q_channel, k_channel) = u(h, k_channel) + A(h, q, r) @ B(h, k, r)
    followed by row-centering. A is small random and B/u are zero, so the
    initialized forward contribution is neutral while B and u receive gradients.
    """
    def __init__(self, num_heads, head_dim, rank=2, use_low_rank=True, use_key_bias=True,
                 use_query_gate=False):
        super(EOTChannelPrior, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = int(rank)
        self.use_low_rank = bool(use_low_rank and self.rank > 0)
        self.use_key_bias = bool(use_key_bias)
        self.use_query_gate = bool(use_query_gate)

        if self.use_key_bias:
            self.key_bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.register_parameter('key_bias', None)

        if self.use_low_rank:
            # B stays at zero, so the initial prior remains exactly neutral.
            # A uses a visible scale so B receives a useful first-step gradient.
            self.low_rank_a = nn.Parameter(torch.randn(num_heads, head_dim, self.rank) * 1e-2)
            self.low_rank_b = nn.Parameter(torch.zeros(num_heads, head_dim, self.rank))
        else:
            self.register_parameter('low_rank_a', None)
            self.register_parameter('low_rank_b', None)

        if self.use_query_gate:
            self.query_gate_scale_raw = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.query_gate_bias = nn.Parameter(torch.full((num_heads, head_dim), 2.0))
        else:
            self.register_parameter('query_gate_scale_raw', None)
            self.register_parameter('query_gate_bias', None)

    def build_prior(self, dtype=None, device=None):
        if device is None:
            device = self.key_bias.device if self.key_bias is not None else self.low_rank_a.device
        if dtype is None:
            dtype = self.key_bias.dtype if self.key_bias is not None else self.low_rank_a.dtype

        prior = torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
        if self.use_key_bias:
            prior = prior + self.key_bias.to(device=device, dtype=torch.float32).unsqueeze(1)
        if self.use_low_rank:
            a = self.low_rank_a.to(device=device, dtype=torch.float32)
            b = self.low_rank_b.to(device=device, dtype=torch.float32)
            prior = prior + torch.einsum('hqr,hkr->hqk', a, b)

        prior = prior - prior.mean(dim=-1, keepdim=True)
        return prior.to(dtype=dtype)

    def forward(self, attn_logits, q_raw=None):
        prior = self.build_prior(dtype=attn_logits.dtype, device=attn_logits.device).unsqueeze(0)
        if self.use_query_gate:
            assert q_raw is not None, 'q_raw is required when use_query_gate=True'
            q_signal = q_raw.float().pow(2).mean(dim=-1).sqrt()
            scale = F.softplus(self.query_gate_scale_raw).to(q_signal.device).unsqueeze(0)
            bias = self.query_gate_bias.to(q_signal.device).unsqueeze(0)
            gate = torch.sigmoid(bias - scale * q_signal).to(dtype=attn_logits.dtype).unsqueeze(-1)
            prior = prior * gate
        return attn_logits + prior


class FourierRoutingLogPriorV2(nn.Module):
    """
    Absolute Fourier log-odds prior for FFT routing.

    This module is a routing prior, not a GOAT spatial attention prior: it adds
    a lightweight absolute-position logit to the two-expert routing decision.
    Raw x/y coordinates and non-integer frequencies avoid identical endpoints.
    """
    def __init__(self, channel, num_freqs=4, hidden=16, bias=False):
        super(FourierRoutingLogPriorV2, self).__init__()
        # Keep this numerically sensitive branch in FP32 under AMP and
        # half-precision validation/checkpoint conversion.
        self.force_fp32 = True
        self.num_freqs = num_freqs
        freq = torch.tensor([1.0, 2.3, 4.7, 7.1], dtype=torch.float32)
        if num_freqs != 4:
            freq = torch.linspace(1.0, 1.0 + 1.7 * max(num_freqs - 1, 0), num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freq, persistent=False)
        self._grid_cache = {}

        in_channels = 2 + 4 * num_freqs
        self.prior_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        context_hidden = max(channel // 16, 8)
        self.context_gate = nn.Sequential(
            nn.Conv2d(3 * channel, context_hidden, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(context_hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.prior_scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.prior_head[0].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.prior_head[0].bias)
        # A strictly zero output layer received gradients too small to survive
        # AMP/checkpoint conversion in full detector training. Keep the prior
        # near-neutral while opening a real gradient path from the first step.
        nn.init.normal_(self.prior_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.prior_head[-1].bias)

    def build_fourier_grid(self, h, w, device, dtype):
        key = (h, w, str(device), str(dtype))
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)
        freqs = self.freqs.to(device=device, dtype=torch.float32) * math.pi

        y_phase = y[:, None] * freqs[None, :]
        x_phase = x[:, None] * freqs[None, :]
        y_feat = torch.cat([torch.cos(y_phase), torch.sin(y_phase)], dim=1)
        x_feat = torch.cat([torch.cos(x_phase), torch.sin(x_phase)], dim=1)

        y_grid = y_feat[:, None, :].expand(h, w, 2 * self.num_freqs)
        x_grid = x_feat[None, :, :].expand(h, w, 2 * self.num_freqs)
        yy = y.view(h, 1, 1).expand(h, w, 1)
        xx = x.view(1, w, 1).expand(h, w, 1)
        grid = torch.cat([yy, xx, y_grid, x_grid], dim=2).permute(2, 0, 1).unsqueeze(0)
        grid = grid.to(dtype=dtype)
        if len(self._grid_cache) > 8:
            self._grid_cache.clear()
        self._grid_cache[key] = grid
        return grid

    def forward(self, q_feat, k_feat):
        b, _, h, w = q_feat.shape
        dtype = q_feat.dtype
        with amp.autocast(enabled=False):
            q_fp32 = q_feat.float()
            k_fp32 = k_feat.float()
            grid = self.build_fourier_grid(h, w, q_feat.device, torch.float32)
            spatial_prior = self.prior_head(grid).clamp(min=-8.0, max=8.0)

            context = torch.cat([q_fp32, k_fp32, torch.abs(q_fp32 - k_fp32)], dim=1)
            context_gate = self.context_gate(context)
            routing_prior = self.prior_scale.float() * context_gate * spatial_prior
        return routing_prior.expand(b, -1, -1, -1).to(dtype=dtype)


SpatialGOATRoutingPrior = FourierRoutingLogPriorV2


class Windowed2DGOATCrossAttention(nn.Module):
    """
    Optional low-resolution 2D GOAT cross-attention for RGB-IR fusion.

    Q comes from the primary modality and K/V from the complementary modality.
    The relative 2D Fourier prior is factorized into q_position @ k_position.T
    and folded into one SDPA call with GOAT-compatible scaling.
    """
    def __init__(self, dim, num_heads=4, window_size=8, rank_x=1, rank_y=1,
                 use_null_sink=True, bias=False):
        super(Windowed2DGOATCrossAttention, self).__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.rank_x = rank_x
        self.rank_y = rank_y
        self.use_null_sink = use_null_sink
        self.head_dim = dim // num_heads
        self.sink_dim = 1 if use_null_sink else 0
        self.d_pos = 2 * (rank_x + rank_y) + self.sink_dim
        self.d_content = self.head_dim - self.d_pos
        assert self.d_content > 0, 'head_dim must be larger than GOAT positional dimension'

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.alpha_y = nn.Parameter(torch.zeros(num_heads, rank_y))
        self.beta_y = nn.Parameter(torch.zeros(num_heads, rank_y))
        self.alpha_x = nn.Parameter(torch.zeros(num_heads, rank_x))
        self.beta_x = nn.Parameter(torch.zeros(num_heads, rank_x))

        if use_null_sink:
            self.null_key_bias = nn.Parameter(torch.zeros(num_heads))
            self.null_value = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        else:
            self.register_parameter('null_key_bias', None)
            self.register_parameter('null_value', None)
        self.last_null_attention = None

        fy = torch.arange(1, rank_y + 1, dtype=torch.float32) * math.pi
        fx = torch.arange(1, rank_x + 1, dtype=torch.float32) * math.pi
        self.register_buffer('freq_y', fy, persistent=False)
        self.register_buffer('freq_x', fx, persistent=False)

    def _window_partition(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        return x, (h, w, hp, wp)

    def _window_reverse(self, x, shape):
        h, w, hp, wp = shape
        ws = self.window_size
        b = x.shape[0] // ((hp // ws) * (wp // ws))
        x = x.view(b, hp // ws, wp // ws, ws, ws, self.dim)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.dim, hp, wp)
        return x[:, :, :h, :w]

    def _coords(self, device, dtype):
        ws = self.window_size
        coord = torch.arange(ws, device=device, dtype=torch.float32)
        denom = max(ws - 1, 1)
        coord = (coord / denom) * 2.0 - 1.0
        yy, xx = torch.meshgrid(coord, coord, indexing='ij')
        return yy.reshape(-1).to(dtype=dtype), xx.reshape(-1).to(dtype=dtype)

    def build_position_factors(self, device, dtype, coords=None):
        if coords is None:
            y, x = self._coords(device, dtype)
        else:
            y, x = coords
            y = y.to(device=device, dtype=dtype)
            x = x.to(device=device, dtype=dtype)

        fy = self.freq_y.to(device=device, dtype=dtype)
        fx = self.freq_x.to(device=device, dtype=dtype)
        cos_y, sin_y = torch.cos(y[:, None] * fy[None, :]), torch.sin(y[:, None] * fy[None, :])
        cos_x, sin_x = torch.cos(x[:, None] * fx[None, :]), torch.sin(x[:, None] * fx[None, :])

        ay = self.alpha_y.to(device=device, dtype=dtype)
        by = self.beta_y.to(device=device, dtype=dtype)
        ax = self.alpha_x.to(device=device, dtype=dtype)
        bx = self.beta_x.to(device=device, dtype=dtype)

        qy0 = ay[:, None, :] * cos_y[None, :, :] + by[:, None, :] * sin_y[None, :, :]
        qy1 = ay[:, None, :] * sin_y[None, :, :] - by[:, None, :] * cos_y[None, :, :]
        qx0 = ax[:, None, :] * cos_x[None, :, :] + bx[:, None, :] * sin_x[None, :, :]
        qx1 = ax[:, None, :] * sin_x[None, :, :] - bx[:, None, :] * cos_x[None, :, :]
        q_pos = torch.cat([qy0, qy1, qx0, qx1], dim=-1)

        ky = torch.cat([cos_y, sin_y], dim=-1).unsqueeze(0).expand(self.num_heads, -1, -1)
        kx = torch.cat([cos_x, sin_x], dim=-1).unsqueeze(0).expand(self.num_heads, -1, -1)
        k_pos = torch.cat([ky, kx], dim=-1)
        return q_pos, k_pos

    def explicit_prior(self, coords=None, device=None, dtype=torch.float32):
        device = device or self.alpha_y.device
        q_pos, k_pos = self.build_position_factors(device, dtype, coords=coords)
        return torch.matmul(q_pos, k_pos.transpose(-2, -1))

    def compose_qk(self, q, k):
        n, h, l, d = q.shape
        dtype = q.dtype
        q_content = q[..., :self.d_content]
        k_content = k[..., :self.d_content]
        q_pos, k_pos = self.build_position_factors(q.device, dtype)
        q_pos = q_pos.unsqueeze(0).expand(n, -1, -1, -1)
        k_pos = k_pos.unsqueeze(0).expand(n, -1, -1, -1)

        parts_q = [q_content / math.sqrt(self.d_content), q_pos]
        parts_k = [k_content, k_pos]
        if self.use_null_sink:
            q_sink = q.new_ones(n, h, l, 1)
            k_sink = q.new_zeros(n, h, l, 1)
            parts_q.append(q_sink)
            parts_k.append(k_sink)
        q_total = torch.cat(parts_q, dim=-1)
        k_total = torch.cat(parts_k, dim=-1)
        return q_total, k_total

    def forward(self, primary, complementary):
        q_win, shape = self._window_partition(self.q_proj(primary))
        k_win, _ = self._window_partition(self.k_proj(complementary))
        v_win, _ = self._window_partition(self.v_proj(complementary))
        n, l, _ = q_win.shape

        q = q_win.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_win.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_win.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)

        q_total, k_total = self.compose_qk(q, k)
        v_total = v
        if self.use_null_sink:
            null_k = k_total.new_zeros(n, self.num_heads, 1, self.head_dim)
            null_k[..., -1] = self.null_key_bias.view(1, self.num_heads, 1)
            null_v = self.null_value.to(dtype=v.dtype, device=v.device).view(1, self.num_heads, 1, self.head_dim)
            null_v = null_v.expand(n, -1, -1, -1)
            k_total = torch.cat([k_total, null_k], dim=2)
            v_total = torch.cat([v_total, null_v], dim=2)

        q_sdpa = q_total * math.sqrt(self.head_dim)
        out = F.scaled_dot_product_attention(q_sdpa.contiguous(), k_total.contiguous(), v_total.contiguous())
        if self.use_null_sink:
            with torch.no_grad():
                logits = torch.matmul(q_sdpa, k_total.transpose(-2, -1)) / math.sqrt(self.head_dim)
                self.last_null_attention = logits.softmax(dim=-1)[..., -1].mean().detach()

        out = out.transpose(1, 2).reshape(n, l, self.dim)
        out = self._window_reverse(out, shape)
        return self.out_proj(out)


class CrossAttention_S(nn.Module):
    """
    Cross Attention with dynamic frequency filtering.

    Original CrossAttention_S:
        q, k are generated from fea_0;
        v is generated from fea_1;
        attention is computed by q and k, then applied to v.

    Modified version:
        1. q and k still generate the attention matrix.
        2. The q-k attention response is converted into a routing attention map.
        3. The routing attention map dynamically combines learnable complex frequency bases.
        4. The v branch is transformed by rfft2, filtered in the frequency domain,
           then transformed back by irfft2.
        5. The filtered v is used in the original attention aggregation.
    """
    def __init__(self, dim, num_heads, bias, kernel_size=80, num_filters=4,
                 prior_mode='eot_channel_v2', routing_mode='entropy_v2',
                 channel_score_mode='entropy', channel_prior_rank=2,
                 use_channel_prior_gate=False, filter_init='v2', beta_init=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_filters = num_filters
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.prior_mode = str(prior_mode)
        self.routing_mode = str(routing_mode)
        self.channel_score_mode = str(channel_score_mode)
        self.filter_init = str(filter_init)
        self.temperature_eps = 1e-6

        self.raw_temperature = nn.Parameter(init_positive_temperature(num_heads, value=1.0))
        if self.prior_mode == 'legacy':
            self.attn_log_prior = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 1e-3)
            self.attn_prior_gamma = nn.Parameter(torch.full((1,), 1e-3))
            self.attn_prior_gate = nn.Sequential(
                nn.Conv2d(3 * dim, num_heads, kernel_size=1, stride=1, padding=0, bias=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Sigmoid()
            )
            self.channel_prior = None
        elif self.prior_mode == 'eot_channel_v2':
            self.register_parameter('attn_log_prior', None)
            self.register_parameter('attn_prior_gamma', None)
            self.attn_prior_gate = None
            self.channel_prior = EOTChannelPrior(num_heads, self.head_dim, rank=channel_prior_rank,
                                                 use_low_rank=True, use_key_bias=True,
                                                 use_query_gate=use_channel_prior_gate)
        elif self.prior_mode in ('key_only_channel_v2', 'disabled'):
            self.register_parameter('attn_log_prior', None)
            self.register_parameter('attn_prior_gamma', None)
            self.attn_prior_gate = None
            self.channel_prior = None if self.prior_mode == 'disabled' else EOTChannelPrior(
                num_heads, self.head_dim, rank=0, use_low_rank=False, use_key_bias=True,
                use_query_gate=use_channel_prior_gate)
        else:
            raise ValueError(f'Unsupported prior_mode={prior_mode}')

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,
                                  bias=bias)

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                   bias=bias)

        # Generate a spatial-channel routing map from q and k.
        self.route_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        )
        self.spatial_routing_prior = FourierRoutingLogPriorV2(dim, bias=bias)
        # Preserve spatial routing information before route_att is globally
        # pooled for filter selection. A small residual gain keeps the initial
        # V branch close to the previous implementation.
        self.spatial_route_gain = nn.Parameter(torch.full((1,), 0.05))

        # Dynamic routing MLP: [B, C] -> [B, num_filters, C]
        hidden = max(dim // 4, 16)
        self.reweight_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_filters * dim)
        )

        h_base = kernel_size
        w_freq_base = kernel_size // 2 + 1
        if self.filter_init == 'legacy':
            init_weight = torch.randn(h_base, w_freq_base, num_filters, 2) * 0.02
            init_weight[..., 0] += 1.0
        else:
            init_weight = init_complex_residual_bases(h_base, w_freq_base, num_filters)
        self.complex_weights = nn.Parameter(init_weight)

        beta_value = 0.0 if self.filter_init == 'legacy' else float(beta_init)
        self.beta = nn.Parameter(torch.full((1,), beta_value))

        # Response gate for the difference between filtered v and original v.
        self.freq_response_gate = FrequencyResponseGate(dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def apply_attention_prior(self, attn_logits, q_feat=None, k_feat=None):
        if self.prior_mode == 'disabled':
            return attn_logits
        if self.prior_mode == 'legacy':
            prior = self.attn_log_prior.unsqueeze(0).to(dtype=attn_logits.dtype, device=attn_logits.device)
            gamma = torch.tanh(self.attn_prior_gamma).to(dtype=attn_logits.dtype, device=attn_logits.device)
            if q_feat is not None and k_feat is not None and self.attn_prior_gate is not None:
                context = torch.cat([q_feat, k_feat, torch.abs(q_feat - k_feat)], dim=1)
                gate = self.attn_prior_gate(context).view(context.shape[0], self.num_heads, 1, 1)
                gamma = gamma * gate.to(dtype=attn_logits.dtype)
            return attn_logits + gamma * prior

        q_raw = None
        if q_feat is not None:
            q_raw = rearrange(q_feat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        return self.channel_prior(attn_logits, q_raw=q_raw)

    def get_temperature(self):
        return positive_temperature(self.raw_temperature, eps=self.temperature_eps)

    def compute_channel_score(self, attn):
        """
        Convert softmax attention to [B, heads, head_dim] channel confidence.
        entropy/max are informative; legacy_mean is the old constant path.
        """
        if self.channel_score_mode == 'legacy_mean':
            return attn.mean(dim=-1).sigmoid()
        if self.channel_score_mode == 'max_prob':
            return attn.max(dim=-1).values.clamp(0.0, 1.0)
        if self.channel_score_mode != 'entropy':
            raise ValueError(f'Unsupported channel_score_mode={self.channel_score_mode}')
        p = attn.float().clamp_min(1e-8)
        denom = math.log(max(self.head_dim, 2))
        entropy = -(p * p.log()).sum(dim=-1) / denom
        return (1.0 - entropy).clamp(0.0, 1.0).to(dtype=attn.dtype)

    @staticmethod
    def prob_to_logit(prob, dtype):
        eps = 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        return torch.logit(prob.float().clamp(eps, 1.0 - eps)).to(dtype=dtype)

    def get_dynamic_weight(self, route_att, target_h, target_w_freq):
        """
        Args:
            route_att: [B, C, H, W]
            target_h: FFT height.
            target_w_freq: FFT width after rfft2.

        Returns:
            dynamic_weight: [B, C, target_h, target_w_freq], complex tensor.
        """
        b, c, _, _ = route_att.shape

        # [B, C, H, W] -> [B, C]
        route_vec = route_att.mean(dim=(2, 3))

        # Important for AMP/model.half(): Linear input dtype must match Linear weight dtype.
        # FFT still runs in FP32 later, but the routing MLP follows the model dtype.
        mlp_dtype = self.reweight_mlp[0].weight.dtype
        route_vec = route_vec.to(dtype=mlp_dtype)

        # [B, C] -> [B, F, C]
        routing = self.reweight_mlp(route_vec).view(b, self.num_filters, c)
        routing = F.softmax(routing.float(), dim=1).to(torch.complex64)

        resized_weights = resize_complex_weight(
            self.complex_weights.float(),
            target_h,
            target_w_freq
        )

        resized_weights_complex = torch.view_as_complex(resized_weights.contiguous())

        # [B, F, C] x [H, Wf, F] -> [B, C, H, Wf]
        dynamic_weight = torch.einsum(
            'bfc,hwf->bchw',
            routing,
            resized_weights_complex
        )

        return dynamic_weight

    def build_route_attention(self, q_feat, k_feat, attn, h, w):
        """
        Convert q-k attention response into [B, C, H, W] routing attention map.
        Here attn is the original channel attention matrix: [B, heads, head_dim, head_dim].
        """
        b = q_feat.shape[0]
        channel_score = self.compute_channel_score(attn).reshape(b, self.dim, 1, 1)
        content_logit = self.route_proj(q_feat + k_feat)
        routing_prior = content_logit.new_zeros((b, 1, h, w))

        if self.routing_mode == 'disabled':
            route_att = torch.sigmoid(content_logit)
        elif self.routing_mode == 'legacy':
            spatial_score = torch.sigmoid(content_logit)
            legacy_channel = attn.mean(dim=-1).reshape(b, self.dim, 1, 1).sigmoid()
            route_att = spatial_score * legacy_channel
        elif self.routing_mode == 'entropy_v2':
            channel_logit = self.prob_to_logit(channel_score, dtype=content_logit.dtype)
            routing_prior = self.spatial_routing_prior(q_feat, k_feat)
            route_att = torch.sigmoid(content_logit + channel_logit + routing_prior)
        elif self.routing_mode == 'entropy_channel_only':
            channel_logit = self.prob_to_logit(channel_score, dtype=content_logit.dtype)
            route_att = torch.sigmoid(content_logit + channel_logit)
        else:
            raise ValueError(f'Unsupported routing_mode={self.routing_mode}')

        route_att = torch.nan_to_num(route_att, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        routing_prior = torch.nan_to_num(routing_prior, nan=0.0, posinf=1.0, neginf=-1.0)
        return route_att, routing_prior

    def frequency_filter_v(self, v_feat, route_att, routing_prior):
        """
        Apply FFT -> dynamic frequency filtering -> IFFT on the V branch.

        Args:
            v_feat: [B, C, H, W]
            route_att: [B, C, H, W]
            routing_prior: [B, 1, H, W]

        Returns:
            v_out: [B, C, H, W]
        """
        b, c, h, w = v_feat.shape
        origin_dtype = v_feat.dtype

        # cuFFT does not support FP16 FFT for non-power-of-two feature sizes
        # such as [136, 168]. Therefore, run FFT/IFFT in FP32 and cast back.
        v_fft = v_feat.float()
        route_att_fp32 = route_att.float()
        routing_prior_fp32 = routing_prior.float()

        # The previous router only consumed the spatial mean of route_att.
        # Fourier priors are approximately zero-mean, so their gradient was
        # almost annihilated. Apply a centered, bounded spatial residual before
        # FFT to give the routing prior a direct path to the detection loss.
        route_centered = route_att_fp32 - route_att_fp32.mean(dim=(2, 3), keepdim=True)
        prior_centered = routing_prior_fp32 - routing_prior_fp32.mean(dim=(2, 3), keepdim=True)
        spatial_gain = torch.tanh(self.spatial_route_gain.float())
        # The direct prior term is bounded to +/-10% even if the learned logits
        # grow later. At initialization its actual effect is around 1e-5.
        prior_residual = 0.1 * torch.tanh(prior_centered)
        v_fft = v_fft * (1.0 + spatial_gain * route_centered + prior_residual)

        fft_v = torch.fft.rfft2(v_fft, norm='ortho')
        target_h = fft_v.shape[2]
        target_w_freq = fft_v.shape[3]

        dynamic_weight = self.get_dynamic_weight(
            route_att_fp32,
            target_h,
            target_w_freq
        )

        # Residual frequency filtering.
        fft_v_filtered = fft_v * (1.0 + self.beta.float() * dynamic_weight)

        v_raw = torch.fft.irfft2(
            fft_v_filtered,
            s=(h, w),
            norm='ortho'
        )

        v_raw = v_raw.to(dtype=origin_dtype)

        diff = torch.abs(v_raw - v_feat)
        gate = self.freq_response_gate(diff)

        v_out = v_feat + gate * (v_raw - v_feat)

        return v_out

    def forward(self, x):
        fea_0 = x[0]
        fea_1 = x[1]
        b, c, h, w = fea_0.shape

        qk = self.qk_dwconv(self.qk(fea_0))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(fea_1))

        q_re = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_re = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_re = torch.nn.functional.normalize(q_re, dim=-1, eps=1e-6)
        k_re = torch.nn.functional.normalize(k_re, dim=-1, eps=1e-6)

        attn = (q_re @ k_re.transpose(-2, -1)) * self.get_temperature()
        attn = self.apply_attention_prior(attn, q, k)
        attn = attn.softmax(dim=-1)

        # q/k attention map -> dynamic frequency filter routing.
        route_att, routing_prior = self.build_route_attention(q, k, attn, h, w)

        # FFT filter is applied to the V branch.
        v_filtered = self.frequency_filter_v(v, route_att, routing_prior)

        v_re = rearrange(v_filtered, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        out = (attn @ v_re)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


FrequencyCrossAttention_S = CrossAttention_S
FrequencyCrossAttention_S.__name__ = 'FrequencyCrossAttention_S'
FrequencyCrossAttention_S.__qualname__ = 'FrequencyCrossAttention_S'


class ExpMaskFrequencyCrossAttention(FrequencyCrossAttention_S):
    """
    Dynamic frequency cross-attention used by the original exp_mask model.

    This path intentionally excludes GOAT channel/spatial priors and the later
    direct spatial FFT compensation so GFB ablations change only the fusion
    operation after frequency cross-attention.
    """
    def __init__(self, dim, num_heads, bias, kernel_size=80, num_filters=4):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.dim = dim
        self.num_filters = num_filters
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.head_dim = dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias)

        self.route_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        hidden = max(dim // 4, 16)
        self.reweight_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_filters * dim)
        )

        h_base = kernel_size
        w_freq_base = kernel_size // 2 + 1
        init_weight = torch.randn(h_base, w_freq_base, num_filters, 2) * 0.02
        init_weight[..., 0] += 1.0
        self.complex_weights = nn.Parameter(init_weight)
        # A small non-zero residual scale keeps the initial frequency response
        # close to identity while allowing the router and filter bases to
        # receive gradients from the first optimizer step.
        self.beta = nn.Parameter(torch.full((1,), 0.01))
        self.freq_response_gate = FrequencyResponseGate(dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def compute_channel_confidence(self, attn):
        """
        Convert row-normalized channel attention into an informative score.

        mean(attn, dim=-1) is always 1 / head_dim after softmax, so it
        cannot guide routing. Normalized inverse entropy is near zero for
        uniform attention and near one for a confident, concentrated row.
        """
        # Full-model checkpoints created before this fix do not contain the
        # plain Python head_dim attribute because unpickling skips __init__.
        head_dim = getattr(self, 'head_dim', self.dim // self.num_heads)
        p = attn.float().clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=-1) / math.log(max(head_dim, 2))
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        confidence = torch.nan_to_num(
            confidence, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=attn.dtype)
        if getattr(self, 'track_route_stats', False):
            tracked = confidence.detach().float()
            self.last_channel_confidence_stats = (
                float(tracked.mean()),
                float(tracked.std()),
                float(tracked.min()),
                float(tracked.max()),
            )
            self.track_route_stats = False
        return confidence

    def build_route_attention(self, q_feat, k_feat, attn, h, w):
        del h, w
        b = q_feat.shape[0]
        channel_confidence = self.compute_channel_confidence(attn)
        channel_gate = 0.5 + 0.5 * channel_confidence
        channel_gate = channel_gate.reshape(b, self.dim, 1, 1)
        spatial_score = self.route_proj(q_feat + k_feat)
        return (spatial_score * channel_gate).clamp(0.0, 1.0)

    def frequency_filter_v(self, v_feat, route_att):
        b, c, h, w = v_feat.shape
        origin_dtype = v_feat.dtype
        fft_v = torch.fft.rfft2(v_feat.float(), norm='ortho')
        dynamic_weight = self.get_dynamic_weight(
            route_att.float(), fft_v.shape[2], fft_v.shape[3])
        fft_v_filtered = fft_v * (1.0 + self.beta.float() * dynamic_weight)
        v_raw = torch.fft.irfft2(
            fft_v_filtered, s=(h, w), norm='ortho').to(dtype=origin_dtype)
        gate = self.freq_response_gate(torch.abs(v_raw - v_feat))
        return v_feat + gate * (v_raw - v_feat)

    def forward(self, x):
        fea_0, fea_1 = x
        b, c, h, w = fea_0.shape

        q, k = self.qk_dwconv(self.qk(fea_0)).chunk(2, dim=1)
        v = self.v_dwconv(self.v(fea_1))
        q_re = rearrange(
            q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_re = rearrange(
            k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_re = F.normalize(q_re, dim=-1, eps=1e-6)
        k_re = F.normalize(k_re, dim=-1, eps=1e-6)
        attn = (q_re @ k_re.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        route_att = self.build_route_attention(q, k, attn, h, w)
        v_filtered = self.frequency_filter_v(v, route_att)
        v_re = rearrange(
            v_filtered, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads)
        out = attn @ v_re
        out = rearrange(
            out, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class OriginalCrossAttention_S(nn.Module):
    """
    Original LCAFNet cross attention without FFT/frequency filtering.

    q and k are generated from the first modality, v is generated from the
    second modality, and the attention result is projected back to dim channels.
    """
    def __init__(self, dim, num_heads, bias):
        super(OriginalCrossAttention_S, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        fea_0 = x[0]
        fea_1 = x[1]
        b, c, h, w = fea_0.shape

        qk = self.qk_dwconv(self.qk(fea_0))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(fea_1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class CrossAttention_S(OriginalCrossAttention_S):
    """
    Original LCAFNet cross attention kept under the historical class name.

    Old baseline checkpoints, such as LCAFNet_FLIR.pt, were saved with modules
    named CrossAttention_S. Keeping this class original makes those checkpoints
    load and run with the baseline implementation.
    """
    pass



class CrossAttention_M(nn.Module):
    """
    Mutual Cross Attention with dynamic frequency filtering for both modalities.
    This class is kept compatible with the original return format: [out_rgb, out_ir].
    """
    def __init__(self, dim, num_heads, bias, kernel_size=80, num_filters=4):
        super(CrossAttention_M, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_filters = num_filters

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)

        self.route_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        hidden = max(dim // 4, 16)
        self.reweight_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_filters * dim)
        )

        h_base = kernel_size
        w_freq_base = kernel_size // 2 + 1

        init_rgb = torch.randn(h_base, w_freq_base, num_filters, 2) * 0.02
        init_rgb[..., 0] += 1.0
        self.complex_weights_rgb = nn.Parameter(init_rgb)

        init_ir = torch.randn(h_base, w_freq_base, num_filters, 2) * 0.02
        init_ir[..., 0] += 1.0
        self.complex_weights_ir = nn.Parameter(init_ir)

        self.beta_rgb = nn.Parameter(torch.full((1,), 0.01))
        self.beta_ir = nn.Parameter(torch.full((1,), 0.01))

        self.rgb_freq_response_gate = FrequencyResponseGate(dim, bias=bias)
        self.ir_freq_response_gate = FrequencyResponseGate(dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def get_dynamic_weight(self, route_att, complex_weights_basis, target_h, target_w_freq):
        b, c, _, _ = route_att.shape

        route_vec = route_att.mean(dim=(2, 3))

        routing = self.reweight_mlp(route_vec).view(b, self.num_filters, c)
        routing = F.softmax(routing, dim=1).to(torch.complex64)

        resized_weights = resize_complex_weight(
            complex_weights_basis,
            target_h,
            target_w_freq
        )

        resized_weights_complex = torch.view_as_complex(resized_weights.contiguous())

        dynamic_weight = torch.einsum(
            'bfc,hwf->bchw',
            routing,
            resized_weights_complex
        )

        return dynamic_weight

    def build_route_attention(self, q_feat, k_feat, v_feat, attn):
        b, c, h, w = q_feat.shape

        channel_score = attn.mean(dim=-1).reshape(b, self.dim, 1, 1).sigmoid()
        spatial_score = self.route_proj(q_feat + k_feat + v_feat)

        route_att = spatial_score * channel_score
        route_att = torch.clamp(route_att, 0.0, 1.0)

        return route_att

    def frequency_filter_v(self, v_feat, route_att, complex_weights_basis, beta, response_gate):
        b, c, h, w = v_feat.shape
        origin_dtype = v_feat.dtype

        # cuFFT does not support FP16 FFT for non-power-of-two feature sizes.
        # Use FP32 for FFT/IFFT, then cast the filtered feature back.
        v_fft = v_feat.float()
        route_att_fp32 = route_att.float()

        fft_v = torch.fft.rfft2(v_fft, norm='ortho')
        target_h = fft_v.shape[2]
        target_w_freq = fft_v.shape[3]

        dynamic_weight = self.get_dynamic_weight(
            route_att_fp32,
            complex_weights_basis,
            target_h,
            target_w_freq
        )

        fft_v_filtered = fft_v * (1.0 + beta.float() * dynamic_weight)

        v_raw = torch.fft.irfft2(
            fft_v_filtered,
            s=(h, w),
            norm='ortho'
        )

        v_raw = v_raw.to(dtype=origin_dtype)

        diff = torch.abs(v_raw - v_feat)
        gate = response_gate(diff)

        v_out = v_feat + gate * (v_raw - v_feat)

        return v_out

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        b, c, h, w = rgb_fea.shape

        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        rgb_q_re = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k_re = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q_re = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k_re = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q_re = torch.nn.functional.normalize(rgb_q_re, dim=-1, eps=1e-6)
        rgb_k_re = torch.nn.functional.normalize(rgb_k_re, dim=-1, eps=1e-6)

        ir_q_re = torch.nn.functional.normalize(ir_q_re, dim=-1, eps=1e-6)
        ir_k_re = torch.nn.functional.normalize(ir_k_re, dim=-1, eps=1e-6)

        # RGB branch receives IR value information:
        # query = RGB q, key = IR k, value = IR v
        attn_ir = (rgb_q_re @ ir_k_re.transpose(-2, -1)) * self.temperature
        # IR branch receives RGB value information:
        # query = IR q, key = RGB k, value = RGB v
        attn_rgb = (ir_q_re @ rgb_k_re.transpose(-2, -1)) * self.temperature

        attn_ir = attn_ir.softmax(dim=-1)
        attn_rgb = attn_rgb.softmax(dim=-1)

        # q/k attention map -> dynamic frequency filtering route.
        ir_route_att = self.build_route_attention(rgb_q, ir_k, ir_v, attn_ir)
        rgb_route_att = self.build_route_attention(ir_q, rgb_k, rgb_v, attn_rgb)

        # Filter V branches in frequency domain.
        ir_v_filtered = self.frequency_filter_v(
            ir_v,
            ir_route_att,
            self.complex_weights_ir,
            self.beta_ir,
            self.ir_freq_response_gate
        )

        rgb_v_filtered = self.frequency_filter_v(
            rgb_v,
            rgb_route_att,
            self.complex_weights_rgb,
            self.beta_rgb,
            self.rgb_freq_response_gate
        )

        ir_v_re = rearrange(ir_v_filtered, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v_re = rearrange(rgb_v_filtered, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        out_ir = (attn_ir @ ir_v_re)
        out_rgb = (attn_rgb @ rgb_v_re)

        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_ir = self.project_out(out_ir)
        out_rgb = self.project_out(out_rgb)

        return [out_rgb, out_ir]



class HAFFormerAdaptiveGNNArchive(nn.Module):
    # Archived adaptive Bayesian-GNN version: frequency attention + GNN node attention + adaptive gate.
    def __init__(self, dim):
        super(HAFFormerAdaptiveGNNArchive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim
        self.num_nodes = 3
        self.gnn_rd_sc = 2
        self.gnn_channels = dim // self.gnn_rd_sc

        self.mhca_rgb = FrequencyCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = FrequencyCrossAttention_S(dim, num_heads, bias)

        self.concat = Concat(dimension=1)

        self.base_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.base_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.gnn_fusion = GraphReasoning(chnn_in=dim, rd_sc=self.gnn_rd_sc, dila=[1, 2, 3], n_iter=1)
        node_hidden = max(self.gnn_channels // 4, 8)
        self.node_score = nn.Sequential(
            nn.Conv2d(self.gnn_channels, node_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(node_hidden, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.rgb_node_bias = nn.Parameter(torch.zeros(self.num_nodes))
        self.ir_node_bias = nn.Parameter(torch.zeros(self.num_nodes))

        self.gnn_fuse = nn.Sequential(
            nn.Conv2d(2 * self.gnn_channels, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )

        gate_hidden = max(dim // 4, 16)
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(2 * dim, gate_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.adaptive_gate[-2].bias, -0.5)

    def get_bayesian_loss(self):
        return self.gnn_fusion.get_bayesian_loss()

    def _node_attention_fuse(self, feats, node_bias):
        stacked = torch.stack(feats, dim=1)
        b, n, c, h, w = stacked.shape
        scores = self.node_score(stacked.reshape(b * n, c, h, w))
        scores = scores.reshape(b, n, 1, h, w)
        scores = scores + node_bias.view(1, n, 1, 1, 1)
        weights = torch.softmax(scores, dim=1)
        return (stacked * weights).sum(dim=1)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        base_seed = self.base_conv(self.concat([out_fea_rgb, out_fea_ir]))
        base_weight = self.base_dwconv(base_seed).sigmoid()
        base_fea = base_weight * out_fea_rgb + (1.0 - base_weight) * out_fea_ir

        gnn_rgb, gnn_ir = self.gnn_fusion((out_fea_rgb, out_fea_ir, out_fea_rgb, out_fea_ir), node=False)
        gnn_rgb = self._node_attention_fuse(gnn_rgb, self.rgb_node_bias)
        gnn_ir = self._node_attention_fuse(gnn_ir, self.ir_node_bias)
        gnn_fea = self.gnn_fuse(self.concat([gnn_rgb, gnn_ir]))

        gate = self.adaptive_gate(self.concat([base_fea, gnn_fea]))
        return base_fea + gate * (gnn_fea - base_fea)


class HAFFormerBayesianGFBArchive(nn.Module):
    """
    Active lightweight Bayesian-GFB version.

    It keeps the original LCAFNet GFB as the main fusion path and only adds a
    Bayesian reliability gate to recalibrate the RGB/IR fusion weights. The
    heavier Bayesian-GNN implementation is archived above and remains available.
    """
    def __init__(self, dim):
        super(HAFFormerBayesianGFBArchive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim
        self.bayes_prior = 0.80
        self.bayes_eps = 1e-6
        self.preserve_gain = 0.05
        self.is_bayesian_fusion = True
        self.bayesian_loss = None

        self.mhca_rgb = FrequencyCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = FrequencyCrossAttention_S(dim, num_heads, bias)

        self.concat = Concat(dimension=1)

        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        hidden = max(dim // 4, 16)
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(4 * dim, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2 * dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

        prior_logit = torch.logit(torch.tensor(self.bayes_prior)).item()
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, prior_logit)

    def _bernoulli_kl(self, q, prior):
        q = q.clamp(self.bayes_eps, 1.0 - self.bayes_eps)
        p = q.new_tensor(prior).clamp(self.bayes_eps, 1.0 - self.bayes_eps)
        return (q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))).mean()

    def _update_bayesian_loss(self, reliability, fused, base_fea):
        reliability_kl = self._bernoulli_kl(reliability, self.bayes_prior)
        preserve_loss = (fused - base_fea).pow(2).mean()
        self.bayesian_loss = reliability_kl + self.preserve_gain * preserve_loss

    def get_bayesian_loss(self):
        if self.bayesian_loss is None:
            return next(self.parameters()).new_zeros(())
        return self.bayesian_loss

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        # Original GFB path. This is the stable baseline fusion result.
        fea_cat = self.concat([out_fea_rgb, out_fea_ir])
        fea_conv = self.conv(fea_cat)
        gfb_weight = self.dwconv(fea_conv).sigmoid().clamp(self.bayes_eps, 1.0 - self.bayes_eps)
        base_fea = gfb_weight * out_fea_rgb + (1.0 - gfb_weight) * out_fea_ir

        # Bayesian reliability posterior q(z_rgb), q(z_ir). It is initialized
        # to the prior, so the module starts nearly identical to original GFB.
        reliability_input = self.concat([
            out_fea_rgb,
            out_fea_ir,
            torch.abs(out_fea_rgb - out_fea_ir),
            base_fea
        ])
        reliability = torch.sigmoid(self.reliability_gate(reliability_input)).clamp(
            self.bayes_eps, 1.0 - self.bayes_eps
        )
        rgb_reliability, ir_reliability = reliability.chunk(2, dim=1)

        rgb_score = gfb_weight * rgb_reliability
        ir_score = (1.0 - gfb_weight) * ir_reliability
        norm = (rgb_score + ir_score).clamp_min(self.bayes_eps)
        rgb_weight = (rgb_score / norm).clamp(self.bayes_eps, 1.0 - self.bayes_eps)
        ir_weight = 1.0 - rgb_weight

        new_fea = rgb_weight * out_fea_rgb + ir_weight * out_fea_ir
        self._update_bayesian_loss(reliability, new_fea, base_fea)

        return new_fea


class GatedModalFusion(nn.Module):
    """Original LCAFNet gated fusion block with a C-channel output."""
    def __init__(self, dim, bias=False):
        super().__init__()
        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def compute_logits(self, rgb_fea, ir_fea):
        seed = self.conv(self.concat([rgb_fea, ir_fea]))
        return self.dwconv(seed)

    def forward(self, rgb_fea, ir_fea):
        weight = self.compute_logits(rgb_fea, ir_fea).sigmoid()
        return weight * rgb_fea + (1.0 - weight) * ir_fea


class SharedLatentDiagonalChannelPrior(nn.Module):
    """
    Maximum-entropy correspondence prior for a shared RGB-IR latent space.

    For a query channel ``a``, the prior gives the matching key channel
    ``b=a`` a positive logit margin and otherwise remains uniform.  It is the
    maximum-entropy distribution subject to a prescribed same-channel
    preference.  Since RGB and IR use the same QKV projection, a diagonal
    correspondence has a stable meaning and is invariant to a simultaneous
    permutation of the shared latent channels.

    The returned matrix is row-centered for identifiability.  Row centering
    does not change softmax probabilities, so this is exactly equivalent to
    adding ``margin * I`` to the content logits.  Unlike the GOAT positional
    prior, this module contains no Fourier features and no key-only sink.
    """

    def __init__(self, num_heads, head_dim, margin_init=1.0,
                 margin_min=0.05, margin_max=1.5):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError('num_heads and head_dim must be positive')
        if not margin_min < margin_max:
            raise ValueError('margin_min must be smaller than margin_max')
        if not margin_min <= margin_init <= margin_max:
            raise ValueError(
                f'margin_init={margin_init} must be inside '
                f'[{margin_min}, {margin_max}]')

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.margin_min = float(margin_min)
        self.margin_max = float(margin_max)
        raw_margin = inverse_bounded_sigmoid(
            float(margin_init), self.margin_min, self.margin_max)
        self.raw_margin = nn.Parameter(raw_margin.repeat(self.num_heads))
        self.register_buffer(
            '_channel_identity', torch.eye(self.head_dim, dtype=torch.float32),
            persistent=False)

    def get_margin(self):
        span = self.margin_max - self.margin_min
        return self.margin_min + span * torch.sigmoid(self.raw_margin.float())

    def build_log_prior(self, dtype=None, device=None):
        if device is None:
            device = self.raw_margin.device
        if dtype is None:
            dtype = self.raw_margin.dtype
        margin = self.get_margin().to(device=device).view(
            self.num_heads, 1, 1)
        identity = self._channel_identity.to(device=device)
        prior = margin * identity.unsqueeze(0)
        prior = prior - prior.mean(dim=-1, keepdim=True)
        return prior.to(dtype=dtype)

    def diagonal_probability(self):
        """Prior probability assigned to the matching channel per head."""
        margin = self.get_margin()
        return torch.exp(margin) / (
            torch.exp(margin) + float(self.head_dim - 1))

    def forward(self, logits):
        prior = self.build_log_prior(
            dtype=logits.dtype, device=logits.device).unsqueeze(0)
        return logits + prior


class BayesianEvidentialCorrespondencePrior(nn.Module):
    """Content-conditioned Beta evidence gate for a diagonal channel prior.

    The gate predicts positive/negative evidence for the proposition that the
    shared RGB-IR latent channels are diagonally correspondent.  It consumes
    only statistics of the already-computed channel-attention logits, so the
    inference overhead is independent of the spatial resolution.  With no
    evidence the Beta(1, 1) posterior has unit subjective uncertainty and the
    prior is exactly (up to numerical precision) bypassed.
    """

    def __init__(self, num_heads, head_dim, hidden_dim=8, max_margin=0.5,
                 risk_kappa=1.0, evidence_bias=-2.0):
        super().__init__()
        if num_heads <= 0 or head_dim <= 1:
            raise ValueError('num_heads must be positive and head_dim must exceed one')
        if hidden_dim <= 0 or max_margin <= 0 or risk_kappa < 0:
            raise ValueError('hidden_dim/max_margin must be positive and risk_kappa non-negative')

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_margin = float(max_margin)
        self.risk_kappa = float(risk_kappa)
        # Digamma/Beta variance are numerically sensitive under AMP.  The
        # checkpoint and inference helpers retain this small submodule in FP32.
        self.force_fp32 = True
        # The five statistics have very different numerical ranges (raw logits,
        # standardized diagonal advantage, entropy concentration, cosine
        # agreement).  Normalizing them per sample prevents one statistic from
        # fixing the evidence head at an almost content-independent output.
        self.descriptor_norm = nn.LayerNorm(5)
        self.evidence_head = nn.Sequential(
            nn.Linear(5, int(hidden_dim), bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), 2, bias=True),
        )
        nn.init.kaiming_uniform_(
            self.evidence_head[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.evidence_head[0].bias)
        # A small non-zero initialization retains a first-step gradient path,
        # while 1e-2 is large enough to avoid the experimentally observed
        # constant-score plateau.  Bias -2 keeps the initial evidence cautious.
        nn.init.normal_(self.evidence_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.evidence_head[-1].bias, float(evidence_bias))
        self.register_buffer(
            '_channel_identity', torch.eye(self.head_dim, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            '_prior_warmup_scale', torch.zeros((), dtype=torch.float32),
            persistent=True)

        self.track_becp_stats = False
        self.last_becp_stats = None
        self._loss_components = None
        self._last_effective_margin = None
        self._last_log_odds = None

    def set_prior_warmup_scale(self, scale):
        self._prior_warmup_scale.fill_(float(min(max(scale, 0.0), 1.0)))

    def _descriptor(self, rgb_logits, ir_logits):
        # Stop gradients into Q/K: the evidence head must read content quality,
        # not encourage the attention projections to manufacture confidence.
        rgb = rgb_logits.detach().float()
        ir = ir_logits.detach().float()
        symmetric = 0.5 * (rgb + ir.transpose(-2, -1))
        diagonal = symmetric.diagonal(dim1=-2, dim2=-1)
        diagonal_mean = diagonal.mean(dim=-1)
        count = float(self.head_dim * (self.head_dim - 1))
        off_sum = symmetric.sum(dim=(-2, -1)) - diagonal.sum(dim=-1)
        off_mean = off_sum / count
        off_centered = symmetric - off_mean[..., None, None]
        off_mask = (1.0 - self._channel_identity).to(
            device=symmetric.device, dtype=symmetric.dtype)
        off_variance = (
            off_centered.square() * off_mask).sum(dim=(-2, -1)) / count
        diagonal_advantage = (
            diagonal_mean - off_mean) / off_variance.clamp_min(1e-8).sqrt()

        probability = symmetric.softmax(dim=-1).clamp_min(1e-8)
        entropy = -(probability * probability.log()).sum(dim=-1).mean(dim=-1)
        concentration = 1.0 - entropy / math.log(float(self.head_dim))
        directional_agreement = F.cosine_similarity(
            rgb.flatten(-2), ir.transpose(-2, -1).flatten(-2), dim=-1,
            eps=1e-6)
        return torch.stack([
            diagonal_mean,
            off_mean,
            diagonal_advantage,
            concentration,
            directional_agreement,
        ], dim=-1)

    def _posterior(self, descriptor):
        with amp.autocast(enabled=False):
            descriptor = descriptor.float()
            # Pickled checkpoints produced by the first BECP experiment predate
            # descriptor_norm.  They remain valid for evaluation; state-dict
            # transfer into the corrected model initializes the new norm.
            descriptor_norm = getattr(self, 'descriptor_norm', None)
            if descriptor_norm is not None:
                descriptor = descriptor_norm(descriptor)
            evidence = F.softplus(self.evidence_head(descriptor))
            alpha = 1.0 + evidence[..., 0]
            beta = 1.0 + evidence[..., 1]
        return alpha, beta

    @staticmethod
    def _beta_kl_uniform(alpha, beta):
        total = alpha + beta
        return (
            torch.lgamma(total) - torch.lgamma(alpha) - torch.lgamma(beta)
            + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(total))
            + (beta - 1.0) * (torch.digamma(beta) - torch.digamma(total)))

    def _confidence(self, alpha, beta):
        total = alpha + beta
        mean = alpha / total
        uncertainty = 2.0 / total
        variance = alpha * beta / (
            total.square() * (total + 1.0)).clamp_min(1e-8)
        lower_bound = (mean - self.risk_kappa * variance.sqrt()).clamp(0.0, 1.0)
        confidence = (lower_bound * (1.0 - uncertainty)).clamp(0.0, 1.0)
        return confidence, mean, uncertainty, variance

    def _compute_training_losses(self, positive, negative):
        pos_alpha, pos_beta = positive
        neg_alpha, neg_beta = negative
        pos_total = pos_alpha + pos_beta
        neg_total = neg_alpha + neg_beta
        evidential = 0.5 * (
            (torch.digamma(pos_total) - torch.digamma(pos_alpha)).mean()
            + (torch.digamma(neg_total) - torch.digamma(neg_beta)).mean())
        # Penalize evidence for the incorrect proposition only.  This avoids
        # suppressing correctly accumulated evidence as a full KL would do.
        kl = 0.5 * (
            self._beta_kl_uniform(torch.ones_like(pos_alpha), pos_beta).mean()
            + self._beta_kl_uniform(neg_alpha, torch.ones_like(neg_beta)).mean())
        pos_gate = self._confidence(pos_alpha, pos_beta)[0]
        neg_gate = self._confidence(neg_alpha, neg_beta)[0]
        # Optimize the uncompressed Beta log-odds rather than the final gate.
        # The old gate is the product of a lower confidence bound and an
        # uncertainty discount; around the cautious initialization its gradient
        # is so small that both classes can remain at the same score.  Averaging
        # each class independently also supports multiple negative constructions
        # per positive pair.
        pos_score = (pos_alpha.log() - pos_beta.log()).mean(dim=0)
        neg_score = (neg_alpha.log() - neg_beta.log()).mean(dim=0)
        ranking = F.relu(0.5 - pos_score + neg_score).mean()
        self._last_log_odds = (
            float(pos_score.detach().mean()),
            float(neg_score.detach().mean()),
            float((pos_score - neg_score).detach().mean()),
        )
        self._loss_components = (evidential, kl, ranking)
        return pos_gate, neg_gate

    def get_becp_loss_components(self):
        components = self._loss_components
        self._loss_components = None
        return components

    def get_margin(self):
        if self._last_effective_margin is None:
            return self._prior_warmup_scale.new_zeros(self.num_heads)
        return self._last_effective_margin

    def diagonal_probability(self):
        margin = self.get_margin()
        return torch.exp(margin) / (
            torch.exp(margin) + float(self.head_dim - 1))

    def forward(self, rgb_logits, ir_logits,
                negative_rgb_logits=None, negative_ir_logits=None):
        descriptor = self._descriptor(rgb_logits, ir_logits)
        alpha, beta = self._posterior(descriptor)
        gate, posterior_mean, uncertainty, _ = self._confidence(alpha, beta)
        negative_gate = gate.new_full(gate.shape, float('nan'))

        if (self.training and negative_rgb_logits is not None
                and negative_ir_logits is not None):
            negative_descriptor = self._descriptor(
                negative_rgb_logits, negative_ir_logits)
            neg_alpha, neg_beta = self._posterior(negative_descriptor)
            _, negative_gate = self._compute_training_losses(
                (alpha, beta), (neg_alpha, neg_beta))
        else:
            self._loss_components = None
            self._last_log_odds = None

        effective_margin = (
            self.max_margin * self._prior_warmup_scale.float() * gate)
        self._last_effective_margin = effective_margin.detach().mean(dim=0)
        identity = self._channel_identity.to(
            device=rgb_logits.device, dtype=torch.float32)
        prior = effective_margin[..., None, None] * identity
        prior = prior - prior.mean(dim=-1, keepdim=True)

        if self.track_becp_stats:
            strength = alpha + beta
            neg_mean = (float(negative_gate.detach().mean())
                        if torch.isfinite(negative_gate).any() else float('nan'))
            components = self._loss_components
            component_values = (
                tuple(float(x.detach()) for x in components)
                if components is not None else (float('nan'),) * 3)
            log_odds_values = (
                self._last_log_odds
                if self._last_log_odds is not None
                else (float('nan'),) * 3)
            gate_stats = gate.detach().float()
            strength_stats = strength.detach().float()
            self.last_becp_stats = (
                float(gate_stats.mean()),
                float(gate_stats.std()),
                float(gate_stats.min()),
                float(gate_stats.max()),
                float(posterior_mean.detach().mean()),
                float(uncertainty.detach().mean()),
                float(strength_stats.mean()),
                float(strength_stats.std()),
                float(gate_stats.mean()),
                neg_mean,
                *log_odds_values,
                *component_values,
                float(self._prior_warmup_scale),
            )
            self.track_becp_stats = False

        return rgb_logits.float() + prior, ir_logits.float() + prior


class DetectionAlignedForegroundBackgroundPrior(nn.Module):
    """Box-supervised, content-aligned prior for cross-modal channel attention.

    A tiny spatial head predicts objectness from RGB/IR agreement statistics.
    Its foreground-minus-background weights pool the already computed Q/K
    features into a *non-diagonal* channel-correlation residual.  The Q/K path
    is detached so the detector cannot manufacture an easy prior signal.  A
    signed per-direction/per-head coefficient starts at a small bounded value,
    while an explicit warmup buffer is initialized to zero.  The complete
    residual is capped at ``rho_max`` times the content-logit RMS. Consequently,
    loading a no-prior RGCA checkpoint remains function preserving throughout
    the mask-calibration phase without relying on a zero-gradient cold start.
    """

    def __init__(self, num_heads, head_dim, hidden_dim=8, rho_max=0.1,
                 rho_init=0.01):
        super().__init__()
        if num_heads <= 0 or head_dim <= 1:
            raise ValueError('num_heads must be positive and head_dim must exceed one')
        if hidden_dim <= 0 or not 0.0 < rho_max <= 1.0:
            raise ValueError('hidden_dim must be positive and rho_max must be in (0, 1]')
        if not abs(float(rho_init)) < float(rho_max):
            raise ValueError('abs(rho_init) must be smaller than rho_max')
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rho_max = float(rho_max)
        # The mask normalization and RMS cap are numerically sensitive.  The
        # checkpoint/inference helpers preserve this 1.2k-parameter module in
        # FP32 even when the rest of the detector runs in FP16.
        self.force_fp32 = True
        # Four channel-free statistics keep the parameter cost identical at
        # every pyramid scale: 4*8*3*3 + 8 + 8 + 1 = 305 parameters.
        self.mask_head = nn.Sequential(
            nn.Conv2d(4, int(hidden_dim), 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden_dim), 1, 1, bias=True),
        )
        nn.init.kaiming_normal_(
            self.mask_head[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.mask_head[0].bias)
        nn.init.normal_(self.mask_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.mask_head[-1].bias)
        raw_rho_init = math.atanh(float(rho_init) / self.rho_max)
        self.raw_rho = nn.Parameter(torch.full(
            (2, self.num_heads), raw_rho_init, dtype=torch.float32))
        self.register_buffer(
            '_dacp_warmup_scale', torch.zeros((), dtype=torch.float32),
            persistent=True)

        self.track_dacp_stats = False
        self.last_dacp_stats = None
        self._last_objectness_mask = None
        self._last_objectness_logits = None

    def set_dacp_warmup_scale(self, scale):
        self._dacp_warmup_scale.fill_(float(min(max(scale, 0.0), 1.0)))

    def get_effective_rho(self):
        return (self.rho_max * self._dacp_warmup_scale.float()
                * torch.tanh(self.raw_rho.float()))

    def get_margin(self):
        # Retain the generic RGCA diagnostic interface.  This is a bounded
        # residual ratio, not a diagonal logit margin.
        return self.get_effective_rho().detach().abs().mean(dim=0)

    def diagonal_probability(self):
        # DACP makes no same-index assumption; its neutral diagonal statistic
        # is therefore the uniform channel probability.
        return self.raw_rho.new_full(
            (self.num_heads,), 1.0 / float(self.head_dim))

    @staticmethod
    def _standardize_maps(descriptor):
        mean = descriptor.mean(dim=(-2, -1), keepdim=True)
        variance = descriptor.var(
            dim=(-2, -1), keepdim=True, unbiased=False)
        return (descriptor - mean) / variance.add(1e-6).sqrt()

    def predict_objectness(self, rgb_fea, ir_fea):
        with amp.autocast(enabled=False):
            rgb_float, ir_float = rgb_fea.float(), ir_fea.float()
            rgb_unit = F.normalize(rgb_float, dim=1, eps=1e-6)
            ir_unit = F.normalize(ir_float, dim=1, eps=1e-6)
            descriptor = torch.cat([
                rgb_float.mean(dim=1, keepdim=True),
                ir_float.mean(dim=1, keepdim=True),
                (rgb_float - ir_float).abs().mean(dim=1, keepdim=True),
                (rgb_unit * ir_unit).mean(dim=1, keepdim=True),
            ], dim=1)
            objectness_logits = self.mask_head(
                self._standardize_maps(descriptor))
            mask = torch.sigmoid(objectness_logits)
        if self.training:
            # The training loop consumes this tensor exactly once to attach
            # Gaussian box supervision at the corresponding feature scale.
            self._last_objectness_mask = mask
            self._last_objectness_logits = objectness_logits
        return mask

    def pop_objectness_prediction(self):
        """Return cached pre-sigmoid logits and probabilities exactly once."""
        prediction = (self._last_objectness_logits,
                      self._last_objectness_mask)
        self._last_objectness_logits = None
        self._last_objectness_mask = None
        return prediction

    def pop_objectness_mask(self):
        # Historical helper retained for compatibility with older diagnostics.
        mask = self._last_objectness_mask
        self._last_objectness_mask = None
        self._last_objectness_logits = None
        return mask

    @staticmethod
    def _contrastive_residual(query, key, mask):
        mask_flat = mask.float().flatten(2).squeeze(1)
        foreground = mask_flat / mask_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        background = (1.0 - mask_flat)
        background = background / background.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        signed_weight = foreground - background
        query = query.detach().float()
        key = key.detach().float()
        residual = ((query * signed_weight[:, None, None, :])
                    @ key.transpose(-2, -1))
        return residual - residual.mean(dim=-1, keepdim=True)

    def _bounded_prior(self, residual, content, rho):
        residual_rms = residual.square().mean(
            dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        content_rms = content.detach().float().square().mean(
            dim=(-2, -1), keepdim=True).sqrt()
        return (residual / residual_rms) * content_rms * rho[..., None, None]

    def forward(self, rgb_logits, ir_logits, q_rgb, k_rgb, q_ir, k_ir,
                objectness_mask):
        with amp.autocast(enabled=False):
            residual_rgb = self._contrastive_residual(
                q_rgb, k_ir, objectness_mask)
            residual_ir = self._contrastive_residual(
                q_ir, k_rgb, objectness_mask)
            rho = self.get_effective_rho().to(device=rgb_logits.device)
            prior_rgb = self._bounded_prior(
                residual_rgb, rgb_logits, rho[0][None, :])
            prior_ir = self._bounded_prior(
                residual_ir, ir_logits, rho[1][None, :])

        if self.track_dacp_stats:
            mask_stats = objectness_mask.detach().float()
            ratios = []
            positive = []
            for prior, content in ((prior_rgb, rgb_logits), (prior_ir, ir_logits)):
                prior_rms = prior.detach().float().square().mean(
                    dim=(-2, -1)).sqrt()
                content_rms = content.detach().float().square().mean(
                    dim=(-2, -1)).sqrt().clamp_min(1e-8)
                ratios.append(prior_rms / content_rms)
                positive.append((prior.detach() > 0).float().mean())
            ratio = torch.cat(ratios, dim=0)
            rho_stats = rho.detach().float()
            self.last_dacp_stats = (
                float(mask_stats.mean()), float(mask_stats.std()),
                float(mask_stats.min()), float(mask_stats.max()),
                float(rho_stats.mean()), float(rho_stats.min()),
                float(rho_stats.max()), float(ratio.mean()),
                float(ratio.max()), float(torch.stack(positive).mean()),
                float(self._dacp_warmup_scale),
            )
            self.track_dacp_stats = False

        return (rgb_logits.float() + prior_rgb,
                ir_logits.float() + prior_ir)


class ReliabilityGuidedBidirectionalCrossAttention(nn.Module):
    """
    Lightweight, frequency-free cross-modal attention for aligned RGB-IR features.

    The module has three deliberately separate paths:

    1. True bidirectional cross-covariance attention. RGB queries correlate
       with IR keys/values and vice versa; unlike the historical LCAF block,
       q and k do not come from the same modality.
    2. An optional attention prior is retained only for historical ablations.
       The active no-prior RGCA path uses content logits directly.
    3. A depthwise-convolutional value path supplies an efficient local bias.
       It is mixed with the global channel interaction instead of constructing
       a quadratic H*W by H*W spatial attention matrix.
    4. A content-dependent reliability gate is an explicit no-transfer path.
       When the complementary modality is unreliable, the outer residual can
       preserve the primary modality rather than forcing softmax attention to
       import complementary features.

    Projection weights are shared between RGB and IR so both streams are
    compared in the same latent space. Modality-specific LayerNorm parameters
    retain enough freedom to handle their different feature statistics.
    Complexity is linear in the number of spatial positions and the active
    path contains no FFT, complex parameters, frequency routing, positional
    Fourier features, or attention-sink parameters.
    """

    def __init__(self, dim, reduction=2, num_heads=4, bias=False,
                 local_kernel_size=5, residual_init=0.1,
                 channel_prior_init=1.0, channel_prior_min=0.05,
                 channel_prior_max=1.5, residual_min=0.02,
                 residual_max=0.15, prior_mode='none',
                 becp_max_margin=0.5, dacp_rho_max=0.1,
                 dacp_rho_init=0.01, residual_mode='legacy_channel',
                 use_reliability_gate=True):
        super().__init__()
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
            raise ValueError(f'Unsupported RGCA residual mode: {residual_mode}')
        self.residual_mode = str(residual_mode)

        self.rgb_norm = LayerNorm(dim, 'WithBias')
        self.ir_norm = LayerNorm(dim, 'WithBias')

        # The same projection/local kernel is applied to both modalities. This
        # creates a common latent coordinate system without doubling weights.
        self.qkv = nn.Conv2d(dim, 3 * self.reduced_dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            3 * self.reduced_dim,
            3 * self.reduced_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=3 * self.reduced_dim,
            bias=bias)

        local_padding = local_kernel_size // 2
        self.local_value = nn.Conv2d(
            self.reduced_dim,
            self.reduced_dim,
            kernel_size=local_kernel_size,
            stride=1,
            padding=local_padding,
            groups=self.reduced_dim,
            bias=bias)

        # One positive temperature and one global-vs-local mixing coefficient
        # per head. Equal mixing is a neutral starting point.
        self.raw_temperature = nn.Parameter(
            init_positive_temperature(
                self.num_heads, value=math.sqrt(self.head_dim)))
        self.cross_mix_logit = nn.Parameter(torch.zeros(self.num_heads))
        if prior_mode == 'none':
            # Explicit no-prior path.  Keeping this as a real constructor mode
            # makes fresh models reproduce the historical best RGCA run rather
            # than depending on a missing attribute in an old pickled model.
            self.channel_prior = None
        elif prior_mode == 'fixed':
            self.channel_prior = SharedLatentDiagonalChannelPrior(
                self.num_heads, self.head_dim,
                margin_init=float(channel_prior_init),
                margin_min=float(channel_prior_min),
                margin_max=float(channel_prior_max))
        elif prior_mode == 'becp':
            self.channel_prior = BayesianEvidentialCorrespondencePrior(
                self.num_heads, self.head_dim, max_margin=becp_max_margin)
        elif prior_mode == 'dacp':
            self.channel_prior = DetectionAlignedForegroundBackgroundPrior(
                self.num_heads, self.head_dim, rho_max=dacp_rho_max,
                rho_init=dacp_rho_init)
        else:
            raise ValueError(f'Unsupported RGCA prior mode: {prior_mode}')
        self.prior_mode = str(prior_mode)

        self.use_reliability_gate = bool(use_reliability_gate)
        gate_hidden = max(self.reduced_dim // 2, 8)
        reliability_gate = nn.Sequential(
            nn.Conv2d(
                4 * self.reduced_dim, gate_hidden,
                kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                gate_hidden, gate_hidden, kernel_size=3, stride=1,
                padding=1, groups=gate_hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid())
        # Start close to a neutral 0.5 gate, while retaining a first-step
        # gradient path through the complete reliability network.
        nn.init.kaiming_normal_(
            reliability_gate[-2].weight, mode='fan_in',
            nonlinearity='linear')
        nn.init.zeros_(reliability_gate[-2].bias)
        if self.use_reliability_gate:
            self.reliability_gate = reliability_gate
        else:
            # Construct, initialize, then discard the gate.  This consumes the
            # same RNG sequence as the full RGCA block, so every retained
            # tensor is bit-identical at the same seed while no hidden gate
            # parameter remains in the ablation checkpoint.
            self.reliability_gate = None

        self.out_proj = nn.Conv2d(self.reduced_dim, dim, kernel_size=1, bias=bias)
        if self.residual_mode == 'legacy_channel':
            # Exact parameterization used by the best no-prior experiment:
            # one trainable scale for every direction and output channel.
            self.residual_scale = nn.Parameter(torch.full(
                (2, dim, 1, 1), float(residual_init)))
        else:
            # Prior ablations retain the later bounded per-direction repair.
            raw_residual = inverse_bounded_sigmoid(
                float(residual_init), self.residual_min, self.residual_max)
            self.raw_residual_scale = nn.Parameter(
                raw_residual.repeat(2).view(2, 1, 1, 1))

        self.track_attention_stats = False
        self.last_attention_stats = None
        # Opt-in training hook for Reliability-Supervised RGCA (RS-RGCA).
        # It is intentionally transient: no parameter/buffer is added, so
        # historical RGCA checkpoints retain a byte-for-byte compatible state
        # dict and the detector forward path is unchanged when supervision is
        # enabled.  The auxiliary prediction is recomputed from a detached
        # descriptor so its loss updates only the existing reliability gate.
        self.enable_rs_reliability_supervision = False
        self.last_rs_reliability_predictions = None

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

    def _from_heads(self, x, h, w):
        return rearrange(
            x, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=h, w=w)

    def _content_logits(self, query, key):
        temperature = self.get_temperature().to(dtype=query.dtype).unsqueeze(0)
        return (query @ key.transpose(-2, -1)) * temperature

    def _reliability(self, primary, complementary):
        difference = torch.abs(primary - complementary)
        agreement = primary * complementary
        descriptor = torch.cat(
            [primary, complementary, difference, agreement], dim=1)
        return self.reliability_gate(descriptor)

    def _reliability_with_auxiliary(self, primary, complementary):
        """Return the original gate and an optional gate-only aux estimate.

        The first output is exactly the historical RGCA computation.  When
        RS-RGCA supervision is active, the same lightweight gate is evaluated
        on a detached copy of its descriptor.  This preserves detection-loss
        gradients through the original path while preventing the reliability
        auxiliary loss from teaching the backbone/QKV projection shortcuts.
        """
        difference = torch.abs(primary - complementary)
        agreement = primary * complementary
        descriptor = torch.cat(
            [primary, complementary, difference, agreement], dim=1)
        reliability = self.reliability_gate(descriptor)
        auxiliary = None
        if (self.training
                and getattr(self, 'enable_rs_reliability_supervision', False)):
            auxiliary = self.reliability_gate(descriptor.detach())
        return reliability, auxiliary

    def _record_stats(self, rgb_attention, ir_attention,
                      rgb_reliability, ir_reliability):
        if not self.track_attention_stats:
            return

        def normalized_entropy(attention):
            probability = attention.detach().float().clamp_min(1e-8)
            denom = math.log(max(self.head_dim, 2))
            return float((-(probability * probability.log()).sum(-1) / denom).mean())

        rgb_att = rgb_attention.detach().float()
        ir_att = ir_attention.detach().float()
        rgb_gate = rgb_reliability.detach().float()
        ir_gate = ir_reliability.detach().float()
        residual_scale = self.get_residual_scale().detach().float()
        channel_prior = getattr(self, 'channel_prior', None)
        if channel_prior is None:
            prior_margin = residual_scale.new_zeros(self.num_heads)
            prior_diagonal_probability = residual_scale.new_full(
                (self.num_heads,), 1.0 / float(self.head_dim))
        else:
            prior_margin = channel_prior.get_margin().detach().float()
            prior_diagonal_probability = (
                channel_prior.diagonal_probability().detach().float())
        self.last_attention_stats = (
            normalized_entropy(rgb_attention),
            normalized_entropy(ir_attention),
            float(rgb_att.std()),
            float(ir_att.std()),
            float(rgb_gate.mean()),
            float(rgb_gate.std()),
            float(ir_gate.mean()),
            float(ir_gate.std()),
            float(torch.sigmoid(self.cross_mix_logit.detach().float()).mean()),
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
        _, _, h, w = rgb_fea.shape

        normalized = torch.cat(
            [self.rgb_norm(rgb_fea), self.ir_norm(ir_fea)], dim=0)
        projected = self.qkv_dwconv(self.qkv(normalized))
        rgb_projected, ir_projected = projected.chunk(2, dim=0)
        q_rgb, k_rgb, v_rgb = rgb_projected.chunk(3, dim=1)
        q_ir, k_ir, v_ir = ir_projected.chunk(3, dim=1)

        # RGB receives IR values and IR receives RGB values. The channel
        # correlation is global over H*W but never materializes token attention.
        q_rgb_heads = F.normalize(self._to_heads(q_rgb), dim=-1, eps=1e-6)
        k_rgb_heads = F.normalize(self._to_heads(k_rgb), dim=-1, eps=1e-6)
        q_ir_heads = F.normalize(self._to_heads(q_ir), dim=-1, eps=1e-6)
        k_ir_heads = F.normalize(self._to_heads(k_ir), dim=-1, eps=1e-6)
        v_rgb_heads = self._to_heads(v_rgb)
        v_ir_heads = self._to_heads(v_ir)
        content_rgb = self._content_logits(q_rgb_heads, k_ir_heads)
        content_ir = self._content_logits(q_ir_heads, k_rgb_heads)

        channel_prior = getattr(self, 'channel_prior', None)
        if isinstance(channel_prior, DetectionAlignedForegroundBackgroundPrior):
            objectness_mask = channel_prior.predict_objectness(rgb_fea, ir_fea)
            channel_prior.track_dacp_stats = self.track_attention_stats
            attention_logits_rgb, attention_logits_ir = channel_prior(
                content_rgb.float(), content_ir.float(),
                q_rgb_heads, k_rgb_heads, q_ir_heads, k_ir_heads,
                objectness_mask)
        elif isinstance(channel_prior, BayesianEvidentialCorrespondencePrior):
            negative_rgb = negative_ir = None
            if self.training:
                # Two complementary negatives teach actual correspondence:
                # (1) a random cross-sample modality mismatch when B > 1;
                # (2) a controlled one-channel displacement in the shared
                # latent space, which remains available for B == 1.
                negative_rgb_parts = [self._content_logits(
                    q_rgb_heads, torch.roll(k_ir_heads, shifts=1, dims=-2))]
                negative_ir_parts = [self._content_logits(
                    torch.roll(q_ir_heads, shifts=1, dims=-2), k_rgb_heads)]
                if content_rgb.shape[0] > 1:
                    mismatch_shift = int(torch.randint(
                        1, content_rgb.shape[0], (),
                        device=content_rgb.device).item())
                    negative_rgb_parts.append(self._content_logits(
                        q_rgb_heads,
                        torch.roll(k_ir_heads, shifts=mismatch_shift, dims=0)))
                    negative_ir_parts.append(self._content_logits(
                        torch.roll(q_ir_heads, shifts=mismatch_shift, dims=0),
                        k_rgb_heads))
                negative_rgb = torch.cat(negative_rgb_parts, dim=0)
                negative_ir = torch.cat(negative_ir_parts, dim=0)
            channel_prior.track_becp_stats = self.track_attention_stats
            attention_logits_rgb, attention_logits_ir = channel_prior(
                content_rgb.float(), content_ir.float(),
                negative_rgb.float() if negative_rgb is not None else None,
                negative_ir.float() if negative_ir is not None else None)
        elif channel_prior is not None:
            attention_logits_rgb = channel_prior(content_rgb.float())
            attention_logits_ir = channel_prior(content_ir.float())
        else:
            # Backward compatibility for serialized no-prior RGCA checkpoints.
            attention_logits_rgb = content_rgb.float()
            attention_logits_ir = content_ir.float()

        attention_rgb = attention_logits_rgb.softmax(dim=-1).to(
            dtype=v_ir_heads.dtype)
        attention_ir = attention_logits_ir.softmax(dim=-1).to(
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
        mixed_rgb = self._from_heads(mixed_rgb, h, w)
        mixed_ir = self._from_heads(mixed_ir, h, w)

        # v_rgb and v_ir share projection weights, making their agreement and
        # discrepancy meaningful inputs for the bidirectional trust decision.
        if getattr(self, 'use_reliability_gate', True):
            reliability_rgb, auxiliary_rgb = self._reliability_with_auxiliary(
                v_rgb, v_ir)
            reliability_ir, auxiliary_ir = self._reliability_with_auxiliary(
                v_ir, v_rgb)
        else:
            # Identity transfer is the mathematically clean removal of the
            # learned reliability gate.  A fixed 0.5 multiplier would still
            # be a gate and would confound this ablation with attenuation.
            reliability_rgb = mixed_rgb.new_ones(
                (mixed_rgb.shape[0], 1, h, w))
            reliability_ir = mixed_ir.new_ones(
                (mixed_ir.shape[0], 1, h, w))
            auxiliary_rgb = auxiliary_ir = None

        # Direction semantics are source based: ``reliability_rgb`` controls
        # IR values injected into RGB, whereas ``reliability_ir`` controls RGB
        # values injected into IR.  Historical gates were initialized and
        # trained around 0.5, so RS-RGCA exposes 2*GAP(gate) as a normalized
        # reliability whose neutral clean value is one.  Only per-sample
        # scalars are retained, keeping the auxiliary graph small.
        if auxiliary_rgb is not None and auxiliary_ir is not None:
            self.last_rs_reliability_predictions = {
                'r_ir_to_rgb': 2.0 * auxiliary_rgb.mean(dim=(2, 3)),
                'r_rgb_to_ir': 2.0 * auxiliary_ir.mean(dim=(2, 3)),
            }
        else:
            self.last_rs_reliability_predictions = None
        # Evaluation-only visualization hook. Direction semantics are fixed:
        # reliability_rgb gates IR values injected into RGB, while
        # reliability_ir gates RGB values injected into IR. Keeping this
        # behind an opt-in flag avoids changing training, outputs, or
        # checkpoint state.
        if getattr(self, 'export_reliability_maps', False):
            self.last_reliability_maps = {
                'r_ir_to_rgb': reliability_rgb.detach().float().cpu(),
                'r_rgb_to_ir': reliability_ir.detach().float().cpu(),
            }
        mixed = torch.cat(
            [mixed_rgb * reliability_rgb, mixed_ir * reliability_ir], dim=0)
        delta_rgb, delta_ir = self.out_proj(mixed).chunk(2, dim=0)
        residual_scale = self.get_residual_scale().to(
            device=delta_rgb.device, dtype=delta_rgb.dtype)
        delta_rgb = delta_rgb * residual_scale[0]
        delta_ir = delta_ir * residual_scale[1]

        # Evaluation-only hook for a fair with/without-reliability comparison.
        # These are the actual cross-modal residual updates after projection
        # and residual scaling; the gated model includes its learned spatial
        # reliability modulation, while the strict ablation uses identity
        # transfer.  No return value, parameter, or checkpoint key changes.
        if getattr(self, 'export_cross_modal_response_maps', False):
            self.last_cross_modal_response_maps = {
                'rgb_update': delta_rgb.detach().float().cpu(),
                'ir_update': delta_ir.detach().float().cpu(),
            }

        self._record_stats(
            attention_rgb, attention_ir, reliability_rgb, reliability_ir)
        return delta_rgb, delta_ir


class ReliabilityConditionedBidirectionalCrossAttentionV2(nn.Module):
    """Reliability-conditioned, frequency-free RGB/IR cross attention.

    RC-RGCA v2 separates *source quality* from *pair correspondence* instead
    of asking one post-attention sigmoid map to represent both.  A detached,
    explicit low/high-frequency descriptor predicts ``q_rgb``, ``q_ir`` and
    ``c_pair``.  The resulting directional reliabilities condition the source
    keys and values before channel cross-covariance attention and compete with
    a learned null (no-transfer) key inside the same softmax.  Consequently an
    unreliable complementary modality can reduce cross-modal transfer rather
    than merely rescale an already-computed update.

    The descriptor uses pooling/high-pass residuals and finite differences;
    it contains no FFT or complex-valued operation.  Its feature input is
    stopped-gradient, following the reliability-controller separation used by
    reliability-conditioned routing methods, while the controller itself is
    fully trainable by detection and explicit degradation supervision.
    """

    descriptor_dim = 13

    def __init__(self, dim, reduction=2, num_heads=4, bias=False,
                 local_kernel_size=5, residual_init=0.1,
                 residual_min=0.02, residual_max=0.25,
                 reliability_init=0.8, null_logit_limit=2.0):
        super().__init__()
        if dim <= 0 or reduction <= 0:
            raise ValueError('dim and reduction must be positive')
        if local_kernel_size % 2 == 0:
            raise ValueError('local_kernel_size must be odd')
        if not residual_min < residual_init < residual_max:
            raise ValueError('residual_init must be inside the residual bounds')
        if not 0.0 < reliability_init < 1.0:
            raise ValueError('reliability_init must be in (0, 1)')

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
        self.null_logit_limit = float(null_logit_limit)

        self.rgb_norm = LayerNorm(dim, 'WithBias')
        self.ir_norm = LayerNorm(dim, 'WithBias')
        self.qkv = nn.Conv2d(dim, 3 * self.reduced_dim, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            3 * self.reduced_dim, 3 * self.reduced_dim, 3, padding=1,
            groups=3 * self.reduced_dim, bias=bias)
        padding = local_kernel_size // 2
        self.local_value = nn.Conv2d(
            self.reduced_dim, self.reduced_dim, local_kernel_size,
            padding=padding, groups=self.reduced_dim, bias=bias)
        self.out_proj = nn.Conv2d(self.reduced_dim, dim, 1, bias=bias)

        controller_hidden = max(24, min(self.reduced_dim, 96))
        self.descriptor_norm = nn.LayerNorm(self.descriptor_dim)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(self.descriptor_dim, controller_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(controller_hidden, controller_hidden),
            nn.SiLU(inplace=True))
        self.reliability_head = nn.Linear(controller_hidden, 3)
        self.mix_head = nn.Linear(controller_hidden, 2 * self.num_heads)

        local_hidden = max(8, min(self.reduced_dim // 4, 32))
        self.local_quality = nn.Sequential(
            nn.Conv2d(3, local_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_hidden, local_hidden, 3, padding=1,
                      groups=local_hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_hidden, 1, 1, bias=True))

        reliability_bias = math.log(
            reliability_init / (1.0 - reliability_init))
        nn.init.normal_(self.reliability_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reliability_head.bias, reliability_bias)
        nn.init.normal_(self.mix_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mix_head.bias)
        nn.init.normal_(self.local_quality[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.local_quality[-1].bias, reliability_bias)

        self.raw_temperature = nn.Parameter(init_positive_temperature(
            self.num_heads, value=math.sqrt(self.head_dim)))
        self.raw_null_logit = nn.Parameter(torch.zeros(2, self.num_heads))
        raw_residual = inverse_bounded_sigmoid(
            float(residual_init), self.residual_min, self.residual_max)
        self.raw_residual_scale = nn.Parameter(
            raw_residual.repeat(2).view(2, 1, 1, 1))

        self.track_attention_stats = False
        self.last_attention_stats = None
        self.last_reliability_predictions = None

    def get_temperature(self):
        return positive_temperature(self.raw_temperature, eps=1e-6)

    def get_residual_scale(self):
        span = self.residual_max - self.residual_min
        return self.residual_min + span * torch.sigmoid(
            self.raw_residual_scale.float())

    def get_null_logit(self):
        return self.null_logit_limit * torch.tanh(self.raw_null_logit.float())

    def _to_heads(self, x):
        return rearrange(
            x, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads)

    def _from_heads(self, x, h, w):
        return rearrange(
            x, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=h, w=w)

    @staticmethod
    def _standardize_map(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        variance = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mean) / variance.add(1e-6).sqrt()

    @staticmethod
    def _feature_statistics(feature):
        low = F.avg_pool2d(feature, 5, stride=1, padding=2)
        high = feature - low
        dx = feature[..., :, 1:] - feature[..., :, :-1]
        dy = feature[..., 1:, :] - feature[..., :-1, :]

        def log_energy(value):
            return value.square().mean(dim=(1, 2, 3)).add(1e-6).log()

        return torch.stack([
            log_energy(low),
            log_energy(high),
            log_energy(dx),
            log_energy(dy),
            feature.flatten(1).std(dim=1, unbiased=False).add(1e-6).log(),
        ], dim=1), low, high

    @staticmethod
    def _correlation(a, b):
        a = a.flatten(1)
        b = b.flatten(1)
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        numerator = (a * b).mean(dim=1)
        denominator = (a.square().mean(dim=1) * b.square().mean(dim=1))
        return numerator / denominator.add(1e-8).sqrt()

    def _explicit_descriptor(self, rgb, ir):
        # Stop-gradient protects the detector backbone from learning shortcuts
        # whose only purpose is to manipulate the reliability controller.
        with amp.autocast(enabled=False):
            rgb = rgb.detach().float()
            ir = ir.detach().float()
            rgb_stats, rgb_low, rgb_high = self._feature_statistics(rgb)
            ir_stats, ir_low, ir_high = self._feature_statistics(ir)
            cross = torch.stack([
                self._correlation(rgb_low, ir_low),
                self._correlation(rgb_high, ir_high),
                F.cosine_similarity(rgb.flatten(1), ir.flatten(1), dim=1),
            ], dim=1)
            descriptor = torch.cat([rgb_stats, ir_stats, cross], dim=1)
            descriptor = torch.nan_to_num(
                descriptor, nan=0.0, posinf=10.0, neginf=-10.0)
        return descriptor.to(dtype=self.descriptor_norm.weight.dtype)

    def _local_descriptor(self, value):
        value = value.detach().float()
        mean_abs = value.abs().mean(dim=1, keepdim=True)
        low = F.avg_pool2d(value, 5, stride=1, padding=2)
        high = (value - low).abs().mean(dim=1, keepdim=True)
        dx = F.pad((value[..., :, 1:] - value[..., :, :-1]).abs(),
                   (0, 1, 0, 0))
        dy = F.pad((value[..., 1:, :] - value[..., :-1, :]).abs(),
                   (0, 0, 0, 1))
        gradient = (dx + dy).mean(dim=1, keepdim=True)
        descriptor = torch.cat([
            self._standardize_map(mean_abs),
            self._standardize_map(high),
            self._standardize_map(gradient),
        ], dim=1)
        return descriptor.to(dtype=self.local_quality[0].weight.dtype)

    def _conditioned_attention(self, query, key, value, spatial_reliability,
                               global_reliability, direction_index):
        spatial = spatial_reliability.flatten(2).unsqueeze(1).clamp(1e-4, 1.0)
        root = spatial.sqrt()
        query = F.normalize(query * root, dim=-1, eps=1e-6)
        key = F.normalize(key * root, dim=-1, eps=1e-6)
        value = value * spatial
        temperature = self.get_temperature().to(
            device=query.device, dtype=query.dtype).view(1, -1, 1, 1)
        content = (query @ key.transpose(-2, -1)) * temperature

        reliability = global_reliability.float().clamp(1e-4, 1.0 - 1e-4)
        # The -log(head_dim) term makes the aggregate real-key probability
        # equal to r when all content logits are neutral.  Without it, d real
        # keys would overwhelm the single null key even at low reliability.
        real_logits = (content.float() + reliability.log().view(-1, 1, 1, 1)
                       - math.log(float(self.head_dim)))
        null_bias = self.get_null_logit()[direction_index].view(1, -1, 1, 1)
        null_logits = null_bias + (1.0 - reliability).log().view(-1, 1, 1, 1)
        null_logits = null_logits.expand(-1, -1, self.head_dim, -1)
        attention = torch.cat([real_logits, null_logits], dim=-1).softmax(dim=-1)
        attention = attention.to(dtype=value.dtype)
        return attention[..., :-1] @ value, attention

    def _record_stats(self, rgb_attention, ir_attention,
                      r_ir_to_rgb, r_rgb_to_ir, mix):
        if not self.track_attention_stats:
            return

        def normalized_entropy(attention):
            probability = attention.detach().float().clamp_min(1e-8)
            denom = math.log(max(self.head_dim + 1, 2))
            return float((-(probability * probability.log()).sum(-1)
                          / denom).mean())

        rgb_att = rgb_attention.detach().float()
        ir_att = ir_attention.detach().float()
        rgb_rel = r_ir_to_rgb.detach().float()
        ir_rel = r_rgb_to_ir.detach().float()
        residual = self.get_residual_scale().detach().float()
        null_mass = torch.cat([
            rgb_att[..., -1].reshape(-1), ir_att[..., -1].reshape(-1)])
        self.last_attention_stats = (
            normalized_entropy(rgb_attention),
            normalized_entropy(ir_attention),
            float(rgb_att.std()), float(ir_att.std()),
            float(rgb_rel.mean()), float(rgb_rel.std()),
            float(ir_rel.mean()), float(ir_rel.std()),
            float(mix.detach().float().mean()),
            float(residual.mean()), float(residual.min()),
            float(null_mass.mean()), float(null_mass.mean()),
        )
        self.track_attention_stats = False

    def forward(self, rgb_fea, ir_fea):
        if rgb_fea.shape != ir_fea.shape:
            raise ValueError(
                'RGB and IR features must have identical shapes, got '
                f'{tuple(rgb_fea.shape)} and {tuple(ir_fea.shape)}')
        batch, _, height, width = rgb_fea.shape

        normalized = torch.cat(
            [self.rgb_norm(rgb_fea), self.ir_norm(ir_fea)], dim=0)
        projected = self.qkv_dwconv(self.qkv(normalized))
        rgb_projected, ir_projected = projected.chunk(2, dim=0)
        q_rgb, k_rgb, v_rgb = rgb_projected.chunk(3, dim=1)
        q_ir, k_ir, v_ir = ir_projected.chunk(3, dim=1)

        descriptor = self._explicit_descriptor(rgb_fea, ir_fea)
        encoded = self.descriptor_encoder(self.descriptor_norm(descriptor))
        global_quality = torch.sigmoid(self.reliability_head(encoded))
        q_rgb_global = global_quality[:, 0:1]
        q_ir_global = global_quality[:, 1:2]
        correspondence = global_quality[:, 2:3]
        mix = torch.sigmoid(self.mix_head(encoded)).view(
            batch, 2, self.num_heads, 1, 1)

        local_rgb = torch.sigmoid(
            self.local_quality(self._local_descriptor(v_rgb)))
        local_ir = torch.sigmoid(
            self.local_quality(self._local_descriptor(v_ir)))
        effective_rgb = (q_rgb_global.view(batch, 1, 1, 1)
                         * local_rgb).clamp_min(1e-6).sqrt()
        effective_ir = (q_ir_global.view(batch, 1, 1, 1)
                        * local_ir).clamp_min(1e-6).sqrt()
        r_ir_to_rgb = correspondence.view(batch, 1, 1, 1) * effective_ir
        r_rgb_to_ir = correspondence.view(batch, 1, 1, 1) * effective_rgb
        r_ir_global = q_ir_global * correspondence
        r_rgb_global = q_rgb_global * correspondence

        q_rgb_heads, k_rgb_heads = self._to_heads(q_rgb), self._to_heads(k_rgb)
        q_ir_heads, k_ir_heads = self._to_heads(q_ir), self._to_heads(k_ir)
        v_rgb_heads, v_ir_heads = self._to_heads(v_rgb), self._to_heads(v_ir)
        cross_rgb, attention_rgb = self._conditioned_attention(
            q_rgb_heads, k_ir_heads, v_ir_heads, r_ir_to_rgb,
            r_ir_global, direction_index=0)
        cross_ir, attention_ir = self._conditioned_attention(
            q_ir_heads, k_rgb_heads, v_rgb_heads, r_rgb_to_ir,
            r_rgb_global, direction_index=1)

        local_values = self.local_value(torch.cat([v_rgb, v_ir], dim=0))
        local_rgb_source, local_ir_source = local_values.chunk(2, dim=0)
        imported_rgb = self._to_heads(local_ir_source) * (
            r_ir_to_rgb.flatten(2).unsqueeze(1))
        imported_ir = self._to_heads(local_rgb_source) * (
            r_rgb_to_ir.flatten(2).unsqueeze(1))
        mixed_rgb = mix[:, 0] * cross_rgb + (1.0 - mix[:, 0]) * imported_rgb
        mixed_ir = mix[:, 1] * cross_ir + (1.0 - mix[:, 1]) * imported_ir
        mixed = torch.cat([
            self._from_heads(mixed_rgb, height, width),
            self._from_heads(mixed_ir, height, width),
        ], dim=0)
        delta_rgb, delta_ir = self.out_proj(mixed).chunk(2, dim=0)
        residual = self.get_residual_scale().to(
            device=delta_rgb.device, dtype=delta_rgb.dtype)
        delta_rgb = delta_rgb * residual[0]
        delta_ir = delta_ir * residual[1]

        # Retained with gradients until the training loop consumes the
        # reliability calibration loss.  Evaluation simply overwrites it on
        # the next batch, so no tensor is serialized in checkpoints.
        self.last_reliability_predictions = {
            'q_rgb': q_rgb_global,
            'q_ir': q_ir_global,
            'correspondence': correspondence,
            'q_rgb_local_mean': local_rgb.mean(dim=(2, 3)),
            'q_ir_local_mean': local_ir.mean(dim=(2, 3)),
            'effective_rgb': effective_rgb,
            'effective_ir': effective_ir,
            'r_ir_to_rgb': r_ir_to_rgb,
            'r_rgb_to_ir': r_rgb_to_ir,
            'null_rgb': attention_rgb[..., -1].mean(dim=(1, 2)),
            'null_ir': attention_ir[..., -1].mean(dim=(1, 2)),
        }
        if getattr(self, 'export_reliability_maps', False):
            self.last_reliability_maps = {
                'q_rgb': effective_rgb.detach().float().cpu(),
                'q_ir': effective_ir.detach().float().cpu(),
                'correspondence': correspondence.detach().float().cpu(),
                'r_ir_to_rgb': r_ir_to_rgb.detach().float().cpu(),
                'r_rgb_to_ir': r_rgb_to_ir.detach().float().cpu(),
            }
        if getattr(self, 'export_cross_modal_response_maps', False):
            self.last_cross_modal_response_maps = {
                'rgb_update': delta_rgb.detach().float().cpu(),
                'ir_update': delta_ir.detach().float().cpu(),
            }
        self._record_stats(
            attention_rgb, attention_ir, r_ir_to_rgb, r_rgb_to_ir, mix)
        return delta_rgb, delta_ir


class HAFFormerAverage(nn.Module):
    """Parameter-free dual-modal mean fusion used as the strict baseline."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return 0.5 * (x[0] + x[1])


class HAFFormerOriginalGFB(nn.Module):
    """Original spatial cross-attention followed by the original GFB."""
    def __init__(self, dim):
        super().__init__()
        bias = False
        num_heads = 8
        # Construct the shared fusion component first. With layer-wise seeded
        # initialization this makes its initial state identical to the
        # frequency-attention counterpart in strict ablations.
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.mhca_rgb = OriginalCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = OriginalCrossAttention_S(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea, ir_fea = x
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea
        return self.fusion(out_rgb, out_ir)


class HAFFormerFrequencyGFB(nn.Module):
    """Corrected entropy-routed frequency attention followed by original GFB."""
    def __init__(self, dim):
        super().__init__()
        bias = False
        num_heads = 8
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.mhca_rgb = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)
        self.mhca_ir = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea, ir_fea = x
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea
        return self.fusion(out_rgb, out_ir)


class HAFFormerFrequencyMaskGFB(nn.Module):
    """
    Corrected frequency cross-attention + foreground mask + original GFB.

    The GFB predicts a per-channel RGB-vs-IR logit. The one-channel foreground
    mask contributes a bounded spatial log-odds correction: high mask response
    shifts the gate toward IR and low response shifts it toward RGB. A neutral
    mask value of 0.5 leaves the original GFB unchanged, which gives this
    combined model a stable pretrained-training start.
    """
    def __init__(self, dim, mask_logit_gain=1.0):
        super().__init__()
        bias = False
        num_heads = 8
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.mhca_rgb = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)
        self.mhca_ir = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)

    def forward(self, x):
        rgb_fea, ir_fea = x
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        fg_mask = self.foreground_mask(out_rgb, out_ir)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * fg_mask - 1.0)
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if getattr(self, 'track_fusion_stats', False):
            mask_stats = fg_mask.detach().float()
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


class HAFFormerMaskGFBOnly(nn.Module):
    """Attention-free RGB/IR ablation with foreground Mask + original GFB.

    The two aligned modal features are passed directly to exactly the same
    foreground-mask-guided GFB rule used after RGCA.  There is deliberately no
    cross-attention, reliability gate, local-value branch, frequency operator,
    or modal residual enhancement in this block.  Keeping this as a separately
    named class makes the ablation checkpoint independent from the historical
    mutable ``HAFFormer`` implementation.
    """

    def __init__(self, dim, mask_logit_gain=1.0):
        super().__init__()
        bias = False
        # Match the post-attention component construction order in the RGCA
        # block so layer-isolated initialization remains comparable.
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)

    def forward(self, x):
        rgb_fea, ir_fea = x
        fg_mask = self.foreground_mask(rgb_fea, ir_fea)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gfb_logit = self.fusion.compute_logits(rgb_fea, ir_fea)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * fg_mask - 1.0)
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * rgb_fea + (1.0 - rgb_weight) * ir_fea

        if getattr(self, 'track_fusion_stats', False):
            mask_stats = fg_mask.detach().float()
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


class HAFFormerGFBOnly(nn.Module):
    """Strict GFB-only RGB/IR fusion ablation.

    The aligned modal features are passed directly to the original LCAFNet
    gated fusion block.  No cross-attention, foreground mask, reliability
    gate, local-value branch, frequency operator, or modal enhancement is
    instantiated.  A dedicated class keeps GFB-only checkpoints unambiguous
    and prevents later edits to the historical ``HAFFormer`` class from
    changing this ablation.
    """

    def __init__(self, dim):
        super().__init__()
        self.fusion = GatedModalFusion(dim, bias=False)

    def forward(self, x):
        rgb_fea, ir_fea = x
        return self.fusion(rgb_fea, ir_fea)


class HAFFormerLCAFMaskGFB(nn.Module):
    """Original LCAFNet cross-attention + foreground mask-guided GFB.

    This is the strict attention ablation of :class:`HAFFormerRGCAMaskGFB`.
    Only the attention path changes: two independent original LCAFNet
    ``OriginalCrossAttention_S`` modules replace RGCA.  Consequently there is
    no reliability gate, shared reduced QKV projection, local/global mixer,
    attention prior, or RGCA residual-scale parameter.  The foreground mask,
    GFB logits and mask log-odds correction are byte-for-byte the same
    post-attention fusion rule used by the RGCA block.
    """

    def __init__(self, dim, mask_logit_gain=1.0):
        super().__init__()
        bias = False
        num_heads = 8
        # Keep the construction order aligned with HAFFormerRGCAMaskGFB so
        # layer-isolated seeded initialisation gives Mask/GFB a fair ablation.
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.mhca_rgb = OriginalCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = OriginalCrossAttention_S(dim, num_heads, bias)
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)

    def forward(self, x):
        rgb_fea, ir_fea = x
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        fg_mask = self.foreground_mask(out_rgb, out_ir)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * fg_mask - 1.0)
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if getattr(self, 'track_fusion_stats', False):
            mask_stats = fg_mask.detach().float()
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


class HAFFormerRGCAMaskGFB(nn.Module):
    """
    Frequency-free RGCA enhancement + foreground mask-guided GFB.

    RGCA (Reliability-Guided Cross-modal Attention) first performs shared,
    bidirectional RGB-IR interaction and returns a bounded residual for each
    modality. The existing foreground-aware mask and GFB are intentionally
    preserved so replacing the attention mechanism does not silently change
    the post-attention fusion rule.
    """

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 channel_prior_init=1.0, prior_mode='none',
                 becp_max_margin=0.5, dacp_rho_max=0.1,
                 dacp_rho_init=0.01,
                 residual_mode='legacy_channel',
                 use_reliability_gate=True):
        super().__init__()
        bias = False
        # Construct the shared post-attention components first. Layer-isolated
        # strict ablations can then initialize them consistently across blocks.
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.cross_modal_attention = ReliabilityGuidedBidirectionalCrossAttention(
            dim, reduction=attention_reduction, num_heads=4, bias=bias,
            local_kernel_size=5, residual_init=0.1,
            channel_prior_init=channel_prior_init, prior_mode=prior_mode,
            becp_max_margin=becp_max_margin,
            dacp_rho_max=dacp_rho_max,
            dacp_rho_init=dacp_rho_init,
            residual_mode=residual_mode,
            use_reliability_gate=use_reliability_gate)
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)

    def forward(self, x):
        rgb_fea, ir_fea = x
        delta_rgb, delta_ir = self.cross_modal_attention(rgb_fea, ir_fea)
        out_rgb = rgb_fea + delta_rgb
        out_ir = ir_fea + delta_ir

        fg_mask = self.foreground_mask(out_rgb, out_ir)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * fg_mask - 1.0)
        rgb_weight = torch.sigmoid(gfb_logit - mask_correction)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if getattr(self, 'track_fusion_stats', False):
            mask_stats = fg_mask.detach().float()
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


class HAFFormerRCRGCAV2MaskGFB(nn.Module):
    """RC-RGCA v2 enhancement with reliability-conditioned GFB reuse.

    The same source-quality estimates that control K/V and null attention also
    bias the final RGB/IR fusion logit.  This avoids training a second,
    potentially contradictory definition of modality reliability after the
    attention block.
    """

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 reliability_logit_gain=0.5):
        super().__init__()
        bias = False
        self.fusion = GatedModalFusion(dim, bias=bias)
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.cross_modal_attention = (
            ReliabilityConditionedBidirectionalCrossAttentionV2(
                dim, reduction=attention_reduction, num_heads=4, bias=bias,
                local_kernel_size=5, residual_init=0.1))
        self.register_buffer(
            'mask_logit_gain', torch.tensor(float(mask_logit_gain)),
            persistent=True)
        self.register_buffer(
            'reliability_logit_gain',
            torch.tensor(float(reliability_logit_gain)), persistent=True)

    @staticmethod
    def _safe_logit(probability):
        probability = probability.clamp(1e-4, 1.0 - 1e-4)
        return probability.log() - (1.0 - probability).log()

    def forward(self, x):
        rgb_fea, ir_fea = x
        delta_rgb, delta_ir = self.cross_modal_attention(rgb_fea, ir_fea)
        out_rgb = rgb_fea + delta_rgb
        out_ir = ir_fea + delta_ir

        fg_mask = self.foreground_mask(out_rgb, out_ir)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gfb_logit = self.fusion.compute_logits(out_rgb, out_ir)
        mask_correction = self.mask_logit_gain.to(
            dtype=gfb_logit.dtype) * (2.0 * fg_mask - 1.0)

        reliability = self.cross_modal_attention.last_reliability_predictions
        effective_rgb = reliability['effective_rgb'].to(dtype=gfb_logit.dtype)
        effective_ir = reliability['effective_ir'].to(dtype=gfb_logit.dtype)
        reliability_bias = self.reliability_logit_gain.to(
            dtype=gfb_logit.dtype) * (
                self._safe_logit(effective_rgb)
                - self._safe_logit(effective_ir))
        rgb_weight = torch.sigmoid(
            gfb_logit - mask_correction + reliability_bias)
        output = rgb_weight * out_rgb + (1.0 - rgb_weight) * out_ir

        if getattr(self, 'track_fusion_stats', False):
            mask_stats = fg_mask.detach().float()
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
        if not self.training:
            # Reliability maps needed for visualization are stored separately
            # as detached CPU tensors.  Do not let validation activations ride
            # along in pickled EMA checkpoints.
            self.cross_modal_attention.last_reliability_predictions = None
        return output


class HAFFormerRGCANoReliabilityMaskGFB(HAFFormerRGCAMaskGFB):
    """RGCA cross-modal interaction with only its reliability gate removed.

    Shared bidirectional cross-covariance attention, the local complementary
    value branch, global/local mixing, residual projection, foreground Mask,
    and mask-guided GFB are identical to :class:`HAFFormerRGCAMaskGFB`.
    The learned HxW reliability networks are not instantiated and both
    transfer multipliers are fixed to the identity value one.
    """

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 channel_prior_init=1.0, prior_mode='none',
                 becp_max_margin=0.5, dacp_rho_max=0.1,
                 dacp_rho_init=0.01,
                 residual_mode='legacy_channel'):
        super().__init__(
            dim, mask_logit_gain=mask_logit_gain,
            attention_reduction=attention_reduction,
            channel_prior_init=channel_prior_init,
            prior_mode=prior_mode,
            becp_max_margin=becp_max_margin,
            dacp_rho_max=dacp_rho_max,
            dacp_rho_init=dacp_rho_init,
            residual_mode=residual_mode,
            use_reliability_gate=False)


class HAFFormerBECPMaskGFB(HAFFormerRGCAMaskGFB):
    """RGCA + foreground mask + GFB with a Beta-evidential channel prior."""

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 becp_max_margin=0.5):
        super().__init__(
            dim, mask_logit_gain=mask_logit_gain,
            attention_reduction=attention_reduction,
            channel_prior_init=1.0, prior_mode='becp',
            becp_max_margin=becp_max_margin,
            residual_mode='bounded_scalar')


class HAFFormerDACPMaskGFB(HAFFormerRGCAMaskGFB):
    """RGCA + mask-guided GFB with a detection-aligned contrastive prior."""

    def __init__(self, dim, mask_logit_gain=1.0, attention_reduction=2,
                 dacp_rho_max=0.1, dacp_rho_init=0.01):
        super().__init__(
            dim, mask_logit_gain=mask_logit_gain,
            attention_reduction=attention_reduction,
            channel_prior_init=1.0, prior_mode='dacp',
            dacp_rho_max=dacp_rho_max,
            dacp_rho_init=dacp_rho_init,
            residual_mode='bounded_scalar')


class HAFFormerOriginalMask(nn.Module):
    """Original spatial cross-attention followed by foreground-mask fusion."""
    def __init__(self, dim):
        super().__init__()
        bias = False
        num_heads = 8
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.mhca_rgb = OriginalCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = OriginalCrossAttention_S(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea, ir_fea = x
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea
        fg_mask = self.foreground_mask(out_rgb, out_ir)
        fg_mask = torch.nan_to_num(
            fg_mask, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return out_rgb + fg_mask * (out_ir - out_rgb)


class HAFFormerOriginalCAConcat(nn.Module):
    """
    Ablation block: original LCAFNet cross-attention + feature concat.

    This block keeps the original spatial-domain modal-guided cross-attention
    only. It removes dynamic frequency filtering, GFB, and foreground masking,
    then concatenates the enhanced VIS/IR features for the detection head.
    """
    def __init__(self, dim):
        super(HAFFormerOriginalCAConcat, self).__init__()
        bias = False
        num_heads = 8

        self.mhca_rgb = OriginalCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = OriginalCrossAttention_S(dim, num_heads, bias)
        self.concat = Concat(dimension=1)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_fea_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        return self.concat([out_fea_rgb, out_fea_ir])


class HAFFormerFrequencyCAConcat(nn.Module):
    """
    Ablation block: dynamic frequency cross-attention + feature concat.

    This block keeps the proposed frequency-enhanced cross-attention only. It
    removes GFB and foreground masking, then concatenates the enhanced VIS/IR
    features for the detection head.
    """
    def __init__(self, dim):
        super(HAFFormerFrequencyCAConcat, self).__init__()
        bias = False
        num_heads = 8

        self.mhca_rgb = FrequencyCrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = FrequencyCrossAttention_S(dim, num_heads, bias)
        self.concat = Concat(dimension=1)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_fea_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        return self.concat([out_fea_rgb, out_fea_ir])


class HAFFormer(nn.Module):
    """
    GFB ablation: exp_mask frequency attention + foreground-mask gate.

    The learned one-channel foreground mask directly routes between enhanced
    RGB and IR features. The original GFB projection and depthwise gate are
    deliberately absent, and no auxiliary mask loss or attention prior is used.
    """
    def __init__(self, dim):
        super(HAFFormer, self).__init__()
        bias = False
        num_heads = 8

        # Keep shared fusion initialization independent of the attention
        # implementation used by a paired strict-ablation variant.
        self.foreground_mask = ForegroundSparseFusionMask(
            dim, bias=bias, output_bias=0.0)
        self.mhca_rgb = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)
        self.mhca_ir = ExpMaskFrequencyCrossAttention(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_fea_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        # Original LCAFNet checkpoints pickle the historical HAFFormer object
        # directly.  That layout contains concat/conv/dwconv (the original
        # GFB), but predates foreground_mask.  Preserve its exact inference
        # equation instead of requiring the old weight to acquire freshly
        # initialized modules at load time.
        if (not hasattr(self, 'foreground_mask')
                and hasattr(self, 'concat')
                and hasattr(self, 'conv')
                and hasattr(self, 'dwconv')):
            fused_seed = self.conv(self.concat([out_fea_rgb, out_fea_ir]))
            rgb_weight = self.dwconv(fused_seed).sigmoid()
            return (rgb_weight * out_fea_rgb
                    + (1.0 - rgb_weight) * out_fea_ir)

        fg_mask = self.foreground_mask(out_fea_rgb, out_fea_ir)
        fg_mask = torch.nan_to_num(fg_mask, nan=0.5, posinf=1.0, neginf=0.0)
        fg_mask = torch.clamp(fg_mask, 0.0, 1.0)
        return out_fea_rgb + fg_mask * (out_fea_ir - out_fea_rgb)
