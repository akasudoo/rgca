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

    return new_weight


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


class FD2HighFrequencyUnit(nn.Module):
    """
    FD2-Net inspired high-frequency unit for spatial features.

    It uses fixed DCT basis maps to emphasize several high-frequency
    components, then learns a lightweight spatial attention mask.
    """
    def __init__(self, channel, num_groups=4, bias=False):
        super(FD2HighFrequencyUnit, self).__init__()
        self.num_groups = max(1, min(num_groups, channel))
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=bias),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True)
        )

    @staticmethod
    def _dct_basis(height, width, u_ratio, v_ratio, device, dtype):
        u = max(1, min(height - 1, int(round((height - 1) * u_ratio))))
        v = max(1, min(width - 1, int(round((width - 1) * v_ratio))))
        y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
        x = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
        basis = torch.cos(math.pi * u / height * (y + 0.5)) * torch.cos(math.pi * v / width * (x + 0.5))
        return basis.to(dtype=dtype).view(1, 1, height, width)

    def forward(self, x):
        b, c, h, w = x.shape
        chunks = torch.chunk(x, self.num_groups, dim=1)
        # Several representative high-frequency DCT components.
        freq_pairs = ((0.50, 0.50), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
        high_chunks = []
        for i, chunk in enumerate(chunks):
            u_ratio, v_ratio = freq_pairs[i % len(freq_pairs)]
            basis = self._dct_basis(h, w, u_ratio, v_ratio, x.device, x.dtype)
            high_chunks.append(chunk * basis)
        high = torch.cat(high_chunks, dim=1)

        avg_map = high.mean(dim=1, keepdim=True)
        max_map = high.amax(dim=1, keepdim=True)
        mask = self.spatial_attention(torch.cat([avg_map, max_map], dim=1))
        return self.project(high * mask)


class FD2LowFrequencyUnit(nn.Module):
    """
    FD2-Net inspired low-frequency context unit.

    Parallel depthwise dilated convolutions model low-frequency structure
    and multi-scale context, followed by channel mixing.
    """
    def __init__(self, channel, bias=False):
        super(FD2LowFrequencyUnit, self).__init__()
        branches = []
        for kernel_size, dilation in ((7, 1), (3, 1), (3, 2), (3, 3)):
            padding = dilation * (kernel_size - 1) // 2
            branches.append(
                nn.Conv2d(
                    channel, channel, kernel_size=kernel_size, stride=1,
                    padding=padding, dilation=dilation, groups=channel, bias=bias
                )
            )
        self.branches = nn.ModuleList(branches)
        hidden = max(channel // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        low = sum(branch(x) for branch in self.branches) / len(self.branches)
        low = low * self.channel_gate(low)
        return self.project(low)


class FD2VBranchHighLowRefine(nn.Module):
    """
    Lightweight high/low-frequency refinement for the filtered V branch.

    The residual scale is initialized to 0, so training starts from the
    original frequency-transformer behavior and learns the FD2-style
    refinement gradually.
    """
    def __init__(self, channel, high_ratio=0.5, bias=False):
        super(FD2VBranchHighLowRefine, self).__init__()
        high_channel = max(1, int(channel * high_ratio))
        low_channel = max(1, channel - high_channel)
        self.high_in = nn.Conv2d(channel, high_channel, kernel_size=1, bias=bias)
        self.low_in = nn.Conv2d(channel, low_channel, kernel_size=1, bias=bias)
        self.high_unit = FD2HighFrequencyUnit(high_channel, bias=bias)
        self.low_unit = FD2LowFrequencyUnit(low_channel, bias=bias)
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        high = self.high_unit(self.high_in(x))
        low = self.low_unit(self.low_in(x))
        refined = self.fuse(torch.cat([high, low], dim=1))
        return x + self.gamma.to(dtype=x.dtype) * refined


class FD2CrossModalHighLowEnhance(nn.Module):
    """
    FD2-Net inspired cross-modal high/low-frequency complementary enhancement.

    Visible high-frequency details are used to enhance the infrared high-frequency
    branch, while infrared low-frequency structure is used to enhance the visible
    low-frequency branch. The enhanced RGB/IR features are then sent to the
    original GFB fusion path.
    """
    def __init__(self, channel, high_ratio=0.5, bias=False):
        super(FD2CrossModalHighLowEnhance, self).__init__()
        if channel < 2:
            raise ValueError('FD2CrossModalHighLowEnhance requires channel >= 2')

        high_channel = max(1, int(channel * high_ratio))
        high_channel = min(channel - 1, high_channel)
        low_channel = channel - high_channel

        def high_branch():
            return nn.Sequential(
                nn.Conv2d(channel, high_channel, kernel_size=1, stride=1, padding=0, bias=bias),
                FD2HighFrequencyUnit(high_channel, bias=bias)
            )

        def low_branch():
            return nn.Sequential(
                nn.Conv2d(channel, low_channel, kernel_size=1, stride=1, padding=0, bias=bias),
                FD2LowFrequencyUnit(low_channel, bias=bias)
            )

        self.rgb_high = high_branch()
        self.rgb_low = low_branch()
        self.ir_high = high_branch()
        self.ir_low = low_branch()

        self.ir_high_gate = nn.Sequential(
            nn.Conv2d(high_channel * 2, high_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.rgb_low_gate = nn.Sequential(
            nn.Conv2d(low_channel * 2, low_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

        def separable_fuse():
            return nn.Sequential(
                nn.Conv2d(
                    channel, channel, kernel_size=3, stride=1, padding=1,
                    groups=channel, bias=bias
                ),
                nn.BatchNorm2d(channel),
                nn.SiLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(channel),
                nn.SiLU(inplace=True)
            )

        self.rgb_fuse = separable_fuse()
        self.ir_fuse = separable_fuse()

        self.rgb_gamma = nn.Parameter(torch.ones(1) * 1e-3)
        self.ir_gamma = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, rgb_fea, ir_fea):
        rgb_high = self.rgb_high(rgb_fea)
        rgb_low = self.rgb_low(rgb_fea)
        ir_high = self.ir_high(ir_fea)
        ir_low = self.ir_low(ir_fea)

        # CSS: visible high-frequency details enhance IR high-frequency features.
        ir_high_gate = self.ir_high_gate(torch.cat([ir_high, rgb_high], dim=1))
        ir_high = ir_high + ir_high_gate * rgb_high

        # CSS: infrared low-frequency structure enhances visible low-frequency features.
        rgb_low_gate = self.rgb_low_gate(torch.cat([rgb_low, ir_low], dim=1))
        rgb_low = rgb_low + rgb_low_gate * ir_low

        rgb_refined = self.rgb_fuse(torch.cat([rgb_high, rgb_low], dim=1))
        ir_refined = self.ir_fuse(torch.cat([ir_high, ir_low], dim=1))

        rgb_out = rgb_fea + self.rgb_gamma.to(dtype=rgb_fea.dtype) * rgb_refined
        ir_out = ir_fea + self.ir_gamma.to(dtype=ir_fea.dtype) * ir_refined

        return rgb_out, ir_out


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
    def __init__(self, dim, num_heads, bias, kernel_size=80, num_filters=4):
        super(CrossAttention_S, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_filters = num_filters

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

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
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Dynamic routing MLP: [B, C] -> [B, num_filters, C]
        hidden = max(dim // 4, 16)
        self.reweight_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_filters * dim)
        )

        # Learnable complex frequency bases.
        # Last dimension stores real and imaginary parts.
        h_base = kernel_size
        w_freq_base = kernel_size // 2 + 1
        init_weight = torch.randn(h_base, w_freq_base, num_filters, 2) * 0.02
        init_weight[..., 0] += 1.0
        self.complex_weights = nn.Parameter(init_weight)

        # Residual frequency filtering strength.
        # Initialized to 0 for stable training; it will be learned automatically.
        self.beta = nn.Parameter(torch.zeros(1))

        # Response gate for the difference between filtered v and original v.
        self.freq_response_gate = FrequencyResponseGate(dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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

        # Channel response from q-k attention: [B, heads, head_dim] -> [B, C, 1, 1]
        channel_score = attn.mean(dim=-1).reshape(b, self.dim, 1, 1).sigmoid()

        # Spatial-channel map from q and k features: [B, C, H, W]
        spatial_score = self.route_proj(q_feat + k_feat)

        route_att = spatial_score * channel_score
        route_att = torch.clamp(route_att, 0.0, 1.0)

        return route_att

    def frequency_filter_v(self, v_feat, route_att):
        """
        Apply FFT -> dynamic frequency filtering -> IFFT on the V branch.

        Args:
            v_feat: [B, C, H, W]
            route_att: [B, C, H, W]

        Returns:
            v_out: [B, C, H, W]
        """
        b, c, h, w = v_feat.shape
        origin_dtype = v_feat.dtype

        # cuFFT does not support FP16 FFT for non-power-of-two feature sizes
        # such as [136, 168]. Therefore, run FFT/IFFT in FP32 and cast back.
        v_fft = v_feat.float()
        route_att_fp32 = route_att.float()

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

        attn = (q_re @ k_re.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # q/k attention map -> dynamic frequency filter routing.
        route_att = self.build_route_attention(q, k, attn, h, w)

        # FFT filter is applied to the V branch.
        v_filtered = self.frequency_filter_v(v, route_att)

        v_re = rearrange(v_filtered, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        out = (attn @ v_re)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out



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

        self.beta_rgb = nn.Parameter(torch.zeros(1))
        self.beta_ir = nn.Parameter(torch.zeros(1))

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

    def build_route_attention(self, q_feat, k_feat, attn):
        b, c, h, w = q_feat.shape

        channel_score = attn.mean(dim=-1).reshape(b, self.dim, 1, 1).sigmoid()
        spatial_score = self.route_proj(q_feat + k_feat)

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
        ir_route_att = self.build_route_attention(rgb_q, ir_k, attn_ir)
        rgb_route_att = self.build_route_attention(ir_q, rgb_k, attn_rgb)

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



class HAFFormerGFBArchive(nn.Module):
    # Archived original HAFFormer implementation with gated fusion.
    def __init__(self, dim):
        super(HAFFormerGFBArchive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        # Concat
        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
                                  nn.GELU())
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        # Cross Attention
        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        # Gated Fusion
        fea_cat = self.concat([out_fea_rgb, out_fea_ir])
        fea_conv = self.conv(fea_cat)
        w = self.dwconv(fea_conv).sigmoid()
        new_fea = w * out_fea_rgb + (1 - w) * out_fea_ir

        return new_fea


class HAFFormerGNN3Node3IterArchive(nn.Module):
    # Archived heavy GNN version: rd_sc=1, dila=[1, 2, 3], n_iter=3.
    def __init__(self, dim):
        super(HAFFormerGNN3Node3IterArchive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        self.gnn_fusion = GraphReasoning(chnn_in=dim, rd_sc=1, dila=[1, 2, 3], n_iter=3)
        self.concat = Concat(dimension=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )

    def get_bayesian_loss(self):
        return self.gnn_fusion.get_bayesian_loss()

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        gnn_rgb, gnn_ir = self.gnn_fusion((out_fea_rgb, out_fea_ir, out_fea_rgb, out_fea_ir), node=False)
        gnn_rgb = torch.stack(gnn_rgb, dim=0).mean(dim=0)
        gnn_ir = torch.stack(gnn_ir, dim=0).mean(dim=0)
        new_fea = self.fuse(self.concat([gnn_rgb, gnn_ir]))

        return new_fea


class HAFFormerCompactResidualArchive(nn.Module):
    # Archived compact residual version: GFB-style base fusion plus rd_sc=4 GNN residual.
    def __init__(self, dim):
        super(HAFFormerCompactResidualArchive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim
        self.gnn_rd_sc = 4
        self.gnn_channels = dim // self.gnn_rd_sc

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        self.concat = Concat(dimension=1)
        self.base_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.base_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.gnn_fusion = GraphReasoning(chnn_in=dim, rd_sc=self.gnn_rd_sc, dila=[1, 2, 3], n_iter=1)
        self.gnn_fuse = nn.Sequential(
            nn.Conv2d(2 * self.gnn_channels, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.gnn_scale = nn.Parameter(torch.tensor(-2.0))

    def get_bayesian_loss(self):
        return self.gnn_fusion.get_bayesian_loss()

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        base_fea = self.base_conv(self.concat([out_fea_rgb, out_fea_ir]))
        base_weight = self.base_dwconv(base_fea).sigmoid()
        base_fea = base_weight * out_fea_rgb + (1 - base_weight) * out_fea_ir

        gnn_rgb, gnn_ir = self.gnn_fusion((out_fea_rgb, out_fea_ir, out_fea_rgb, out_fea_ir), node=False)
        gnn_rgb = torch.stack(gnn_rgb, dim=0).mean(dim=0)
        gnn_ir = torch.stack(gnn_ir, dim=0).mean(dim=0)
        gnn_fea = self.gnn_fuse(self.concat([gnn_rgb, gnn_ir]))

        return base_fea + torch.sigmoid(self.gnn_scale) * gnn_fea


class HAFFormerPlanBRd2Archive(nn.Module):
    # Archived plan-B version: rd_sc=2 GNN main fusion with mean node aggregation.
    def __init__(self, dim):
        super(HAFFormerPlanBRd2Archive, self).__init__()
        bias = False
        num_heads = 8
        self.dim = dim
        self.gnn_rd_sc = 2
        self.gnn_channels = dim // self.gnn_rd_sc

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        self.gnn_fusion = GraphReasoning(chnn_in=dim, rd_sc=self.gnn_rd_sc, dila=[1, 2, 3], n_iter=1)
        self.concat = Concat(dimension=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * self.gnn_channels, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )

    def get_bayesian_loss(self):
        return self.gnn_fusion.get_bayesian_loss()

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea = self.mhca_rgb([rgb_fea, ir_fea])
        out_fea_rgb = out_fea + rgb_fea

        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea

        gnn_rgb, gnn_ir = self.gnn_fusion((out_fea_rgb, out_fea_ir, out_fea_rgb, out_fea_ir), node=False)
        gnn_rgb = torch.stack(gnn_rgb, dim=0).mean(dim=0)
        gnn_ir = torch.stack(gnn_ir, dim=0).mean(dim=0)
        new_fea = self.fuse(self.concat([gnn_rgb, gnn_ir]))

        return new_fea


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

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

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

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

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


class HAFFormer(nn.Module):
    """
    Active frequency-transformer + original GFB fusion block.

    It uses dynamic frequency-filtered cross attention for RGB/IR feature
    interaction, then keeps the original LCAFNet GFB weighted fusion.
    """
    def __init__(self, dim):
        super(HAFFormer, self).__init__()
        bias = False
        num_heads = 8

        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        out_fea_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_fea_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        fea_cat = self.concat([out_fea_rgb, out_fea_ir])
        fea_conv = self.conv(fea_cat)
        w = self.dwconv(fea_conv).sigmoid()
        new_fea = w * out_fea_rgb + (1.0 - w) * out_fea_ir

        return new_fea
