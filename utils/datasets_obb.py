"""Paired RGB/IR DroneVehicle loader for native DOTA oriented boxes."""

import glob
import logging
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.datasets import InfiniteDataLoader, augment_hsv, img_formats, letterbox
from utils.obb_utils import ANGLE_BINS, poly2rbox
from utils.torch_utils import torch_distributed_zero_first


LOGGER = logging.getLogger(__name__)
# Match the per-class column order in LCAFNet paper Table 1.
CLASS_NAMES = ('car', 'truck', 'freight_car', 'bus', 'van')


def _image_files(source):
    files = []
    for item in source if isinstance(source, list) else [source]:
        path = Path(item)
        if path.is_dir():
            files.extend(glob.glob(str(path / '**' / '*.*'), recursive=True))
        elif path.is_file():
            parent = str(path.parent) + os.sep
            files.extend(x.replace('./', parent) if x.startswith('./') else x
                         for x in path.read_text().strip().splitlines())
        else:
            raise FileNotFoundError(path)
    return sorted(x for x in files if x.rsplit('.', 1)[-1].lower() in img_formats)


def _dota_label_path(image_path):
    image_path = Path(image_path)
    return image_path.parent.parent / 'dota_labels' / (image_path.stem + '.txt')


def _read_dota(path, single_cls=False):
    labels = []
    if path.is_file():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 10:
                raise ValueError(
                    f'{path}:{line_number}: expected DOTA 10 fields, got {len(fields)}')
            name, difficulty = fields[8], int(float(fields[9]))
            if name not in CLASS_NAMES:
                raise ValueError(f'{path}:{line_number}: unknown class {name!r}')
            if difficulty >= 2:
                continue
            polygon = np.asarray(fields[:8], dtype=np.float32)
            if not np.isfinite(polygon).all():
                raise ValueError(f'{path}:{line_number}: non-finite polygon')
            class_id = 0 if single_cls else CLASS_NAMES.index(name)
            labels.append(np.concatenate(([class_id], polygon)))
    if not labels:
        return np.zeros((0, 9), dtype=np.float32)
    return np.unique(np.asarray(labels, dtype=np.float32), axis=0)


def _transform_polygons(labels, matrix, perspective=False):
    if not len(labels):
        return labels
    points = np.ones((len(labels) * 4, 3), dtype=np.float32)
    points[:, :2] = labels[:, 1:].reshape(-1, 2)
    points = points @ matrix.T
    if perspective:
        points = points[:, :2] / points[:, 2:3]
    else:
        points = points[:, :2]
    output = labels.copy()
    output[:, 1:] = points.reshape(-1, 8)
    return output


def _filter_polygons(labels, width, height, min_edge=2.0):
    if not len(labels):
        return labels
    boxes = poly2rbox(labels[:, 1:])
    keep = ((boxes[:, 0] >= 0) & (boxes[:, 0] < width) &
            (boxes[:, 1] >= 0) & (boxes[:, 1] < height) &
            (boxes[:, 2] >= min_edge) & (boxes[:, 3] >= min_edge))
    return labels[keep]


def paired_random_perspective(rgb, ir, labels, degrees=0.0, translate=0.1,
                              scale=0.5, shear=0.0, perspective=0.0,
                              border=(0, 0)):
    """Apply one geometric transform to both modalities and all four points."""
    height = rgb.shape[0] + border[0] * 2
    width = rgb.shape[1] + border[1] * 2
    center = np.eye(3)
    center[0, 2], center[1, 2] = -rgb.shape[1] / 2, -rgb.shape[0] / 2
    projective = np.eye(3)
    projective[2, 0] = random.uniform(-perspective, perspective)
    projective[2, 1] = random.uniform(-perspective, perspective)
    rotation = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_factor)
    skew = np.eye(3)
    skew[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    skew[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    translation = np.eye(3)
    translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    matrix = translation @ skew @ rotation @ projective @ center
    if perspective:
        rgb = cv2.warpPerspective(rgb, matrix, (width, height),
                                  borderValue=(114, 114, 114))
        ir = cv2.warpPerspective(ir, matrix, (width, height),
                                 borderValue=(114, 114, 114))
    else:
        rgb = cv2.warpAffine(rgb, matrix[:2], (width, height),
                             borderValue=(114, 114, 114))
        ir = cv2.warpAffine(ir, matrix[:2], (width, height),
                            borderValue=(114, 114, 114))
    labels = _transform_polygons(labels, matrix, bool(perspective))
    return rgb, ir, _filter_polygons(labels, width, height)


class LoadPairedDOTAOBB(Dataset):
    """Use visible DOTA annotations as the shared target for an aligned pair."""

    def __init__(self, path_rgb, path_ir, img_size=640, batch_size=16,
                 augment=False, hyp=None, rect=False, cache_images=False,
                 single_cls=False, stride=32, pad=0.0, prefix=''):
        del batch_size, rect, cache_images, stride, pad
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.hyp = hyp or {}
        self.mosaic = self.augment
        self.mosaic_border = (-self.img_size // 2, -self.img_size // 2)
        self.img_files_rgb = _image_files(path_rgb)
        self.img_files_ir = _image_files(path_ir)
        rgb_stems = [Path(x).stem for x in self.img_files_rgb]
        ir_stems = [Path(x).stem for x in self.img_files_ir]
        if not rgb_stems or rgb_stems != ir_stems:
            raise ValueError(
                f'{prefix}RGB/IR files are empty or not paired in identical order')
        self.label_files = [_dota_label_path(x) for x in self.img_files_rgb]
        self.labels_poly = [_read_dota(x, single_cls) for x in self.label_files]
        self.indices = range(len(self.img_files_rgb))
        self.labels = []
        for labels in self.labels_poly:
            if len(labels):
                boxes = poly2rbox(labels[:, 1:])
                stats = np.concatenate((labels[:, :1], boxes[:, :4]), 1)
                stats[:, [1, 3]] /= 640.0
                stats[:, [2, 4]] /= 512.0
                self.labels.append(stats.astype(np.float32))
            else:
                self.labels.append(np.zeros((0, 5), dtype=np.float32))
        count = sum(len(x) for x in self.labels_poly)
        LOGGER.info('%sverified %d paired images and %d native DOTA OBB labels '
                    '(shared target: visible/RGB annotations)',
                    prefix, len(self.img_files_rgb), count)

    def __len__(self):
        return len(self.img_files_rgb)

    def _load_pair(self, index):
        rgb = cv2.imread(self.img_files_rgb[index])
        ir = cv2.imread(self.img_files_ir[index])
        if rgb is None or ir is None:
            raise FileNotFoundError(
                f'failed to read pair {self.img_files_rgb[index]}, '
                f'{self.img_files_ir[index]}')
        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(f'pair size mismatch at {self.img_files_rgb[index]}')
        h0, w0 = rgb.shape[:2]
        ratio = self.img_size / max(h0, w0)
        labels = self.labels_poly[index].copy()
        if ratio != 1:
            size = (int(w0 * ratio), int(h0 * ratio))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
            ir = cv2.resize(ir, size, interpolation=cv2.INTER_LINEAR)
            labels[:, 1:] *= ratio
        return rgb, ir, labels, (h0, w0), rgb.shape[:2]

    def _load_mosaic(self, index):
        size = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * size + x))
                  for x in self.mosaic_border)
        indices = [index] + random.choices(list(self.indices), k=3)
        labels_all = []
        for tile, sample_index in enumerate(indices):
            rgb, ir, labels, _, (height, width) = self._load_pair(sample_index)
            if tile == 0:
                canvas_rgb = np.full((2 * size, 2 * size, 3), 114, np.uint8)
                canvas_ir = np.full((2 * size, 2 * size, 3), 114, np.uint8)
                x1a, y1a, x2a, y2a = max(xc-width, 0), max(yc-height, 0), xc, yc
                x1b, y1b, x2b, y2b = width-(x2a-x1a), height-(y2a-y1a), width, height
            elif tile == 1:
                x1a, y1a, x2a, y2a = xc, max(yc-height, 0), min(xc+width, 2*size), yc
                x1b, y1b, x2b, y2b = 0, height-(y2a-y1a), min(width, x2a-x1a), height
            elif tile == 2:
                x1a, y1a, x2a, y2a = max(xc-width, 0), yc, xc, min(2*size, yc+height)
                x1b, y1b, x2b, y2b = width-(x2a-x1a), 0, width, min(y2a-y1a, height)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc+width, 2*size), min(2*size, yc+height)
                x1b, y1b, x2b, y2b = 0, 0, min(width, x2a-x1a), min(height, y2a-y1a)
            canvas_rgb[y1a:y2a, x1a:x2a] = rgb[y1b:y2b, x1b:x2b]
            canvas_ir[y1a:y2a, x1a:x2a] = ir[y1b:y2b, x1b:x2b]
            if len(labels):
                labels[:, [1, 3, 5, 7]] += x1a - x1b
                labels[:, [2, 4, 6, 8]] += y1a - y1b
                labels_all.append(labels)
        labels = (np.concatenate(labels_all, 0) if labels_all else
                  np.zeros((0, 9), dtype=np.float32))
        return paired_random_perspective(
            canvas_rgb, canvas_ir, labels,
            degrees=self.hyp.get('degrees', 0.0),
            translate=self.hyp.get('translate', 0.1),
            scale=self.hyp.get('scale', 0.5),
            shear=self.hyp.get('shear', 0.0),
            perspective=self.hyp.get('perspective', 0.0),
            border=self.mosaic_border)

    def __getitem__(self, index):
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp.get('mosaic', 1.0)
        if mosaic:
            rgb, ir, labels = self._load_mosaic(index)
            shapes = None
            if random.random() < hyp.get('mixup', 0.0):
                rgb2, ir2, labels2 = self._load_mosaic(random.randrange(len(self)))
                mix = np.random.beta(8.0, 8.0)
                rgb = (rgb * mix + rgb2 * (1-mix)).astype(np.uint8)
                ir = (ir * mix + ir2 * (1-mix)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            rgb, ir, labels, (h0, w0), (h, w) = self._load_pair(index)
            rgb, ratio, pad = letterbox(rgb, self.img_size, auto=False,
                                        scaleup=self.augment)
            ir, ratio_ir, pad_ir = letterbox(ir, self.img_size, auto=False,
                                             scaleup=self.augment)
            if ratio != ratio_ir or pad != pad_ir:
                raise RuntimeError('paired letterbox transforms differ')
            if len(labels):
                labels[:, [1, 3, 5, 7]] = labels[:, [1, 3, 5, 7]] * ratio[0] + pad[0]
                labels[:, [2, 4, 6, 8]] = labels[:, [2, 4, 6, 8]] * ratio[1] + pad[1]
            shapes = (h0, w0), ((h/h0, w/w0), pad)
            if self.augment:
                rgb, ir, labels = paired_random_perspective(
                    rgb, ir, labels,
                    degrees=hyp.get('degrees', 0.0),
                    translate=hyp.get('translate', 0.1),
                    scale=hyp.get('scale', 0.5),
                    shear=hyp.get('shear', 0.0),
                    perspective=hyp.get('perspective', 0.0))
        if self.augment:
            augment_hsv(rgb, hyp.get('hsv_h', 0), hyp.get('hsv_s', 0),
                        hyp.get('hsv_v', 0))
            augment_hsv(ir, hyp.get('hsv_h', 0), hyp.get('hsv_s', 0),
                        hyp.get('hsv_v', 0))
            if random.random() < hyp.get('flipud', 0.0):
                rgb, ir = np.flipud(rgb), np.flipud(ir)
                labels[:, 2::2] = rgb.shape[0] - labels[:, 2::2] - 1
            if random.random() < hyp.get('fliplr', 0.0):
                rgb, ir = np.fliplr(rgb), np.fliplr(ir)
                labels[:, 1::2] = rgb.shape[1] - labels[:, 1::2] - 1
        if len(labels):
            boxes, csl = poly2rbox(
                labels[:, 1:], bins=int(hyp.get('cls_theta', ANGLE_BINS)),
                radius=float(hyp.get('csl_radius', 2.0)), with_csl=True)
            valid = ((boxes[:, 0] >= 0) & (boxes[:, 0] < rgb.shape[1]) &
                     (boxes[:, 1] >= 0) & (boxes[:, 1] < rgb.shape[0]) &
                     (boxes[:, 2] >= 2) & (boxes[:, 3] >= 2))
            target = np.concatenate((labels[:, :1], boxes, csl), 1)[valid]
        else:
            target = np.zeros((0, 1 + 5 + ANGLE_BINS), dtype=np.float32)
        labels_out = torch.zeros((len(target), 7 + ANGLE_BINS))
        if len(target):
            labels_out[:, 1:] = torch.from_numpy(target)
        rgb = np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1))
        ir = np.ascontiguousarray(ir[:, :, ::-1].transpose(2, 0, 1))
        image = np.concatenate((rgb, ir), 0)
        return torch.from_numpy(image), labels_out, self.img_files_rgb[index], shapes

    @staticmethod
    def collate_fn(batch):
        images, labels, paths, shapes = zip(*batch)
        for index, target in enumerate(labels):
            target[:, 0] = index
        return torch.stack(images), torch.cat(labels), paths, shapes


def create_dataloader_obb_rgb_ir(path_rgb, path_ir, imgsz, batch_size, stride,
                                 opt, hyp=None, augment=False, cache=False,
                                 pad=0.0, rect=False, rank=-1, world_size=1,
                                 workers=8, image_weights=False, quad=False,
                                 prefix='', sampler=None,
                                 mutable_augmentations=False):
    del image_weights, quad
    with torch_distributed_zero_first(rank):
        dataset = LoadPairedDOTAOBB(
            path_rgb, path_ir, imgsz, batch_size, augment, hyp, rect, cache,
            getattr(opt, 'single_cls', False), stride, pad, prefix)
    batch_size = min(batch_size, len(dataset))
    worker_count = min(os.cpu_count() // world_size,
                       batch_size if batch_size > 1 else 0, workers)
    if callable(sampler):
        sampler = sampler(dataset)
    sampler = sampler or (torch.utils.data.distributed.DistributedSampler(dataset)
                          if rank != -1 else None)
    loader_class = (torch.utils.data.DataLoader if mutable_augmentations or
                    sampler is not None else InfiniteDataLoader)
    loader = loader_class(
        dataset, batch_size=batch_size, num_workers=worker_count,
        sampler=sampler, pin_memory=True, collate_fn=dataset.collate_fn)
    return loader, dataset
