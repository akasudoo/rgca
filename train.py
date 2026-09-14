"""Train RGB/IR detectors, including reliability-supervised RGCA, on M3FD.

The default entrypoint fine-tunes the converged historical C0 RGCA detector.
RS-RGCA keeps its original attention/fusion forward path and directly
supervises the existing directional reliability gates with mild asymmetric
modality degradation.  The test split remains unused during training.
"""

import argparse
import importlib.util
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

# Reduce CUDA allocation fragmentation for the four-level dual-stream model.
# This must be set before torch initializes CUDA; an explicit user setting wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import test
import test_obb
from models.experimental import attempt_load
from models.yolo_test import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir, label_names_from_image_source
from utils.datasets_obb import create_dataloader_obb_rgb_ir
from utils.general import (
    check_dataset,
    check_file,
    check_img_size,
    colorstr,
    fitness,
    get_latest_run,
    increment_path,
    init_seeds,
    keep_force_fp32_modules,
    labels_to_class_weights,
    logger,
    one_cycle,
    set_logging,
    strip_optimizer,
)
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.loss_obb import ComputeLossOBB
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import (
    ModelEMA,
    intersect_dicts,
    is_parallel,
    select_device,
    torch_distributed_zero_first,
    torch_load,
)


ROOT = Path(__file__).resolve().parent

# =============================================================================
# 训练参数集中配置区
# -----------------------------------------------------------------------------
# 日常实验只需要修改下面两个字典，然后运行：python train.py
# 命令行参数仍可临时覆盖 TRAINING_CONFIG 中的同名项目。
# 当前任务：从原版LCAFNet_LLVIP.pt迁移到C0 RGCA+Mask+GFB并在1024下
# 微调。双主干和PAN/Detect完整加载并冻结主干，训练新融合层与检测头；该
# 路径沿用本地已验证在第7轮达到0.976898 mAP50的配置。
# =============================================================================
TRAINING_CONFIG = {
    # 文件路径
    'weights': str(ROOT / 'LCAFNet_LLVIP.pt'),
    'model_cfg': str(
        ROOT / 'models/transformer/'
        'yolov5s_LCAFNet_LLVIP_C0_FullDual.yaml'),
    'data_cfg': str(
        ROOT / 'data/multispectral/LLVIP.yaml'),
    'hyp_cfg': str(ROOT / 'data/hyp.rgca_llvip_lcafnet_finetune15_1024.yaml'),

    # 最常修改的训练参数
    # 历史同路径第7轮达到峰值，训练15轮足以覆盖峰值并减少后期过拟合。
    'epochs': 15,
    # 原LLVIP实验已验证1024下物理batch=8；累积8步保持effective batch=64。
    'batch_size': 8,
    'nominal_batch_size': 64,
    # 两个值分别是训练/验证尺寸。
    'image_size': (1024, 1024),
    'device': '0',                  # '0'、'0,1' 或 'cpu'
    'workers': 8,
    'seed': 1,                      # 与旧实验opt.yaml一致
    'optimizer': 'SGD',             # Nesterov SGD，见_build_optimizer
    'lr_schedule': 'cosine',
    'lr_drop_epochs': (),
    'lr_drop_factor': 0.1,
    'best_metric': 'map50',         # LLVIP微调严格按mAP50保存best.pt
    'trainable_scope': 'fusion_head',  # 冻结双主干，训练融合/PAN/Detect
    'amp': True,
    'gradient_clip_norm': 10.0,     # 与旧实验opt.yaml一致

    # 仅供命令行复现历史 C4 配置；当前 C0 模型不会读取该值。
    'd3_kernel_size': 7,

    # 数据增强与运行设置
    # 前10轮使用温和Mosaic，最后5轮关闭Mosaic完成干净分布收敛。
    'close_mosaic': 5,
    'close_augmentations': 0,
    'cache_images': False,
    'rect': False,
    'multi_scale': False,
    'single_class': False,
    'sync_batch_norm': False,
    # Detect和anchors均从同一LLVIP权重复制，避免再次改变已训练anchors。
    'autoanchor': False,
    'validate_each_epoch': True,
    'validate_initialization': True,
    'save_each_epoch': True,
    # 以下 OBB 参数仅为兼容历史命令；FLIR HBB 路径不会使用。
    'obb_val_conf_thres': 0.001,
    'obb_val_iou_thres': 0.5,
    'obb_val_max_nms': 3000,
    'obb_val_max_det': 300,
    'obb_val_multi_label': False,
    'cudnn_benchmark': False,
    'strict_determinism': False,

    # 输出设置
    'project': str(ROOT / 'runs/train'),
    'experiment_name': 'llvip_rgca_from_lcafnet_finetune15_1024',
    'exist_ok': False,              # False：同名目录自动递增，避免覆盖旧实验
}

# LLVIP原版LCAFNet到C0 RGCA迁移：沿用本地最佳历史路径的SGD 2e-3、
# 轻量配对增强和fusion_head范围，双主干保持冻结。
HYPERPARAMETERS = {
    'lr0': 0.002,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 1.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,
    'box': 0.05,
    'cls': 0.5,
    'cls_pw': 1.0,
    'obj': 1.0,
    'obj_pw': 1.0,
    'iou_t': 0.2,
    'anchor_t': 4.0,
    'fl_gamma': 0.0,
    'hsv_h': 0.01,
    'hsv_s': 0.4,
    'hsv_v': 0.3,
    'degrees': 0.0,
    'translate': 0.05,
    'scale': 0.2,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.5,
    'mixup': 0.0,
}


DEFAULT_ATTENTION_CONTROL_LR_RATIO = 1.0
DEFAULT_TRAIN_WEIGHTS = TRAINING_CONFIG['weights']
DEFAULT_TRAIN_CFG = TRAINING_CONFIG['model_cfg']
DEFAULT_TRAIN_DATA = TRAINING_CONFIG['data_cfg']
DEFAULT_TRAIN_HYP = TRAINING_CONFIG['hyp_cfg']
DEFAULT_TRAIN_EPOCHS = TRAINING_CONFIG['epochs']
DEFAULT_TRAIN_BATCH_SIZE = TRAINING_CONFIG['batch_size']
DEFAULT_TRAIN_IMAGE_SIZE = TRAINING_CONFIG['image_size']
DEFAULT_TRAIN_OPTIMIZER = TRAINING_CONFIG['optimizer']
DEFAULT_TRAIN_SEED = TRAINING_CONFIG['seed']
DEFAULT_TRAIN_WORKERS = TRAINING_CONFIG['workers']
DEFAULT_TRAIN_CUDNN_BENCHMARK = TRAINING_CONFIG['cudnn_benchmark']
DEFAULT_TRAIN_CLOSE_MOSAIC = TRAINING_CONFIG['close_mosaic']
DEFAULT_TRAIN_CLOSE_AUGMENTATIONS = TRAINING_CONFIG['close_augmentations']
DEFAULT_TRAIN_PRETRAINED_SCOPE = 'backbone'
DEFAULT_TRAIN_EXPERIMENT_NAME = TRAINING_CONFIG['experiment_name']

# Compatibility constants retained for old read-only tests and opt.yaml files.
DEFAULT_TRAIN_AFSS = False
DEFAULT_TRAIN_PRIOR_ADAPT_EPOCHS = 0
DEFAULT_TRAIN_DIFFERENTIAL_LR = False


def is_attention_control_parameter(name):
    """Return parameters that should not receive weight decay."""
    return (
        '.channel_prior.' in name
        or '.spatial_routing_prior.' in name
        or name.endswith((
            '.spatial_route_gain', '.temperature', '.raw_temperature',
            '.raw_residual_scale', '.cross_mix_logit', '.context_scale',
            '.beta', '.attn_log_prior', '.attn_prior_gamma',
            '.private_scale_logit', '.sod_residual_scale',
        )))


def is_frequency_prior_parameter(name):
    return is_attention_control_parameter(name)


def configure_trainable_scope(model, scope):
    """Select trainable parameters without changing forward connectivity."""
    if scope == 'all':
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return []
    if scope not in {
            'fusion_only', 'fusion_head', 'pan_head', 'sod_head',
            'ccall_finetune'}:
        raise ValueError(
            "trainable_scope must be 'all', 'fusion_only', 'fusion_head', "
            "'pan_head', 'sod_head', or 'ccall_finetune'.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if scope == 'ccall_finetune':
        architecture = model.yaml.get('architecture_variant')
        if architecture != 'c0_full_dual_rgca_ccall':
            raise RuntimeError(
                'ccall_finetune requires c0_full_dual_rgca_ccall, got '
                f'{architecture!r}.')
        crossconv_layers = {2, 4, 6, 8, 12, 14, 16, 18}
        expected_layers = crossconv_layers | set(range(20, 46))
        actual_crossconv = {
            index for index, module in enumerate(model.model)
            if module.__class__.__name__ == 'C3CrossConv'}
        if actual_crossconv != crossconv_layers or len(model.model) != 46:
            raise RuntimeError(
                'C0-CCAll layout mismatch: expected C3CrossConv at '
                f'{sorted(crossconv_layers)} and 46 total layers, got '
                f'{sorted(actual_crossconv)} and {len(model.model)} layers.')
        trainable_modules = []
        for layer_index, module in enumerate(model.model):
            if layer_index not in expected_layers:
                continue
            trainable_modules.append(f'model.{layer_index}')
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if {int(name.split('.')[1]) for name in trainable_modules} != expected_layers:
            raise RuntimeError('C0-CCAll trainable-module selection failed.')
        return trainable_modules
    if scope == 'sod_head':
        architecture = model.yaml.get('architecture_variant')
        if architecture not in {
                'c0_full_dual_becp_sod_asf',
                'c0_full_dual_rgca_sod_asf'}:
            raise RuntimeError(
                'sod_head requires a verified SOD-ASF architecture, got '
                f'{architecture!r}.')
        trainable_modules = []
        for layer_index in range(45, len(model.model)):
            module = model.model[layer_index]
            trainable_modules.append(f'model.{layer_index}')
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if trainable_modules != [f'model.{i}' for i in range(45, 50)]:
            raise RuntimeError(
                'sod_head expects SOD layers 45--48 and Detect layer 49, '
                f'got {trainable_modules}.')
        return trainable_modules
    if scope in {'fusion_head', 'pan_head'}:
        architecture = model.yaml.get('architecture_variant')
        expected_last_layers = {
            'c0_full_dual_becp': 45,
            'c0_full_dual_becp_sod_asf': 49,
            'c0_full_dual_rgca': 45,
            'c0_full_dual_rgca_no_reliability_mask_gfb': 45,
            'c0_full_dual_rgca_ccall': 45,
            'c0_full_dual_rgca_sod_asf': 49,
            'c0_full_dual_lcafca_mask_gfb': 45,
            'c0_full_dual_mask_gfb_only': 45,
            'c0_full_dual_gfb_only': 45,
        }
        expected_last_layer = expected_last_layers.get(architecture)
        if expected_last_layer is None:
            raise RuntimeError(
                f'{scope} has no verified layer layout for architecture '
                f'{architecture!r}.')
        # ``pan_head`` is the conservative recovery scope used after RS-RGCA
        # calibration: layers 0--19 are the two modal backbones and 20--23 are
        # the four RGCA+Mask+GFB blocks.  Freezing all of them preserves the
        # learned reliability behaviour exactly while PAN/FPN and Detect
        # (layers 24 onward) recover the clean-validation detection objective.
        first_trainable_layer = 20 if scope == 'fusion_head' else 24
        trainable_modules = []
        for layer_index, module in enumerate(model.model):
            if layer_index < first_trainable_layer:
                continue
            trainable_modules.append(f'model.{layer_index}')
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if (not trainable_modules
                or trainable_modules[-1] != f'model.{expected_last_layer}'):
            raise RuntimeError(
                f'{scope} expects the verified {architecture} layout at layers '
                f'{first_trainable_layer}--{expected_last_layer}, got '
                f'{trainable_modules[:1]}...'
                f'{trainable_modules[-1:]}.')
        return trainable_modules

    fusion_modules = []
    architecture = model.yaml.get('architecture_variant')
    if architecture == 'c0_full_dual_gfb_only':
        expected_counts = {'HAFFormerGFBOnly': 4}
    elif architecture == 'c0_full_dual_mask_gfb_only':
        expected_counts = {'HAFFormerMaskGFBOnly': 4}
    elif architecture == 'c0_full_dual_lcafca_mask_gfb':
        expected_counts = {'HAFFormerLCAFMaskGFB': 4}
    elif architecture == 'c0_full_dual_rgca_no_reliability_mask_gfb':
        expected_counts = {'HAFFormerRGCANoReliabilityMaskGFB': 4}
    elif architecture == 'c0_full_dual_rc_rgca_v2':
        expected_counts = {'HAFFormerRCRGCAV2MaskGFB': 4}
    elif architecture in {
            'c0_full_dual_becp', 'c0_full_dual_becp_sod_asf'}:
        expected_counts = {'HAFFormerBECPMaskGFB': 4}
    else:
        expected_counts = {'HAFFormerRGCAMaskGFB': 4}
    if architecture in {
            'c0_full_dual_becp_sod_asf',
            'c0_full_dual_rgca_sod_asf'}:
        expected_counts.update({
            'SODScaleSequenceFusion': 2,
            'SODResidualAdd': 2,
        })
    found_counts = {name: 0 for name in expected_counts}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type in expected_counts:
            fusion_modules.append(name)
            found_counts[module_type] += 1
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if found_counts != expected_counts:
        raise RuntimeError(
            f'fusion_only module layout mismatch: expected {expected_counts}, '
            f'found {found_counts}: {fusion_modules}')
    return fusion_modules


def keep_frozen_batch_norm_eval(model):
    """Prevent frozen YOLO BatchNorm buffers from changing during fine-tuning."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.BatchNorm3d, nn.SyncBatchNorm)):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(p.requires_grad for p in parameters):
                module.eval()


def enable_rs_rgca_reliability_supervision(model):
    """Enable transient auxiliary predictions on all four original RGCA gates."""
    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != 'ReliabilityGuidedBidirectionalCrossAttention':
            continue
        if not getattr(module, 'use_reliability_gate', False):
            continue
        module.enable_rs_reliability_supervision = True
        module.last_rs_reliability_predictions = None
        modules.append(name)
    if len(modules) != 4:
        raise RuntimeError(
            f'RS-RGCA expects four supervised reliability gates, got '
            f'{len(modules)}: {modules}')
    return modules


def set_rs_rgca_gate_only_stage(model, enabled):
    """Freeze/unfreeze around the existing reliability gates without rewiring."""
    gate_parameters = 0
    for name, parameter in model.named_parameters():
        is_gate = '.reliability_gate.' in name
        parameter.requires_grad_(is_gate if enabled else True)
        if is_gate:
            gate_parameters += parameter.numel()
    if gate_parameters == 0:
        raise RuntimeError('RS-RGCA gate-only stage found no reliability parameters.')
    return gate_parameters


def set_flir_lcafnet_migration_stage(model, fusion_only):
    """Select the safe trainable stage for LCAFNet-to-RGCA FLIR migration.

    Layers 0--19 are the already-trained RGB/IR backbones, 20--23 are the
    replacement RGCA+Mask+GFB blocks, and 24--45 are the compatible PAN and
    Detect head.  The optimizer is built for layers 20--45 before this staged
    switch, so head parameters are available when stage two starts.
    """
    architecture = model.yaml.get('architecture_variant')
    if architecture != 'c0_full_dual_rgca' or len(model.model) != 46:
        raise RuntimeError(
            'FLIR LCAFNet migration requires the verified 46-layer C0 RGCA, '
            f'got architecture={architecture!r}, layers={len(model.model)}.')
    last_trainable = 23 if fusion_only else 45
    for layer_index, module in enumerate(model.model):
        requires_grad = 20 <= layer_index <= last_trainable
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError('FLIR migration stage selected no trainable parameters.')
    return trainable


def compute_warmup_iterations(warmup_epochs, batches_per_epoch, min_iters,
                              remaining_epochs, max_fraction, resumed=False):
    """Bound warmup so short smoke runs do not spend their whole run warming up."""
    if resumed or min(warmup_epochs, batches_per_epoch, remaining_epochs) <= 0:
        return 0
    requested = max(round(warmup_epochs * batches_per_epoch), int(min_iters))
    cap = max(1, round(remaining_epochs * batches_per_epoch * float(max_fraction)))
    return min(requested, cap)


def enable_cross_modal_attention_tracking(model):
    for module in model.modules():
        if hasattr(module, 'track_attention_stats'):
            module.track_attention_stats = True
            module.last_attention_stats = None


def collect_cross_modal_attention_diagnostics(model):
    rows = []
    for module_name, module in model.named_modules():
        stats = getattr(module, 'last_attention_stats', None)
        if stats is not None:
            rows.append((module_name, *stats))
            module.last_attention_stats = None
    return rows


def enable_fusion_gate_tracking(model):
    for module in model.modules():
        if hasattr(module, 'mask_logit_gain') and hasattr(module, 'foreground_mask'):
            module.track_fusion_stats = True
            module.last_fusion_stats = None


def collect_fusion_gate_diagnostics(model):
    rows = []
    for module_name, module in model.named_modules():
        stats = getattr(module, 'last_fusion_stats', None)
        if stats is not None:
            gain = float(module.mask_logit_gain.detach().float())
            rows.append((module_name, *stats, gain))
            module.last_fusion_stats = None
    return rows


def _shift_image_without_wrap(image, shift_x, shift_y):
    """Translate BCHW data while filling uncovered pixels with its mean."""
    _, _, height, width = image.shape
    shift_x = int(max(-width + 1, min(width - 1, shift_x)))
    shift_y = int(max(-height + 1, min(height - 1, shift_y)))
    output = image.mean(dim=(-2, -1), keepdim=True).expand_as(image).clone()
    source_x0, source_x1 = max(0, -shift_x), min(width, width - shift_x)
    source_y0, source_y1 = max(0, -shift_y), min(height, height - shift_y)
    target_x0, target_x1 = source_x0 + shift_x, source_x1 + shift_x
    target_y0, target_y1 = source_y0 + shift_y, source_y1 + shift_y
    output[..., target_y0:target_y1, target_x0:target_x1] = image[
        ..., source_y0:source_y1, source_x0:source_x1]
    return output


def _degrade_visible_weather(image, severity):
    """Apply one differentiable proxy for low light, fog, or rain blur."""
    severity = float(min(max(severity, 0.0), 1.0))
    mode = random.randrange(3)
    if mode == 0:  # low light: photon starvation, mild cast and sensor noise
        gamma = 1.0 + 1.8 * severity
        color = image.new_tensor([0.92, 0.97, 1.05]).view(1, 3, 1, 1)
        output = image.clamp(0.0, 1.0).pow(gamma)
        output = output * (1.0 - 0.55 * severity) * color
        signal_noise = (0.008 + 0.035 * severity) * torch.randn_like(output)
        output = output + signal_noise * (1.15 - output).clamp_min(0.25)
    elif mode == 1:  # atmospheric veil and contrast/edge loss
        softened = F.avg_pool2d(image, 7, stride=1, padding=3)
        transmission = 1.0 - 0.68 * severity
        veil = image.mean(dim=1, keepdim=True).mean(
            dim=(-2, -1), keepdim=True).clamp(0.45, 0.8)
        output = transmission * (
            (1.0 - 0.45 * severity) * image + 0.45 * severity * softened)
        output = output + (1.0 - transmission) * veil
    else:  # streak occlusion plus camera motion/defocus under heavy weather
        softened = F.avg_pool2d(image, 5, stride=1, padding=2)
        streak_seed = (torch.rand_like(image[:, :1]) > 0.965).float()
        streak = F.avg_pool2d(
            streak_seed, kernel_size=(17, 1), stride=1, padding=(8, 0))
        streak = streak / streak.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        output = (1.0 - 0.50 * severity) * image + 0.50 * severity * softened
        output = output + 0.22 * severity * streak
    return output.clamp(0.0, 1.0)


def _degrade_thermal_sensor(image, severity):
    """Model thermal crossover, optics blur, fixed-pattern noise and clutter."""
    severity = float(min(max(severity, 0.0), 1.0))
    mean = image.mean(dim=(-2, -1), keepdim=True)
    contrast = 1.0 - 0.82 * severity
    output = mean + contrast * (image - mean)
    output = ((1.0 - 0.60 * severity) * output
              + 0.60 * severity * F.avg_pool2d(
                  output, 5, stride=1, padding=2))
    # Low-frequency temperature clutter and column-wise fixed-pattern noise
    # represent realistic IR failure modes without pretending weather affects
    # thermal cameras in exactly the same way as visible cameras.
    clutter = torch.randn(
        image.shape[0], 1, 8, 8, device=image.device, dtype=image.dtype)
    clutter = F.interpolate(
        clutter, size=image.shape[-2:], mode='bilinear', align_corners=False)
    columns = torch.randn(
        image.shape[0], 1, 1, image.shape[-1],
        device=image.device, dtype=image.dtype)
    output = output + 0.055 * severity * clutter + 0.025 * severity * columns
    output = output + 0.012 * severity * torch.randn_like(output)
    return output.clamp(0.0, 1.0)


def apply_rc_rgca_v2_training_corruption(images_rgb, images_ir, hyp):
    """Create asymmetric reliability supervision from known perturbations.

    Returns corrupted modalities and per-sample targets ``[q_rgb, q_ir, c]``.
    Clean samples remain in every batch in expectation, preserving clean-data
    accuracy while making reliability identifiable from counterexamples.
    """
    batch = images_rgb.shape[0]
    quality = images_rgb.new_ones((batch, 3))
    probability = float(hyp.get('reliability_corruption_probability', 0.0))
    pair_probability = float(
        hyp.get('reliability_pair_corruption_probability', 0.0))
    dropout_probability = float(
        hyp.get('reliability_modality_dropout_probability', 0.0))
    max_shift = float(hyp.get('reliability_max_shift_fraction', 0.04))
    rgb_outputs, ir_outputs = [], []

    for index in range(batch):
        rgb = images_rgb[index:index + 1]
        ir = images_ir[index:index + 1]
        if random.random() < probability:
            branch = random.random()
            if branch < 0.45:
                severity = random.uniform(0.25, 1.0)
                rgb = _degrade_visible_weather(rgb, severity)
                quality[index, 0] = 1.0 - 0.80 * severity
            elif branch < 0.90:
                severity = random.uniform(0.25, 1.0)
                ir = _degrade_thermal_sensor(ir, severity)
                quality[index, 1] = 1.0 - 0.80 * severity
            else:
                rgb_severity = random.uniform(0.20, 0.75)
                ir_severity = random.uniform(0.20, 0.75)
                rgb = _degrade_visible_weather(rgb, rgb_severity)
                ir = _degrade_thermal_sensor(ir, ir_severity)
                quality[index, 0] = 1.0 - 0.80 * rgb_severity
                quality[index, 1] = 1.0 - 0.80 * ir_severity

        if random.random() < dropout_probability:
            if random.random() < 0.5:
                rgb = rgb.mean(dim=(-2, -1), keepdim=True).expand_as(rgb)
                rgb = (rgb + 0.003 * torch.randn_like(rgb)).clamp(0.0, 1.0)
                quality[index, 0] = 0.02
            else:
                ir = ir.mean(dim=(-2, -1), keepdim=True).expand_as(ir)
                ir = (ir + 0.003 * torch.randn_like(ir)).clamp(0.0, 1.0)
                quality[index, 1] = 0.02

        if random.random() < pair_probability:
            severity = random.uniform(0.35, 1.0)
            shift_limit = max(1, round(max_shift * max(ir.shape[-2:])))
            shift_x = random.choice((-1, 1)) * max(
                1, round(shift_limit * severity))
            shift_y = random.choice((-1, 1)) * max(
                1, round(0.6 * shift_limit * severity))
            ir = _shift_image_without_wrap(ir, shift_x, shift_y)
            quality[index, 2] = 1.0 - 0.95 * severity

        rgb_outputs.append(rgb)
        ir_outputs.append(ir)
    return torch.cat(rgb_outputs, 0), torch.cat(ir_outputs, 0), quality


def apply_rs_rgca_training_corruption(images_rgb, images_ir, hyp):
    """Apply the mild RS-RGCA curriculum using the audited corruption code.

    Keeping one implementation for the physical degradation proxies prevents
    RC-RGCA v2 and RS-RGCA from silently assigning different meanings to the
    same quality targets.  Only the probabilities are protocol specific.
    """
    mapped = {
        'reliability_corruption_probability': float(
            hyp['rs_reliability_corruption_probability']),
        'reliability_pair_corruption_probability': float(
            hyp.get('rs_reliability_pair_corruption_probability', 0.0)),
        'reliability_modality_dropout_probability': float(
            hyp.get('rs_reliability_modality_dropout_probability', 0.0)),
        'reliability_max_shift_fraction': float(
            hyp.get('rs_reliability_max_shift_fraction', 0.04)),
    }
    return apply_rc_rgca_v2_training_corruption(
        images_rgb, images_ir, mapped)


def compute_rc_rgca_v2_reliability_loss(model, target, hyp):
    """Calibrate all RC-RGCA v2 levels to the known corruption state."""
    def probability_bce(prediction, supervision):
        # torch 1.13 rejects F.binary_cross_entropy inside autocast even when
        # its inputs are explicitly float.  This is the same Bernoulli NLL,
        # evaluated in FP32 with explicit clamping for AMP compatibility.
        prediction = prediction.float().clamp(1e-5, 1.0 - 1e-5)
        supervision = supervision.float()
        return -(supervision * prediction.log()
                 + (1.0 - supervision) * (1.0 - prediction).log()).mean()

    losses = []
    local_gain = float(hyp.get('reliability_local_gain', 0.5))
    ranking_gain = float(hyp.get('reliability_ranking_gain', 0.25))
    target_rgb, target_ir, target_pair = target.split(1, dim=1)
    for module in model.modules():
        prediction = getattr(module, 'last_reliability_predictions', None)
        if prediction is None:
            continue
        calibration = (
            probability_bce(prediction['q_rgb'], target_rgb)
            + probability_bce(prediction['q_ir'], target_ir)
            + probability_bce(
                prediction['correspondence'], target_pair)) / 3.0
        local = (
            probability_bce(prediction['q_rgb_local_mean'], target_rgb)
            + probability_bce(
                prediction['q_ir_local_mean'], target_ir)) / 2.0
        predicted_difference = prediction['q_rgb'] - prediction['q_ir']
        target_difference = target_rgb - target_ir
        ranking = F.smooth_l1_loss(
            predicted_difference.float(), target_difference.float())
        expected_null_rgb = 1.0 - target_ir * target_pair
        expected_null_ir = 1.0 - target_rgb * target_pair
        null_calibration = (
            F.smooth_l1_loss(
                prediction['null_rgb'].view_as(expected_null_rgb).float(),
                expected_null_rgb.float())
            + F.smooth_l1_loss(
                prediction['null_ir'].view_as(expected_null_ir).float(),
                expected_null_ir.float())) / 2.0
        losses.append(
            calibration + local_gain * local
            + ranking_gain * (ranking + null_calibration))
        module.last_reliability_predictions = None
    if not losses:
        raise RuntimeError(
            'RC-RGCA v2 architecture produced no reliability predictions.')
    return torch.stack(losses).mean()


def summarize_rc_rgca_v2_reliability(model, target):
    """Return batch means for an auditable reliability-training CSV."""
    target = target.detach().float()
    module_rows = []
    for module in model.modules():
        prediction = getattr(module, 'last_reliability_predictions', None)
        if prediction is None:
            continue
        module_rows.append(torch.stack([
            prediction['q_rgb'].detach().float().mean(),
            prediction['q_ir'].detach().float().mean(),
            prediction['correspondence'].detach().float().mean(),
            prediction['null_rgb'].detach().float().mean(),
            prediction['null_ir'].detach().float().mean(),
            prediction['r_ir_to_rgb'].detach().float().mean(),
            prediction['r_rgb_to_ir'].detach().float().mean(),
        ]))
    if not module_rows:
        raise RuntimeError('No RC-RGCA v2 state is available for diagnostics.')
    prediction_mean = torch.stack(module_rows).mean(dim=0)
    return torch.cat([target.mean(dim=0), prediction_mean]).cpu()


def summarize_rs_rgca_reliability(model, target):
    """Summarize normalized directional gate predictions for diagnostics."""
    module_rows = []
    for module in model.modules():
        prediction = getattr(module, 'last_rs_reliability_predictions', None)
        if prediction is None:
            continue
        module_rows.append(torch.stack([
            prediction['r_ir_to_rgb'].detach().float().mean(),
            prediction['r_rgb_to_ir'].detach().float().mean(),
        ]))
    if not module_rows:
        raise RuntimeError('No RS-RGCA reliability state is available.')
    return torch.cat([
        target.detach().float().mean(dim=0),
        torch.stack(module_rows).mean(dim=0),
    ]).cpu()


def compute_rs_rgca_reliability_loss(model, target, hyp):
    """Supervise the original RGCA gates without gradients into their inputs.

    ``r_ir_to_rgb`` is supervised by IR source quality and pair
    correspondence; ``r_rgb_to_ir`` is supervised by RGB source quality and
    correspondence.  This direction assignment is the core reliability
    invariant and is deliberately asserted in one centralized loss.
    """
    target_rgb, target_ir, target_pair = target.float().split(1, dim=1)
    target_ir_to_rgb = target_ir * target_pair
    target_rgb_to_ir = target_rgb * target_pair
    calibration_losses, ranking_losses = [], []
    for module in model.modules():
        prediction = getattr(module, 'last_rs_reliability_predictions', None)
        if prediction is None:
            continue
        pred_ir_to_rgb = prediction['r_ir_to_rgb'].float()
        pred_rgb_to_ir = prediction['r_rgb_to_ir'].float()
        calibration_losses.append((
            F.smooth_l1_loss(pred_ir_to_rgb, target_ir_to_rgb)
            + F.smooth_l1_loss(pred_rgb_to_ir, target_rgb_to_ir)) / 2.0)
        ranking_losses.append(F.smooth_l1_loss(
            pred_ir_to_rgb - pred_rgb_to_ir,
            target_ir_to_rgb - target_rgb_to_ir))
        module.last_rs_reliability_predictions = None
    if not calibration_losses:
        raise RuntimeError(
            'RS-RGCA architecture produced no reliability predictions.')
    calibration = torch.stack(calibration_losses).mean()
    ranking = torch.stack(ranking_losses).mean()
    ranking_gain = float(hyp.get('rs_reliability_ranking_gain', 0.25))
    total = calibration + ranking_gain * ranking
    return total, calibration.detach(), ranking.detach()


def focus_weight_to_conv6(weight):
    """Convert a YOLOv5 Focus 12x3x3 kernel to an equivalent 3x6x6 kernel."""
    if weight.ndim != 4 or weight.shape[1] % 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f'Not a YOLOv5 Focus kernel: shape={tuple(weight.shape)}')
    in_channels = weight.shape[1] // 4
    converted = weight.new_zeros((weight.shape[0], in_channels, 6, 6))
    for group, (row, col) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        source = weight[:, group * in_channels:(group + 1) * in_channels]
        converted[:, :, row::2, col::2] = source
    return converted


def _model_state_layer_index(name):
    parts = name.split('.', 2)
    return int(parts[1]) if (
        len(parts) == 3 and parts[0] == 'model' and parts[1].isdigit()) else None


def build_dual_stream_backbone_state_dict(
        source_state, model_state, target_layer_map=None):
    """Map a single-stream YOLOv5s backbone into a progressive RGB/IR backbone."""
    old_focus_key = 'model.0.conv.conv.weight'
    old_focus = (
        old_focus_key in source_state
        and 'model.0.conv.weight' in model_state
        and tuple(source_state[old_focus_key].shape[2:]) == (3, 3)
        and tuple(model_state['model.0.conv.weight'].shape[2:]) == (6, 6))
    new_to_old = {9: 8, 8: 9} if old_focus else {}

    if target_layer_map is None:
        canonical_map = {index: index for index in range(10)}
    else:
        canonical_map = {
            int(target): int(source) for target, source in target_layer_map.items()}
        if not canonical_map or any(x not in range(10) for x in canonical_map.values()):
            raise ValueError(
                'pretrained_backbone_map must map target layers to stages 0-9')

    transferred, missing = {}, []
    converted_focus = False
    target_keys = [
        key for key in model_state
        if _model_state_layer_index(key) in canonical_map]
    for target_key in target_keys:
        _, layer_text, suffix = target_key.split('.', 2)
        target_layer = int(layer_text)
        canonical_layer = canonical_map[target_layer]
        source_layer = new_to_old.get(canonical_layer, canonical_layer)

        if old_focus and canonical_layer == 0:
            if suffix == 'conv.weight':
                value = focus_weight_to_conv6(source_state[old_focus_key])
                converted_focus = True
            elif suffix.startswith('bn.'):
                value = source_state.get(f'model.0.conv.{suffix}')
            else:
                value = source_state.get(f'model.0.{suffix}')
        else:
            value = source_state.get(f'model.{source_layer}.{suffix}')

        if value is None or value.shape != model_state[target_key].shape:
            missing.append(target_key)
            continue
        transferred[target_key] = value

        # Historical full-dual configs omitted an explicit map.
        if target_layer_map is None:
            ir_key = f'model.{target_layer + 10}.{suffix}'
            if ir_key not in model_state or value.shape != model_state[ir_key].shape:
                missing.append(ir_key)
            else:
                transferred[ir_key] = value

    if target_layer_map is None:
        rgb_layers, ir_layers, shared_layers = (
            set(range(10)), set(range(10, 20)), set())
    else:
        # A canonical stage mapped twice initializes the independent RGB/IR
        # copies; a stage mapped once initializes the shared deep path. This
        # supports both C1 (dual through stage 4) and C1.5 (dual through stage
        # 6) without hard-coding a branch length.
        stage_targets = {}
        for target, stage in canonical_map.items():
            stage_targets.setdefault(stage, []).append(target)
        invalid = {
            stage: targets for stage, targets in stage_targets.items()
            if len(targets) not in (1, 2)}
        if invalid:
            raise ValueError(
                'Each pretrained backbone stage must map to one shared layer '
                f'or two modal layers, got {invalid}.')
        rgb_layers = {
            targets[0] for targets in stage_targets.values()
            if len(targets) == 2}
        ir_layers = {
            targets[1] for targets in stage_targets.values()
            if len(targets) == 2}
        shared_layers = {
            targets[0] for targets in stage_targets.values()
            if len(targets) == 1}

    report = {
        'format': 'focus_spp' if old_focus else 'conv6_sppf',
        'layout': 'legacy_dual' if target_layer_map is None else 'explicit_map',
        'rgb_items': sum(_model_state_layer_index(k) in rgb_layers for k in transferred),
        'ir_items': sum(_model_state_layer_index(k) in ir_layers for k in transferred),
        'shared_items': sum(_model_state_layer_index(k) in shared_layers for k in transferred),
        'target_items': len(transferred),
        'expected_target_items': (
            len(target_keys) * 2 if target_layer_map is None else len(target_keys)),
        'expected_per_branch': len(target_keys),
        'converted_focus': converted_focus,
        'missing': missing,
    }
    return transferred, report


def build_yolov5s_to_c0_ccall_state_dict(
        source_state, model_state, target_layer_map):
    """Initialize both C0-CCAll backbones from a single YOLOv5s backbone.

    Conv/downsampling, SPP/SPPF-compatible tensors and the outer CSP
    projections of all eight target C3CrossConv stages are transferred.  The
    source C3 Bottleneck sequence is never mixed into CrossConv, even when a
    BatchNorm tensor happens to have the same shape.  Fusion, PAN and Detect
    intentionally retain target initialization because the single-stream
    three-scale YOLOv5s head is not semantically aligned with this dual-stream
    four-scale detector.
    """
    transferred, base_report = build_dual_stream_backbone_state_dict(
        source_state, model_state, target_layer_map=target_layer_map)
    replacement_layers = {2, 4, 6, 8, 12, 14, 16, 18}
    backbone_layers = set(range(20))

    for key in list(transferred):
        layer = _model_state_layer_index(key)
        if (layer in replacement_layers
                and key.startswith(f'model.{layer}.m.')):
            del transferred[key]

    backbone_keys = [
        key for key in model_state
        if _model_state_layer_index(key) in backbone_layers]
    fresh_crossconv = []
    unexpected_missing = []
    for key in backbone_keys:
        if key in transferred:
            continue
        layer = _model_state_layer_index(key)
        if (layer in replacement_layers
                and key.startswith(f'model.{layer}.m.')):
            fresh_crossconv.append(key)
        else:
            unexpected_missing.append(key)

    report = {
        'mode': 'yolov5s_to_c0_ccall',
        'format': base_report['format'],
        'target_items': len(transferred),
        'expected_target_items': len(backbone_keys) - len(fresh_crossconv),
        'rgb_items': sum(
            _model_state_layer_index(k) in range(10)
            for k in transferred),
        'ir_items': sum(
            _model_state_layer_index(k) in range(10, 20)
            for k in transferred),
        'fresh_crossconv_items': fresh_crossconv,
        'fresh_detector_items': [
            key for key in model_state
            if _model_state_layer_index(key) not in backbone_layers],
        'converted_focus': base_report['converted_focus'],
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_full_dual_lcafnet_to_c1_state_dict(source_state, model_state):
    """Migrate a trained full-dual LCAFNet checkpoint into the C1 topology.

    The source layout has RGB layers 0--9, IR layers 10--19, fusion layers
    20--23, and PAN/Detect layers 24--45. C1 keeps the two shallow branches
    through C3, merges them at layers 10/11, shares C4/C5 at layers 12--16,
    and places the unchanged PAN/Detect at layers 17--38.

    RGB/IR shallow tensors and the entire detection neck/head are copied
    exactly. Corresponding RGB/IR C4/C5 tensors are averaged to initialize the
    shared deep stages symmetrically. The old fusion block's GFB tensors are
    compatible and retained at P2/P3; the new RGCA and foreground-mask tensors
    deliberately remain freshly initialized.
    """
    transferred = {}
    counts = {'rgb': 0, 'ir': 0, 'shared': 0, 'head': 0, 'gfb': 0}

    def target_layer_keys(layer):
        prefix = f'model.{layer}.'
        return [key for key in model_state if key.startswith(prefix)]

    def copy_layer(target_layer, source_layer, group):
        for target_key in target_layer_keys(target_layer):
            suffix = target_key.split('.', 2)[2]
            source_key = f'model.{source_layer}.{suffix}'
            value = source_state.get(source_key)
            if value is not None and value.shape == model_state[target_key].shape:
                transferred[target_key] = value
                counts[group] += 1

    def average_layer(target_layer, rgb_layer, ir_layer):
        for target_key in target_layer_keys(target_layer):
            suffix = target_key.split('.', 2)[2]
            rgb_value = source_state.get(f'model.{rgb_layer}.{suffix}')
            ir_value = source_state.get(f'model.{ir_layer}.{suffix}')
            target_value = model_state[target_key]
            if (rgb_value is None or ir_value is None
                    or rgb_value.shape != target_value.shape
                    or ir_value.shape != target_value.shape):
                continue
            if torch.is_floating_point(rgb_value) or torch.is_complex(rgb_value):
                value = (rgb_value + ir_value) * 0.5
            else:
                # BatchNorm num_batches_tracked is integral. Retain the more
                # mature counter rather than performing an invalid mean.
                value = torch.maximum(rgb_value, ir_value)
            transferred[target_key] = value
            counts['shared'] += 1

    # Independent RGB and IR feature extraction through C3.
    for target_layer, source_layer in zip(range(0, 5), range(0, 5)):
        copy_layer(target_layer, source_layer, 'rgb')
    for target_layer, source_layer in zip(range(5, 10), range(10, 15)):
        copy_layer(target_layer, source_layer, 'ir')

    # Symmetric initialization for the single shared C4/C5 path.
    for target_layer, rgb_layer, ir_layer in (
            (12, 5, 15), (13, 6, 16), (14, 7, 17),
            (15, 8, 18), (16, 9, 19)):
        average_layer(target_layer, rgb_layer, ir_layer)

    # The PAN and four-scale Detect topology are unchanged apart from indices.
    for target_layer, source_layer in zip(range(17, 39), range(24, 46)):
        copy_layer(target_layer, source_layer, 'head')

    # Preserve the old learned GFB at the two fusion levels retained by C1.
    for target_layer, source_layer in ((10, 20), (11, 21)):
        for source_suffix, target_suffix in (
                ('conv.0.weight', 'fusion.conv.0.weight'),
                ('dwconv.weight', 'fusion.dwconv.weight')):
            source_key = f'model.{source_layer}.{source_suffix}'
            target_key = f'model.{target_layer}.{target_suffix}'
            value = source_state.get(source_key)
            if (value is not None and target_key in model_state
                    and value.shape == model_state[target_key].shape):
                transferred[target_key] = value
                counts['gfb'] += 1

    fresh = [key for key in model_state if key not in transferred]
    expected_fresh_prefixes = ('model.10.', 'model.11.')
    unexpected_missing = [
        key for key in fresh if not key.startswith(expected_fresh_prefixes)]
    report = {
        'mode': 'full_dual_lcafnet_to_c1',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'rgb_items': counts['rgb'],
        'ir_items': counts['ir'],
        'shared_items': counts['shared'],
        'head_items': counts['head'],
        'gfb_items': counts['gfb'],
        'fresh_fusion_items': fresh,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_full_dual_lcafnet_to_c0_rgca_state_dict(source_state, model_state):
    """Migrate an original full-dual LCAFNet checkpoint into C0 RGCA.

    Source and target share the complete RGB/IR backbone (layers 0--19) and
    PAN/Detect topology (layers 24--45).  At fusion layers 20--23, the learned
    GFB tensors are copied into the target ``fusion`` submodule while the new
    RGCA and foreground-mask tensors deliberately retain model initialization.
    """
    transferred = {}
    counts = {'backbone': 0, 'head': 0, 'gfb': 0}
    fusion_layers = set(range(20, 24))

    for target_key, target_value in model_state.items():
        layer = _model_state_layer_index(target_key)
        if layer in fusion_layers:
            continue
        source_value = source_state.get(target_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        transferred[target_key] = source_value
        if layer is not None and layer < 20:
            counts['backbone'] += 1
        else:
            counts['head'] += 1

    for layer in sorted(fusion_layers):
        for source_suffix, target_suffix in (
                ('conv.0.weight', 'fusion.conv.0.weight'),
                ('dwconv.weight', 'fusion.dwconv.weight')):
            source_key = f'model.{layer}.{source_suffix}'
            target_key = f'model.{layer}.{target_suffix}'
            source_value = source_state.get(source_key)
            if (source_value is not None and target_key in model_state
                    and source_value.shape == model_state[target_key].shape):
                transferred[target_key] = source_value
                counts['gfb'] += 1

    fresh = [key for key in model_state if key not in transferred]
    unexpected_missing = [
        key for key in fresh
        if _model_state_layer_index(key) not in fusion_layers]
    report = {
        'mode': 'full_dual_lcafnet_to_c0_rgca',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'backbone_items': counts['backbone'],
        'head_items': counts['head'],
        'gfb_items': counts['gfb'],
        'fresh_fusion_items': fresh,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_c0_rgca_to_ccall_state_dict(source_state, model_state):
    """Migrate trained C0 weights into the YOLO-TLA C3CrossConv variant.

    The eight backbone C3 modules keep the same outer CSP cv1/cv2/cv3 paths,
    so those tensors are loaded exactly.  Their internal ``m`` blocks change
    from Bottleneck (1x1 + 3x3) to the paper's residual CrossConv
    (1x3 + 3x1) and must remain freshly initialized.  Every tensor outside
    those internal mixers is required to transfer exactly; this guards the
    unchanged RGCA/Mask/GFB, dense-C3 PAN and four-scale Detect head.
    """
    replacement_layers = {2, 4, 6, 8, 12, 14, 16, 18}
    transferred = {}
    counts = {'outer_csp': 0, 'unchanged': 0}

    for target_key, target_value in model_state.items():
        layer = _model_state_layer_index(target_key)
        is_new_internal = (
            layer in replacement_layers
            and target_key.startswith(f'model.{layer}.m.'))
        if is_new_internal:
            continue
        source_value = source_state.get(target_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        transferred[target_key] = source_value
        if layer in replacement_layers:
            counts['outer_csp'] += 1
        else:
            counts['unchanged'] += 1

    fresh = [key for key in model_state if key not in transferred]
    unexpected_missing = []
    for key in fresh:
        layer = _model_state_layer_index(key)
        if not (layer in replacement_layers
                and key.startswith(f'model.{layer}.m.')):
            unexpected_missing.append(key)
    report = {
        'mode': 'c0_rgca_to_ccall',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'outer_csp_items': counts['outer_csp'],
        'unchanged_items': counts['unchanged'],
        'fresh_crossconv_items': fresh,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_lcafnet_to_c0_ccall_state_dict(source_state, model_state):
    """Migrate original LCAFNet into C0-CCAll without silent partial loads.

    Original LCAFNet and the target share the dual-backbone stage layout,
    dense-C3 PAN and P2-P5 Detect.  The outer CSP projections of each replaced
    backbone C3 are also shape/semantics compatible.  Only the new asymmetric
    CrossConv mixers and RGCA/mask tensors remain fresh; the compatible GFB
    convolution and depthwise convolution at every fusion level are remapped.
    """
    replacement_layers = {2, 4, 6, 8, 12, 14, 16, 18}
    fusion_layers = set(range(20, 24))
    transferred = {}
    counts = {'unchanged': 0, 'outer_csp': 0, 'gfb': 0}

    for target_key, target_value in model_state.items():
        layer = _model_state_layer_index(target_key)
        if layer in fusion_layers:
            continue
        if (layer in replacement_layers
                and target_key.startswith(f'model.{layer}.m.')):
            continue
        source_value = source_state.get(target_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        transferred[target_key] = source_value
        if layer in replacement_layers:
            counts['outer_csp'] += 1
        else:
            counts['unchanged'] += 1

    for layer in sorted(fusion_layers):
        for source_suffix, target_suffix in (
                ('conv.0.weight', 'fusion.conv.0.weight'),
                ('dwconv.weight', 'fusion.dwconv.weight')):
            source_key = f'model.{layer}.{source_suffix}'
            target_key = f'model.{layer}.{target_suffix}'
            source_value = source_state.get(source_key)
            if (source_value is not None and target_key in model_state
                    and source_value.shape == model_state[target_key].shape):
                transferred[target_key] = source_value
                counts['gfb'] += 1

    fresh = [key for key in model_state if key not in transferred]
    fresh_crossconv = []
    fresh_fusion = []
    unexpected_missing = []
    for key in fresh:
        layer = _model_state_layer_index(key)
        if (layer in replacement_layers
                and key.startswith(f'model.{layer}.m.')):
            fresh_crossconv.append(key)
        elif layer in fusion_layers:
            fresh_fusion.append(key)
        else:
            unexpected_missing.append(key)
    report = {
        'mode': 'lcafnet_to_c0_ccall',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'unchanged_items': counts['unchanged'],
        'outer_csp_items': counts['outer_csp'],
        'gfb_items': counts['gfb'],
        'fresh_crossconv_items': fresh_crossconv,
        'fresh_fusion_items': fresh_fusion,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_c0_rgca_to_sod_asf_state_dict(source_state, model_state):
    """Transfer a trained FLIR C0 detector into the SOD-ASF extension.

    Target layers 0--44 are structurally identical to C0.  Its Detect layer is
    moved from 45 to 49 because layers 45--48 add P3/P2 scale fusion.  Every
    established detector tensor is required to transfer exactly; only the new
    scale-sequence modules may remain freshly initialized.
    """
    transferred = {}
    counts = {'existing': 0, 'detect': 0}
    new_layers = set(range(45, 49))

    for target_key, target_value in model_state.items():
        target_layer = _model_state_layer_index(target_key)
        if target_layer in new_layers:
            continue
        if target_layer == 49:
            suffix = target_key.split('.', 2)[2]
            source_key = f'model.45.{suffix}'
            group = 'detect'
        else:
            source_key = target_key
            group = 'existing'
        source_value = source_state.get(source_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        transferred[target_key] = source_value
        counts[group] += 1

    fresh = [key for key in model_state if key not in transferred]
    unexpected_missing = [
        key for key in fresh
        if _model_state_layer_index(key) not in new_layers]
    report = {
        'mode': 'c0_rgca_to_sod_asf',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'excluded_items': len(fresh),
        'existing_items': counts['existing'],
        'detect_items': counts['detect'],
        'fresh_sod_items': fresh,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_c0_becp_to_becp_sod_asf_state_dict(source_state, model_state):
    """Add SOD-ASF while retaining every trained M3FD BECP tensor.

    Layers 0--44 are identical between source and target. The four new SOD
    layers occupy 45--48, while the trained source Detect layer 45 is remapped
    exactly to target layer 49. Only parameters belonging to layers 45--48 may
    remain freshly initialized.
    """
    transferred, report = build_c0_rgca_to_sod_asf_state_dict(
        source_state, model_state)
    report['mode'] = 'c0_becp_to_becp_sod_asf'
    return transferred, report


def build_lcafnet_to_c0_sod_asf_state_dict(source_state, model_state):
    """Migrate the original trained FLIR LCAFNet into C0-SOD-ASF.

    The original checkpoint and target share both complete backbones and PAN.
    The four learned GFB kernels are retained, while RGCA/mask and SOD-ASF are
    the intended new trainable modules.  The original Detect layer 45 is
    remapped to target layer 49 without reinitialization.
    """
    transferred = {}
    counts = {'backbone': 0, 'pan': 0, 'detect': 0, 'gfb': 0}
    fusion_layers = set(range(20, 24))
    new_sod_layers = set(range(45, 49))

    for target_key, target_value in model_state.items():
        target_layer = _model_state_layer_index(target_key)
        if target_layer in fusion_layers or target_layer in new_sod_layers:
            continue
        if target_layer == 49:
            suffix = target_key.split('.', 2)[2]
            source_key = f'model.45.{suffix}'
            group = 'detect'
        else:
            source_key = target_key
            group = 'backbone' if target_layer is not None and target_layer < 20 else 'pan'
        source_value = source_state.get(source_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        transferred[target_key] = source_value
        counts[group] += 1

    for layer in sorted(fusion_layers):
        for source_suffix, target_suffix in (
                ('conv.0.weight', 'fusion.conv.0.weight'),
                ('dwconv.weight', 'fusion.dwconv.weight')):
            source_key = f'model.{layer}.{source_suffix}'
            target_key = f'model.{layer}.{target_suffix}'
            source_value = source_state.get(source_key)
            if (source_value is not None and target_key in model_state
                    and source_value.shape == model_state[target_key].shape):
                transferred[target_key] = source_value
                counts['gfb'] += 1

    fresh = [key for key in model_state if key not in transferred]
    allowed_fresh_layers = fusion_layers | new_sod_layers
    unexpected_missing = [
        key for key in fresh
        if _model_state_layer_index(key) not in allowed_fresh_layers]
    report = {
        'mode': 'lcafnet_to_c0_sod_asf',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(fresh),
        'excluded_items': len(fresh),
        'backbone_items': counts['backbone'],
        'pan_items': counts['pan'],
        'detect_items': counts['detect'],
        'gfb_items': counts['gfb'],
        'fresh_items': fresh,
        'unexpected_missing': unexpected_missing,
    }
    return transferred, report


def build_full_dual_rgca_to_c1_5_state_dict(source_state, model_state):
    """Migrate the trained M3FD C0 RGCA detector into C1.5 exactly.

    C0 layout: RGB 0--9, IR 10--19, P2/P3/P4/P5 fusion 20--23, and
    PAN/Detect 24--45. C1.5 keeps independent RGB/IR stages only through C4,
    retains fusion 20--22 as target 14--16, averages the two C5/SPPF branches
    into target 17--19, drops the now-redundant P5 fusion, and shifts the
    unchanged PAN/Detect to target 20--41.

    Every target tensor must be initialized. Floating weights and BatchNorm
    statistics in shared C5 are averaged; integral BatchNorm counters retain
    the larger value. No source optimizer, scheduler, or epoch is restored.
    """
    transferred = {}
    counts = {'rgb': 0, 'ir': 0, 'fusion': 0, 'shared_c5': 0, 'head': 0}

    def target_layer_keys(layer):
        prefix = f'model.{layer}.'
        return [key for key in model_state if key.startswith(prefix)]

    def copy_layer(target_layer, source_layer, group):
        for target_key in target_layer_keys(target_layer):
            suffix = target_key.split('.', 2)[2]
            source_key = f'model.{source_layer}.{suffix}'
            value = source_state.get(source_key)
            if value is not None and value.shape == model_state[target_key].shape:
                transferred[target_key] = value
                counts[group] += 1

    def average_layer(target_layer, rgb_layer, ir_layer):
        for target_key in target_layer_keys(target_layer):
            suffix = target_key.split('.', 2)[2]
            rgb_value = source_state.get(f'model.{rgb_layer}.{suffix}')
            ir_value = source_state.get(f'model.{ir_layer}.{suffix}')
            target_value = model_state[target_key]
            if (rgb_value is None or ir_value is None
                    or rgb_value.shape != target_value.shape
                    or ir_value.shape != target_value.shape):
                continue
            if torch.is_floating_point(rgb_value) or torch.is_complex(rgb_value):
                value = (rgb_value + ir_value) * 0.5
            else:
                value = torch.maximum(rgb_value, ir_value)
            transferred[target_key] = value
            counts['shared_c5'] += 1

    for target_layer, source_layer in zip(range(0, 7), range(0, 7)):
        copy_layer(target_layer, source_layer, 'rgb')
    for target_layer, source_layer in zip(range(7, 14), range(10, 17)):
        copy_layer(target_layer, source_layer, 'ir')
    for target_layer, source_layer in zip(range(14, 17), range(20, 23)):
        copy_layer(target_layer, source_layer, 'fusion')
    for target_layer, rgb_layer, ir_layer in (
            (17, 7, 17), (18, 8, 18), (19, 9, 19)):
        average_layer(target_layer, rgb_layer, ir_layer)
    for target_layer, source_layer in zip(range(20, 42), range(24, 46)):
        copy_layer(target_layer, source_layer, 'head')

    missing = [key for key in model_state if key not in transferred]
    report = {
        'mode': 'full_dual_rgca_to_c1_5',
        'target_items': len(transferred),
        'expected_target_items': len(model_state),
        'rgb_items': counts['rgb'],
        'ir_items': counts['ir'],
        'fusion_items': counts['fusion'],
        'shared_c5_items': counts['shared_c5'],
        'head_items': counts['head'],
        'missing': missing,
    }
    return transferred, report


NEW_D3_INTERNAL_PREFIXES = ('model.34.m.', 'model.37.m.')


def build_c1_to_deep_p4p5_state_dict(source_state, model_state):
    """Transfer C1 exactly while leaving only the new D3 mixers fresh.

    Layer numbers are stable across C1 and the conservative C4 YAML.  Every
    Every target tensor outside ``model.34.m`` and ``model.37.m`` must exist
    with the same shape in C1.  This deliberately transfers the retained D3C3
    outer ``cv1/cv2/cv3`` CSP projections, as well as fusion, dense
    top-down/P2/P3 PAN, downsampling, and Detect.  The replaced C1 Bottleneck
    states cannot leak into either new D3 mixer.
    """
    transferred, missing = {}, []
    excluded = []
    for key, target_value in model_state.items():
        if key.startswith(NEW_D3_INTERNAL_PREFIXES):
            excluded.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None or source_value.shape != target_value.shape:
            missing.append(key)
        else:
            transferred[key] = source_value
    report = {
        'mode': 'c1_to_deep_p4p5_surgery',
        'target_items': len(transferred),
        'expected_target_items': len(model_state) - len(excluded),
        'excluded_items': len(excluded),
        'excluded_prefixes': NEW_D3_INTERNAL_PREFIXES,
        'missing': missing,
    }
    return transferred, report


def resolve_resume_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    try:
        torch_load(checkpoint, map_location='cpu')
        return str(checkpoint)
    except Exception as error:
        fallback = checkpoint.with_name('best.pt')
        if fallback.is_file() and fallback != checkpoint:
            try:
                torch_load(fallback, map_location='cpu')
                logger.warning(
                    "Resume checkpoint '%s' is unreadable (%s); using '%s'.",
                    checkpoint, error, fallback)
                return str(fallback)
            except Exception:
                pass
        raise


try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


def _write_diagnostics_headers(save_dir, rank):
    attention = save_dir / 'attention_diagnostics.csv'
    fusion = save_dir / 'fusion_diagnostics.csv'
    reliability = save_dir / 'reliability_calibration.csv'
    if rank in (-1, 0) and not attention.exists():
        attention.write_text(
            'epoch,module,rgb_entropy,ir_entropy,rgb_attention_std,'
            'ir_attention_std,rgb_reliability_mean,rgb_reliability_std,'
            'ir_reliability_mean,ir_reliability_std,cross_mix,'
            'residual_scale,residual_scale_min,prior_margin,'
            'prior_diagonal_probability\n')
    if rank in (-1, 0) and not fusion.exists():
        fusion.write_text(
            'epoch,module,mask_mean,mask_std,mask_min,mask_max,'
            'rgb_weight_mean,rgb_weight_std,mask_logit_gain\n')
    if rank in (-1, 0) and not reliability.exists():
        reliability.write_text(
            'epoch,target_q_rgb,target_q_ir,target_correspondence,'
            'pred_q_rgb,pred_q_ir,pred_correspondence,'
            'null_ir_to_rgb,null_rgb_to_ir,r_ir_to_rgb,r_rgb_to_ir,'
            'reliability_aux_loss\n')
    return attention, fusion, reliability


def _configured_model_definition(opt):
    """Load YAML and apply the centralized C4 D3 kernel control."""
    with open(opt.cfg) as stream:
        definition = yaml.safe_load(stream)
    if getattr(opt, 'detection_task', 'hbb') == 'obb':
        final_layer = definition['head'][-1]
        if final_layer[2] not in ('Detect', 'DetectOBB'):
            raise RuntimeError(
                f'OBB training requires a Detect head, got {final_layer[2]!r}.')
        final_layer[2] = 'DetectOBB'
        definition['detection_task'] = 'obb'
        definition['angle_bins'] = 180
    architecture = definition.get('architecture_variant')
    expected_indices = {
        'c4_c1_d3c3_slim_pan': [20, 24, 28, 31, 34, 37],
        'c4_c1_deep_p4p5_d3c3_pan': [34, 37],
    }.get(architecture)
    if expected_indices is None:
        return definition

    kernel_size = int(opt.d3_kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError('d3_kernel_size must be an odd integer >= 3')
    block_indices = []
    for index, layer in enumerate(definition['backbone'] + definition['head']):
        if layer[2] == 'D3C3':
            layer[3][-1] = kernel_size
            block_indices.append(index)
    if block_indices != expected_indices:
        raise RuntimeError(
            f'{architecture} must place D3C3 at {expected_indices}, '
            f'got {block_indices}')
    definition['d3_kernel_size'] = kernel_size
    return definition


def _load_active_model(opt, hyp, device, nc, rank):
    weights = opt.weights
    with torch_distributed_zero_first(rank):
        attempt_download(weights)
    checkpoint = torch_load(weights, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get('model') is None:
        raise RuntimeError(
            f"Checkpoint '{weights}' does not contain a YOLOv5 model entry.")

    model = Model(
        _configured_model_definition(opt), ch=3, nc=nc,
        anchors=hyp.get('anchors')).to(device)
    architecture = model.yaml.get('architecture_variant')
    supported = {
        'c0_full_dual_becp',
        'c0_full_dual_becp_sod_asf',
        'c0_full_dual_rgca',
        'c0_full_dual_rc_rgca_v2',
        'c0_full_dual_rgca_no_reliability_mask_gfb',
        'c0_full_dual_rgca_ccall',
        'c0_full_dual_rgca_sod_asf',
        'c0_full_dual_lcafca_mask_gfb',
        'c0_full_dual_mask_gfb_only',
        'c0_full_dual_gfb_only',
        'c1_dual_to_c3_shared_c4_c5',
        'c1_5_dual_to_c4_shared_c5',
        'c4_c1_d3c3_slim_pan',
        'c4_c1_deep_p4p5_d3c3_pan',
    }
    if architecture not in supported:
        raise RuntimeError(
            f'train.py supports only the active C0 and archived C1/C1.5/C4 paths, '
            f'got {architecture!r}.')

    source_model = checkpoint['model'].float()
    source_state = source_model.state_dict()
    source_yaml = getattr(source_model, 'yaml', {}) or {}
    source_architecture = source_yaml.get('architecture_variant')
    source_detection_task = str(
        source_yaml.get('detection_task', 'hbb')).lower()
    model_state = model.state_dict()
    if opt.resume:
        transferred = intersect_dicts(source_state, model_state, exclude=[])
        missing = [key for key in model_state if key not in transferred and 'anchor' not in key]
        if missing:
            raise RuntimeError(
                f'Resume checkpoint is not the same architecture; missing {len(missing)} tensors.')
        report = None
        initialization_summary = f'resume exact checkpoint `{weights}`'
    else:
        if (source_architecture == architecture
                and source_detection_task == 'obb'
                and getattr(opt, 'detection_task', 'hbb') == 'obb'):
            transferred = intersect_dicts(source_state, model_state, exclude=[])
            missing = [key for key in model_state if key not in transferred]
            if missing:
                preview = ', '.join(missing[:5])
                raise RuntimeError(
                    f'Exact C0-OBB fine-tune misses {len(missing)} tensors '
                    f'(first: {preview}).')
            report = {
                'mode': 'same_architecture_obb_finetune',
                'target_items': len(transferred),
                'expected_target_items': len(model_state),
                'missing': [],
            }
            initialization_summary = (
                f'complete same-architecture C0-OBB fine-tune checkpoint '
                f'`{weights}`')
        elif (source_architecture == architecture
                and getattr(opt, 'detection_task', 'hbb') == 'obb'):
            transferred = intersect_dicts(source_state, model_state, exclude=[])
            missing = [key for key in model_state if key not in transferred]
            unexpected = [key for key in missing
                          if not key.startswith('model.45.m.')]
            if unexpected:
                preview = ', '.join(unexpected[:5])
                raise RuntimeError(
                    f'HBB-to-OBB C0 transfer misses {len(unexpected)} '
                    f'non-head tensors (first: {preview}).')
            report = {
                'mode': 'same_architecture_hbb_to_obb',
                'target_items': len(transferred),
                'expected_target_items': len(model_state),
                'excluded_items': len(missing),
            }
            initialization_summary = (
                f'C0 checkpoint `{weights}`; backbone/fusion/PAN loaded and '
                'incompatible OBB prediction convolutions freshly initialized')
        elif source_architecture == architecture:
            transferred = intersect_dicts(source_state, model_state, exclude=[])
            missing = [key for key in model_state if key not in transferred]
            if missing:
                preview = ', '.join(missing[:5])
                raise RuntimeError(
                    f'Exact-architecture fine-tune misses {len(missing)} '
                    f'tensors (first: {preview}).')
            report = {
                'mode': 'same_architecture_finetune',
                'target_items': len(transferred),
                'expected_target_items': len(model_state),
                'missing': [],
            }
            initialization_summary = (
                f'full same-architecture fine-tune checkpoint `{weights}`')
        elif (source_architecture == 'c0_full_dual_rgca'
              and architecture == 'c0_full_dual_rgca_ccall'
              and source_detection_task == 'hbb'
              and getattr(opt, 'detection_task', 'hbb') == 'hbb'):
            transferred, report = build_c0_rgca_to_ccall_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'C0 RGCA to C0-CCAll transfer leaves '
                    f'{len(report["unexpected_missing"])} unchanged tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'MFAD C0 best `{weights}` -> C0-CCAll: all unchanged '
                'backbone tensors, eight outer CSP projections, '
                'RGCA/Mask/GFB, dense-C3 PAN and P2-P5 Detect loaded; only '
                'the eight C3CrossConv internal mixers freshly initialized')
        elif (architecture == 'c0_full_dual_rgca_ccall'
              and source_architecture is None
              and source_detection_task == 'hbb'
              and getattr(opt, 'detection_task', 'hbb') == 'hbb'
              and 'model.20.mhca_rgb.v.weight' in source_state
              and 'model.23.mhca_ir.project_out.weight' in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_lcafnet_to_c0_ccall_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'Original LCAFNet to C0-CCAll transfer leaves '
                    f'{len(report["unexpected_missing"])} established '
                    f'tensors uninitialized (first: {preview}).')
            initialization_summary = (
                f'original FLIR LCAFNet `{weights}` -> C0-CCAll: unchanged '
                'dual-backbone tensors, eight outer CSP projections, '
                'compatible GFB, dense-C3 PAN and P2-P5 Detect loaded; only '
                'CrossConv internals and RGCA/mask freshly initialized')
        elif (architecture == 'c0_full_dual_rgca_ccall'
              and source_architecture is None
              and len(source_yaml.get('backbone', [])) == 10
              and len(source_yaml.get('head', [])) >= 10):
            transferred, report = build_yolov5s_to_c0_ccall_state_dict(
                source_state, model_state,
                target_layer_map=model.yaml.get('pretrained_backbone_map'))
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'YOLOv5s to C0-CCAll transfer leaves '
                    f'{len(report["unexpected_missing"])} compatible '
                    f'backbone tensors uninitialized (first: {preview}).')
            initialization_summary = (
                f'canonical YOLOv5s `{weights}` -> both C0-CCAll '
                'backbones: compatible Conv/SPP(F) and outer CSP tensors '
                'loaded symmetrically; CrossConv internals, RGCA/Mask/GFB, '
                'dense-C3 PAN and P2-P5 Detect freshly initialized')
        elif (architecture == 'c0_full_dual_rgca'
              and source_architecture is None
              and source_detection_task == 'hbb'
              and getattr(opt, 'detection_task', 'hbb') == 'hbb'
              and 'model.20.cross_modal_attention.qkv.weight' in source_state
              and ('model.20.cross_modal_attention.reliability_gate.0.weight'
                   in source_state)
              and 'model.23.cross_modal_attention.out_proj.weight' in source_state
              and 'model.45.m.0.weight' in source_state
              and not any('.channel_prior.' in key for key in source_state)):
            # The converged M3FD pure-RGCA checkpoint predates the explicit
            # architecture_variant field.  Its four RGCA gates and four-scale
            # Detect layout uniquely identify the exact model.  RS-RGCA adds
            # only transient hooks, so every state tensor must transfer.
            transferred = intersect_dicts(
                source_state, model_state, exclude=[])
            missing = [key for key in model_state if key not in transferred]
            if missing:
                preview = ', '.join(missing[:5])
                raise RuntimeError(
                    f'Legacy pure-RGCA fine-tune misses {len(missing)} '
                    f'tensors (first: {preview}).')
            report = {
                'mode': 'legacy_pure_rgca_exact_finetune',
                'target_items': len(transferred),
                'expected_target_items': len(model_state),
                'missing': [],
            }
            initialization_summary = (
                f'exact legacy pure-RGCA checkpoint `{weights}`; all RGB/IR '
                'backbone, RGCA reliability gates, Mask/GFB, PAN and Detect '
                'tensors loaded')
        elif (architecture == 'c0_full_dual_becp_sod_asf'
              and source_architecture in {None, 'c0_full_dual_becp'}
              and ('model.20.cross_modal_attention.channel_prior.'
                   'evidence_head.0.weight') in source_state
              and ('model.23.cross_modal_attention.channel_prior.'
                   'evidence_head.2.bias') in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_c0_becp_to_becp_sod_asf_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'M3FD BECP to BECP-SOD transfer leaves '
                    f'{len(report["unexpected_missing"])} established tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'M3FD BECP best `{weights}` -> BECP-SOD-ASF: complete '
                'RGB/IR backbone, BECP/RGCA/mask/GFB, PAN and HBB Detect '
                'loaded; only new P2/P3 SOD-ASF tensors freshly initialized')
        elif (architecture == 'c0_full_dual_becp'
              and source_architecture is None
              and ('model.20.cross_modal_attention.channel_prior.'
                   'evidence_head.0.weight') in source_state
              and ('model.23.cross_modal_attention.channel_prior.'
                   'evidence_head.2.bias') in source_state
              and 'model.45.m.0.weight' in source_state):
            # The historical M3FD checkpoint predates architecture_variant,
            # but its BECP controller and Detect layout uniquely identify this
            # exact model.  Require every target tensor so a partial load can
            # never masquerade as a fine-tune.
            transferred = intersect_dicts(source_state, model_state, exclude=[])
            missing = [key for key in model_state if key not in transferred]
            if missing:
                preview = ', '.join(missing[:5])
                raise RuntimeError(
                    f'Legacy M3FD BECP fine-tune misses {len(missing)} '
                    f'tensors (first: {preview}).')
            report = {
                'mode': 'legacy_becp_exact_finetune',
                'target_items': len(transferred),
                'expected_target_items': len(model_state),
                'missing': [],
            }
            initialization_summary = (
                f'exact legacy M3FD BECP checkpoint `{weights}`; all RGB/IR '
                'backbone, BECP/RGCA, mask/GFB, PAN and Detect tensors loaded')
        elif (architecture == 'c0_full_dual_rgca_sod_asf'
              and 'model.20.mhca_rgb.v.weight' in source_state
              and 'model.23.mhca_ir.project_out.weight' in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_lcafnet_to_c0_sod_asf_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'Original LCAFNet to C0-SOD-ASF transfer leaves '
                    f'{len(report["unexpected_missing"])} established tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'original FLIR LCAFNet `{weights}` -> C0-SOD-ASF: complete '
                'RGB/IR backbone, PAN, HBB Detect and compatible GFB loaded; '
                'only RGCA/mask and gated P2/P3 SOD-ASF are freshly initialized')
        elif (source_architecture == 'c0_full_dual_rgca'
              and architecture == 'c0_full_dual_rgca_sod_asf'
              and source_detection_task == 'hbb'
              and getattr(opt, 'detection_task', 'hbb') == 'hbb'):
            transferred, report = build_c0_rgca_to_sod_asf_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'C0 RGCA to SOD-ASF transfer leaves '
                    f'{len(report["unexpected_missing"])} established tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'FLIR C0 RGCA best `{weights}` -> SOD-ASF: complete RGB/IR '
                'backbone, RGCA/GFB, PAN and HBB Detect loaded; only the new '
                'P2/P3 scale-sequence fusion tensors freshly initialized')
        elif (source_architecture == 'c1_dual_to_c3_shared_c4_c5'
              and architecture == 'c4_c1_deep_p4p5_d3c3_pan'):
            transferred, report = build_c1_to_deep_p4p5_state_dict(
                source_state, model_state)
            if report['missing']:
                preview = ', '.join(report['missing'][:5])
                raise RuntimeError(
                    f'C1-to-C4 transfer misses {len(report["missing"])} '
                    f'unchanged tensors (first: {preview}).')
            initialization_summary = (
                f'C1 best `{weights}`; all compatible tensors including '
                'layer-34/37 outer CSP projections loaded, only their '
                'internal D3 mixers randomly initialized')
        elif (architecture == 'c0_full_dual_rgca'
              and 'model.20.mhca_rgb.v.weight' in source_state
              and 'model.23.mhca_ir.project_out.weight' in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_full_dual_lcafnet_to_c0_rgca_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'Full-dual LCAFNet to C0 RGCA transfer leaves '
                    f'{len(report["unexpected_missing"])} non-fusion tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'full-dual LCAFNet `{weights}` -> C0 RGCA: complete RGB/IR '
                'backbone, PAN/Detect and GFB copied; RGCA/mask freshly initialized')
        elif (architecture == 'c1_5_dual_to_c4_shared_c5'
              and 'model.20.cross_modal_attention.qkv.weight' in source_state
              and 'model.22.cross_modal_attention.qkv.weight' in source_state
              and 'model.23.cross_modal_attention.qkv.weight' in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_full_dual_rgca_to_c1_5_state_dict(
                source_state, model_state)
            if report['missing']:
                preview = ', '.join(report['missing'][:5])
                raise RuntimeError(
                    'Full-dual RGCA C0 to C1.5 transfer leaves '
                    f'{len(report["missing"])} target tensors uninitialized '
                    f'(first: {preview}).')
            initialization_summary = (
                f'full-dual RGCA C0 `{weights}` -> C1.5: RGB/IR C2-C4, '
                'P2/P3/P4 fusion, and PAN/Detect copied exactly; RGB/IR '
                'C5/SPPF averaged into the shared path; no fresh tensors')
        elif (architecture == 'c1_dual_to_c3_shared_c4_c5'
              and 'model.19.cv1.conv.weight' in source_state
              and 'model.20.mhca_rgb.v.weight' in source_state
              and 'model.45.m.0.weight' in source_state):
            transferred, report = build_full_dual_lcafnet_to_c1_state_dict(
                source_state, model_state)
            if report['unexpected_missing']:
                preview = ', '.join(report['unexpected_missing'][:5])
                raise RuntimeError(
                    'Full-dual LCAFNet to C1 transfer leaves '
                    f'{len(report["unexpected_missing"])} non-fusion tensors '
                    f'uninitialized (first: {preview}).')
            initialization_summary = (
                f'full-dual LCAFNet `{weights}` -> C1: shallow modal '
                'branches and PAN/Detect copied, C4/C5 RGB/IR tensors '
                'averaged, P2/P3 GFB copied, RGCA/mask freshly initialized')
        elif source_architecture in {
                'c0_full_dual_rgca_ccall',
                'c0_full_dual_rgca_sod_asf',
                'c0_full_dual_rgca_no_reliability_mask_gfb',
                'c0_full_dual_lcafca_mask_gfb',
                'c0_full_dual_mask_gfb_only',
                'c0_full_dual_gfb_only',
                'c1_dual_to_c3_shared_c4_c5',
                'c2_c1_private_low_rank_c4_c5',
                'c3_c2_gt_private_foreground',
                'c4_c1_d3c3_slim_pan',
                'c4_c1_deep_p4p5_d3c3_pan'}:
            raise RuntimeError(
                f'Unsupported checkpoint transfer: {source_architecture} -> '
                f'{architecture}.')
        else:
            transferred, report = build_dual_stream_backbone_state_dict(
                source_state, model_state,
                target_layer_map=model.yaml.get('pretrained_backbone_map'))
            if report['missing']:
                preview = ', '.join(report['missing'][:5])
                raise RuntimeError(
                    f'Pretrained backbone transfer misses {len(report["missing"])} '
                    f'tensors (first: {preview}).')
            report['mode'] = 'canonical_backbone_mapping'
            initialization_summary = (
                f'canonical YOLOv5 `{weights}` mapped backbone only')

    model.load_state_dict(transferred, strict=False)
    for module in model.modules():
        if hasattr(module, 'last_reliability_predictions'):
            module.last_reliability_predictions = None
        if hasattr(module, 'last_rs_reliability_predictions'):
            module.last_rs_reliability_predictions = None
    model.initialization_summary = initialization_summary
    # Transient audit metadata used to guard staged structural-migration
    # protocols.  These are plain Python attributes, not checkpoint tensors.
    model.initialization_mode = report.get('mode') if report else 'resume_exact'
    model.initialization_report = report
    if report and report.get('mode') == 'canonical_backbone_mapping':
        logger.info(
            '%s backbone transfer: %d/%d tensors; RGB=%d, IR=%d, shared stages=%d, '
            'Focus->Conv6=%s.',
            architecture, report['target_items'], report['expected_target_items'],
            report['rgb_items'], report['ir_items'], report['shared_items'],
            report['converted_focus'])
    elif report and report.get('mode') == 'full_dual_lcafnet_to_c0_rgca':
        logger.info(
            '%s initialization (%s): loaded %d/%d tensors from %s; '
            'backbone=%d, PAN/Detect=%d, GFB=%d, fresh RGCA/mask=%d.',
            architecture, report['mode'], report['target_items'],
            len(model_state), weights, report['backbone_items'],
            report['head_items'], report['gfb_items'],
            len(report['fresh_fusion_items']))
    elif report and report.get('mode') == 'full_dual_lcafnet_to_c1':
        logger.info(
            '%s initialization (%s): loaded %d/%d tensors from %s; '
            'RGB=%d, IR=%d, shared C4/C5=%d, PAN/Detect=%d, GFB=%d, '
            'fresh RGCA/mask=%d.',
            architecture, report['mode'], report['target_items'],
            len(model_state), weights, report['rgb_items'], report['ir_items'],
            report['shared_items'], report['head_items'], report['gfb_items'],
            len(report['fresh_fusion_items']))
    elif report and report.get('mode') == 'full_dual_rgca_to_c1_5':
        logger.info(
            '%s initialization (%s): loaded %d/%d target tensors from %s; '
            'RGB C2-C4=%d, IR C2-C4=%d, P2/P3/P4 fusion=%d, '
            'averaged shared C5/SPPF=%d, PAN/Detect=%d, fresh=0.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights, report['rgb_items'],
            report['ir_items'], report['fusion_items'],
            report['shared_c5_items'], report['head_items'])
    elif report and report.get('mode') == 'c0_rgca_to_sod_asf':
        logger.info(
            '%s initialization (%s): loaded %d/%d established tensors from '
            '%s; unchanged C0=%d, remapped Detect=%d, fresh SOD-ASF=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['existing_items'], report['detect_items'],
            len(report['fresh_sod_items']))
    elif report and report.get('mode') == 'c0_rgca_to_ccall':
        logger.info(
            '%s initialization (%s): loaded %d/%d established tensors from '
            '%s; unchanged=%d, outer C3/CSP=%d, fresh CrossConv internals=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['unchanged_items'], report['outer_csp_items'],
            len(report['fresh_crossconv_items']))
    elif report and report.get('mode') == 'lcafnet_to_c0_ccall':
        logger.info(
            '%s initialization (%s): loaded %d/%d compatible tensors from '
            '%s; unchanged=%d, outer CSP=%d, GFB=%d, fresh CrossConv=%d, '
            'fresh RGCA/mask=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['unchanged_items'], report['outer_csp_items'],
            report['gfb_items'], len(report['fresh_crossconv_items']),
            len(report['fresh_fusion_items']))
    elif report and report.get('mode') == 'yolov5s_to_c0_ccall':
        logger.info(
            '%s initialization (%s): loaded %d/%d compatible backbone '
            'tensors from %s; RGB=%d, IR=%d, fresh CrossConv=%d, fresh '
            'fusion/PAN/Detect=%d, Focus->Conv6=%s.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['rgb_items'], report['ir_items'],
            len(report['fresh_crossconv_items']),
            len(report['fresh_detector_items']), report['converted_focus'])
    elif report and report.get('mode') == 'c0_becp_to_becp_sod_asf':
        logger.info(
            '%s initialization (%s): loaded %d/%d established tensors from '
            '%s; unchanged BECP detector=%d, remapped Detect=%d, fresh '
            'SOD-ASF=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['existing_items'], report['detect_items'],
            len(report['fresh_sod_items']))
    elif report and report.get('mode') == 'lcafnet_to_c0_sod_asf':
        logger.info(
            '%s initialization (%s): loaded %d/%d compatible tensors from '
            '%s; backbone=%d, PAN=%d, remapped Detect=%d, GFB=%d, fresh=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report['backbone_items'], report['pan_items'],
            report['detect_items'], report['gfb_items'],
            len(report['fresh_items']))
    elif report:
        logger.info(
            '%s initialization (%s): %d/%d tensors loaded from %s; '
            'excluded replacement tensors=%d.',
            architecture, report['mode'], report['target_items'],
            report['expected_target_items'], weights,
            report.get('excluded_items', 0))
    else:
        logger.info('Resumed %d/%d model tensors from %s.',
                    len(transferred), len(model_state), weights)
    return model, checkpoint


def _build_optimizer(model, hyp, total_batch_size, optimizer_name='SGD',
                     nominal_batch_size=64):
    nbs = nominal_batch_size
    accumulate = max(round(nbs / total_batch_size), 1)
    weight_decay = hyp['weight_decay'] * total_batch_size * accumulate / nbs
    gate_lr_ratio = float(hyp.get('rs_reliability_gate_lr_ratio', 1.0))
    if gate_lr_ratio <= 0:
        raise ValueError('rs_reliability_gate_lr_ratio must be positive.')
    split_gate_groups = not math.isclose(gate_lr_ratio, 1.0)
    no_decay, decay, biases = [], [], []
    gate_no_decay, gate_decay, gate_biases = [], [], []
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f'{module_name}.{parameter_name}' if module_name else parameter_name
            is_rs_gate = split_gate_groups and '.reliability_gate.' in full_name
            target_no_decay = gate_no_decay if is_rs_gate else no_decay
            target_decay = gate_decay if is_rs_gate else decay
            target_biases = gate_biases if is_rs_gate else biases
            if parameter_name == 'bias':
                target_biases.append(parameter)
            elif isinstance(module, (
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.SyncBatchNorm)):
                target_no_decay.append(parameter)
            elif is_attention_control_parameter(full_name):
                target_no_decay.append(parameter)
            else:
                target_decay.append(parameter)

    groups = []
    def append_group(parameters, decay_value, role, lr_ratio,
                     warmup_start_lr):
        if parameters:
            groups.append({
                'params': parameters,
                'weight_decay': decay_value,
                'role': role,
                'lr': hyp['lr0'] * lr_ratio,
                'warmup_start_lr': warmup_start_lr,
            })

    append_group(no_decay, 0.0, 'no_decay', 1.0, 0.0)
    append_group(decay, weight_decay, 'decay', 1.0, 0.0)
    append_group(biases, 0.0, 'bias', 1.0, hyp['warmup_bias_lr'])
    append_group(gate_no_decay, 0.0, 'rs_gate_no_decay', gate_lr_ratio, 0.0)
    append_group(gate_decay, weight_decay, 'rs_gate_decay', gate_lr_ratio, 0.0)
    append_group(
        gate_biases, 0.0, 'rs_gate_bias', gate_lr_ratio,
        hyp['warmup_bias_lr'])
    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            groups, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            groups, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            groups, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        raise ValueError('optimizer must be SGD, Adam, or AdamW')
    grouped = sum(p.numel() for group in groups for p in group['params'])
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if grouped != expected:
        raise RuntimeError(f'Optimizer grouped {grouped} params, expected {expected}.')
    logger.info(
        'optimizer: %s base_lr=%g, RS-gate_lr=%g, weight_decay=%g; '
        'nominal_batch=%d, accumulate=%d, effective_batch=%d; groups=%s',
        type(optimizer).__name__, hyp['lr0'], hyp['lr0'] * gate_lr_ratio,
        weight_decay, nbs, accumulate, total_batch_size * accumulate,
        [(group['role'], len(group['params'])) for group in groups])
    return optimizer, accumulate


def _record_experiment_identity(save_dir, opt, model, hyp):
    params = sum(parameter.numel() for parameter in model.parameters())
    architecture = model.yaml.get('architecture_variant')
    full_dual_architectures = {
        'c0_full_dual_becp', 'c0_full_dual_becp_sod_asf',
        'c0_full_dual_rgca', 'c0_full_dual_rc_rgca_v2',
        'c0_full_dual_rgca_no_reliability_mask_gfb',
        'c0_full_dual_rgca_ccall',
        'c0_full_dual_rgca_sod_asf',
        'c0_full_dual_lcafca_mask_gfb',
        'c0_full_dual_mask_gfb_only', 'c0_full_dual_gfb_only'}
    if (architecture == 'c0_full_dual_rgca'
            and bool(hyp.get('llvip_lcafnet_to_rgca_finetune', False))):
        title = '# LLVIP C0-RGCA -- transfer from original LCAFNet at 1024\n\n'
        replacement = (
            '- Original LCAFNet_LLVIP.pt supplies both modal backbones, '
            'PAN/Detect and compatible GFB tensors.\n'
            '- The new RGCA and foreground-mask tensors retain their verified '
            'neutral initialization.\n'
            '- RGB/IR backbones are frozen; fusion, PAN/FPN and Detect are '
            'fine-tuned at 1024x1024.\n'
            '- The run uses the locally validated SGD/mild-augmentation '
            'protocol and selects best.pt strictly by mAP50.\n')
    elif (architecture == 'c0_full_dual_rgca'
            and bool(hyp.get('llvip_rgca_same_architecture_finetune', False))):
        title = '# LLVIP C0-RGCA -- conservative 1024 fine-tuning\n\n'
        replacement = (
            '- The converged LLVIP RGCA+Mask+GFB best.pt is loaded exactly.\n'
            '- RGB/IR training and validation resolution is 1024x1024.\n'
            '- The full model is fine-tuned with a 2e-5 learning rate.\n'
            '- Geometry, color jitter and Mosaic are disabled to preserve '
            'paired alignment; only synchronized horizontal flipping remains.\n'
            '- All augmentation closes for the final five epochs and best.pt '
            'is selected strictly by mAP50.\n')
    elif (architecture == 'c0_full_dual_rgca'
            and bool(hyp.get('flir_lcafnet_to_rgca_staged_finetune', False))):
        title = '# FLIR C0-RGCA -- staged migration from original LCAFNet\n\n'
        replacement = (
            '- The trained original three-class LCAFNet_FLIR detector supplies '
            'both modal backbones, PAN/Detect and compatible GFB tensors.\n'
            '- RGCA and foreground-mask tensors are freshly initialized.\n'
            '- Stage 1 trains only four RGCA/Mask/GFB fusion blocks.\n'
            '- Stage 2 jointly trains fusion, PAN/FPN and Detect while both '
            'modal backbones remain frozen.\n'
            '- best.pt is protected at initialization and selected by mAP50.\n')
    elif (architecture == 'c0_full_dual_rgca'
            and bool(hyp.get('rgca_map50_recovery', False))):
        title = '# M3FD RS-RGCA -- mAP50 recovery stage\n\n'
        replacement = (
            '- The reliability-supervised checkpoint is used as initialization.\n'
            '- RGB/IR backbones and all four RGCA+Mask+GFB blocks are frozen.\n'
            '- Only PAN/FPN and Detect are optimized on clean aligned pairs.\n'
            '- Reliability corruption and auxiliary gate loss are disabled.\n'
            '- Mosaic is disabled and best.pt is selected strictly by mAP50.\n')
    elif (architecture == 'c0_full_dual_rgca'
            and bool(hyp.get('rs_reliability_supervision', False))):
        title = '# M3FD RS-RGCA -- reliability-supervised original RGCA\n\n'
        replacement = (
            '- The converged original no-prior RGCA forward path, foreground '
            'Mask and GFB are retained exactly.\n'
            '- Existing directional spatial gates receive mild asymmetric '
            'source-quality/correspondence supervision.\n'
            '- Auxiliary gate descriptors are detached, so reliability loss '
            'does not backpropagate into RGB/IR backbones or QKV features.\n'
            f'- Gate-only adaptation lasts '
            f'{int(hyp.get("rs_gate_only_epochs", 0))} epochs; base/gate LR '
            f'ratio is 1:{float(hyp.get("rs_reliability_gate_lr_ratio", 1.0)):g}.\n')
    elif architecture == 'c0_full_dual_gfb_only':
        title = '# M3FD C0-GFB-only -- strict fusion ablation\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 contain no cross-attention or modal enhancement.\n'
            '- Raw aligned RGB/IR features use the original GFB only.\n'
            '- Foreground Mask, RGCA reliability/local branches and frequency '
            'operators are absent.\n'
            '- All six PAN aggregation blocks retain dense C3.\n')
    elif architecture == 'c0_full_dual_mask_gfb_only':
        title = '# M3FD C0-Mask-GFB-only -- attention-free ablation\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 contain no cross-attention or modal enhancement.\n'
            '- Raw aligned RGB/IR features use foreground Mask + GFB only.\n'
            '- RGCA reliability/local branches and frequency operators are absent.\n'
            '- All six PAN aggregation blocks retain dense C3.\n')
    elif architecture == 'c0_full_dual_lcafca_mask_gfb':
        title = '# M3FD C0-LCAFCA-Mask-GFB -- attention ablation\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 use original LCAFNet bidirectional cross-attention '
            'without RGCA reliability gating.\n'
            '- Foreground Mask and mask-guided GFB remain unchanged.\n'
            '- All six PAN aggregation blocks retain dense C3.\n')
    elif architecture == 'c0_full_dual_rc_rgca_v2':
        title = '# M3FD RC-RGCA v2 -- reliability-conditioned fusion\n\n'
        replacement = (
            '- P2/P3/P4/P5 explicitly estimate RGB quality, IR quality and '
            'cross-modal correspondence from stopped-gradient descriptors.\n'
            '- Directional reliability conditions source K/V tensors and a '
            'null no-transfer key inside attention.\n'
            '- The same reliability state guides the final Mask-GFB fusion; '
            'training uses supervised asymmetric modality degradation.\n')
    elif architecture == 'c0_full_dual_rgca_no_reliability_mask_gfb':
        title = '# M3FD C0-RGCA-NoReliability-Mask-GFB -- gate ablation\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 retain shared bidirectional cross-covariance '
            'attention, complementary local values and global/local mixing.\n'
            '- Learned spatial reliability networks are absent; both '
            'cross-modal transfer multipliers are the identity value 1.\n'
            '- Foreground Mask, mask-guided GFB, dense-C3 PAN and P2-P5 '
            'Detect remain unchanged.\n')
    elif architecture == 'c0_full_dual_becp_sod_asf':
        title = '# M3FD C0-BECP-SOD-ASF -- small-object fine-tuning\n\n'
        replacement = (
            '- RGB and IR retain trained independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 retain trained BECP + RGCA + mask + GFB fusion.\n'
            '- SOD-YOLO scale-sequence fusion residually enriches only the '
            'P2/P3 Detect inputs; original PAN remains intact.\n')
    elif architecture == 'c0_full_dual_becp':
        title = '# M3FD C0-BECP -- full dual RGB/IR fine-tuning\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 use Bayesian evidential correspondence prior + '
            'RGCA + foreground mask + GFB.\n'
            '- All six PAN aggregation blocks retain dense C3.\n')
    elif architecture == 'c0_full_dual_rgca_sod_asf':
        title = '# C0-SOD-ASF -- FLIR small-target optimization\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 use no-prior RGCA + foreground mask + GFB.\n'
            '- Dense C3 PAN is unchanged; SOD-YOLO scale-sequence fusion '
            'residually enriches only the P2/P3 Detect inputs.\n')
    elif architecture == 'c0_full_dual_rgca_ccall':
        title = '# C0-CCAll -- dual-backbone YOLO-TLA C3CrossConv\n\n'
        replacement = (
            '- All four C3 stages in each RGB/IR backbone (eight total) use '
            'C3CrossConv: the C3 CSP shell is retained and each internal '
            'Bottleneck is replaced by residual 1x3 -> 3x1 CrossConv.\n'
            '- P2/P3/P4/P5 retain no-prior RGCA + foreground mask + GFB.\n'
            '- All six PAN aggregation blocks remain the original dense C3; '
            'the P2-P5 four-scale Detect head is unchanged.\n')
    elif architecture == 'c0_full_dual_rgca':
        title = '# C0 -- complete dual RGB/IR backbone\n\n'
        replacement = (
            '- RGB and IR retain independent C2/C3/C4/C5/SPPF extraction.\n'
            '- P2/P3/P4/P5 use no-prior RGCA + foreground mask + GFB.\n'
            '- All six PAN aggregation blocks retain dense C3.\n')
    elif architecture == 'c4_c1_deep_p4p5_d3c3_pan':
        title = '# C4 conservative -- C1 + deep P4/P5 D3C3\n\n'
        replacement = (
            '- Dense C3 is retained at top-down layers 20/24 and P2/P3 '
            'layers 28/31.\n'
            '- Only PAN output layers 34(P4) and 37(P5) use residual D3C3.\n')
    elif architecture == 'c4_c1_d3c3_slim_pan':
        title = '# C4 historical full replacement -- C1 + D3C3 Slim-PAN\n\n'
        replacement = '- All six PAN C3 blocks use residual D3C3.\n'
    elif architecture == 'c1_5_dual_to_c4_shared_c5':
        title = '# C1.5 M3FD -- dual through C4, shared C5\n\n'
        replacement = (
            '- RGB and IR retain independent C4 extraction.\n'
            '- P2/P3/P4 use no-prior RGCA + foreground mask + GFB.\n'
            '- Only C5/SPPF is shared; all six PAN blocks retain dense C3.\n')
    else:
        title = '# C1 lightweight transfer/fine-tune\n\n'
        replacement = '- All six PAN aggregation blocks use dense C3.\n'
    d3_description = (
        f'- D3 large depthwise kernel: {opt.d3_kernel_size}x{opt.d3_kernel_size}.\n'
        if architecture in {
            'c4_c1_deep_p4p5_d3c3_pan', 'c4_c1_d3c3_slim_pan'} else '')
    identity = ''.join((
        title,
        ('- RGB and IR use independent C2/C3/C4/C5 feature extraction.\n'
         if architecture in full_dual_architectures else
         '- RGB and IR use independent C2/C3/C4 feature extraction.\n'
         if architecture == 'c1_5_dual_to_c4_shared_c5' else
         '- RGB and IR use independent C2/C3 feature extraction.\n'),
        ('- P2/P3/P4/P5 are fused independently for the PAN/Detect path.\n'
         if architecture in full_dual_architectures else
         '- P4 fusion is the merge point before shared C5/SPPF.\n'
         if architecture == 'c1_5_dual_to_c4_shared_c5' else
         '- C2 fusion is the P2 skip; C3 fusion is the single-stream merge point.\n'),
        '- P2/P3/P4/P5 Detect inputs are retained.\n',
        replacement,
        d3_description,
        '- No private bypass, foreground auxiliary loss, RepDown, Mamba, or '
        'feature distillation.\n',
        f'- Parameters: {params:,}\n',
        f'- Trainable parameters: '
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n',
        f'- Trainable scope: {opt.trainable_scope}\n',
        f'- Detection task: {getattr(opt, "detection_task", "hbb")}\n',
        ('- OBB target: native RGB DOTA quadrilaterals; 180-bin CSL angle; '
         'exact rotated-IoU validation.\n'
         if getattr(opt, 'detection_task', 'hbb') == 'obb' else ''),
        f'- Initialization: {getattr(model, "initialization_summary", opt.weights)}\n',
        f'- Dataset: `{opt.data}`\n',
        f'- Config: `{opt.cfg}`\n',
        f'- Training: {opt.epochs} epochs, physical batch {opt.batch_size}, '
        f'effective batch {opt.nominal_batch_size}, '
        f'image size {opt.img_size}, seed {opt.seed}, optimizer {opt.optimizer}, '
        f'lr0 {hyp["lr0"]}, LR schedule {opt.lr_schedule}, '
        f'LR drop epochs/factor {opt.lr_drop_epochs}/{opt.lr_drop_factor}, '
        f'best metric {opt.best_metric}\n'))
    (save_dir / 'EXPERIMENT_IDENTITY.md').write_text(identity)


def _save_epoch_diagnostics(model, epoch, attention_file, fusion_file):
    attention_rows = collect_cross_modal_attention_diagnostics(model)
    if attention_rows:
        with attention_file.open('a') as stream:
            for row in attention_rows:
                stream.write(
                    f'{epoch},{row[0]},' +
                    ','.join(f'{value:.8g}' for value in row[1:]) + '\n')
        attention_std = [x for row in attention_rows for x in (row[3], row[4])]
        reliability_std = [x for row in attention_rows for x in (row[6], row[8])]
        logger.info(
            'RGCA diagnostics: attention std %.3e..%.3e, reliability std '
            '%.3e..%.3e across %d fusion levels.',
            min(attention_std), max(attention_std), min(reliability_std),
            max(reliability_std), len(attention_rows))

    fusion_rows = collect_fusion_gate_diagnostics(model)
    if fusion_rows:
        with fusion_file.open('a') as stream:
            for row in fusion_rows:
                stream.write(
                    f'{epoch},{row[0]},' +
                    ','.join(f'{value:.8g}' for value in row[1:]) + '\n')
        mask_std = [row[2] for row in fusion_rows]
        gate_std = [row[6] for row in fusion_rows]
        logger.info(
            'Mask/GFB diagnostics: mask std %.3e..%.3e, RGB gate std '
            '%.3e..%.3e across %d fusion levels.',
            min(mask_std), max(mask_std), min(gate_std), max(gate_std),
            len(fusion_rows))

def train_rgb_ir(hyp, opt, device, tb_writer=None):
    """Train the current RGB/IR detector."""
    save_dir = Path(opt.save_dir)
    rank = opt.global_rank
    cuda = device.type != 'cpu'
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last, best = wdir / 'last.pt', wdir / 'best.pt'
    attention_file, fusion_file, reliability_file = (
        _write_diagnostics_headers(save_dir, rank))

    with (save_dir / 'hyp.yaml').open('w') as stream:
        yaml.safe_dump(hyp, stream, sort_keys=False)
    with (save_dir / 'opt.yaml').open('w') as stream:
        yaml.safe_dump(vars(opt), stream, sort_keys=False)

    init_seeds(int(opt.seed) + max(rank, 0), deterministic=opt.strict_determinism)
    torch.backends.cudnn.benchmark = bool(
        cuda and opt.cudnn_benchmark and not opt.multi_scale
        and not opt.strict_determinism)
    torch.backends.cudnn.deterministic = bool(opt.strict_determinism)

    with open(opt.data) as stream:
        data_dict = yaml.safe_load(stream)
    detection_task = str(data_dict.get('task', 'hbb')).lower()
    if detection_task not in ('hbb', 'obb'):
        raise ValueError(f'Unsupported dataset task {detection_task!r}.')
    opt.detection_task = detection_task
    # detection_task is inferred from the dataset rather than exposed as an
    # error-prone CLI switch, so persist the resolved value in opt.yaml.
    if rank in (-1, 0):
        with (save_dir / 'opt.yaml').open('w') as stream:
            yaml.safe_dump(vars(opt), stream, sort_keys=False)
    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls else data_dict['names']
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if len(names) != nc:
        raise ValueError(f'Dataset declares nc={nc} but has {len(names)} names.')

    model, checkpoint = _load_active_model(opt, hyp, device, nc, rank)
    rc_rgca_v2_active = (
        model.yaml.get('architecture_variant') == 'c0_full_dual_rc_rgca_v2')
    rs_rgca_active = (
        model.yaml.get('architecture_variant') == 'c0_full_dual_rgca'
        and bool(hyp.get('rs_reliability_supervision', False)))
    rgca_map50_recovery_active = (
        model.yaml.get('architecture_variant') == 'c0_full_dual_rgca'
        and bool(hyp.get('rgca_map50_recovery', False)))
    flir_lcafnet_migration_active = bool(
        hyp.get('flir_lcafnet_to_rgca_staged_finetune', False))
    llvip_rgca_finetune_active = bool(
        hyp.get('llvip_rgca_same_architecture_finetune', False))
    llvip_lcafnet_migration_active = bool(
        hyp.get('llvip_lcafnet_to_rgca_finetune', False))
    flir_fusion_only_epochs = 0
    if llvip_lcafnet_migration_active:
        if getattr(model, 'initialization_mode', None) != (
                'full_dual_lcafnet_to_c0_rgca'):
            raise RuntimeError(
                'LLVIP LCAFNet migration requires original LCAFNet -> C0 RGCA '
                'structural transfer, got initialization mode '
                f'{getattr(model, "initialization_mode", None)!r}.')
        report = getattr(model, 'initialization_report', None) or {}
        if (report.get('target_items') != 630
                or len(report.get('fresh_fusion_items', [])) != 128):
            raise RuntimeError(
                'Unexpected LLVIP migration coverage: expected 630 transferred '
                'and 128 fresh fusion state items, got '
                f'{report.get("target_items")} and '
                f'{len(report.get("fresh_fusion_items", []))}.')
        if nc != 1 or list(names) != ['person']:
            raise RuntimeError(
                'LLVIP migration requires nc=1 and names=[person], got '
                f'nc={nc}, names={names}.')
        if list(opt.img_size) != [1024, 1024]:
            raise ValueError('LLVIP migration requires --img-size 1024 1024.')
        if opt.trainable_scope != 'fusion_head':
            raise ValueError(
                'LLVIP migration requires --trainable-scope fusion_head.')
        if opt.best_metric != 'map50':
            raise ValueError('LLVIP migration requires --best-metric map50.')
        if (rs_rgca_active or rgca_map50_recovery_active
                or flir_lcafnet_migration_active or llvip_rgca_finetune_active):
            raise ValueError(
                'LLVIP LCAFNet migration cannot be combined with another '
                'reliability/migration protocol.')
        logger.info(
            'LLVIP LCAFNet->C0-RGCA fine-tuning enabled: 630/758 state '
            'items transferred, 128 fresh RGCA/mask items, 1024x1024, '
            'fusion+PAN+Detect trainable, mAP50 selection.')
    if llvip_rgca_finetune_active:
        if getattr(model, 'initialization_mode', None) != (
                'same_architecture_finetune'):
            raise RuntimeError(
                'LLVIP RGCA fine-tuning requires an exact same-architecture '
                f'checkpoint, got {getattr(model, "initialization_mode", None)!r}.')
        if nc != 1 or list(names) != ['person']:
            raise RuntimeError(
                'LLVIP RGCA fine-tuning requires nc=1 and names=[person], got '
                f'nc={nc}, names={names}.')
        if list(opt.img_size) != [1024, 1024]:
            raise ValueError(
                'LLVIP RGCA fine-tuning requires --img-size 1024 1024.')
        if opt.trainable_scope != 'all':
            raise ValueError(
                'LLVIP same-architecture fine-tuning requires '
                '--trainable-scope all.')
        if opt.best_metric != 'map50':
            raise ValueError(
                'LLVIP RGCA fine-tuning requires --best-metric map50.')
        if float(hyp.get('mosaic', 0.0)) != 0.0:
            raise ValueError('LLVIP RGCA fine-tuning requires mosaic=0.')
        if (rs_rgca_active or rgca_map50_recovery_active
                or flir_lcafnet_migration_active):
            raise ValueError(
                'LLVIP same-architecture fine-tuning cannot be combined with '
                'another reliability/migration protocol.')
        logger.info(
            'LLVIP exact C0-RGCA fine-tuning enabled: 1024x1024, all tensors '
            'loaded, full model low-LR, aligned augmentation, mAP50 selection.')
    if flir_lcafnet_migration_active:
        if getattr(model, 'initialization_mode', None) != (
                'full_dual_lcafnet_to_c0_rgca'):
            raise RuntimeError(
                'Staged FLIR migration requires original LCAFNet -> C0 RGCA '
                'structural transfer, got initialization mode '
                f'{getattr(model, "initialization_mode", None)!r}.')
        if nc != 3 or list(names) != ['person', 'car', 'bicycle']:
            raise RuntimeError(
                'Staged FLIR migration requires the verified three-class FLIR '
                f'dataset, got nc={nc}, names={names}.')
        if rs_rgca_active or rgca_map50_recovery_active:
            raise ValueError(
                'FLIR structural migration cannot be combined with an RS '
                'reliability-supervision/recovery protocol.')
        if opt.trainable_scope != 'fusion_head':
            raise ValueError(
                'Staged FLIR migration requires --trainable-scope fusion_head.')
        if opt.best_metric != 'map50':
            raise ValueError(
                'Staged FLIR migration requires --best-metric map50.')
        configured_fusion_epochs = int(hyp.get('flir_fusion_only_epochs', 0))
        if configured_fusion_epochs <= 0:
            raise ValueError('flir_fusion_only_epochs must be positive.')
        flir_fusion_only_epochs = min(configured_fusion_epochs, opt.epochs)
        if configured_fusion_epochs not in opt.lr_drop_epochs:
            raise ValueError(
                'The FLIR stage transition must also appear in '
                '--lr-drop-epochs.')
        logger.info(
            'Staged FLIR LCAFNet->RGCA migration enabled: epochs 0--%d '
            'fusion-only, then fusion+PAN+Detect; RGB/IR backbones remain '
            'frozen; mAP50 selection.', flir_fusion_only_epochs - 1)
    if bool(hyp.get('rgca_map50_recovery', False)):
        if not rgca_map50_recovery_active:
            raise RuntimeError(
                'RGCA mAP50 recovery requires c0_full_dual_rgca, got '
                f'{model.yaml.get("architecture_variant")!r}.')
        if rs_rgca_active:
            raise ValueError(
                'mAP50 recovery must not enable RS reliability corruption/loss.')
        if opt.trainable_scope != 'pan_head':
            raise ValueError(
                'RGCA mAP50 recovery requires --trainable-scope pan_head.')
        if opt.best_metric != 'map50':
            raise ValueError(
                'RGCA mAP50 recovery requires --best-metric map50.')
        if float(hyp.get('mosaic', 0.0)) != 0.0:
            raise ValueError('RGCA mAP50 recovery requires mosaic=0.')
        logger.info(
            'RGCA mAP50 recovery enabled: frozen layers 0--23, trainable '
            'PAN/Detect layers 24--45, no reliability corruption/auxiliary loss.')
    if rc_rgca_v2_active:
        required_reliability_keys = (
            'reliability_corruption_probability',
            'reliability_aux_gain')
        missing_keys = [key for key in required_reliability_keys if key not in hyp]
        if missing_keys:
            raise KeyError(
                f'RC-RGCA v2 hyperparameters are missing {missing_keys}.')
        logger.info(
            'RC-RGCA v2 reliability curriculum enabled: corruption=%.3f, '
            'pair=%.3f, dropout=%.3f, auxiliary gain=%.3f.',
            hyp['reliability_corruption_probability'],
            hyp.get('reliability_pair_corruption_probability', 0.0),
            hyp.get('reliability_modality_dropout_probability', 0.0),
            hyp['reliability_aux_gain'])
    rs_reliability_file = save_dir / 'rs_reliability_calibration.csv'
    rs_gate_only_epochs = 0
    if rs_rgca_active:
        required_rs_keys = (
            'rs_reliability_corruption_probability',
            'rs_reliability_aux_gain')
        missing_keys = [key for key in required_rs_keys if key not in hyp]
        if missing_keys:
            raise KeyError(
                f'RS-RGCA hyperparameters are missing {missing_keys}.')
        configured_gate_only_epochs = int(
            hyp.get('rs_gate_only_epochs', 0))
        if configured_gate_only_epochs < 0:
            raise ValueError('rs_gate_only_epochs must be non-negative.')
        rs_gate_only_epochs = min(configured_gate_only_epochs, opt.epochs)
        if (rank in (-1, 0)
                and configured_gate_only_epochs > opt.epochs):
            logger.warning(
                'RS-RGCA gate-only stage shortened from %d to %d epochs '
                'for this short run.',
                configured_gate_only_epochs, rs_gate_only_epochs)
        if rs_gate_only_epochs and opt.trainable_scope != 'all':
            raise ValueError(
                'RS-RGCA staged gate adaptation requires trainable_scope=all.')
        rs_modules = enable_rs_rgca_reliability_supervision(model)
        if rank in (-1, 0) and not rs_reliability_file.exists():
            rs_reliability_file.write_text(
                'epoch,target_q_rgb,target_q_ir,target_correspondence,'
                'pred_r_ir_to_rgb,pred_r_rgb_to_ir,calibration_loss,'
                'ranking_loss,reliability_aux_loss\n')
        logger.info(
            'RS-RGCA supervision enabled on %s: corruption=%.3f, pair=%.3f, '
            'dropout=%.3f, auxiliary gain=%.3f, gate-only epochs=%d, '
            'gate LR ratio=%g.',
            rs_modules, hyp['rs_reliability_corruption_probability'],
            hyp.get('rs_reliability_pair_corruption_probability', 0.0),
            hyp.get('rs_reliability_modality_dropout_probability', 0.0),
            hyp['rs_reliability_aux_gain'], rs_gate_only_epochs,
            hyp.get('rs_reliability_gate_lr_ratio', 1.0))
    fusion_modules = configure_trainable_scope(model, opt.trainable_scope)
    _record_experiment_identity(save_dir, opt, model, hyp)
    logger.info(
        'Trainable scope=%s, modules=%s, parameters=%s/%s',
        opt.trainable_scope, fusion_modules or ['all'],
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}',
        f'{sum(p.numel() for p in model.parameters()):,}')

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    train_rgb, train_ir = data_dict['train_rgb'], data_dict['train_ir']
    val_rgb, val_ir = data_dict['val_rgb'], data_dict['val_ir']
    labels_list = (None if detection_task == 'obb'
                   else label_names_from_image_source(val_rgb))

    optimizer, accumulate = _build_optimizer(
        model, hyp, opt.total_batch_size, opt.optimizer,
        opt.nominal_batch_size)
    if opt.lr_schedule == 'linear':
        denominator = max(opt.epochs - 1, 1)
        base_lr_function = lambda epoch: (
            (1.0 - epoch / denominator) * (1.0 - hyp['lrf']) + hyp['lrf'])
    elif opt.lr_schedule == 'cosine':
        base_lr_function = one_cycle(1, hyp['lrf'], opt.epochs)
    else:
        raise ValueError("lr_schedule must be 'linear' or 'cosine'.")
    lr_function = lambda epoch: (
        base_lr_function(epoch) * opt.lr_drop_factor ** sum(
            epoch >= milestone for milestone in opt.lr_drop_epochs))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_function)
    logger.info(
        'Learning-rate schedule: %s decay over %d epochs; milestones %s each '
        'apply a cumulative x%g multiplier.',
        opt.lr_schedule, opt.epochs, opt.lr_drop_epochs, opt.lr_drop_factor)
    ema = ModelEMA(model) if rank in (-1, 0) else None

    start_epoch, best_fitness = 0, 0.0
    if opt.resume:
        if checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness = float(checkpoint.get('best_fitness', 0.0))
        if ema and checkpoint.get('ema') is not None:
            ema.ema.load_state_dict(checkpoint['ema'].float().state_dict())
            ema.updates = int(checkpoint.get('updates', 0))
        start_epoch = int(checkpoint['epoch']) + 1
        if start_epoch >= opt.epochs:
            raise RuntimeError(
                f'{opt.weights} already reached epoch {start_epoch - 1}; '
                f'target epochs={opt.epochs}.')
    del checkpoint

    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if opt.sync_bn and cuda and rank != -1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    dataloader_factory = (create_dataloader_obb_rgb_ir
                          if detection_task == 'obb'
                          else create_dataloader_rgb_ir)
    trainloader, dataset = dataloader_factory(
        train_rgb, train_ir, imgsz, opt.batch_size, gs, opt,
        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
        rank=rank, world_size=opt.world_size, workers=opt.workers,
        image_weights=False, quad=False, prefix=colorstr('train: '),
        mutable_augmentations=(
            opt.close_augmentations > 0 or opt.close_mosaic > 0))
    augmentations_closed = (
        opt.close_augmentations > 0
        and start_epoch >= opt.epochs - opt.close_augmentations)
    if augmentations_closed:
        dataset.augment = False
        dataset.mosaic = False
        logger.info(
            'All training augmentations are disabled from resumed epoch %d.',
            start_epoch)
    nb = len(trainloader)
    labels = np.concatenate(dataset.labels, 0)
    if labels[:, 0].max() >= nc:
        raise ValueError('A training label class index exceeds dataset nc.')

    val_batch_size = min(opt.batch_size, 16)
    if rank in (-1, 0):
        testloader, _ = dataloader_factory(
            val_rgb, val_ir, imgsz_test, val_batch_size, gs, opt,
            hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
            rank=-1, world_size=opt.world_size, workers=opt.workers,
            pad=0.5, prefix=colorstr('val: '))
        if not opt.resume and detection_task == 'hbb':
            plot_labels(labels, names, save_dir, {'wandb': None})
            if tb_writer:
                tb_writer.add_histogram('classes', torch.tensor(labels[:, 0]), 0)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()

    if cuda and rank != -1:
        model = DDP(
            model, device_ids=[opt.local_rank], output_device=opt.local_rank,
            find_unused_parameters=any(
                isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    hyp['box'] *= 3.0 / nl
    hyp['cls'] *= nc / 80.0 * 3.0 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3.0 / nl
    if detection_task == 'obb':
        hyp['theta'] *= 3.0 / nl
    hyp['label_smoothing'] = 0.0
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda and opt.amp)
    compute_loss = (ComputeLossOBB(model) if detection_task == 'obb'
                    else ComputeLoss(model))
    warmup_iters = compute_warmup_iterations(
        hyp['warmup_epochs'], nb, 0, opt.epochs - start_epoch, 0.2,
        resumed=bool(opt.resume))
    logger.info(
        'Starting %s training: %d epochs, batch=%d, image=%d/%d, workers=%d, '
        'warmup=%d iterations, AMP=%s, output=%s',
        model.yaml.get('architecture_variant'), opt.epochs,
        opt.total_batch_size, imgsz, imgsz_test,
        trainloader.num_workers, warmup_iters, bool(cuda and opt.amp), save_dir)

    if (rank in (-1, 0) and opt.validate_initialization and not opt.resume):
        ema.update_attr(
            model, include=[
                'yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        validation_module = test_obb if detection_task == 'obb' else test
        initial_results, _, _, _ = validation_module.test(
            data_dict, batch_size=val_batch_size, imgsz=imgsz_test,
            conf_thres=opt.obb_val_conf_thres,
            iou_thres=opt.obb_val_iou_thres,
            model=ema.ema, single_cls=opt.single_cls,
            dataloader=testloader, save_dir=save_dir,
            save_txt=False, save_conf=False, verbose=False, plots=False,
            wandb_logger=None, compute_loss=compute_loss,
            is_coco=opt.data.endswith('coco.yaml'), opt=opt,
            labels_list=labels_list,
            **({
                'max_det': opt.obb_val_max_det,
                'max_nms': opt.obb_val_max_nms,
                'multi_label': opt.obb_val_multi_label,
            } if detection_task == 'obb' else {}))
        if opt.best_metric == 'map50':
            best_fitness = float(initial_results[6])
        else:
            best_fitness = float(np.asarray(
                fitness(np.asarray(initial_results).reshape(1, -1))).item())
        initial_checkpoint = {
            'epoch': -1,
            'best_fitness': best_fitness,
            'model': keep_force_fp32_modules(deepcopy(
                model.module if is_parallel(model) else model).half()),
            'ema': keep_force_fp32_modules(deepcopy(ema.ema).half()),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(initial_checkpoint, best)
        del initial_checkpoint
        logger.info(
            'Initialization protected as best.pt before epoch 0: '
            'mAP50=%.6f, mAP50:95=%.6f, selection fitness=%.6f.',
            initial_results[6], initial_results[7], best_fitness)
        if cuda:
            torch.cuda.empty_cache()

    results = (0.0,) * 7
    maps = np.zeros(nc)
    mr_result = [0.0] * 10
    batches_seen = start_epoch * nb
    started = time.time()

    for epoch in range(start_epoch, opt.epochs):
        rs_gate_only_active = bool(
            rs_rgca_active and epoch < rs_gate_only_epochs)
        if rs_rgca_active and (epoch == start_epoch
                               or epoch == rs_gate_only_epochs):
            gate_parameter_count = set_rs_rgca_gate_only_stage(
                model, rs_gate_only_active)
            logger.info(
                'RS-RGCA stage at epoch %d: %s (%s gate parameters).',
                epoch,
                ('reliability-gate-only adaptation'
                 if rs_gate_only_active else 'full-model fine-tuning'),
                f'{gate_parameter_count:,}')
        flir_fusion_only_active = bool(
            flir_lcafnet_migration_active
            and epoch < flir_fusion_only_epochs)
        if (flir_lcafnet_migration_active
                and (epoch == start_epoch
                     or epoch == flir_fusion_only_epochs)):
            flir_trainable_parameters = set_flir_lcafnet_migration_stage(
                model, flir_fusion_only_active)
            logger.info(
                'FLIR migration stage at epoch %d: %s (%s trainable '
                'parameters).', epoch,
                ('new RGCA/Mask/GFB fusion only'
                 if flir_fusion_only_active else
                 'fusion + PAN/FPN + Detect joint fine-tuning'),
                f'{flir_trainable_parameters:,}')
        model.train()
        if (opt.trainable_scope != 'all' or rs_gate_only_active
                or flir_fusion_only_active):
            keep_frozen_batch_norm_eval(model)
        if rank in (-1, 0):
            enable_cross_modal_attention_tracking(model)
            enable_fusion_gate_tracking(model)

        if (opt.close_augmentations > 0
                and epoch >= opt.epochs - opt.close_augmentations
                and not augmentations_closed):
            dataset.augment = False
            dataset.mosaic = False
            augmentations_closed = True
            logger.info(
                'All training augmentations disabled for the final %d epochs.',
                opt.close_augmentations)

        if (opt.close_mosaic > 0 and epoch >= opt.epochs - opt.close_mosaic
                and getattr(dataset, 'mosaic', False)):
            dataset.mosaic = False
            logger.info('Mosaic disabled for the final %d epochs.', opt.close_mosaic)

        if hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)
        progress = enumerate(trainloader)
        if rank in (-1, 0):
            progress = tqdm(progress, total=nb)
        mean_loss = torch.zeros(4, device=device)
        reliability_epoch_sum = torch.zeros(11)
        reliability_epoch_count = 0
        rs_reliability_epoch_sum = torch.zeros(8)
        rs_reliability_epoch_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (images, targets, paths, _) in progress:
            iteration = batches_seen
            batches_seen += 1
            images = images.to(device, non_blocking=True).float() / 255.0

            if opt.multi_scale:
                size = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs
                scale = size / max(images.shape[2:])
                if scale != 1:
                    shape = [math.ceil(x * scale / gs) * gs for x in images.shape[2:]]
                    images = F.interpolate(
                        images, size=shape, mode='bilinear', align_corners=False)

            images_rgb, images_ir = images[:, :3], images[:, 3:]
            reliability_target = None
            if rc_rgca_v2_active:
                images_rgb, images_ir, reliability_target = (
                    apply_rc_rgca_v2_training_corruption(
                        images_rgb, images_ir, hyp))
            elif rs_rgca_active:
                images_rgb, images_ir, reliability_target = (
                    apply_rs_rgca_training_corruption(
                        images_rgb, images_ir, hyp))
            if warmup_iters and iteration <= warmup_iters:
                interval = [0, warmup_iters]
                accumulate = max(
                    1, np.interp(
                        iteration, interval,
                        [1, opt.nominal_batch_size /
                         opt.total_batch_size]).round())
                for group in optimizer.param_groups:
                    target_lr = group['initial_lr'] * lr_function(epoch)
                    group['lr'] = np.interp(
                        iteration, interval,
                        [group.get('warmup_start_lr', 0.0), target_lr])
                    if 'momentum' in group:
                        group['momentum'] = np.interp(
                            iteration, interval,
                            [hyp['warmup_momentum'], hyp['momentum']])
                    elif 'betas' in group:
                        group['betas'] = (
                            float(np.interp(
                                iteration, interval,
                                [hyp['warmup_momentum'], hyp['momentum']])),
                            group['betas'][1])

            targets_device = targets.to(device)
            reliability_batch_stats = None
            reliability_loss_value = None
            rs_reliability_batch_stats = None
            rs_reliability_loss_values = None
            with amp.autocast(enabled=cuda and opt.amp):
                predictions = model(images_rgb, images_ir)
                loss, loss_items = compute_loss(predictions, targets_device)
                if rc_rgca_v2_active:
                    if rank in (-1, 0):
                        reliability_batch_stats = (
                            summarize_rc_rgca_v2_reliability(
                                model, reliability_target))
                    reliability_loss = compute_rc_rgca_v2_reliability_loss(
                        model, reliability_target, hyp)
                    reliability_loss_value = float(
                        reliability_loss.detach().float())
                    weighted_reliability = (
                        float(hyp['reliability_aux_gain']) * reliability_loss)
                    # ComputeLoss returns a batch-scaled training loss but
                    # unscaled logging components. Preserve that convention.
                    loss = loss + weighted_reliability * images.shape[0]
                    loss_items[3] = (
                        loss_items[3] + weighted_reliability.detach())
                elif rs_rgca_active:
                    if rank in (-1, 0):
                        rs_reliability_batch_stats = (
                            summarize_rs_rgca_reliability(
                                model, reliability_target))
                    (rs_reliability_loss, rs_calibration_loss,
                     rs_ranking_loss) = compute_rs_rgca_reliability_loss(
                         model, reliability_target, hyp)
                    weighted_reliability = (
                        float(hyp['rs_reliability_aux_gain'])
                        * rs_reliability_loss)
                    rs_reliability_loss_values = (
                        float(rs_calibration_loss.float()),
                        float(rs_ranking_loss.float()),
                        float(rs_reliability_loss.detach().float()))
                    # Match ComputeLoss' batch-scaled optimization convention;
                    # the CSV/logging item remains per sample.
                    loss = loss + weighted_reliability * images.shape[0]
                    loss_items[3] = (
                        loss_items[3] + weighted_reliability.detach())
                if rank != -1:
                    loss *= opt.world_size

            if batch_index == 0 and rank in (-1, 0):
                _save_epoch_diagnostics(model, epoch, attention_file, fusion_file)

            scaler.scale(loss).backward()
            if iteration % accumulate == 0:
                scaler.unscale_(optimizer)
                if opt.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), opt.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

            if rank in (-1, 0):
                if reliability_batch_stats is not None:
                    batch_samples = images.shape[0]
                    stats_with_loss = torch.cat([
                        reliability_batch_stats,
                        torch.tensor([reliability_loss_value])])
                    reliability_epoch_sum += stats_with_loss * batch_samples
                    reliability_epoch_count += batch_samples
                if rs_reliability_batch_stats is not None:
                    batch_samples = images.shape[0]
                    stats_with_loss = torch.cat([
                        rs_reliability_batch_stats,
                        torch.tensor(rs_reliability_loss_values),
                    ])
                    rs_reliability_epoch_sum += (
                        stats_with_loss * batch_samples)
                    rs_reliability_epoch_count += batch_samples
                mean_loss = (mean_loss * batch_index + loss_items) / (batch_index + 1)
                memory = torch.cuda.memory_reserved() / 1e9 if cuda else 0
                progress.set_description(
                    ('%10s%10.3gG' + '%10.4g' * 6) % (
                        f'{epoch}/{opt.epochs - 1}', memory, *mean_loss,
                        targets.shape[0], images.shape[-1]))
                if (detection_task == 'hbb' and epoch == start_epoch
                        and batch_index < 3):
                    Thread(
                        target=plot_images,
                        args=(images_rgb, targets, paths,
                              save_dir / f'train_batch{batch_index}_vis.jpg'),
                        daemon=True).start()
                    Thread(
                        target=plot_images,
                        args=(images_ir, targets, paths,
                              save_dir / f'train_batch{batch_index}_ir.jpg'),
                        daemon=True).start()

        learning_rates = [group['lr'] for group in optimizer.param_groups]
        learning_rates = (learning_rates + [learning_rates[-1]] * 3)[:3]
        scheduler.step()
        if cuda:
            torch.cuda.empty_cache()

        if rank in (-1, 0):
            if rc_rgca_v2_active and reliability_epoch_count:
                reliability_mean = (
                    reliability_epoch_sum / reliability_epoch_count)
                with reliability_file.open('a') as stream:
                    stream.write(
                        str(epoch) + ',' + ','.join(
                            f'{float(value):.8g}'
                            for value in reliability_mean) + '\n')
                logger.info(
                    'RC-RGCA v2 calibration: target q=(%.3f, %.3f), c=%.3f; '
                    'pred q=(%.3f, %.3f), c=%.3f; null=(%.3f, %.3f); '
                    'transfer=(%.3f, %.3f), aux=%.4f.',
                    *[float(value) for value in reliability_mean])
            if rs_rgca_active and rs_reliability_epoch_count:
                rs_reliability_mean = (
                    rs_reliability_epoch_sum
                    / rs_reliability_epoch_count)
                with rs_reliability_file.open('a') as stream:
                    stream.write(
                        str(epoch) + ',' + ','.join(
                            f'{float(value):.8g}'
                            for value in rs_reliability_mean) + '\n')
                logger.info(
                    'RS-RGCA calibration: target q=(%.3f, %.3f), c=%.3f; '
                    'directional pred=(IR->RGB %.3f, RGB->IR %.3f); '
                    'cal=%.4f, rank=%.4f, aux=%.4f.',
                    *[float(value) for value in rs_reliability_mean])
            ema.update_attr(
                model, include=[
                    'yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == opt.epochs
            if not opt.notest or final_epoch:
                validation_module = test_obb if detection_task == 'obb' else test
                results, maps, mr_result, _ = validation_module.test(
                    data_dict, batch_size=val_batch_size, imgsz=imgsz_test,
                    conf_thres=opt.obb_val_conf_thres,
                    iou_thres=opt.obb_val_iou_thres,
                    model=ema.ema, single_cls=opt.single_cls,
                    dataloader=testloader, save_dir=save_dir,
                    save_txt=False, save_conf=False,
                    verbose=nc < 50 and final_epoch,
                    plots=final_epoch, wandb_logger=None,
                    compute_loss=compute_loss,
                    is_coco=opt.data.endswith('coco.yaml'), opt=opt,
                    labels_list=labels_list,
                    **({
                        'max_det': opt.obb_val_max_det,
                        'max_nms': opt.obb_val_max_nms,
                        'multi_label': opt.obb_val_multi_label,
                    } if detection_task == 'obb' else {}))

            keys = [
                'train/box_loss', 'train/obj_loss', 'train/cls_loss',
                ('train/theta_loss' if detection_task == 'obb'
                 else 'train/rank_loss'), 'TP', 'FP', 'FN', 'F1',
                'metrics/precision', 'metrics/recall',
                ('metrics/OBB_mAP50' if detection_task == 'obb'
                 else 'metrics/mAP_0.5'),
                ('metrics/OBB_mAP50:0.95' if detection_task == 'obb'
                 else 'metrics/mAP_0.5:0.95'),
                'val/box_loss', 'val/obj_loss',
                'val/cls_loss',
                ('val/theta_loss' if detection_task == 'obb'
                 else 'val/rank_loss'), 'x/lr0', 'x/lr1', 'x/lr2',
                'MR_all', 'MR_day', 'MR_night', 'MR_near', 'MR_medium',
                'MR_far', 'MR_none', 'MR_partial', 'MR_heavy', 'Recall_all']
            values = list(mean_loss) + list(results) + learning_rates + list(mr_result)
            if detection_task == 'obb':
                # Keep the historical columns in their original positions so
                # plot_results remains compatible, and append native DOTA OBB
                # AP50 for every class at the end of each epoch row.
                keys += [f'val/OBB_AP50_{name}' for name in names]
                values += list(maps)
            result_file = save_dir / 'results.csv'
            if not result_file.exists():
                result_file.write_text('epoch,' + ','.join(keys) + '\n')
            with result_file.open('a') as stream:
                stream.write(
                    str(epoch) + ',' + ','.join(f'{float(x):g}' for x in values) + '\n')

            if opt.best_metric == 'map50':
                current_fitness = float(results[6])
            elif opt.best_metric == 'fitness':
                current_fitness = float(np.asarray(
                    fitness(np.asarray(results).reshape(1, -1))).item())
            else:
                raise ValueError("best_metric must be 'map50' or 'fitness'.")
            # Always materialize best.pt after the first validation, even if
            # the initial metric is exactly zero.
            is_best = not best.exists() or current_fitness > best_fitness
            best_fitness = max(best_fitness, current_fitness)
            if not opt.nosave or final_epoch:
                checkpoint = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': keep_force_fp32_modules(deepcopy(
                        model.module if is_parallel(model) else model).half()),
                    'ema': keep_force_fp32_modules(deepcopy(ema.ema).half()),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, last)
                if is_best:
                    torch.save(checkpoint, best)
                del checkpoint

    if rank in (-1, 0):
        plot_results(file=save_dir / 'results.csv')
        logger.info(
            '%d epochs completed in %.3f hours.',
            opt.epochs - start_epoch, (time.time() - started) / 3600)
        if best.exists():
            validation_module = test_obb if detection_task == 'obb' else test
            final_kwargs = ({
                'max_det': opt.obb_val_max_det,
                'max_nms': opt.obb_val_max_nms,
                'multi_label': opt.obb_val_multi_label,
                'task': 'test',
            } if detection_task == 'obb' else {})
            # Reload validation-selected best.pt once to verify the serialized
            # checkpoint. OBB uses its held-out test split; HBB reuses the
            # already constructed validation loader.
            final_results, final_class_ap50, _, final_timing = validation_module.test(
                opt.data, batch_size=val_batch_size, imgsz=imgsz_test,
                conf_thres=opt.obb_val_conf_thres,
                iou_thres=opt.obb_val_iou_thres,
                model=attempt_load(best, device).half(),
                single_cls=opt.single_cls,
                dataloader=None if detection_task == 'obb' else testloader,
                save_dir=save_dir, save_txt=False, save_conf=False,
                save_json=False, plots=False,
                is_coco=opt.data.endswith('coco.yaml'), opt=opt,
                labels_list=labels_list, verbose=nc > 1,
                **final_kwargs)
            if detection_task == 'obb':
                test_keys = [
                    'precision', 'recall', 'OBB_mAP50', 'OBB_mAP50:95',
                    *[f'AP50_{name}' for name in names],
                    'inference_ms_per_image', 'nms_ms_per_image']
                test_values = [
                    final_results[4], final_results[5], final_results[6],
                    final_results[7], *final_class_ap50,
                    final_timing[0], final_timing[1]]
                test_result_file = save_dir / 'test_results.csv'
                test_result_file.write_text(
                    ','.join(test_keys) + '\n' +
                    ','.join(f'{float(value):g}' for value in test_values) + '\n')
                logger.info(
                    'Final test metrics from validation-selected best.pt saved to %s.',
                    test_result_file)
        else:
            raise RuntimeError(
                'best.pt was not created; final test evaluation cannot run.')
        for checkpoint_path in (last, best):
            if checkpoint_path.exists():
                strip_optimizer(checkpoint_path)
    elif dist.is_initialized():
        dist.destroy_process_group()
    if cuda:
        torch.cuda.empty_cache()
    return results


def parse_opt(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Transfer the trained original LCAFNet_LLVIP.pt detector to C0 '
            'RGCA+Mask+GFB and fine-tune at 1024x1024. Both modal backbones '
            'remain frozen while fusion, PAN/FPN and Detect are optimized with '
            'mAP50 checkpoint selection.'))
    parser.add_argument('--weights', default=DEFAULT_TRAIN_WEIGHTS)
    parser.add_argument('--cfg', default=DEFAULT_TRAIN_CFG)
    parser.add_argument('--data', default=DEFAULT_TRAIN_DATA)
    parser.add_argument('--hyp', default=DEFAULT_TRAIN_HYP)
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument(
        '--nominal-batch-size', type=int,
        default=TRAINING_CONFIG['nominal_batch_size'],
        help='target effective batch size used for gradient accumulation')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=list(DEFAULT_TRAIN_IMAGE_SIZE))
    parser.add_argument('--device', default=TRAINING_CONFIG['device'])
    parser.add_argument('--workers', type=int, default=DEFAULT_TRAIN_WORKERS)
    parser.add_argument('--seed', type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        '--optimizer', choices=['SGD', 'Adam', 'AdamW'],
        default=DEFAULT_TRAIN_OPTIMIZER)
    parser.add_argument(
        '--lr-schedule', choices=['linear', 'cosine'],
        default=TRAINING_CONFIG['lr_schedule'])
    parser.add_argument(
        '--lr-drop-epochs', nargs='+', type=int,
        default=list(TRAINING_CONFIG['lr_drop_epochs']))
    parser.add_argument(
        '--lr-drop-factor', type=float,
        default=TRAINING_CONFIG['lr_drop_factor'])
    parser.add_argument(
        '--best-metric', choices=['map50', 'fitness'],
        default=TRAINING_CONFIG['best_metric'])
    parser.add_argument(
        '--obb-val-conf-thres', type=float,
        default=TRAINING_CONFIG['obb_val_conf_thres'])
    parser.add_argument(
        '--obb-val-iou-thres', type=float,
        default=TRAINING_CONFIG['obb_val_iou_thres'])
    parser.add_argument(
        '--obb-val-max-nms', type=int,
        default=TRAINING_CONFIG['obb_val_max_nms'])
    parser.add_argument(
        '--obb-val-max-det', type=int,
        default=TRAINING_CONFIG['obb_val_max_det'])
    parser.add_argument(
        '--obb-val-multi-label', action='store_true',
        default=TRAINING_CONFIG['obb_val_multi_label'])
    parser.add_argument(
        '--trainable-scope',
        choices=['all', 'fusion_only', 'fusion_head', 'pan_head', 'sod_head',
                 'ccall_finetune'],
        default=TRAINING_CONFIG['trainable_scope'])
    parser.add_argument(
        '--lr0', type=float, default=None,
        help='optional command-line override for HYPERPARAMETERS["lr0"]')
    parser.add_argument(
        '--gradient-clip-norm', type=float,
        default=TRAINING_CONFIG['gradient_clip_norm'])
    parser.add_argument(
        '--d3-kernel-size', type=int,
        default=TRAINING_CONFIG['d3_kernel_size'])
    parser.add_argument('--project', default=TRAINING_CONFIG['project'])
    parser.add_argument('--name', default=DEFAULT_TRAIN_EXPERIMENT_NAME)
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument(
        '--exist-ok', action='store_true', default=TRAINING_CONFIG['exist_ok'])
    parser.add_argument(
        '--cache-images', action='store_true',
        default=TRAINING_CONFIG['cache_images'])
    parser.add_argument('--rect', action='store_true', default=TRAINING_CONFIG['rect'])
    parser.add_argument(
        '--multi-scale', action='store_true',
        default=TRAINING_CONFIG['multi_scale'])
    parser.add_argument(
        '--single-cls', action='store_true',
        default=TRAINING_CONFIG['single_class'])
    parser.add_argument(
        '--sync-bn', action='store_true',
        default=TRAINING_CONFIG['sync_batch_norm'])
    parser.add_argument(
        '--noautoanchor', action='store_true',
        default=not TRAINING_CONFIG['autoanchor'])
    parser.add_argument(
        '--notest', action='store_true',
        default=not TRAINING_CONFIG['validate_each_epoch'])
    parser.add_argument(
        '--validate-initialization', dest='validate_initialization',
        action='store_true', default=TRAINING_CONFIG['validate_initialization'])
    parser.add_argument(
        '--no-validate-initialization', dest='validate_initialization',
        action='store_false')
    parser.add_argument(
        '--nosave', action='store_true',
        default=not TRAINING_CONFIG['save_each_epoch'])
    parser.add_argument(
        '--amp', dest='amp', action='store_true',
        default=TRAINING_CONFIG['amp'])
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument(
        '--cudnn-benchmark', dest='cudnn_benchmark', action='store_true',
        default=DEFAULT_TRAIN_CUDNN_BENCHMARK)
    parser.add_argument(
        '--no-cudnn-benchmark', dest='cudnn_benchmark', action='store_false')
    parser.add_argument(
        '--strict-determinism', action='store_true',
        default=TRAINING_CONFIG['strict_determinism'])
    parser.add_argument('--close-mosaic', type=int,
                        default=DEFAULT_TRAIN_CLOSE_MOSAIC)
    parser.add_argument(
        '--close-augmentations', type=int,
        default=DEFAULT_TRAIN_CLOSE_AUGMENTATIONS,
        help='disable all training augmentations for the final N epochs')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args(args)


def _restore_resume_options(opt):
    if not opt.resume:
        return opt
    checkpoint = opt.resume if isinstance(opt.resume, str) else get_latest_run()
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError('No checkpoint was found for --resume.')
    checkpoint = resolve_resume_checkpoint(checkpoint)
    saved_opt_path = Path(checkpoint).parent.parent / 'opt.yaml'
    if not saved_opt_path.is_file():
        raise FileNotFoundError(f'Resume options not found: {saved_opt_path}')
    ranks = opt.local_rank, opt.global_rank, opt.world_size
    with saved_opt_path.open() as stream:
        saved = argparse.Namespace(**yaml.safe_load(stream))
    saved_hyp_path = saved_opt_path.with_name('hyp.yaml')
    if saved_hyp_path.is_file():
        saved.hyp = str(saved_hyp_path)
    saved.cfg = getattr(saved, 'cfg', DEFAULT_TRAIN_CFG)
    saved.weights = checkpoint
    saved.resume = True
    saved.local_rank, saved.global_rank, saved.world_size = ranks
    # Old opt.yaml files may not contain the streamlined runtime fields.
    for name, value in {
        'strict_determinism': False,
        'cudnn_benchmark': False,
        'close_mosaic': DEFAULT_TRAIN_CLOSE_MOSAIC,
        'close_augmentations': DEFAULT_TRAIN_CLOSE_AUGMENTATIONS,
        'amp': True,
        'sync_bn': False,
        'single_cls': False,
        'cache_images': False,
        'rect': False,
        'multi_scale': False,
        'noautoanchor': False,
        'notest': False,
        'validate_initialization': False,
        'nosave': False,
        'optimizer': DEFAULT_TRAIN_OPTIMIZER,
        # Historical runs used cosine LR and YOLO composite fitness. Preserve
        # those semantics when an old opt.yaml is resumed.
        'lr_schedule': 'cosine',
        'lr_drop_epochs': [],
        'lr_drop_factor': 1.0,
        'best_metric': 'fitness',
        'obb_val_conf_thres': TRAINING_CONFIG['obb_val_conf_thres'],
        'obb_val_iou_thres': TRAINING_CONFIG['obb_val_iou_thres'],
        'obb_val_max_nms': TRAINING_CONFIG['obb_val_max_nms'],
        'obb_val_max_det': TRAINING_CONFIG['obb_val_max_det'],
        'obb_val_multi_label': TRAINING_CONFIG['obb_val_multi_label'],
        'trainable_scope': 'all',
        'lr0': None,
        'gradient_clip_norm': TRAINING_CONFIG['gradient_clip_norm'],
        # Preserve the historical effective-batch=64 behavior when resuming
        # checkpoints whose opt.yaml predates this explicit option.
        'nominal_batch_size': 64,
        'd3_kernel_size': TRAINING_CONFIG['d3_kernel_size'],
        'workers': DEFAULT_TRAIN_WORKERS,
        'seed': DEFAULT_TRAIN_SEED,
    }.items():
        if not hasattr(saved, name):
            setattr(saved, name, value)
    if not saved.lr_drop_epochs and hasattr(saved, 'lr_drop_epoch'):
        saved.lr_drop_epochs = (
            [saved.lr_drop_epoch] if saved.lr_drop_epoch > 0 else [])
    return saved


def main(args=None):
    os.chdir(ROOT)
    opt = parse_opt(args)
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.global_rank = int(os.environ.get('RANK', -1))
    set_logging(opt.global_rank)
    opt = _restore_resume_options(opt)

    if not 0 <= opt.close_augmentations <= opt.epochs:
        raise ValueError('--close-augmentations must be between 0 and --epochs.')
    if not 0 <= opt.close_mosaic <= opt.epochs:
        raise ValueError('--close-mosaic must be between 0 and --epochs.')
    if opt.nominal_batch_size <= 0:
        raise ValueError('--nominal-batch-size must be positive.')
    if any(epoch < 0 or epoch >= opt.epochs for epoch in opt.lr_drop_epochs):
        raise ValueError(
            '--lr-drop-epochs values must be between 0 and epochs - 1.')
    if len(set(opt.lr_drop_epochs)) != len(opt.lr_drop_epochs):
        raise ValueError('--lr-drop-epochs must not contain duplicates.')
    opt.lr_drop_epochs = sorted(opt.lr_drop_epochs)
    if not 0 < opt.lr_drop_factor <= 1:
        raise ValueError('--lr-drop-factor must be in the interval (0, 1].')
    if not 0 <= opt.obb_val_conf_thres <= 1:
        raise ValueError('--obb-val-conf-thres must be in the interval [0, 1].')
    if not 0 <= opt.obb_val_iou_thres <= 1:
        raise ValueError('--obb-val-iou-thres must be in the interval [0, 1].')
    if opt.obb_val_max_nms <= 0 or opt.obb_val_max_det <= 0:
        raise ValueError('OBB validation candidate limits must be positive.')
    if opt.obb_val_max_det > opt.obb_val_max_nms:
        raise ValueError('--obb-val-max-det must not exceed --obb-val-max-nms.')

    if not opt.resume:
        opt.data = check_file(opt.data)
        opt.cfg = check_file(opt.cfg)
        opt.hyp = check_file(opt.hyp)
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        if opt.d3_kernel_size < 3 or opt.d3_kernel_size % 2 == 0:
            raise ValueError('d3_kernel_size must be an odd integer >= 3.')
    opt.total_batch_size = opt.batch_size

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        if opt.batch_size % opt.world_size:
            raise ValueError('Batch size must be divisible by DDP world size.')
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        opt.batch_size //= opt.world_size

    with open(opt.hyp) as stream:
        hyp = yaml.safe_load(stream)
    if not opt.resume:
        hyp.update(HYPERPARAMETERS)
        if opt.lr0 is not None:
            if opt.lr0 <= 0:
                raise ValueError('--lr0 must be positive.')
            hyp['lr0'] = opt.lr0
    if bool(hyp.get('llvip_lcafnet_to_rgca_finetune', False)):
        active_protocol = (
            'LLVIP original-LCAFNet to C0-RGCA transfer fine-tuning at 1024')
    elif bool(hyp.get('llvip_rgca_same_architecture_finetune', False)):
        active_protocol = (
            'LLVIP exact C0-RGCA low-LR fine-tuning at 1024x1024')
    elif bool(hyp.get('flir_lcafnet_to_rgca_staged_finetune', False)):
        active_protocol = (
            'FLIR original-LCAFNet to C0-RGCA staged structural migration')
    elif bool(hyp.get('rgca_map50_recovery', False)):
        active_protocol = (
            'RS-RGCA mAP50 recovery (frozen backbones/RGCA, PAN+Detect only)')
    elif bool(hyp.get('rs_reliability_supervision', False)):
        active_protocol = 'RS-RGCA reliability-supervised original directional gates'
    elif 'RCRGCAV2' in str(opt.cfg):
        active_protocol = (
            'RC-RGCA v2 reliability-conditioned K/V + Null Attention + shared '
            'reliability-guided GFB')
    else:
        active_protocol = 'configured RGB/IR fusion architecture'
    logger.info(
        'Multispectral %s protocol is active; initialization=%s; '
        'optimizer=%s, lr0=%g, epochs=%d, physical_batch=%d, '
        'effective_batch=%d, accumulate=%d, image=%s',
        active_protocol, opt.weights,
        opt.optimizer, hyp['lr0'], opt.epochs, opt.total_batch_size,
        opt.nominal_batch_size,
        max(round(opt.nominal_batch_size / opt.total_batch_size), 1),
        opt.img_size)
    logger.info('Runtime options: %s', opt)

    writer = None
    if opt.global_rank in (-1, 0) and SummaryWriter is not None:
        writer = SummaryWriter(opt.save_dir)
    try:
        return train_rgb_ir(hyp, opt, device, writer)
    finally:
        if writer is not None:
            writer.close()


# Historical unit tests can still request retired helpers without placing the
# archived multi-experiment trainer on the active C4 import path.
_LEGACY_MODULE = None


def __getattr__(name):
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        path = ROOT / 'archives/train_experiments_legacy_20260805.py'
        spec = importlib.util.spec_from_file_location('_legacy_train_experiments', path)
        _LEGACY_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_LEGACY_MODULE)
    try:
        return getattr(_LEGACY_MODULE, name)
    except AttributeError as error:
        raise AttributeError(f'module train has no attribute {name!r}') from error


if __name__ == '__main__':
    main()
