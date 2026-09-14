"""Training-only direct foreground supervision for RGB/IR private states."""

import math

import torch
import torch.nn.functional as F


def build_private_foreground_heatmap(targets, batch_size, height, width,
                                     min_sigma=0.5):
    """Rasterize normalized YOLO boxes into a class-agnostic soft heatmap."""
    height, width = int(height), int(width)
    heatmap = torch.zeros((int(batch_size), height, width), dtype=torch.float32)
    if targets.numel() == 0:
        return heatmap
    for image_index, _, x, y, box_width, box_height in targets.detach().cpu().tolist():
        image_index = int(image_index)
        if not 0 <= image_index < batch_size:
            continue
        center_x = float(x) * max(width - 1, 1)
        center_y = float(y) * max(height - 1, 1)
        sigma_x = max(float(box_width) * width / 6.0, float(min_sigma))
        sigma_y = max(float(box_height) * height / 6.0, float(min_sigma))
        x0 = max(0, int(math.floor(center_x - 3.0 * sigma_x)))
        x1 = min(width, int(math.ceil(center_x + 3.0 * sigma_x)) + 1)
        y0 = max(0, int(math.floor(center_y - 3.0 * sigma_y)))
        y1 = min(height, int(math.ceil(center_y + 3.0 * sigma_y)) + 1)
        grid_y = torch.arange(y0, y1, dtype=torch.float32).view(-1, 1)
        grid_x = torch.arange(x0, x1, dtype=torch.float32).view(1, -1)
        gaussian = torch.exp(-0.5 * (
            ((grid_x - center_x) / sigma_x).square()
            + ((grid_y - center_y) / sigma_y).square()))
        current = heatmap[image_index, y0:y1, x0:x1]
        heatmap[image_index, y0:y1, x0:x1] = torch.maximum(current, gaussian)
        center_row = min(height - 1, max(0, int(round(center_y))))
        center_col = min(width - 1, max(0, int(round(center_x))))
        heatmap[image_index, center_row, center_col] = 1.0
    return heatmap


def _soft_focal_bce(logits, target, alpha=0.75, gamma=2.0):
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_true = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
    return (alpha_true * (1.0 - probability_true).pow(float(gamma)) * bce).mean()


def compute_private_foreground_supervision(model, targets, batch_size,
                                           stage_weights, warmup_scale,
                                           alpha=0.75, gamma=2.0,
                                           min_sigma=0.5,
                                           collect_stats=False):
    """Supervise C4/C5 RGB and IR private states with direct GT foreground."""
    stages = []
    for module_name, module in model.named_modules():
        logits = getattr(module, 'last_private_foreground_logits', None)
        if logits is not None:
            stages.append((logits.shape[-2] * logits.shape[-1], module_name,
                           module, logits))
    stages.sort(reverse=True)  # C4 before C5
    if not stages:
        return next(model.parameters()).new_zeros(()), []
    if len(stages) != len(stage_weights):
        raise RuntimeError(
            f'Expected {len(stage_weights)} private foreground stages, got {len(stages)}.')

    total = stages[0][3].new_zeros((), dtype=torch.float32)
    diagnostic_rows = []
    for (_, module_name, module, logits), stage_weight in zip(stages, stage_weights):
        if logits.shape[1] != 2:
            raise RuntimeError('Private foreground logits must contain RGB and IR channels.')
        target = build_private_foreground_heatmap(
            targets, batch_size, logits.shape[-2], logits.shape[-1], min_sigma)
        target = target.to(device=logits.device, non_blocking=True)
        rgb_loss = _soft_focal_bce(logits[:, 0], target, alpha, gamma)
        ir_loss = _soft_focal_bce(logits[:, 1], target, alpha, gamma)
        total = total + float(stage_weight) * float(warmup_scale) * (
            rgb_loss + ir_loss) * 0.5

        if collect_stats:
            with torch.no_grad():
                probabilities = logits.detach().float().sigmoid()
                foreground, background = target > 0.1, target < 0.01

                def masked_mean(values, mask):
                    selected = values[mask]
                    return float(selected.mean()) if selected.numel() else float('nan')

                rgb_fg = masked_mean(probabilities[:, 0], foreground)
                rgb_bg = masked_mean(probabilities[:, 0], background)
                ir_fg = masked_mean(probabilities[:, 1], foreground)
                ir_bg = masked_mean(probabilities[:, 1], background)
                diagnostic_rows.append((
                    module_name, logits.shape[-2], logits.shape[-1],
                    float(warmup_scale), float(stage_weight), float(target.mean()),
                    float(foreground.float().mean()), float(rgb_loss.detach()),
                    float(ir_loss.detach()), rgb_fg, rgb_bg, rgb_fg - rgb_bg,
                    ir_fg, ir_bg, ir_fg - ir_bg))

        # Do not retain an autograd graph on the module or in checkpoints.
        module.last_private_foreground_logits = None
    return total, diagnostic_rows
