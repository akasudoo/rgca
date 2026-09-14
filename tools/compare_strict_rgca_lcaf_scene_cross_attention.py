#!/usr/bin/env python3
"""Render a seeded-random four-scene strict RGCA-vs-LCAF attention figure.

The comparison isolates the attention path by using checkpoints with the same
Foreground Mask and GFB post-attention fusion.  For every M3FD scene, the
script takes the first eligible image in a seeded random permutation, uses the
largest valid GT instance as a pre-declared explanation target, and requires
both detectors to have a same-class raw prediction centred inside the GT box.

The maps are target-conditioned positive cross-modal contribution maps, not a
spatial reshape of channel-attention matrices.  RGCA maps are captured at its
post-reliability-gate deltas; LCAF maps are captured at its two original
cross-attention residuals.  LCAF/RGCA maps in both directions share one global
display scale after spatial-density normalization.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from models.common import HAFFormerLCAFMaskGFB  # noqa: E402
from tools.export_rgca_gate_paired_modal_heatmaps import (  # noqa: E402
    locate_target_prediction,
)
from tools.compare_rgca_gate_gradcam import (  # noqa: E402
    target_mask,
    valid_image_mask,
)
from tools.visualize_rgca_panel_a import (  # noqa: E402
    CLASS_NAMES,
    SCENES,
    image_id_from_manifest_line,
    paired_paths,
    prepare_pair,
    read_nonempty_lines,
    read_yolo_labels,
    resolve_device,
    sha256sum,
)


FUSION_LAYERS = (20, 21, 22, 23)
DIRECTIONS = ("ir_to_rgb", "rgb_to_ir")
PANEL_TITLES = (
    "RGB target",
    "IR target",
    "LCAF | IR→RGB",
    "RGCA | IR→RGB",
    "LCAF | RGB→IR",
    "RGCA | RGB→IR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgca-weights",
        type=Path,
        default=(REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt"),
    )
    parser.add_argument(
        "--lcaf-weights",
        type=Path,
        default=(REPO_ROOT / "runs/train/exp_m3fd_lcaf_ca_mask/weights/best.pt"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/dell/lcp/dataset/M3FD")
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "paper/strict_rgca_lcaf_scene_cross_attention_m3fd"),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-target-area", type=float, default=0.001)
    parser.add_argument("--max-target-area", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--display-quantile", type=float, default=0.995)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


class CrossModalContributionCapture:
    """Capture the directional residuals produced by P2-P5 attention blocks."""

    def __init__(self, model: torch.nn.Module, kind: str):
        if kind not in ("rgca", "lcaf"):
            raise ValueError(kind)
        self.kind = kind
        self.activations: Dict[Tuple[str, int], torch.Tensor] = {}
        self.handles = []
        for level, layer_index in enumerate(FUSION_LAYERS):
            layer = model.model[layer_index]
            if kind == "rgca":
                module = layer.cross_modal_attention
                self.handles.append(
                    module.register_forward_hook(self._rgca_hook(level))
                )
            else:
                self.handles.append(
                    layer.mhca_rgb.register_forward_hook(
                        self._single_hook("ir_to_rgb", level)
                    )
                )
                self.handles.append(
                    layer.mhca_ir.register_forward_hook(
                        self._single_hook("rgb_to_ir", level)
                    )
                )

    def _rgca_hook(self, level: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise TypeError("RGCA attention must return (delta_rgb, delta_ir)")
            if not all(isinstance(value, torch.Tensor) for value in output):
                raise TypeError("RGCA directional deltas must be tensors")
            self.activations[("ir_to_rgb", level)] = output[0]
            self.activations[("rgb_to_ir", level)] = output[1]

        return hook

    def _single_hook(self, direction: str, level: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"LCAF {direction} output is not a tensor")
            self.activations[(direction, level)] = output

        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def restore_strict_lcaf_layers(
    lcaf_model: torch.nn.Module, device: torch.device
) -> List[Dict[str, object]]:
    """Restore legacy flattened LCAF Mask+GFB blocks to the current strict class.

    The historical checkpoint pickled a generic ``HAFFormer`` class whose
    method name was later reused.  Its trained modules are unambiguous:
    directional original LCAF attention, Foreground Mask, and flattened
    concat/conv/dwconv GFB.  Repackage those exact trained modules under the
    current HAFFormerLCAFMaskGFB forward so loading cannot silently execute the
    later generic HAFFormer method.
    """
    audit = []
    for index in FUSION_LAYERS:
        legacy = lcaf_model.model[index]
        if isinstance(legacy, HAFFormerLCAFMaskGFB):
            audit.append({"layer": index, "action": "already_strict_current_class"})
            continue
        required = (
            "mhca_rgb", "mhca_ir", "foreground_mask", "concat", "conv", "dwconv"
        )
        if not all(hasattr(legacy, name) for name in required):
            raise TypeError(
                f"Legacy LCAF layer {index} cannot be restored; missing one of {required}"
            )
        dim = int(legacy.mhca_rgb.project_out.out_channels)
        strict = HAFFormerLCAFMaskGFB(dim)
        strict.mhca_rgb = legacy.mhca_rgb
        strict.mhca_ir = legacy.mhca_ir
        strict.foreground_mask = legacy.foreground_mask
        strict.fusion.conv = legacy.conv
        strict.fusion.dwconv = legacy.dwconv
        for attribute in ("i", "f", "type", "np"):
            if hasattr(legacy, attribute):
                setattr(strict, attribute, getattr(legacy, attribute))
        strict = strict.to(device=device).float().eval()
        lcaf_model.model[index] = strict
        audit.append({
            "layer": index,
            "action": "repackaged_legacy_flat_modules_under_HAFFormerLCAFMaskGFB",
            "legacy_class": type(legacy).__name__,
            "dim": dim,
            "reused_trained_modules": list(required),
            "mask_logit_gain": float(strict.mask_logit_gain.detach().cpu()),
        })
    lcaf_model.eval()
    return audit


def validate_models(
    rgca_model: torch.nn.Module, lcaf_model: torch.nn.Module
) -> Dict[str, object]:
    rgca_classes = []
    lcaf_classes = []
    for index in FUSION_LAYERS:
        rgca_layer = rgca_model.model[index]
        lcaf_layer = lcaf_model.model[index]
        rgca_classes.append(type(rgca_layer).__name__)
        lcaf_classes.append(type(lcaf_layer).__name__)
        if not hasattr(rgca_layer, "cross_modal_attention"):
            raise TypeError(f"RGCA fusion layer {index} has no cross_modal_attention")
        rgca_attention = rgca_layer.cross_modal_attention
        if getattr(rgca_attention, "reliability_gate", None) is None:
            raise TypeError(f"RGCA fusion layer {index} has no reliability gate")
        prior_mode = getattr(rgca_attention, "prior_mode", None)
        if prior_mode not in (None, "none"):
            raise TypeError(
                f"RGCA fusion layer {index} is not the pure prior-free RGCA checkpoint"
            )
        if getattr(rgca_attention, "channel_prior", None) is not None:
            raise TypeError(f"RGCA fusion layer {index} unexpectedly contains a prior module")
        if not all(hasattr(rgca_layer, name) for name in ("foreground_mask", "fusion")):
            raise TypeError(f"RGCA fusion layer {index} is missing Mask/GFB")
        if not all(hasattr(lcaf_layer, name) for name in ("mhca_rgb", "mhca_ir")):
            raise TypeError(f"LCAF fusion layer {index} is missing directional attention")
        if not all(hasattr(lcaf_layer, name) for name in ("foreground_mask", "fusion")):
            raise TypeError(f"LCAF fusion layer {index} is missing the shared Mask/GFB")
    return {
        "rgca_fusion_classes": rgca_classes,
        "lcaf_fusion_classes": lcaf_classes,
        "rgca_prior_modes": [
            getattr(rgca_model.model[index].cross_modal_attention, "prior_mode", None)
            for index in FUSION_LAYERS
        ],
        "rgca_channel_prior_modules": [
            type(getattr(
                rgca_model.model[index].cross_modal_attention, "channel_prior", None
            )).__name__
            for index in FUSION_LAYERS
        ],
        "fusion_layers": list(FUSION_LAYERS),
    }


def build_scene_permutations(
    dataset_root: Path, scene_root: Path, seed: int
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
    manifest_dir = scene_root / "manifests/original_m3fd_split"
    canonical = {
        path.stem for path in (dataset_root / "images/vis_test").glob("*.png")
    }
    memberships = Counter()
    scene_ids: Dict[str, List[str]] = {}
    raw_counts: Dict[str, int] = {}
    for scene, _display in SCENES:
        manifest = manifest_dir / f"{scene}_test_vis.txt"
        ids = sorted(
            {
                image_id_from_manifest_line(line)
                for line in read_nonempty_lines(manifest)
            }
            & canonical
        )
        scene_ids[scene] = ids
        raw_counts[scene] = len(ids)
        memberships.update(ids)
    ambiguous = sorted(image_id for image_id, count in memberships.items() if count > 1)
    ambiguous_set = set(ambiguous)
    rng = random.Random(seed)
    permutations: Dict[str, List[str]] = {}
    for scene, _display in SCENES:
        ids = [image_id for image_id in scene_ids[scene] if image_id not in ambiguous_set]
        rng.shuffle(ids)
        permutations[scene] = ids
    return permutations, ambiguous, raw_counts


def largest_valid_target(
    boxes: np.ndarray, valid: np.ndarray, img_size: int
) -> Tuple[np.ndarray, int, float]:
    candidates = []
    for index, box in enumerate(boxes):
        roi = target_mask(box, img_size) & valid
        area = int(roi.sum())
        if area:
            candidates.append((area, -index, index, box))
    if not candidates:
        raise ValueError("No GT target overlaps the valid image")
    area, _negative_index, index, target = max(candidates, key=lambda item: item[:2])
    return target.copy(), int(index), float(area / max(int(valid.sum()), 1))


def centred_target_confidence(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target: np.ndarray,
    device: torch.device,
) -> Tuple[float, bool]:
    with torch.no_grad():
        batch = tensor.to(device, non_blocking=True)
        output = model(batch[:, :3], batch[:, 3:])
    if not isinstance(output, (tuple, list)) or len(output) < 1:
        raise RuntimeError("Unexpected detector inference output")
    predictions = output[0][0].detach()
    class_id, x1, y1, x2, y2 = target.tolist()
    class_id = int(class_id)
    confidence = predictions[:, 4] * predictions[:, 5 + class_id]
    centres = predictions[:, :2]
    inside = (
        (centres[:, 0] >= x1)
        & (centres[:, 0] <= x2)
        & (centres[:, 1] >= y1)
        & (centres[:, 1] <= y2)
    )
    if not inside.any():
        return 0.0, False
    return float(confidence.masked_fill(~inside, -1.0).max().cpu()), True


def select_scene_samples(
    permutations: Dict[str, List[str]],
    rgca_model: torch.nn.Module,
    lcaf_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    selected = []
    audit: Dict[str, object] = {}
    display_names = dict(SCENES)
    for scene, _display in SCENES:
        rejected = Counter()
        choice: Optional[Dict[str, object]] = None
        for position, image_id in enumerate(permutations[scene]):
            try:
                rgb_path, ir_path, label_path = paired_paths(args.dataset_root, image_id)
            except FileNotFoundError:
                rejected["missing_pair"] += 1
                continue
            labels = read_yolo_labels(label_path)
            if len(labels) < args.min_gt:
                rejected["below_min_gt"] += 1
                continue
            rgb, ir, tensor, boxes = prepare_pair(
                rgb_path, ir_path, labels, args.img_size
            )
            valid = valid_image_mask(rgb_path, args.img_size)
            target, target_index, target_area = largest_valid_target(
                boxes, valid, args.img_size
            )
            if not args.min_target_area <= target_area <= args.max_target_area:
                rejected["target_area"] += 1
                continue
            rgca_conf, rgca_inside = centred_target_confidence(
                rgca_model, tensor, target, device
            )
            lcaf_conf, lcaf_inside = centred_target_confidence(
                lcaf_model, tensor, target, device
            )
            if not rgca_inside or not lcaf_inside:
                rejected["no_shared_centred_prediction"] += 1
                continue
            if min(rgca_conf, lcaf_conf) < args.min_confidence:
                rejected["below_min_confidence"] += 1
                continue
            choice = {
                "scene": scene,
                "scene_display": display_names[scene],
                "image_id": image_id,
                "random_permutation_position_zero_based": position,
                "gt_count": len(boxes),
                "target_index": target_index,
                "target": target,
                "target_area_fraction": target_area,
                "rgca_filter_confidence": rgca_conf,
                "lcaf_filter_confidence": lcaf_conf,
                "rgb_path": str(rgb_path.resolve()),
                "ir_path": str(ir_path.resolve()),
                "label_path": str(label_path.resolve()),
                "rgb": rgb,
                "ir": ir,
                "tensor": tensor,
                "boxes": boxes,
                "valid": valid,
            }
            break
        if choice is None:
            raise RuntimeError(f"No eligible random sample remains for scene {scene}")
        selected.append(choice)
        audit[scene] = {
            "examined_before_selection": int(choice["random_permutation_position_zero_based"]) + 1,
            "rejected_before_selection": dict(rejected),
            "selected_image_id": choice["image_id"],
        }
        print(
            f"Selected {display_names[scene]} ID {choice['image_id']} "
            f"with {choice['gt_count']} GT; target conf "
            f"LCAF/RGCA={lcaf_conf:.3f}/{rgca_conf:.3f}"
        )
    return selected, audit


def aggregate_positive_contribution(
    activations: Sequence[torch.Tensor],
    gradients: Sequence[Optional[torch.Tensor]],
    img_size: int,
) -> np.ndarray:
    total = None
    for activation, gradient in zip(activations, gradients):
        if gradient is None:
            continue
        level = F.relu((activation * gradient).sum(dim=1, keepdim=True))
        level = F.interpolate(
            level,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        total = level if total is None else total + level
    if total is None:
        return np.zeros((img_size, img_size), dtype=np.float32)
    array = total[0, 0].detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("Cross-modal contribution map is non-finite")
    return array


def compute_directional_maps(
    model: torch.nn.Module,
    capture: CrossModalContributionCapture,
    tensor: torch.Tensor,
    target: np.ndarray,
    device: torch.device,
    img_size: int,
) -> Tuple[Dict[str, np.ndarray], float, str, int]:
    capture.clear()
    batch = tensor.to(device, non_blocking=True).detach().requires_grad_(True)
    output = model(batch[:, :3], batch[:, 3:])
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("Expected inference output (predictions, logits, raw_levels)")
    predictions, raw_levels = output[0], output[2]
    score, confidence, detect_level, prediction_index = locate_target_prediction(
        predictions, raw_levels, target
    )
    activation_groups = {
        direction: [capture.activations[(direction, level)] for level in range(4)]
        for direction in DIRECTIONS
    }
    flat = activation_groups["ir_to_rgb"] + activation_groups["rgb_to_ir"]
    gradients = torch.autograd.grad(
        score, flat, retain_graph=False, allow_unused=True
    )
    maps = {
        "ir_to_rgb": aggregate_positive_contribution(
            activation_groups["ir_to_rgb"], gradients[:4], img_size
        ),
        "rgb_to_ir": aggregate_positive_contribution(
            activation_groups["rgb_to_ir"], gradients[4:], img_size
        ),
    }
    return maps, confidence, detect_level, prediction_index


def density_and_metrics(
    cam: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    positive = np.maximum(cam, 0).astype(np.float32)
    positive[~valid] = 0.0
    total = float(positive[valid].sum())
    roi = target_mask(target, positive.shape[0]) & valid
    background = valid & ~roi
    if total <= 1e-12:
        return np.zeros_like(positive), {
            "ebpg": 0.0,
            "background_leakage": 1.0,
            "foreground_background_enrichment": 0.0,
            "pointing_hit": 0.0,
            "target_coverage_top10_valid_pixels": 0.0,
            "raw_positive_mass": 0.0,
        }
    density = positive / max(float(positive[valid].mean()), 1e-12)
    ebpg = float(positive[roi].sum() / total)
    fg_mean = float(density[roi].mean())
    bg_mean = float(density[background].mean()) if background.any() else 0.0
    valid_values = density[valid]
    threshold = float(np.quantile(valid_values, 0.90))
    peak_flat = int(np.argmax(np.where(valid, density, -1.0)))
    peak_y, peak_x = np.unravel_index(peak_flat, density.shape)
    return density.astype(np.float32), {
        "ebpg": ebpg,
        "background_leakage": 1.0 - ebpg,
        "foreground_background_enrichment": fg_mean / max(bg_mean, 1e-12),
        "pointing_hit": float(roi[peak_y, peak_x]),
        "target_coverage_top10_valid_pixels": float((density[roi] >= threshold).mean()),
        "raw_positive_mass": total,
    }


def draw_target_box(
    image: np.ndarray,
    target: np.ndarray,
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    _class_id, x1, y1, x2, y2 = target.tolist()
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(output, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        (width, height), baseline = cv2.getTextSize(label, font, scale, 1)
        tx = max(0, min(p1[0], output.shape[1] - width - 4))
        ty = max(height + 4, p1[1])
        cv2.rectangle(
            output,
            (tx, ty - height - 4),
            (tx + width + 4, ty + baseline),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            output, label, (tx + 2, ty - 2), font, scale, color, 1, cv2.LINE_AA
        )
    return output


def overlay_shared_density(
    base: np.ndarray,
    density: np.ndarray,
    valid: np.ndarray,
    display_max: float,
    target: np.ndarray,
) -> np.ndarray:
    display = np.clip(density / max(display_max, 1e-12), 0.0, 1.0)
    display[~valid] = 0.0
    heat_bgr = cv2.applyColorMap(
        np.uint8(np.clip(display * 255, 0, 255)), cv2.COLORMAP_TURBO
    )
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    alpha = (0.72 * np.power(display, 0.62))[..., None]
    overlay = base.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return draw_target_box(overlay, target, (255, 255, 255), thickness=1)


def render_figure(
    items: Sequence[Dict[str, object]],
    output_dir: Path,
    display_quantile: float,
    dpi: int,
) -> Tuple[Path, Path, float]:
    density_values = []
    for item in items:
        valid = np.asarray(item["valid"], dtype=bool)
        for model_name in ("lcaf", "rgca"):
            for direction in DIRECTIONS:
                density_values.append(
                    np.asarray(item[f"{model_name}_{direction}_density"])[valid]
                )
    display_max = float(np.quantile(np.concatenate(density_values), display_quantile))
    display_max = max(display_max, 1e-6)

    fig, axes = plt.subplots(4, 6, figsize=(17.2, 11.2))
    fig.subplots_adjust(
        left=0.073, right=0.995, top=0.918, bottom=0.215,
        wspace=0.035, hspace=0.16,
    )
    for row_index, (row_axes, item) in enumerate(zip(axes, items)):
        target = np.asarray(item["target"], dtype=np.float32)
        class_id = int(target[0])
        class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        rgb = np.asarray(item["rgb"])
        ir = np.asarray(item["ir"])
        valid = np.asarray(item["valid"], dtype=bool)
        raw_label = f"Target: {class_name}"
        panels = [
            draw_target_box(rgb, target, (255, 230, 0), raw_label, thickness=2),
            draw_target_box(ir, target, (255, 230, 0), raw_label, thickness=2),
            overlay_shared_density(
                rgb, np.asarray(item["lcaf_ir_to_rgb_density"]), valid,
                display_max, target,
            ),
            overlay_shared_density(
                rgb, np.asarray(item["rgca_ir_to_rgb_density"]), valid,
                display_max, target,
            ),
            overlay_shared_density(
                ir, np.asarray(item["lcaf_rgb_to_ir_density"]), valid,
                display_max, target,
            ),
            overlay_shared_density(
                ir, np.asarray(item["rgca_rgb_to_ir_density"]), valid,
                display_max, target,
            ),
        ]
        metric_keys = (
            None, None,
            "lcaf_ir_to_rgb", "rgca_ir_to_rgb",
            "lcaf_rgb_to_ir", "rgca_rgb_to_ir",
        )
        for column, (axis, panel, metric_key) in enumerate(
            zip(row_axes, panels, metric_keys)
        ):
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#707070")
                spine.set_linewidth(0.45)
            if row_index == 0:
                axis.set_title(PANEL_TITLES[column], fontsize=9.4, pad=6)
            if metric_key:
                metrics = item[f"{metric_key}_metrics"]
                axis.set_xlabel(
                    f"E-box {float(metrics['ebpg']):.3f}  |  Cov10 {float(metrics['target_coverage_top10_valid_pixels']):.3f}",
                    fontsize=7.3,
                    labelpad=3,
                )
        row_axes[0].set_ylabel(
            f"{item['scene_display']}\nID {item['image_id']}\n"
            f"GT {item['gt_count']} | target #{int(item['target_index'])}\n"
            f"conf L/R {float(item['lcaf_confidence']):.2f}/{float(item['rgca_confidence']):.2f}",
            fontsize=8.1,
            labelpad=5,
        )

    fig.suptitle(
        "M3FD | Strict attention-path comparison with identical Mask+GFB post-fusion",
        fontsize=14.7,
        y=0.968,
    )
    color_axis = fig.add_axes([0.35, 0.168, 0.30, 0.012])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, display_max), cmap="turbo"),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        "Positive cross-modal contribution density (one global shared scale)",
        fontsize=8,
        labelpad=2,
    )
    colorbar.ax.tick_params(labelsize=7)

    bar_axis = fig.add_axes([0.16, 0.032, 0.68, 0.095])
    groups = (
        ("LCAF\nIR→RGB", "lcaf_ir_to_rgb"),
        ("RGCA\nIR→RGB", "rgca_ir_to_rgb"),
        ("LCAF\nRGB→IR", "lcaf_rgb_to_ir"),
        ("RGCA\nRGB→IR", "rgca_rgb_to_ir"),
    )
    scene_colors = ("#d89000", "#6040b0", "#4682b4", "#bd3f3f")
    for group_index, (label, key) in enumerate(groups):
        values = [float(item[f"{key}_metrics"]["ebpg"]) for item in items]
        bar_axis.bar(
            group_index, float(np.mean(values)), width=0.58,
            color="#d8d8d8", edgecolor="#555555", linewidth=0.7,
        )
        offsets = np.linspace(-0.16, 0.16, len(values))
        for offset, value, color in zip(offsets, values, scene_colors):
            bar_axis.scatter(
                group_index + offset, value, s=22, color=color,
                edgecolor="white", linewidth=0.4, zorder=3,
            )
        bar_axis.text(
            group_index, min(1.0, float(np.mean(values)) + 0.035),
            f"{float(np.mean(values)):.3f}", ha="center", va="bottom", fontsize=7.5,
        )
    bar_axis.set_xticks(range(len(groups)), [label for label, _key in groups], fontsize=7.5)
    bar_axis.set_ylabel("E-box ↑", fontsize=8)
    bar_axis.set_ylim(0, 1.03)
    bar_axis.grid(axis="y", color="#dddddd", linewidth=0.55)
    bar_axis.set_axisbelow(True)
    bar_axis.set_title(
        "Box-energy fraction for the four seeded-random scene exemplars (points=scenes; bars=mean)",
        fontsize=8.4,
        pad=3,
    )
    for spine in ("top", "right"):
        bar_axis.spines[spine].set_visible(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "m3fd_strict_rgca_vs_lcaf_cross_attention_four_scenes.png"
    pdf_path = output_dir / "m3fd_strict_rgca_vs_lcaf_cross_attention_four_scenes.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path, display_max


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def serializable_item(item: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in item.items()
        if not isinstance(value, (np.ndarray, torch.Tensor))
    }


def main() -> None:
    args = parse_args()
    if args.min_gt < 1:
        raise ValueError("--min-gt must be positive")
    if not 0.5 < args.display_quantile < 1.0:
        raise ValueError("--display-quantile must be between 0.5 and 1")
    for path in (args.rgca_weights, args.lcaf_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Loading pure RGCA: {args.rgca_weights}")
    rgca_model = attempt_load(
        str(args.rgca_weights), map_location=device
    ).to(device).float().eval()
    print(f"Loading strict LCAF attention control: {args.lcaf_weights}")
    lcaf_model = attempt_load(
        str(args.lcaf_weights), map_location=device
    ).to(device).float().eval()
    lcaf_restore_audit = restore_strict_lcaf_layers(lcaf_model, device)
    model_audit = validate_models(rgca_model, lcaf_model)
    for model in (rgca_model, lcaf_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    permutations, ambiguous, scene_counts = build_scene_permutations(
        args.dataset_root, args.scene_root, args.seed
    )
    selected, selection_audit = select_scene_samples(
        permutations, rgca_model, lcaf_model, device, args
    )

    rgca_capture = CrossModalContributionCapture(rgca_model, "rgca")
    lcaf_capture = CrossModalContributionCapture(lcaf_model, "lcaf")
    metric_rows: List[Dict[str, object]] = []
    raw_archive: Dict[str, np.ndarray] = {}
    try:
        for item in selected:
            target = np.asarray(item["target"], dtype=np.float32)
            tensor = item["tensor"]
            rgca_maps, rgca_conf, rgca_level, rgca_index = compute_directional_maps(
                rgca_model, rgca_capture, tensor, target, device, args.img_size
            )
            lcaf_maps, lcaf_conf, lcaf_level, lcaf_index = compute_directional_maps(
                lcaf_model, lcaf_capture, tensor, target, device, args.img_size
            )
            item["rgca_confidence"] = rgca_conf
            item["lcaf_confidence"] = lcaf_conf
            item["rgca_detect_level"] = rgca_level
            item["lcaf_detect_level"] = lcaf_level
            item["rgca_prediction_index"] = rgca_index
            item["lcaf_prediction_index"] = lcaf_index
            for model_name, maps in (("rgca", rgca_maps), ("lcaf", lcaf_maps)):
                for direction, cam in maps.items():
                    density, metrics = density_and_metrics(
                        cam, target, np.asarray(item["valid"], dtype=bool)
                    )
                    item[f"{model_name}_{direction}_raw"] = cam
                    item[f"{model_name}_{direction}_density"] = density
                    item[f"{model_name}_{direction}_metrics"] = metrics
                    prefix = f"{item['scene']}_{item['image_id']}_{model_name}_{direction}"
                    raw_archive[f"{prefix}_raw"] = cam.astype(np.float32)
                    raw_archive[f"{prefix}_density"] = density.astype(np.float32)
                    class_id = int(target[0])
                    metric_rows.append({
                        "scene": item["scene"],
                        "image_id": item["image_id"],
                        "gt_count": item["gt_count"],
                        "target_index": item["target_index"],
                        "target_class_id": class_id,
                        "target_class_name": (
                            CLASS_NAMES[class_id]
                            if 0 <= class_id < len(CLASS_NAMES)
                            else str(class_id)
                        ),
                        "model": model_name,
                        "direction": direction,
                        "target_confidence": item[f"{model_name}_confidence"],
                        "target_area_fraction": item["target_area_fraction"],
                        **metrics,
                    })
            print(
                f"Attributed {item['scene_display']} ID {item['image_id']}: "
                f"E-box IR→RGB L/R="
                f"{item['lcaf_ir_to_rgb_metrics']['ebpg']:.3f}/"
                f"{item['rgca_ir_to_rgb_metrics']['ebpg']:.3f}; "
                f"RGB→IR L/R="
                f"{item['lcaf_rgb_to_ir_metrics']['ebpg']:.3f}/"
                f"{item['rgca_rgb_to_ir_metrics']['ebpg']:.3f}"
            )
    finally:
        rgca_capture.close()
        lcaf_capture.close()

    png_path, pdf_path, display_max = render_figure(
        selected, args.output_dir, args.display_quantile, args.dpi
    )
    write_csv(args.output_dir / "selected_scene_metrics.csv", metric_rows)
    np.savez_compressed(args.output_dir / "raw_cross_attention_maps.npz", **raw_archive)

    mean_ebpg = {}
    for model_name in ("lcaf", "rgca"):
        for direction in DIRECTIONS:
            values = [
                float(item[f"{model_name}_{direction}_metrics"]["ebpg"])
                for item in selected
            ]
            mean_ebpg[f"{model_name}_{direction}"] = float(np.mean(values))
    config = {
        "task": "M3FD four-scene strict RGCA-vs-LCAF cross-attention contribution figure",
        "layout": list(PANEL_TITLES),
        "sampling": {
            "method": (
                "per scene: first eligible ID in a seeded uniform random permutation; "
                "eligibility uses pairing, min GT, largest-target area, and shared centred "
                "same-class prediction confidence only; no heatmap metric is used"
            ),
            "seed": args.seed,
            "min_gt": args.min_gt,
            "min_confidence": args.min_confidence,
            "target_area_fraction_range": [args.min_target_area, args.max_target_area],
            "scene_manifest_intersection_counts": scene_counts,
            "excluded_ambiguous_scene_ids": ambiguous,
            "selection_audit": selection_audit,
        },
        "target_definition": "largest valid GT instance in each selected image",
        "model_audit": model_audit,
        "legacy_lcaf_runtime_restore": lcaf_restore_audit,
        "rgca_identity": "pure prior-free RGCA + Foreground Mask + GFB",
        "rgca_weights": str(args.rgca_weights.resolve()),
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_identity": "original LCAF attention + identical Foreground Mask + GFB",
        "lcaf_weights": str(args.lcaf_weights.resolve()),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "map_definition": (
            "target-conditioned positive contribution ReLU(sum_c activation*gradient), "
            "summed over P2-P5 directional cross-attention residuals"
        ),
        "rgca_hook": "post-reliability-gate delta_rgb/delta_ir",
        "lcaf_hook": "mhca_rgb/mhca_ir original cross-attention residual",
        "spatial_density_normalization": "divide each positive map by its valid-image mean",
        "display_scale": {
            "shared_across_all_16_maps": True,
            "quantile": args.display_quantile,
            "maximum_density": display_max,
        },
        "gt_used_to_shape_or_normalize_maps": False,
        "mean_ebpg_selected_four": mean_ebpg,
        "selected": [serializable_item(item) for item in selected],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "metrics_csv": str((args.output_dir / "selected_scene_metrics.csv").resolve()),
            "raw_npz": str((args.output_dir / "raw_cross_attention_maps.npz").resolve()),
        },
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    (args.output_dir / "SCIENTIFIC_CAVEAT.txt").write_text(
        "The four images are seeded-random scene exemplars after non-heatmap eligibility filters. "
        "They are not selected to maximize RGCA visual advantage and do not replace a full-test "
        "distribution. E-box uses GT only for evaluation; GT does not shape or normalize the maps. "
        "The maps explain directional cross-attention residual contributions to one matched target "
        "prediction, not raw channel-attention matrices. Positive contribution maps omit inhibitory "
        "negative evidence.\n",
        encoding="utf-8",
    )
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Mean E-box: {mean_ebpg}")


if __name__ == "__main__":
    main()
