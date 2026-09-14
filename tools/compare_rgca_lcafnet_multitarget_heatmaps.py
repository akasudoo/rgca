#!/usr/bin/env python3
"""Create one seeded-random six-column RGCA-vs-LCAFNet M3FD heatmap row.

The spatial maps are modality-specific, target-conditioned multiscale
HiResCAM at the RGB/IR P2-P5 backbone features.  Both cross-attention modules
use channel-by-channel attention, so their raw attention matrices cannot be
honestly reshaped into HxW maps.  HiResCAM instead answers where each modality
contributed to the detector score after the corresponding cross-attention.

Unlike the older single-target exporter, every GT instance is explained
separately.  Each target-specific map is normalized by its own GT-region 95th
percentile and the maps are merged with a pixelwise maximum.  This prevents a
large/high-confidence instance from hiding the other instances while retaining
background activation as evidence of poor spatial focus.  No box is painted on
the final six-column image.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.export_rgca_gate_paired_modal_heatmaps import (  # noqa: E402
    BranchActivationCapture,
    aggregate_hirescam,
    locate_target_prediction,
    overlay_heatmap,
)
from tools.visualize_rgca_panel_a import (  # noqa: E402
    CLASS_NAMES,
    prepare_pair,
    read_yolo_labels,
    resolve_device,
    sha256sum,
)
from tools.compare_rgca_gate_gradcam import (  # noqa: E402
    target_mask,
    valid_image_mask,
)


PANEL_TITLES = (
    "RGB",
    "IR",
    "RGCA | RGB heatmap",
    "RGCA | IR heatmap",
    "LCAFNet | RGB heatmap",
    "LCAFNet | IR heatmap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgca-weights",
        type=Path,
        default=(REPO_ROOT / "runs/train/m3fd_rgca_mask_gfb_best/weights/best.pt"),
    )
    parser.add_argument(
        "--lcafnet-weights",
        type=Path,
        default=(REPO_ROOT / "runs/train/lcafnet_original_close_mosaic103/weights/best.pt"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/dell/lcp/dataset/M3FD")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "paper/rgca_vs_lcafnet_multitarget_heatmaps_m3fd"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument(
        "--image-id",
        default=None,
        help="Optional exact image ID. If omitted, use seeded uniform random sampling.",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--target-quantile", type=float, default=0.95)
    parser.add_argument("--overlay-quantile", type=float, default=0.995)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def paired_paths(dataset_root: Path, split: str, image_id: str) -> Tuple[Path, Path, Path]:
    rgb_path = dataset_root / f"images/vis_{split}" / f"{image_id}.png"
    ir_path = dataset_root / f"images/Ir_{split}" / f"{image_id}.png"
    label_path = dataset_root / f"labels/vis_{split}" / f"{image_id}.txt"
    for path in (rgb_path, ir_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return rgb_path, ir_path, label_path


def choose_pair(args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, object]]:
    image_dir = args.dataset_root / f"images/vis_{args.split}"
    image_ids = sorted(path.stem for path in image_dir.glob("*.png"))
    if not image_ids:
        raise RuntimeError(f"No PNG images found in {image_dir}")
    missing: List[str] = []
    rejected_gt_count: List[Tuple[str, int]] = []
    if args.image_id is None:
        # A uniformly shuffled order followed by the first eligible element is
        # exactly a uniform draw over the eligible subset.  It avoids hundreds
        # of slow stat/loadtxt calls on network-mounted datasets.
        candidates = image_ids.copy()
        random.Random(args.seed).shuffle(candidates)
        selection = "first eligible ID in a uniformly seeded permutation of all sorted RGB IDs"
    else:
        candidates = [args.image_id]
        selection = "explicit --image-id override"

    chosen = None
    for permutation_position, image_id in enumerate(candidates):
        rgb_path = image_dir / f"{image_id}.png"
        ir_path = args.dataset_root / f"images/Ir_{args.split}" / f"{image_id}.png"
        label_path = args.dataset_root / f"labels/vis_{args.split}" / f"{image_id}.txt"
        if not ir_path.is_file() or not label_path.is_file():
            missing.append(image_id)
            continue
        gt_count = len(read_yolo_labels(label_path))
        if gt_count < args.min_gt:
            rejected_gt_count.append((image_id, gt_count))
            continue
        chosen = (image_id, gt_count, permutation_position, rgb_path, ir_path, label_path)
        break
    if chosen is None:
        if args.image_id is not None:
            raise ValueError(
                f"Requested ID {args.image_id!r} is not an eligible paired {args.split} image"
            )
        raise RuntimeError(
            f"No {args.split} pair has at least {args.min_gt} valid GT instances"
        )
    image_id, gt_count, permutation_position, rgb_path, ir_path, label_path = chosen
    selected = {
        "image_id": image_id,
        "gt_count": gt_count,
        "rgb_path": str(rgb_path.resolve()),
        "ir_path": str(ir_path.resolve()),
        "label_path": str(label_path.resolve()),
        "random_permutation_position_zero_based": permutation_position,
    }
    audit = {
        "selection_method": selection,
        "seed": args.seed,
        "split": args.split,
        "minimum_gt_count": args.min_gt,
        "canonical_rgb_count": len(image_ids),
        "examined_before_selection": permutation_position + 1,
        "missing_pair_ids_before_selection": missing,
        "rejected_below_min_gt_before_selection": [
            {"image_id": image_id, "gt_count": count}
            for image_id, count in rejected_gt_count
        ],
    }
    return selected, audit


def validate_models(rgca_model: torch.nn.Module, lcafnet_model: torch.nn.Module) -> Dict[str, object]:
    if len(rgca_model.model) != 46 or len(lcafnet_model.model) != 46:
        raise RuntimeError("Expected both M3FD detectors to contain 46 top-level layers")
    rgca_fusion_names = [type(rgca_model.model[index]).__name__ for index in range(20, 24)]
    lcaf_fusion_names = [type(lcafnet_model.model[index]).__name__ for index in range(20, 24)]
    for index in range(20, 24):
        rgca_layer = rgca_model.model[index]
        if not hasattr(rgca_layer, "cross_modal_attention"):
            raise TypeError(f"RGCA layer {index} has no cross_modal_attention")
        attention = rgca_layer.cross_modal_attention
        if getattr(attention, "reliability_gate", None) is None:
            raise TypeError(f"RGCA layer {index} has no learned reliability gate")
        if not hasattr(rgca_layer, "foreground_mask") or not hasattr(rgca_layer, "fusion"):
            raise TypeError(f"RGCA layer {index} is missing Mask/GFB")

        lcaf_layer = lcafnet_model.model[index]
        if not hasattr(lcaf_layer, "mhca_rgb") or not hasattr(lcaf_layer, "mhca_ir"):
            raise TypeError(f"LCAFNet layer {index} has no bidirectional LCAF attention")
        # Historical original-LCAFNet checkpoints contain concat/conv/dwconv
        # directly and intentionally predate the foreground-mask branch.
        if not all(hasattr(lcaf_layer, name) for name in ("concat", "conv", "dwconv")):
            raise TypeError(f"LCAFNet layer {index} is missing its original GFB modules")

    return {
        "rgca_top_level_fusion_classes": rgca_fusion_names,
        "lcafnet_top_level_fusion_classes": lcaf_fusion_names,
        "rgca_stride": [float(value) for value in rgca_model.stride],
        "lcafnet_stride": [float(value) for value in lcafnet_model.stride],
    }


def compute_one_target(
    model: torch.nn.Module,
    capture: BranchActivationCapture,
    tensor: torch.Tensor,
    target: np.ndarray,
    device: torch.device,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray, float, str, int]:
    capture.clear()
    batch = tensor.to(device, non_blocking=True).detach().requires_grad_(True)
    output = model(batch[:, :3], batch[:, 3:])
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("Expected inference output (predictions, logits, raw_levels)")
    predictions, raw_levels = output[0], output[2]
    score, confidence, detect_level, prediction_index = locate_target_prediction(
        predictions, raw_levels, target
    )
    rgb_activations = [capture.activations[("rgb", level)] for level in range(4)]
    ir_activations = [capture.activations[("ir", level)] for level in range(4)]
    activations = rgb_activations + ir_activations
    gradients = torch.autograd.grad(
        score, activations, retain_graph=False, allow_unused=True
    )
    rgb_cam = aggregate_hirescam(rgb_activations, gradients[:4], img_size)
    ir_cam = aggregate_hirescam(ir_activations, gradients[4:], img_size)
    return rgb_cam, ir_cam, confidence, detect_level, prediction_index


def target_balanced_map(
    cams: Sequence[np.ndarray],
    boxes: np.ndarray,
    valid: np.ndarray,
    quantile: float,
) -> Tuple[np.ndarray, List[Dict[str, float]], List[np.ndarray]]:
    if len(cams) != len(boxes):
        raise ValueError("The number of target CAMs must match the number of GT boxes")
    balanced: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []
    eps = 1e-12
    for cam, box in zip(cams, boxes):
        positive = np.maximum(np.asarray(cam, dtype=np.float32), 0)
        roi = target_mask(box, positive.shape[0]) & valid
        values = positive[roi]
        scale_source = "gt_region"
        if not len(values) or float(values.max()) <= eps:
            values = positive[valid]
            scale_source = "valid_image_fallback"
        scale = float(np.quantile(values, quantile)) if len(values) else 0.0
        if scale <= eps:
            normalized = np.zeros_like(positive, dtype=np.float32)
        else:
            normalized = np.clip(positive / scale, 0.0, 1.0).astype(np.float32)
        normalized[~valid] = 0.0
        balanced.append(normalized)
        rows.append(
            {
                "normalization_scale": scale,
                "normalization_source_gt_region": float(scale_source == "gt_region"),
                "own_gt_peak_after_normalization": float(normalized[roi].max()) if roi.any() else 0.0,
                "own_gt_mean_after_normalization": float(normalized[roi].mean()) if roi.any() else 0.0,
            }
        )
    merged = np.maximum.reduce(balanced) if balanced else np.zeros_like(valid, dtype=np.float32)
    return merged.astype(np.float32), rows, balanced


def merged_map_metrics(cam: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    foreground = np.zeros_like(valid, dtype=bool)
    per_target_peaks = []
    per_target_active = []
    for box in boxes:
        roi = target_mask(box, cam.shape[0]) & valid
        foreground |= roi
        per_target_peaks.append(float(cam[roi].max()) if roi.any() else 0.0)
        per_target_active.append(float((cam[roi] >= 0.5).mean()) if roi.any() else 0.0)
    background = valid & ~foreground
    fg_mean = float(cam[foreground].mean()) if foreground.any() else 0.0
    bg_mean = float(cam[background].mean()) if background.any() else 0.0
    return {
        "all_target_peak_min": min(per_target_peaks, default=0.0),
        "all_target_peak_mean": float(np.mean(per_target_peaks)) if per_target_peaks else 0.0,
        "all_target_active_fraction_min": min(per_target_active, default=0.0),
        "foreground_mean": fg_mean,
        "background_mean": bg_mean,
        "foreground_background_ratio": fg_mean / max(bg_mean, 1e-12),
    }


def render_six_columns(
    rgb: np.ndarray,
    ir: np.ndarray,
    overlays: Sequence[np.ndarray],
    image_id: str,
    gt_count: int,
    output_dir: Path,
    dpi: int,
) -> Tuple[Path, Path]:
    panels = [rgb, ir, *overlays]
    fig, axes = plt.subplots(1, 6, figsize=(19.2, 3.55))
    fig.subplots_adjust(left=0.006, right=0.997, top=0.83, bottom=0.035, wspace=0.025)
    for axis, panel, title in zip(axes, panels, PANEL_TITLES):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10, pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.45)
    fig.suptitle(
        f"M3FD test ID {image_id} | all {gt_count} GT instances | target-balanced P2-P5 HiResCAM",
        fontsize=13,
        y=0.965,
    )
    png_path = output_dir / f"m3fd_{image_id}_rgca_vs_lcafnet_multitarget_6col.png"
    pdf_path = output_dir / f"m3fd_{image_id}_rgca_vs_lcafnet_multitarget_6col.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.min_gt < 2:
        raise ValueError("--min-gt must be at least 2 for a multi-target comparison")
    if not 0.5 < args.target_quantile < 1.0:
        raise ValueError("--target-quantile must be between 0.5 and 1")
    if not 0.5 < args.overlay_quantile < 1.0:
        raise ValueError("--overlay-quantile must be between 0.5 and 1")
    for path in (args.rgca_weights, args.lcafnet_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected, selection_audit = choose_pair(args)
    rgb_path = Path(str(selected["rgb_path"]))
    ir_path = Path(str(selected["ir_path"]))
    label_path = Path(str(selected["label_path"]))
    labels = read_yolo_labels(label_path)
    rgb, ir, tensor, boxes = prepare_pair(rgb_path, ir_path, labels, args.img_size)
    valid = valid_image_mask(rgb_path, args.img_size)
    device = resolve_device(args.device)

    print(f"Selected M3FD {args.split} ID {selected['image_id']} with {len(boxes)} GT instances")
    print(f"Loading RGCA checkpoint: {args.rgca_weights}")
    rgca_model = attempt_load(str(args.rgca_weights), map_location=device).to(device).float().eval()
    print(f"Loading LCAFNet checkpoint: {args.lcafnet_weights}")
    lcafnet_model = attempt_load(str(args.lcafnet_weights), map_location=device).to(device).float().eval()
    model_audit = validate_models(rgca_model, lcafnet_model)
    for model in (rgca_model, lcafnet_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    model_maps: Dict[str, Dict[str, object]] = {}
    target_rows: List[Dict[str, object]] = []
    for model_name, model in (("rgca", rgca_model), ("lcafnet", lcafnet_model)):
        capture = BranchActivationCapture(model)
        rgb_cams: List[np.ndarray] = []
        ir_cams: List[np.ndarray] = []
        model_target_rows: List[Dict[str, object]] = []
        try:
            for target_index, target in enumerate(boxes):
                print(f"  {model_name}: target {target_index + 1}/{len(boxes)}")
                rgb_cam, ir_cam, confidence, level, prediction_index = compute_one_target(
                    model, capture, tensor, target, device, args.img_size
                )
                rgb_cams.append(rgb_cam)
                ir_cams.append(ir_cam)
                class_id = int(target[0])
                model_target_rows.append(
                    {
                        "image_id": selected["image_id"],
                        "model": model_name,
                        "target_index": target_index,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                        "x1": float(target[1]),
                        "y1": float(target[2]),
                        "x2": float(target[3]),
                        "y2": float(target[4]),
                        "target_prediction_confidence": confidence,
                        "detect_level": level,
                        "prediction_index": prediction_index,
                    }
                )
        finally:
            capture.close()

        rgb_merged, rgb_balance_rows, rgb_balanced = target_balanced_map(
            rgb_cams, boxes, valid, args.target_quantile
        )
        ir_merged, ir_balance_rows, ir_balanced = target_balanced_map(
            ir_cams, boxes, valid, args.target_quantile
        )
        for row, rgb_stats, ir_stats in zip(
            model_target_rows, rgb_balance_rows, ir_balance_rows
        ):
            for key, value in rgb_stats.items():
                row[f"rgb_{key}"] = value
            for key, value in ir_stats.items():
                row[f"ir_{key}"] = value
            target_rows.append(row)
        model_maps[model_name] = {
            "rgb_merged": rgb_merged,
            "ir_merged": ir_merged,
            "rgb_raw": rgb_cams,
            "ir_raw": ir_cams,
            "rgb_balanced": rgb_balanced,
            "ir_balanced": ir_balanced,
        }

    rgca_rgb_overlay, _ = overlay_heatmap(
        rgb, model_maps["rgca"]["rgb_merged"], valid, args.overlay_quantile
    )
    rgca_ir_overlay, _ = overlay_heatmap(
        ir, model_maps["rgca"]["ir_merged"], valid, args.overlay_quantile
    )
    lcaf_rgb_overlay, _ = overlay_heatmap(
        rgb, model_maps["lcafnet"]["rgb_merged"], valid, args.overlay_quantile
    )
    lcaf_ir_overlay, _ = overlay_heatmap(
        ir, model_maps["lcafnet"]["ir_merged"], valid, args.overlay_quantile
    )
    png_path, pdf_path = render_six_columns(
        rgb,
        ir,
        (rgca_rgb_overlay, rgca_ir_overlay, lcaf_rgb_overlay, lcaf_ir_overlay),
        str(selected["image_id"]),
        len(boxes),
        args.output_dir,
        args.dpi,
    )

    raw_dir = args.output_dir / "raw_maps"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for model_name, maps in model_maps.items():
        for modality in ("rgb", "ir"):
            np.save(
                raw_dir / f"{model_name}_{modality}_target_balanced_merged.npy",
                np.asarray(maps[f"{modality}_merged"], dtype=np.float32),
                allow_pickle=False,
            )
    write_csv(args.output_dir / "per_target_metrics.csv", target_rows)

    summary = {}
    for model_name, maps in model_maps.items():
        for modality in ("rgb", "ir"):
            summary[f"{model_name}_{modality}"] = merged_map_metrics(
                np.asarray(maps[f"{modality}_merged"]), boxes, valid
            )
    config = {
        "task": "seeded-random M3FD multi-target RGCA-vs-original-LCAFNet six-column heatmap",
        "layout": list(PANEL_TITLES),
        "selected": selected,
        "selection_audit": selection_audit,
        "model_audit": model_audit,
        "rgca_model_identity": "RGCA + Mask + GFB + BECP fine-tune",
        "rgca_weights": str(args.rgca_weights.resolve()),
        "rgca_weights_sha256": sha256sum(args.rgca_weights),
        "lcafnet_model_identity": "original LCAF cross-attention + original GFB, no Foreground Mask",
        "lcafnet_weights": str(args.lcafnet_weights.resolve()),
        "lcafnet_weights_sha256": sha256sum(args.lcafnet_weights),
        "attribution_method": "modality-specific absolute HiResCAM aggregated over backbone P2-P5",
        "multi_target_method": (
            "one backward pass per GT target; normalize each target map by its own GT-region "
            f"{args.target_quantile:.3f} quantile; merge target maps by pixelwise maximum"
        ),
        "scientific_reason_for_hirescam": (
            "RGCA and original LCAF use channel cross-attention matrices, not spatial token attention; "
            "the channel matrices are not reshaped into HxW maps"
        ),
        "boxes_drawn": False,
        "img_size": args.img_size,
        "display_normalization_quantile": args.overlay_quantile,
        "summary_metrics": summary,
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "per_target_csv": str((args.output_dir / "per_target_metrics.csv").resolve()),
            "raw_maps": str(raw_dir.resolve()),
        },
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    (args.output_dir / "SCIENTIFIC_CAVEAT.txt").write_text(
        "These panels compare two independently trained checkpoints with different post-attention fusion "
        "configurations, so they are a qualitative model-level comparison rather than a strict causal "
        "ablation of attention alone. Each GT instance is used to define a target-specific detector score. "
        "Target-region normalization gives small and large instances equal visual opportunity, so colors "
        "show spatial distribution rather than absolute cross-model attribution magnitude. Raw channel "
        "attention is not visualized as an HxW map.\n",
        encoding="utf-8",
    )

    print(f"Six-column PNG: {png_path}")
    print(f"Six-column PDF: {pdf_path}")
    for key, values in summary.items():
        print(
            f"  {key}: min-target-peak={values['all_target_peak_min']:.3f}, "
            f"foreground/background={values['foreground_background_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
