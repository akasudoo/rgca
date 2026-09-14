#!/usr/bin/env python3
"""Compare target-conditioned detector Grad-CAM with/without RGCA gating.

For every eligible canonical M3FD test image, the largest GT object is used as
the target.  Each checkpoint independently selects its highest-confidence raw
prediction of the same class whose predicted centre lies inside that GT box.
Grad-CAM is then computed at the matching P2/P3/P4/P5 detector-input feature.

The paper figure deliberately omits boxes.  GT is used for target definition,
quantitative foreground enrichment, and transparent exemplar selection only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.select_rgca_reliability_representatives import build_candidates  # noqa: E402
from tools.visualize_rgca_panel_a import (  # noqa: E402
    SCENES,
    find_rgca_layer,
    prepare_pair,
    read_yolo_labels,
    resolve_device,
    sha256sum,
    write_csv,
)


DETECT_INPUT_LAYERS = (35, 38, 41, 44)  # P2, P3, P4, P5
DETECT_LEVELS = ("P2", "P3", "P4", "P5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gated-weights", type=Path,
        default=REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt")
    parser.add_argument(
        "--no-gate-weights", type=Path,
        default=(REPO_ROOT / "runs/train/"
                 "exp_rgca_no_reliability_mask_gfb_uniform_lr_m3fd_full200/weights/best.pt"))
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/dell/lcp/dataset/M3FD"))
    parser.add_argument(
        "--scene-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "paper/rgca_gate_gradcam_comparison_m3fd")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--rgca-layer", default="model.21.cross_modal_attention")
    parser.add_argument("--min-gt", type=int, default=1)
    parser.add_argument("--min-target-area", type=float, default=0.0015)
    parser.add_argument("--max-target-area", type=float, default=0.25)
    parser.add_argument("--min-target-confidence", type=float, default=0.10)
    parser.add_argument("--min-cam-support", type=float, default=0.01)
    parser.add_argument("--max-target-cam-mass", type=float, default=0.98)
    parser.add_argument("--device", default="0")
    parser.add_argument("--display-quantile", type=float, default=0.995)
    return parser.parse_args()


class ActivationCapture:
    def __init__(self, model: torch.nn.Module, layer_indices: Sequence[int]):
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles = []
        for level, layer_index in enumerate(layer_indices):
            layer = model.model[layer_index]
            self.handles.append(layer.register_forward_hook(self._hook(level)))

    def _hook(self, level: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Detector input level {level} did not return a tensor")
            self.activations[level] = output
        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def valid_image_mask(rgb_path: Path, img_size: int) -> np.ndarray:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read {rgb_path}")
    h0, w0 = image.shape[:2]
    # Match prepare_pair exactly: it first resizes the longest side to
    # img_size even when the source is smaller, then letterboxes with ratio 1.
    gain = min(img_size / h0, img_size / w0)
    new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
    dw, dh = (img_size - new_w) / 2.0, (img_size - new_h) / 2.0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    mask = np.zeros((img_size, img_size), dtype=bool)
    mask[top:img_size - bottom, left:img_size - right] = True
    return mask


def largest_target(boxes: np.ndarray) -> Tuple[np.ndarray, int, float]:
    if not len(boxes):
        raise ValueError("At least one GT box is required")
    areas = np.maximum(boxes[:, 3] - boxes[:, 1], 0) * np.maximum(
        boxes[:, 4] - boxes[:, 2], 0)
    index = int(np.argmax(areas))
    return boxes[index].copy(), index, float(areas[index])


def target_mask(target: np.ndarray, img_size: int) -> np.ndarray:
    _, x1, y1, x2, y2 = target.tolist()
    xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xb, yb = min(img_size, int(np.ceil(x2))), min(img_size, int(np.ceil(y2)))
    mask = np.zeros((img_size, img_size), dtype=bool)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = True
    return mask


def global_to_level_index(raw_levels: Sequence[torch.Tensor], index: int) -> Tuple[int, int]:
    offset = 0
    for level, raw in enumerate(raw_levels):
        count = int(np.prod(raw.shape[1:4]))
        if index < offset + count:
            return level, index - offset
        offset += count
    raise IndexError(f"Prediction index {index} exceeds {offset} candidates")


def compute_target_gradcam(
    model: torch.nn.Module,
    capture: ActivationCapture,
    tensor: torch.Tensor,
    target: np.ndarray,
    device: torch.device,
    img_size: int,
) -> Tuple[np.ndarray, float, str, int]:
    capture.clear()
    batch = tensor.to(device, non_blocking=True).detach().requires_grad_(True)
    output = model(batch[:, :3], batch[:, 3:])
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("Expected inference output (predictions, logits, raw_levels)")
    predictions, raw_levels = output[0], output[2]
    if len(raw_levels) != len(DETECT_INPUT_LAYERS):
        raise RuntimeError(f"Expected four detection levels, got {len(raw_levels)}")

    class_id, x1, y1, x2, y2 = target.tolist()
    class_id = int(class_id)
    detached = predictions[0].detach()
    if 5 + class_id >= detached.shape[1]:
        raise IndexError(f"GT class {class_id} is outside the detector output")
    confidence = detached[:, 4] * detached[:, 5 + class_id]
    centres = detached[:, :2]
    inside = ((centres[:, 0] >= x1) & (centres[:, 0] <= x2)
              & (centres[:, 1] >= y1) & (centres[:, 1] <= y2))
    if inside.any():
        masked = confidence.masked_fill(~inside, -1.0)
        global_index = int(masked.argmax())
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        distance = (centres[:, 0] - cx).square() + (centres[:, 1] - cy).square()
        scale = max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1.0)
        global_index = int((confidence - distance / scale).argmax())

    level, local_index = global_to_level_index(raw_levels, global_index)
    raw_flat = raw_levels[level].reshape(1, -1, raw_levels[level].shape[-1])
    raw_target = raw_flat[0, local_index]
    score = raw_target[4].sigmoid() * raw_target[5 + class_id].sigmoid()
    activation = capture.activations[level]
    gradient = torch.autograd.grad(score, activation, retain_graph=False)[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)
    array = cam[0, 0].detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("Grad-CAM contains non-finite values")
    return array, float(score.detach().cpu()), DETECT_LEVELS[level], global_index


def focus_metrics(
    gated: np.ndarray,
    no_gate: np.ndarray,
    foreground: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-12
    foreground = foreground & valid
    background = valid & ~foreground
    if not foreground.any() or not background.any():
        raise ValueError("Invalid foreground/background mask")

    def density_and_enrichment(cam: np.ndarray):
        clipped = np.maximum(cam, 0)
        valid_mean = float(clipped[valid].mean())
        density = clipped / max(valid_mean, eps)
        fg_mean = float(density[foreground].mean())
        bg_mean = float(density[background].mean())
        enrichment = fg_mean / max(bg_mean, eps)
        mass_fraction = float(clipped[foreground].sum() / max(clipped[valid].sum(), eps))
        return density, fg_mean, bg_mean, enrichment, mass_fraction

    gd, gf, gb, ge, gm = density_and_enrichment(gated)
    nd, nf, nb, ne, nm = density_and_enrichment(no_gate)
    return {
        "target_area_fraction": float(foreground.sum() / valid.sum()),
        "gated_foreground_density": gf,
        "gated_background_density": gb,
        "no_gate_foreground_density": nf,
        "no_gate_background_density": nb,
        "gated_foreground_enrichment": ge,
        "no_gate_foreground_enrichment": ne,
        "gated_target_cam_mass_fraction": gm,
        "no_gate_target_cam_mass_fraction": nm,
        "foreground_enrichment_difference": ge - ne,
        "foreground_enrichment_log2_gain": float(np.log2((ge + eps) / (ne + eps))),
        "target_cam_mass_fraction_difference": gm - nm,
        "gated_cam_nonzero_fraction": float((gated[valid] > 0).mean()),
        "no_gate_cam_nonzero_fraction": float((no_gate[valid] > 0).mean()),
        "gated_cam_mean": float(gated[valid].mean()),
        "no_gate_cam_mean": float(no_gate[valid].mean()),
        "gated_density": gd,
        "no_gate_density": nd,
    }


def scan(
    gated_model: torch.nn.Module,
    no_gate_model: torch.nn.Module,
    candidates: Sequence[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
    gated_capture = ActivationCapture(gated_model, DETECT_INPUT_LAYERS)
    no_gate_capture = ActivationCapture(no_gate_model, DETECT_INPUT_LAYERS)
    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    try:
        for candidate in tqdm(candidates, desc="Target-conditioned Grad-CAM", unit="image"):
            labels = read_yolo_labels(Path(str(candidate["label_path"])))
            rgb, ir, tensor, boxes = prepare_pair(
                Path(str(candidate["rgb_path"])), Path(str(candidate["ir_path"])),
                labels, args.img_size)
            target, target_index, area = largest_target(boxes)
            valid = valid_image_mask(Path(str(candidate["rgb_path"])), args.img_size)
            area_fraction = area / max(float(valid.sum()), 1.0)
            if not args.min_target_area <= area_fraction <= args.max_target_area:
                continue
            foreground = target_mask(target, args.img_size)
            if not (foreground & valid).any() or not (valid & ~foreground).any():
                continue
            gated_cam, gated_conf, gated_level, gated_pred_index = compute_target_gradcam(
                gated_model, gated_capture, tensor, target, device, args.img_size)
            no_gate_cam, no_gate_conf, no_gate_level, no_gate_pred_index = compute_target_gradcam(
                no_gate_model, no_gate_capture, tensor, target, device, args.img_size)
            # A completely inactive ReLU Grad-CAM has no spatial interpretation;
            # exclude it instead of creating a divide-by-epsilon focus score.
            if (float(gated_cam[valid].max()) <= 1e-12
                    or float(no_gate_cam[valid].max()) <= 1e-12):
                continue
            metrics = focus_metrics(gated_cam, no_gate_cam, foreground, valid)
            metrics.pop("gated_density")
            metrics.pop("no_gate_density")
            row = {
                **candidate,
                "target_box_index": target_index,
                "target_class": int(target[0]),
                "target_x1": float(target[1]), "target_y1": float(target[2]),
                "target_x2": float(target[3]), "target_y2": float(target[4]),
                "gated_target_confidence": gated_conf,
                "no_gate_target_confidence": no_gate_conf,
                "gated_detect_level": gated_level,
                "no_gate_detect_level": no_gate_level,
                "gated_prediction_index": gated_pred_index,
                "no_gate_prediction_index": no_gate_pred_index,
                **metrics,
            }
            rows.append(row)
            key = (str(candidate["scene"]), str(candidate["image_id"]))
            cache[key] = {
                "gated_cam": gated_cam, "no_gate_cam": no_gate_cam,
                "rgb": rgb, "ir": ir, "target": target, "valid": valid,
            }
    finally:
        gated_capture.close()
        no_gate_capture.close()
    return rows, cache


def rank_and_select(
    rows: Sequence[Dict[str, object]], min_confidence: float,
    min_cam_support: float, max_target_cam_mass: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ranked_all: List[Dict[str, object]] = []
    selected: List[Dict[str, object]] = []
    for scene, _ in SCENES:
        scene_rows = [dict(row) for row in rows if row["scene"] == scene]
        confident = [
            row for row in scene_rows
            if min(float(row["gated_target_confidence"]),
                   float(row["no_gate_target_confidence"])) >= min_confidence
            and min(float(row["gated_cam_nonzero_fraction"]),
                    float(row["no_gate_cam_nonzero_fraction"])) >= min_cam_support
            and max(float(row["gated_target_cam_mass_fraction"]),
                    float(row["no_gate_target_cam_mass_fraction"])) <= max_target_cam_mass
        ]
        pool = confident if confident else scene_rows
        pool.sort(key=lambda row: (
            -float(row["foreground_enrichment_log2_gain"]),
            -min(float(row["gated_target_confidence"]),
                 float(row["no_gate_target_confidence"])),
            str(row["image_id"]),
        ))
        if not pool:
            raise RuntimeError(f"No Grad-CAM result for scene {scene}")
        rank_lookup = {str(row["image_id"]): rank for rank, row in enumerate(pool, 1)}
        for row in scene_rows:
            row["eligible_for_selection"] = str(row["image_id"]) in rank_lookup
            row["scene_rank_by_focus_gain"] = rank_lookup.get(str(row["image_id"]), "")
            ranked_all.append(row)
        chosen = dict(pool[0])
        chosen["eligible_for_selection"] = True
        chosen["scene_rank_by_focus_gain"] = 1
        selected.append(chosen)
    return ranked_all, selected


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    groups = [(scene, [row for row in rows if row["scene"] == scene]) for scene, _ in SCENES]
    groups.append(("all", list(rows)))
    for scene, group in groups:
        gains = np.array([float(row["foreground_enrichment_log2_gain"]) for row in group])
        mass = np.array([float(row["target_cam_mass_fraction_difference"]) for row in group])
        output.append({
            "scene": scene,
            "n": len(group),
            "positive_focus_gain_fraction": float((gains > 0).mean()),
            "log2_focus_gain_mean": float(gains.mean()),
            "log2_focus_gain_median": float(np.median(gains)),
            "target_cam_mass_difference_mean": float(mass.mean()),
            "target_cam_mass_difference_median": float(np.median(mass)),
        })
    return output


def crop_target(
    base: np.ndarray, first: np.ndarray, second: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    _, x1, y1, x2, y2 = target.tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 2.2
    side = max(side, 112.0)
    xa, ya = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    xb = min(base.shape[1], int(cx + side / 2))
    yb = min(base.shape[0], int(cy + side / 2))
    base_crop = base[ya:yb, xa:xb]
    map_crop = np.concatenate((first[ya:yb, xa:xb], second[ya:yb, xa:xb]), axis=1)
    return np.concatenate((base_crop, base_crop), axis=1), map_crop, base_crop.shape[1]


def normalized_for_display(cam: np.ndarray, valid: np.ndarray, quantile: float) -> np.ndarray:
    positive = np.maximum(cam, 0)
    scale = float(np.quantile(positive[valid], quantile))
    return np.clip(positive / max(scale, 1e-12), 0, 1)


def render(
    selected: Sequence[Dict[str, object]],
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_dir: Path,
    quantile: float,
) -> None:
    items = []
    all_differences = []
    raw_dir = output_dir / "raw_maps"
    row_dir = output_dir / "rows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    display_names = dict(SCENES)
    for row in selected:
        key = (str(row["scene"]), str(row["image_id"]))
        data = cache[key]
        valid = data["valid"]
        gated_cam, no_gate_cam = data["gated_cam"], data["no_gate_cam"]
        gated_density = gated_cam / max(float(gated_cam[valid].mean()), 1e-12)
        no_gate_density = no_gate_cam / max(float(no_gate_cam[valid].mean()), 1e-12)
        difference = gated_density - no_gate_density
        all_differences.append(difference[valid])
        prefix = raw_dir / f"{row['scene']}_{row['image_id']}"
        for suffix, array in (
            ("gated_gradcam", gated_cam), ("no_gate_gradcam", no_gate_cam),
            ("gated_density", gated_density), ("no_gate_density", no_gate_density),
            ("density_difference", difference),
        ):
            np.save(str(prefix) + f"_{suffix}.npy", array.astype(np.float32), allow_pickle=False)
        items.append({
            **row, **data,
            "scene_display": display_names[str(row["scene"])],
            "gated_display": normalized_for_display(gated_cam, valid, quantile),
            "no_gate_display": normalized_for_display(no_gate_cam, valid, quantile),
            "difference": difference,
        })
    diff_limit = float(np.quantile(np.abs(np.concatenate(all_differences)), quantile))
    diff_limit = max(diff_limit, 1e-6)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(items), 6, figsize=(16.0, 10.4), constrained_layout=False)
    plt.subplots_adjust(left=0.055, right=0.992, top=0.925, bottom=0.105, wspace=0.065, hspace=0.075)
    titles = (
        "RGB", "IR", "RGCA w/o reliability gate", "RGCA with reliability gate",
        "With gate − without gate", "Target-region zoom")
    cmap = plt.get_cmap("turbo")
    def draw_item(row_axes, item, show_titles):
        rgb = item["rgb"]
        ir = item["ir"]
        base = rgb.astype(np.float32) / 255.0
        row_axes[0].imshow(rgb)
        row_axes[1].imshow(ir)
        for ax, display in ((row_axes[2], item["no_gate_display"]),
                            (row_axes[3], item["gated_display"])):
            ax.imshow(base)
            ax.imshow(display, cmap=cmap, vmin=0, vmax=1, alpha=0.58)
        row_axes[4].imshow(base)
        row_axes[4].imshow(item["difference"], cmap="RdBu_r",
                           vmin=-diff_limit, vmax=diff_limit, alpha=0.72)
        crop_base, crop_map, divider = crop_target(
            base, item["no_gate_display"], item["gated_display"], item["target"])
        row_axes[5].imshow(crop_base)
        row_axes[5].imshow(crop_map, cmap=cmap, vmin=0, vmax=1, alpha=0.62)
        row_axes[5].axvline(divider - 0.5, color="white", lw=0.7)
        row_axes[5].text(0.25, 0.03, "w/o gate", transform=row_axes[5].transAxes,
                         ha="center", va="bottom", color="white", fontsize=7,
                         bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})
        row_axes[5].text(0.75, 0.03, "with gate", transform=row_axes[5].transAxes,
                         ha="center", va="bottom", color="white", fontsize=7,
                         bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})
        row_axes[0].set_ylabel(
            f"{item['scene_display']}\nID {item['image_id']}", fontsize=9, labelpad=5)
        for col, ax in enumerate(row_axes):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45); spine.set_color("#777777")
            if show_titles:
                ax.set_title(titles[col], fontsize=8.5, pad=4)

    for row_index, (row_axes, item) in enumerate(zip(axes, items)):
        draw_item(row_axes, item, row_index == 0)

    fig.suptitle(
        "M3FD | Target-conditioned detector Grad-CAM without vs. with RGCA reliability gating",
        fontsize=15, y=0.965)
    response_axis = fig.add_axes([0.379, 0.060, 0.188, 0.011])
    response_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=response_axis, orientation="horizontal")
    response_bar.set_label("Per-map Grad-CAM intensity", fontsize=8, labelpad=1)
    response_bar.ax.tick_params(labelsize=7)
    difference_axis = fig.add_axes([0.674, 0.060, 0.115, 0.011])
    difference_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-diff_limit, diff_limit), cmap="RdBu_r"),
        cax=difference_axis, orientation="horizontal")
    difference_bar.set_label("Spatial-density difference", fontsize=8, labelpad=1)
    difference_bar.ax.tick_params(labelsize=7)
    fig.text(
        0.5, 0.013,
        "Same GT target class/region for both checkpoints; boxes omitted; one top focus-gain exemplar per scene; see CSV audit.",
        ha="center", fontsize=8)
    basename = "m3fd_rgca_reliability_gate_gradcam_comparison"
    fig.savefig(output_dir / f"{basename}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for item in items:
        one_fig, one_axes = plt.subplots(1, 6, figsize=(16, 2.75))
        one_fig.subplots_adjust(left=0.06, right=0.995, top=0.84, bottom=0.02, wspace=0.065)
        draw_item(one_axes, item, True)
        one_fig.suptitle(
            f"{item['scene_display']} | M3FD ID {item['image_id']} | target-conditioned Grad-CAM",
            fontsize=13, y=0.98)
        one_fig.savefig(
            row_dir / f"{item['scene']}_{item['image_id']}_gradcam_comparison.png",
            dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(one_fig)


def main() -> None:
    args = parse_args()
    for path in (args.gated_weights, args.no_gate_weights):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    gated_model = attempt_load(str(args.gated_weights), map_location=device).to(device).float().eval()
    no_gate_model = attempt_load(str(args.no_gate_weights), map_location=device).to(device).float().eval()
    gated_rgca = find_rgca_layer(gated_model, args.rgca_layer)
    no_gate_rgca = find_rgca_layer(no_gate_model, args.rgca_layer)
    if getattr(gated_rgca, "reliability_gate", None) is None:
        raise RuntimeError("The gated checkpoint does not contain a reliability gate")
    if getattr(no_gate_rgca, "reliability_gate", "missing") is not None:
        raise RuntimeError("The no-gate checkpoint still contains a reliability gate")
    for model in (gated_model, no_gate_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    candidates, ambiguous, scene_counts = build_candidates(
        args.dataset_root, args.scene_root, args.min_gt)
    rows, cache = scan(gated_model, no_gate_model, candidates, device, args)
    ranked, selected = rank_and_select(
        rows, args.min_target_confidence, args.min_cam_support,
        args.max_target_cam_mass)
    summary = aggregate(rows)
    write_csv(args.output_dir / "all_test_target_gradcam_metrics.csv", ranked)
    write_csv(args.output_dir / "selected_manifest.csv", selected)
    write_csv(args.output_dir / "aggregate_target_gradcam_summary.csv", summary)
    render(selected, cache, args.output_dir, args.display_quantile)

    config = {
        "task": "M3FD RGCA reliability-gate target-conditioned detector Grad-CAM comparison",
        "gated_weights": str(args.gated_weights.resolve()),
        "gated_weights_sha256": sha256sum(args.gated_weights),
        "no_gate_weights": str(args.no_gate_weights.resolve()),
        "no_gate_weights_sha256": sha256sum(args.no_gate_weights),
        "split": "canonical M3FD test only",
        "detector_input_layers": dict(zip(DETECT_LEVELS, DETECT_INPUT_LAYERS)),
        "target_definition": "largest GT box; highest-confidence same-class prediction centre inside GT",
        "selection": (
            "top foreground-enrichment log2 gain per scene among shared-confidence, "
            "non-degenerate CAM samples"),
        "selection_bias": True,
        "min_target_confidence": args.min_target_confidence,
        "min_cam_support_for_selection": args.min_cam_support,
        "max_target_cam_mass_for_selection": args.max_target_cam_mass,
        "target_area_fraction_range": [args.min_target_area, args.max_target_area],
        "eligible_scene_counts_before_area_filter": scene_counts,
        "evaluated_after_area_filter": len(rows),
        "excluded_ambiguous_scene_ids": ambiguous,
        "boxes_drawn": False,
        "selected": [
            {
                "scene": row["scene"], "image_id": row["image_id"],
                "target_class": row["target_class"],
                "gated_target_confidence": row["gated_target_confidence"],
                "no_gate_target_confidence": row["no_gate_target_confidence"],
                "foreground_enrichment_log2_gain": row["foreground_enrichment_log2_gain"],
            } for row in selected
        ],
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    (args.output_dir / "SCIENTIFIC_CAVEAT.txt").write_text(
        "The panels compare independently trained best checkpoints, so they show model-level association, "
        "not a controlled causal effect of the gate alone. Exemplars are deliberately selected by GT-based "
        "focus gain. The full per-image distribution and aggregate summary must accompany any claim. "
        "Per-map heatmap contrast is normalized independently, as conventional for Grad-CAM.\n",
        encoding="utf-8")

    print("Selected target-conditioned Grad-CAM exemplars:")
    for row in selected:
        print(
            f"  {row['scene_display']}: {row['image_id']}  "
            f"conf(no/with)={float(row['no_gate_target_confidence']):.3f}/"
            f"{float(row['gated_target_confidence']):.3f}  "
            f"log2-focus-gain={float(row['foreground_enrichment_log2_gain']):+.3f}")
    print("Aggregate target-conditioned Grad-CAM summary:")
    for row in summary:
        print(
            f"  {row['scene']}: n={row['n']}  "
            f"positive={float(row['positive_focus_gain_fraction']):.3f}  "
            f"median={float(row['log2_focus_gain_median']):+.3f}")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
