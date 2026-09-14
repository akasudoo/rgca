#!/usr/bin/env python3
"""Export paired RGB/IR attribution heatmaps for all M3FD scene-test pairs.

Each output row contains six panels:
RGB | IR | no-gate RGB CAM | no-gate IR CAM | gated RGB CAM | gated IR CAM.

Unlike a fused detector Grad-CAM copied onto two images, this script computes
modality-specific multiscale HiResCAM attribution magnitudes from the RGB and
IR backbone P2-P5 features.  The same GT class and target region are used for
both independently trained checkpoints.  No target boxes are drawn.
"""

from __future__ import annotations

import argparse
import csv
import html
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
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.compare_rgca_gate_gradcam import (  # noqa: E402
    global_to_level_index,
    normalized_for_display,
    target_mask,
    valid_image_mask,
)
from tools.visualize_rgca_panel_a import (  # noqa: E402
    SCENES,
    find_rgca_layer,
    image_id_from_manifest_line,
    paired_paths,
    prepare_pair,
    read_nonempty_lines,
    read_yolo_labels,
    resolve_device,
    sha256sum,
    write_csv,
)


RGB_BRANCH_LAYERS = (2, 4, 6, 9)
IR_BRANCH_LAYERS = (12, 14, 16, 19)
FEATURE_LEVELS = ("P2", "P3", "P4", "P5")
PANEL_TITLES = (
    "RGB",
    "IR",
    "No gate | RGB CAM",
    "No gate | IR CAM",
    "With gate | RGB CAM",
    "With gate | IR CAM",
)


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
        default=REPO_ROOT / "paper/rgca_gate_paired_modal_heatmaps_m3fd")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--display-quantile", type=float, default=0.995)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-target-confidence", type=float, default=0.10)
    parser.add_argument("--min-cam-support", type=float, default=0.01)
    parser.add_argument("--max-target-cam-mass", type=float, default=0.98)
    parser.add_argument("--rgca-layer", default="model.21.cross_modal_attention")
    return parser.parse_args()


class BranchActivationCapture:
    """Capture the paired backbone P2-P5 tensors used by the fusion modules."""

    def __init__(self, model: torch.nn.Module):
        self.activations: Dict[Tuple[str, int], torch.Tensor] = {}
        self.handles = []
        for modality, indices in (("rgb", RGB_BRANCH_LAYERS), ("ir", IR_BRANCH_LAYERS)):
            for level, index in enumerate(indices):
                self.handles.append(
                    model.model[index].register_forward_hook(
                        self._make_hook(modality, level)))

    def _make_hook(self, modality: str, level: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{modality} {FEATURE_LEVELS[level]} did not return a tensor")
            self.activations[(modality, level)] = output
        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def build_all_scene_assignments(
    dataset_root: Path, scene_root: Path
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, int]]:
    """Keep every scene-manifest assignment, including the five overlaps."""
    manifest_dir = scene_root / "manifests/original_m3fd_split"
    canonical = {
        path.stem for path in (dataset_root / "images/vis_test").glob("*.png")
    }
    assignments: List[Dict[str, object]] = []
    memberships: List[str] = []
    counts: Dict[str, int] = {}
    for scene, display_name in SCENES:
        manifest = manifest_dir / f"{scene}_test_vis.txt"
        ids = sorted({
            image_id_from_manifest_line(line)
            for line in read_nonempty_lines(manifest)
        } & canonical)
        counts[scene] = len(ids)
        for image_id in ids:
            rgb_path, ir_path, label_path = paired_paths(dataset_root, image_id)
            labels = read_yolo_labels(label_path)
            if not len(labels):
                raise ValueError(f"Empty test label: {label_path}")
            assignments.append({
                "scene": scene,
                "scene_display": display_name,
                "image_id": image_id,
                "gt_count": len(labels),
                "rgb_path": str(rgb_path),
                "ir_path": str(ir_path),
                "label_path": str(label_path),
            })
            memberships.append(image_id)
    membership_counts = Counter(memberships)
    ambiguous = sorted(key for key, value in membership_counts.items() if value > 1)
    ambiguous_set = set(ambiguous)
    for assignment in assignments:
        assignment["ambiguous_scene_assignment"] = (
            str(assignment["image_id"]) in ambiguous_set)
    return assignments, ambiguous, counts


def select_largest_valid_target(
    boxes: np.ndarray, valid: np.ndarray, img_size: int
) -> Tuple[np.ndarray, int, float]:
    candidates = []
    for index, box in enumerate(boxes):
        mask = target_mask(box, img_size) & valid
        area = int(mask.sum())
        if area:
            candidates.append((area, index, box))
    if not candidates:
        raise ValueError("No GT box overlaps the valid letterboxed image region")
    area, index, box = max(candidates, key=lambda item: (item[0], -item[1]))
    return box.copy(), int(index), float(area / valid.sum())


def locate_target_prediction(
    predictions: torch.Tensor,
    raw_levels: Sequence[torch.Tensor],
    target: np.ndarray,
) -> Tuple[torch.Tensor, float, str, int]:
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
        global_index = int(confidence.masked_fill(~inside, -1.0).argmax())
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        distance = (centres[:, 0] - cx).square() + (centres[:, 1] - cy).square()
        scale = max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1.0)
        global_index = int((confidence - distance / scale).argmax())
    level, local_index = global_to_level_index(raw_levels, global_index)
    raw = raw_levels[level].reshape(1, -1, raw_levels[level].shape[-1])
    candidate = raw[0, local_index]
    score = candidate[4].sigmoid() * candidate[5 + class_id].sigmoid()
    return score, float(score.detach().cpu()), FEATURE_LEVELS[level], global_index


def aggregate_hirescam(
    activations: Sequence[torch.Tensor],
    gradients: Sequence[Optional[torch.Tensor]],
    img_size: int,
) -> np.ndarray:
    """Aggregate absolute gradient-weighted activations over P2-P5."""
    total = None
    for activation, gradient in zip(activations, gradients):
        if gradient is None:
            continue
        level_cam = (activation * gradient).sum(dim=1, keepdim=True).abs()
        level_cam = F.interpolate(
            level_cam, size=(img_size, img_size), mode="bilinear",
            align_corners=False)
        total = level_cam if total is None else total + level_cam
    if total is None:
        return np.zeros((img_size, img_size), dtype=np.float32)
    array = total[0, 0].detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("HiResCAM contains non-finite values")
    return array


def compute_paired_modal_cams(
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
        predictions, raw_levels, target)
    rgb_activations = [capture.activations[("rgb", level)] for level in range(4)]
    ir_activations = [capture.activations[("ir", level)] for level in range(4)]
    all_activations = rgb_activations + ir_activations
    gradients = torch.autograd.grad(
        score, all_activations, retain_graph=False, allow_unused=True)
    rgb_cam = aggregate_hirescam(rgb_activations, gradients[:4], img_size)
    ir_cam = aggregate_hirescam(ir_activations, gradients[4:], img_size)
    return rgb_cam, ir_cam, confidence, detect_level, prediction_index


def cam_stats(
    cam: np.ndarray, foreground: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    eps = 1e-12
    positive = np.maximum(cam, 0)
    foreground = foreground & valid
    background = valid & ~foreground
    total = float(positive[valid].sum())
    if total <= eps:
        return {
            "foreground_enrichment": 0.0,
            "target_mass_fraction": 0.0,
            "support_fraction": 0.0,
            "cam_mean": 0.0,
        }
    density = positive / max(float(positive[valid].mean()), eps)
    fg_mean = float(density[foreground].mean())
    bg_mean = float(density[background].mean())
    return {
        "foreground_enrichment": fg_mean / max(bg_mean, eps),
        "target_mass_fraction": float(positive[foreground].sum() / total),
        "support_fraction": float((positive[valid] > 0).mean()),
        "cam_mean": float(positive[valid].mean()),
    }


def overlay_heatmap(
    base_rgb: np.ndarray, cam: np.ndarray, valid: np.ndarray, quantile: float
) -> Tuple[np.ndarray, np.ndarray]:
    display = normalized_for_display(cam, valid, quantile)
    display = display.copy()
    display[~valid] = 0
    heat_bgr = cv2.applyColorMap(
        np.uint8(np.clip(display * 255, 0, 255)), cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    alpha = (0.68 * np.power(display, 0.65))[..., None]
    overlay = (
        base_rgb.astype(np.float32) * (1.0 - alpha)
        + heat_rgb.astype(np.float32) * alpha)
    return np.uint8(np.clip(overlay, 0, 255)), display


def compose_six_panel_row(
    item: Dict[str, object], panel_size: int, include_titles: bool = True
) -> np.ndarray:
    panels = [
        item["rgb"], item["ir"],
        item["no_gate_rgb_overlay"], item["no_gate_ir_overlay"],
        item["gated_rgb_overlay"], item["gated_ir_overlay"],
    ]
    resized = [
        cv2.resize(panel, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
        for panel in panels
    ]
    title_height = 38 if include_titles else 0
    footer_height = 38
    canvas = np.full(
        (title_height + panel_size + footer_height, panel_size * 6, 3),
        255, dtype=np.uint8)
    for index, panel in enumerate(resized):
        x1, x2 = index * panel_size, (index + 1) * panel_size
        canvas[title_height:title_height + panel_size, x1:x2] = panel
        if include_titles:
            font_scale = max(0.36, panel_size / 760.0)
            thickness = 1 if panel_size < 500 else 2
            (width, _), _ = cv2.getTextSize(
                PANEL_TITLES[index], cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, thickness)
            cv2.putText(
                canvas, PANEL_TITLES[index],
                (x1 + max(4, (panel_size - width) // 2), 25),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
                thickness, cv2.LINE_AA)
    footer = (
        f"{item['scene_display']} | ID {item['image_id']} | class {item['target_class']} | "
        f"confidence no/with: {item['no_gate_confidence']:.3f}/{item['gated_confidence']:.3f} | "
        f"paired focus score: {item['paired_focus_score']:+.3f}")
    cv2.putText(
        canvas, footer, (10, title_height + panel_size + 25),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.38, panel_size / 820.0),
        (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def save_rgb_jpeg(path: Path, array: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"Failed to write {path}")


def write_union_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write mixed success/error rows without losing diagnostic columns."""
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selection_eligible(item: Dict[str, object], args: argparse.Namespace) -> bool:
    supports = [
        item[f"{prefix}_{modality}_support_fraction"]
        for prefix in ("no_gate", "gated") for modality in ("rgb", "ir")
    ]
    masses = [
        item[f"{prefix}_{modality}_target_mass_fraction"]
        for prefix in ("no_gate", "gated") for modality in ("rgb", "ir")
    ]
    return (
        not bool(item.get("ambiguous_scene_assignment", False))
        and
        min(item["no_gate_confidence"], item["gated_confidence"])
        >= args.min_target_confidence
        and min(supports) >= args.min_cam_support
        and max(masses) <= args.max_target_cam_mass
    )


def update_best(
    best: Dict[str, Dict[str, object]],
    fallback: Dict[str, Dict[str, object]],
    item: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    scene = str(item["scene"])
    key = (float(item["paired_focus_score"]), float(item["mean_modal_focus_gain"]))
    current_fallback = fallback.get(scene)
    if current_fallback is None or key > (
        float(current_fallback["paired_focus_score"]),
        float(current_fallback["mean_modal_focus_gain"])):
        fallback[scene] = item
    if selection_eligible(item, args):
        current = best.get(scene)
        if current is None or key > (
            float(current["paired_focus_score"]),
            float(current["mean_modal_focus_gain"])):
            best[scene] = item


def render_selected_main(
    selected: Sequence[Dict[str, object]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(16.0, 10.25))
    fig.subplots_adjust(
        left=0.052, right=0.995, top=0.925, bottom=0.035,
        wspace=0.045, hspace=0.075)
    panel_keys = (
        "rgb", "ir", "no_gate_rgb_overlay", "no_gate_ir_overlay",
        "gated_rgb_overlay", "gated_ir_overlay")
    for row_index, (row_axes, item) in enumerate(zip(axes, selected)):
        for column, (ax, key) in enumerate(zip(row_axes, panel_keys)):
            ax.imshow(item[key])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45); spine.set_color("#777777")
            if row_index == 0:
                ax.set_title(PANEL_TITLES[column], fontsize=8.5, pad=4)
        row_axes[0].set_ylabel(
            f"{item['scene_display']}\nID {item['image_id']}",
            fontsize=9, labelpad=5)
    fig.suptitle(
        "M3FD | Paired RGB/IR target attribution without vs. with RGCA reliability gating",
        fontsize=14.5, y=0.965)
    fig.text(
        0.5, 0.010,
        "Modality-specific multiscale HiResCAM (P2-P5); same GT target for both checkpoints; boxes omitted.",
        ha="center", fontsize=8)
    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    basename = "m3fd_rgca_gate_paired_rgb_ir_heatmaps_selected"
    fig.savefig(selected_dir / f"{basename}.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(selected_dir / f"{basename}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    raw_dir = selected_dir / "raw_maps"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        scene, image_id = str(item["scene"]), str(item["image_id"])
        row = compose_six_panel_row(item, panel_size=480, include_titles=True)
        save_rgb_jpeg(
            selected_dir / f"{scene}_{image_id}_paired_heatmaps.jpg", row, 95)
        for key in (
            "no_gate_rgb_cam", "no_gate_ir_cam",
            "gated_rgb_cam", "gated_ir_cam"):
            np.save(
                raw_dir / f"{scene}_{image_id}_{key}.npy",
                np.asarray(item[key], dtype=np.float32), allow_pickle=False)


def write_gallery(
    rows: Sequence[Dict[str, object]], output_dir: Path
) -> None:
    cards = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        relative = f"all_test/{row['scene']}/{row['image_id']}.jpg"
        caption = (
            f"{row['scene_display']} · {row['image_id']} · "
            f"score {float(row['paired_focus_score']):+.3f}")
        cards.append(
            f'<figure data-scene="{html.escape(str(row["scene"]))}">'
            f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>M3FD paired RGCA heatmaps</title><style>
body{font-family:Arial,sans-serif;margin:18px;background:#f5f5f5;color:#222}
h1{font-size:22px}.note{max-width:1050px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:12px}
figure{margin:0;background:white;padding:8px;border:1px solid #ccc}img{width:100%;height:auto;display:block}figcaption{padding-top:5px;font-size:13px}
</style></head><body><h1>M3FD paired RGB/IR heatmaps</h1>
<p class="note">Columns: RGB, IR, no-gate RGB CAM, no-gate IR CAM, gated RGB CAM, gated IR CAM. Click an image to open the full JPEG. Heatmap contrast is normalized independently per map.</p>
<div class="grid">""" + "\n".join(cards) + "</div></body></html>"
    (output_dir / "gallery.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.5 < args.display_quantile < 1.0:
        raise ValueError("--display-quantile must be between 0.5 and 1")
    for path in (args.gated_weights, args.no_gate_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments, ambiguous, scene_counts = build_all_scene_assignments(
        args.dataset_root, args.scene_root)
    device = resolve_device(args.device)
    gated_model = attempt_load(
        str(args.gated_weights), map_location=device).to(device).float().eval()
    no_gate_model = attempt_load(
        str(args.no_gate_weights), map_location=device).to(device).float().eval()
    gated_rgca = find_rgca_layer(gated_model, args.rgca_layer)
    no_gate_rgca = find_rgca_layer(no_gate_model, args.rgca_layer)
    if getattr(gated_rgca, "reliability_gate", None) is None:
        raise RuntimeError("The gated checkpoint does not contain a reliability gate")
    if getattr(no_gate_rgca, "reliability_gate", "missing") is not None:
        raise RuntimeError("The no-gate checkpoint still contains a reliability gate")
    for model in (gated_model, no_gate_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    gated_capture = BranchActivationCapture(gated_model)
    no_gate_capture = BranchActivationCapture(no_gate_model)
    rows: List[Dict[str, object]] = []
    best: Dict[str, Dict[str, object]] = {}
    fallback: Dict[str, Dict[str, object]] = {}
    try:
        for candidate in tqdm(
            assignments, desc="Exporting paired RGB/IR heatmaps", unit="assignment"):
            labels = read_yolo_labels(Path(str(candidate["label_path"])))
            rgb, ir, tensor, boxes = prepare_pair(
                Path(str(candidate["rgb_path"])), Path(str(candidate["ir_path"])),
                labels, args.img_size)
            valid = valid_image_mask(Path(str(candidate["rgb_path"])), args.img_size)
            try:
                target, target_index, target_area_fraction = select_largest_valid_target(
                    boxes, valid, args.img_size)
                gated_rgb, gated_ir, gated_conf, gated_level, gated_index = (
                    compute_paired_modal_cams(
                        gated_model, gated_capture, tensor, target, device,
                        args.img_size))
                no_rgb, no_ir, no_conf, no_level, no_index = (
                    compute_paired_modal_cams(
                        no_gate_model, no_gate_capture, tensor, target, device,
                        args.img_size))
                foreground = target_mask(target, args.img_size) & valid
                map_lookup = {
                    "no_gate_rgb": no_rgb, "no_gate_ir": no_ir,
                    "gated_rgb": gated_rgb, "gated_ir": gated_ir,
                }
                stats = {
                    prefix: cam_stats(cam, foreground, valid)
                    for prefix, cam in map_lookup.items()
                }
                eps = 1e-12
                rgb_gain = float(np.log2(
                    (stats["gated_rgb"]["foreground_enrichment"] + eps)
                    / (stats["no_gate_rgb"]["foreground_enrichment"] + eps)))
                ir_gain = float(np.log2(
                    (stats["gated_ir"]["foreground_enrichment"] + eps)
                    / (stats["no_gate_ir"]["foreground_enrichment"] + eps)))
                row: Dict[str, object] = {
                    **candidate,
                    "status": "ok",
                    "target_box_index": target_index,
                    "target_class": int(target[0]),
                    "target_area_fraction": target_area_fraction,
                    "no_gate_confidence": no_conf,
                    "gated_confidence": gated_conf,
                    "no_gate_detect_level": no_level,
                    "gated_detect_level": gated_level,
                    "no_gate_prediction_index": no_index,
                    "gated_prediction_index": gated_index,
                    "rgb_log2_focus_gain": rgb_gain,
                    "ir_log2_focus_gain": ir_gain,
                    "mean_modal_focus_gain": 0.5 * (rgb_gain + ir_gain),
                    "paired_focus_score": min(rgb_gain, ir_gain),
                }
                for prefix, values in stats.items():
                    for name, value in values.items():
                        row[f"{prefix}_{name}"] = value
                overlays = {}
                for prefix, base in (
                    ("no_gate_rgb", rgb), ("no_gate_ir", ir),
                    ("gated_rgb", rgb), ("gated_ir", ir)):
                    overlays[f"{prefix}_overlay"], _ = overlay_heatmap(
                        base, map_lookup[prefix], valid, args.display_quantile)
                item = {
                    **row, "rgb": rgb, "ir": ir, "valid": valid,
                    "target": target,
                    "no_gate_rgb_cam": no_rgb, "no_gate_ir_cam": no_ir,
                    "gated_rgb_cam": gated_rgb, "gated_ir_cam": gated_ir,
                    **overlays,
                }
                output_path = (
                    args.output_dir / "all_test" / str(candidate["scene"])
                    / f"{candidate['image_id']}.jpg")
                save_rgb_jpeg(
                    output_path,
                    compose_six_panel_row(item, args.panel_size, include_titles=True),
                    args.jpeg_quality)
                row["heatmap_path"] = str(output_path.resolve())
                update_best(best, fallback, item, args)
            except Exception as error:
                row = {
                    **candidate,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            rows.append(row)
    finally:
        gated_capture.close()
        no_gate_capture.close()

    selected = [best.get(scene, fallback[scene]) for scene, _ in SCENES]
    selected_rows = []
    for item in selected:
        selected_rows.append({
            key: value for key, value in item.items()
            if not isinstance(value, np.ndarray)
        })
    write_union_csv(args.output_dir / "all_scene_test_heatmap_metrics.csv", rows)
    write_csv(args.output_dir / "selected_manifest.csv", selected_rows)
    recommended_rows: List[Dict[str, object]] = []
    for scene, _ in SCENES:
        eligible = [
            row for row in rows
            if row.get("scene") == scene and row.get("status") == "ok"
            and selection_eligible(row, args)
        ]
        eligible.sort(
            key=lambda row: (
                float(row["paired_focus_score"]),
                float(row["mean_modal_focus_gain"])),
            reverse=True)
        for rank, row in enumerate(eligible[:20], start=1):
            recommended_rows.append({**row, "scene_rank": rank})
    write_union_csv(
        args.output_dir / "recommended_top20_per_scene.csv",
        recommended_rows)
    render_selected_main(selected, args.output_dir)
    write_gallery(rows, args.output_dir)

    config = {
        "task": "M3FD all-scene paired RGB/IR attribution heatmaps",
        "method": "modality-specific absolute HiResCAM aggregated over backbone P2-P5",
        "layout": list(PANEL_TITLES),
        "gated_weights": str(args.gated_weights.resolve()),
        "gated_weights_sha256": sha256sum(args.gated_weights),
        "no_gate_weights": str(args.no_gate_weights.resolve()),
        "no_gate_weights_sha256": sha256sum(args.no_gate_weights),
        "scene_assignment_counts": scene_counts,
        "assignment_total": len(assignments),
        "unique_image_count": len({str(row["image_id"]) for row in assignments}),
        "overlapping_scene_ids_retained_in_each_scene": ambiguous,
        "successful_exports": sum(row.get("status") == "ok" for row in rows),
        "failed_exports": sum(row.get("status") != "ok" for row in rows),
        "target_definition": "largest valid GT box; same GT class/region for both checkpoints",
        "selection_score": "minimum of RGB and IR log2 foreground-enrichment gains",
        "selection_constraints": {
            "min_target_confidence": args.min_target_confidence,
            "min_cam_support": args.min_cam_support,
            "max_target_cam_mass": args.max_target_cam_mass,
            "exclude_overlapping_scene_assignments": True,
        },
        "per_map_display_normalization_quantile": args.display_quantile,
        "boxes_drawn": False,
        "selected": [
            {
                "scene": item["scene"],
                "image_id": item["image_id"],
                "rgb_log2_focus_gain": item["rgb_log2_focus_gain"],
                "ir_log2_focus_gain": item["ir_log2_focus_gain"],
                "paired_focus_score": item["paired_focus_score"],
            }
            for item in selected
        ],
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    print("Selected paired-modal exemplars:")
    for item in selected:
        print(
            f"  {item['scene_display']}: {item['image_id']}  "
            f"RGB-gain={float(item['rgb_log2_focus_gain']):+.3f}  "
            f"IR-gain={float(item['ir_log2_focus_gain']):+.3f}  "
            f"paired-score={float(item['paired_focus_score']):+.3f}")
    print(
        f"Exported {sum(row.get('status') == 'ok' for row in rows)}/"
        f"{len(rows)} scene assignments to {args.output_dir / 'all_test'}")
    print(f"Gallery: {args.output_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
