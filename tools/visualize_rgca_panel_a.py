#!/usr/bin/env python3
"""Generate the paper-ready RGCA Panel A from real M3FD test samples.

The direction labels in this script follow the actual gating operation in
ReliabilityGuidedBidirectionalCrossAttention:

    reliability_rgb -> gates IR values injected into RGB -> R_IR->RGB
    reliability_ir  -> gates RGB values injected into IR -> R_RGB->IR

No attention matrix is reshaped into a spatial map.  Only the learned HxW
reliability gates are exported and visualized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402


SCENES = (
    ("daytime", "Daytime"),
    ("night", "Night"),
    ("overcast", "Overcast"),
    ("challenge", "Challenging"),
)
CLASS_NAMES = ("People", "Car", "Bus", "Lamp", "Motorcycle", "Truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt",
        help="Pure RGCA checkpoint used for the main M3FD experiment.",
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
        default=REPO_ROOT / "paper/rgca_reliability_visualization/panel_a",
    )
    parser.add_argument("--layer", default="model.21.cross_modal_attention")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument("--device", default="0", help="CUDA index, or 'cpu'.")
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def image_id_from_manifest_line(line: str) -> str:
    return Path(line).stem


def read_yolo_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    labels = np.loadtxt(str(path), dtype=np.float32, ndmin=2)
    if labels.shape[1] > 5:
        labels = labels[:, :5]
    if labels.shape[1] != 5:
        raise ValueError(f"Expected five YOLO columns in {path}, got {labels.shape}")
    if not np.isfinite(labels).all():
        raise ValueError(f"Non-finite label value in {path}")
    if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
        raise ValueError(f"Non-normalized label coordinates in {path}")
    return labels


def paired_paths(dataset_root: Path, image_id: str) -> Tuple[Path, Path, Path]:
    rgb_path = dataset_root / "images/vis_test" / f"{image_id}.png"
    ir_path = dataset_root / "images/Ir_test" / f"{image_id}.png"
    label_path = dataset_root / "labels/vis_test" / f"{image_id}.txt"
    for path in (rgb_path, ir_path, label_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return rgb_path, ir_path, label_path


def select_samples(
    dataset_root: Path, scene_root: Path, seed: int, min_gt: int
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Uniform seeded sampling after declared, non-visual eligibility filters."""
    manifest_dir = scene_root / "manifests/original_m3fd_split"
    canonical_ids = {path.stem for path in (dataset_root / "images/vis_test").glob("*.png")}
    scene_ids: Dict[str, set] = {}
    for key, _ in SCENES:
        manifest = manifest_dir / f"{key}_test_vis.txt"
        ids = {image_id_from_manifest_line(line) for line in read_nonempty_lines(manifest)}
        scene_ids[key] = ids & canonical_ids

    membership = Counter(image_id for ids in scene_ids.values() for image_id in ids)
    ambiguous = {image_id for image_id, count in membership.items() if count > 1}
    rng = np.random.default_rng(seed)
    selected: List[Dict[str, object]] = []
    audit: Dict[str, object] = {
        "sampling_method": "uniform choice from sorted eligible IDs using one NumPy RNG",
        "eligibility": [
            "member of canonical M3FD vis_test",
            "member of original_m3fd_split scene test manifest",
            "RGB, IR, and visible-label files exist",
            f"at least {min_gt} valid GT boxes",
            "excluded IDs assigned to more than one scene",
        ],
        "excluded_ambiguous_ids": sorted(ambiguous),
        "scene_counts": {},
    }

    for selection_order, (key, display_name) in enumerate(SCENES, start=1):
        unambiguous = sorted(scene_ids[key] - ambiguous)
        eligible: List[Tuple[str, int]] = []
        missing_pairs = 0
        for image_id in unambiguous:
            try:
                _, _, label_path = paired_paths(dataset_root, image_id)
            except FileNotFoundError:
                missing_pairs += 1
                continue
            gt_count = len(read_yolo_labels(label_path))
            if gt_count >= min_gt:
                eligible.append((image_id, gt_count))
        if not eligible:
            raise RuntimeError(f"No eligible samples remain for scene {key}")
        chosen_index = int(rng.integers(0, len(eligible)))
        image_id, gt_count = eligible[chosen_index]
        selected.append(
            {
                "scene": key,
                "scene_display": display_name,
                "selection_order": selection_order,
                "image_id": image_id,
                "gt_count": gt_count,
                "eligible_index": chosen_index,
                "eligible_count": len(eligible),
                "unambiguous_scene_count": len(unambiguous),
            }
        )
        audit["scene_counts"][key] = {
            "canonical_manifest_intersection": len(scene_ids[key]),
            "unambiguous": len(unambiguous),
            "missing_pairs": missing_pairs,
            "eligible_min_gt": len(eligible),
        }
    return selected, audit


def prepare_pair(
    rgb_path: Path, ir_path: Path, labels: np.ndarray, img_size: int
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """Match LoadMultiModalImagesAndLabels validation preprocessing exactly."""
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir_bgr = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None or ir_bgr is None:
        raise ValueError(f"Failed to read paired images: {rgb_path}, {ir_path}")
    if rgb_bgr.shape[:2] != ir_bgr.shape[:2]:
        raise ValueError(
            f"Unaligned image shapes for {rgb_path.stem}: "
            f"RGB={rgb_bgr.shape[:2]}, IR={ir_bgr.shape[:2]}"
        )

    h0, w0 = rgb_bgr.shape[:2]
    resize_gain = img_size / max(h0, w0)
    resized_h, resized_w = h0, w0
    if resize_gain != 1:
        resized_w, resized_h = int(w0 * resize_gain), int(h0 * resize_gain)
        interpolation = cv2.INTER_AREA if resize_gain < 1 else cv2.INTER_LINEAR
        rgb_bgr = cv2.resize(rgb_bgr, (resized_w, resized_h), interpolation=interpolation)
        ir_bgr = cv2.resize(ir_bgr, (resized_w, resized_h), interpolation=interpolation)

    rgb_bgr, ratio, pad = letterbox(
        rgb_bgr, new_shape=img_size, auto=False, scaleup=False
    )
    ir_bgr, ir_ratio, ir_pad = letterbox(
        ir_bgr, new_shape=img_size, auto=False, scaleup=False
    )
    if tuple(ratio) != tuple(ir_ratio) or tuple(pad) != tuple(ir_pad):
        raise RuntimeError("RGB and IR received different letterbox transforms")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    ir = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
    chw = np.concatenate(
        (rgb.transpose(2, 0, 1), ir.transpose(2, 0, 1)), axis=0
    )
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).float() / 255.0

    boxes = np.zeros((len(labels), 5), dtype=np.float32)
    if len(labels):
        boxes[:, 0] = labels[:, 0]
        cx = labels[:, 1] * resized_w
        cy = labels[:, 2] * resized_h
        bw = labels[:, 3] * resized_w
        bh = labels[:, 4] * resized_h
        boxes[:, 1] = (cx - bw / 2) * ratio[0] + pad[0]
        boxes[:, 2] = (cy - bh / 2) * ratio[1] + pad[1]
        boxes[:, 3] = (cx + bw / 2) * ratio[0] + pad[0]
        boxes[:, 4] = (cy + bh / 2) * ratio[1] + pad[1]
        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, img_size)
    return rgb, ir, tensor, boxes


def resolve_device(spec: str) -> torch.device:
    if spec.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    index = int(spec.split(",")[0])
    return torch.device(f"cuda:{index}")


def find_rgca_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        candidates = [name for name in modules if name.endswith("cross_modal_attention")]
        raise KeyError(
            f"Layer {layer_name!r} not found. Available cross-modal layers: {candidates}"
        )
    layer = modules[layer_name]
    if not hasattr(layer, "reliability_gate"):
        raise TypeError(f"{layer_name} is not an RGCA reliability-gated module")
    return layer


def draw_boxes(ax: plt.Axes, boxes: np.ndarray, label_text: bool = False) -> None:
    for row in boxes:
        class_id, x1, y1, x2, y2 = row.tolist()
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#ffea00",
                linewidth=0.9,
            )
        )
        if label_text:
            class_index = int(class_id)
            label = CLASS_NAMES[class_index] if 0 <= class_index < len(CLASS_NAMES) else str(class_index)
            ax.text(
                x1,
                max(1.0, y1 - 2.0),
                label,
                color="#ffea00",
                fontsize=5.5,
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.5, "edgecolor": "none"},
            )


def make_gt_zoom(rgb: np.ndarray, ir: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic side-by-side crop centered on the largest GT."""
    if not len(boxes):
        return np.concatenate((rgb, ir), axis=1), boxes
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    target = boxes[int(np.argmax(areas))]
    _, x1, y1, x2, y2 = target.tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 2.0
    side = max(side, 96.0)
    crop_x1 = max(0, int(round(cx - side / 2)))
    crop_y1 = max(0, int(round(cy - side / 2)))
    crop_x2 = min(rgb.shape[1], int(round(cx + side / 2)))
    crop_y2 = min(rgb.shape[0], int(round(cy + side / 2)))
    rgb_crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    ir_crop = ir[crop_y1:crop_y2, crop_x1:crop_x2]
    composite = np.concatenate((rgb_crop, ir_crop), axis=1)
    local_box = np.array(
        [
            [target[0], x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1],
            [
                target[0],
                x1 - crop_x1 + rgb_crop.shape[1],
                y1 - crop_y1,
                x2 - crop_x1 + rgb_crop.shape[1],
                y2 - crop_y1,
            ],
        ],
        dtype=np.float32,
    )
    return composite, local_box


def foreground_mask(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for _, x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return mask


def finite_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def calculate_stats(
    scene: str,
    image_id: str,
    first: np.ndarray,
    second: np.ndarray,
    boxes: np.ndarray,
) -> Dict[str, object]:
    delta = first - second
    flat_first, flat_second = first.ravel(), second.ravel()
    correlation = float(np.corrcoef(flat_first, flat_second)[0, 1])
    fg = foreground_mask(boxes, first.shape[0], first.shape[1])
    bg = ~fg
    return {
        "scene": scene,
        "image_id": image_id,
        "gt_count": len(boxes),
        "r_ir_to_rgb_mean": float(first.mean()),
        "r_ir_to_rgb_std": float(first.std()),
        "r_rgb_to_ir_mean": float(second.mean()),
        "r_rgb_to_ir_std": float(second.std()),
        "delta_mean": float(delta.mean()),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "delta_abs_max": float(np.abs(delta).max()),
        "direction_map_pearson_r": correlation,
        "positive_delta_fraction": float((delta > 0).mean()),
        "foreground_r_ir_to_rgb_mean": finite_mean(first[fg]),
        "foreground_r_rgb_to_ir_mean": finite_mean(second[fg]),
        "background_r_ir_to_rgb_mean": finite_mean(first[bg]),
        "background_r_rgb_to_ir_mean": finite_mean(second[bg]),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_row(
    axes: Sequence[plt.Axes], item: Dict[str, object], show_titles: bool = True,
    reliability_range: Tuple[float, float] = (0.0, 1.0),
    delta_range: Tuple[float, float] = (-1.0, 1.0),
) -> None:
    rgb = item["rgb"]
    ir = item["ir"]
    boxes = item["boxes"]
    first = item["r_ir_to_rgb"]
    second = item["r_rgb_to_ir"]
    delta = first - second

    axes[0].imshow(rgb)
    draw_boxes(axes[0], boxes)
    axes[1].imshow(ir)
    draw_boxes(axes[1], boxes)
    axes[2].imshow(rgb)
    axes[2].imshow(
        first, cmap="magma", vmin=reliability_range[0],
        vmax=reliability_range[1], alpha=0.50)
    draw_boxes(axes[2], boxes)
    axes[3].imshow(ir)
    axes[3].imshow(
        second, cmap="magma", vmin=reliability_range[0],
        vmax=reliability_range[1], alpha=0.50)
    draw_boxes(axes[3], boxes)
    axes[4].imshow(
        delta, cmap="RdBu_r", vmin=delta_range[0], vmax=delta_range[1])
    draw_boxes(axes[4], boxes)
    zoom, zoom_boxes = make_gt_zoom(rgb, ir, boxes)
    axes[5].imshow(zoom)
    draw_boxes(axes[5], zoom_boxes, label_text=True)
    axes[5].text(
        0.25,
        0.03,
        "RGB",
        transform=axes[5].transAxes,
        ha="center",
        va="bottom",
        color="white",
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
    )
    axes[5].text(
        0.75,
        0.03,
        "IR",
        transform=axes[5].transAxes,
        ha="center",
        va="bottom",
        color="white",
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
    )

    titles = (
        "RGB + GT",
        "IR + GT",
        r"$R_{\mathrm{IR}\rightarrow\mathrm{RGB}}$",
        r"$R_{\mathrm{RGB}\rightarrow\mathrm{IR}}$",
        r"$\Delta R$",
        "Largest-GT zoom",
    )
    for index, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#777777")
        if show_titles:
            ax.set_title(titles[index], fontsize=9, pad=4)
    axes[0].set_ylabel(
        f"{item['scene_display']}\nID {item['image_id']}", fontsize=9, labelpad=6
    )


def add_colorbars(
    fig: plt.Figure,
    reliability_range: Tuple[float, float] = (0.0, 1.0),
    delta_range: Tuple[float, float] = (-1.0, 1.0),
) -> None:
    reliability = plt.cm.ScalarMappable(
        norm=Normalize(*reliability_range), cmap="magma")
    difference = plt.cm.ScalarMappable(
        norm=Normalize(*delta_range), cmap="RdBu_r")
    reliability_axis = fig.add_axes([0.382, 0.050, 0.185, 0.010])
    difference_axis = fig.add_axes([0.666, 0.050, 0.115, 0.010])
    colorbar_r = fig.colorbar(reliability, cax=reliability_axis, orientation="horizontal")
    colorbar_r.set_label("Reliability", fontsize=8, labelpad=1)
    colorbar_r.ax.tick_params(labelsize=7)
    colorbar_d = fig.colorbar(difference, cax=difference_axis, orientation="horizontal")
    colorbar_d.set_label(r"$\Delta R$", fontsize=8, labelpad=1)
    colorbar_d.ax.tick_params(labelsize=7)


def save_figures(
    items: Sequence[Dict[str, object]],
    output_dir: Path,
    basename: str = "panel_A_bidirectional_reliability",
    title: str = (
        "Panel A  |  Bidirectional reliability fields of RGCA "
        "on M3FD test scenes"),
    footer: str = (
        "P3 (stride 8); fixed ranges: reliability [0, 1], difference "
        "[-1, 1]; yellow boxes are GT only."),
    reliability_range: Tuple[float, float] = (0.0, 1.0),
    delta_range: Tuple[float, float] = (-1.0, 1.0),
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(4, 6, figsize=(15.6, 10.2), squeeze=False)
    for row_index, item in enumerate(items):
        draw_row(
            axes[row_index], item, show_titles=row_index == 0,
            reliability_range=reliability_range, delta_range=delta_range)
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.002,
        footer,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.055, right=0.965, top=0.945, bottom=0.090, wspace=0.045, hspace=0.08)
    add_colorbars(fig, reliability_range, delta_range)
    fig.savefig(output_dir / f"{basename}.png", dpi=600)
    fig.savefig(output_dir / f"{basename}.pdf", dpi=600)
    plt.close(fig)

    row_dir = output_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        row_fig, row_axes = plt.subplots(1, 6, figsize=(15.6, 2.8), squeeze=False)
        draw_row(
            row_axes[0], item, show_titles=True,
            reliability_range=reliability_range, delta_range=delta_range)
        row_fig.suptitle(
            f"Panel A — {item['scene_display']} (M3FD test ID {item['image_id']})",
            fontsize=11,
            y=0.99,
        )
        row_fig.subplots_adjust(left=0.055, right=0.965, top=0.82, bottom=0.18, wspace=0.045)
        add_colorbars(row_fig, reliability_range, delta_range)
        row_fig.savefig(
            row_dir / f"{basename}_{item['scene']}.png", dpi=600)
        plt.close(row_fig)


def main() -> None:
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.scene_root = args.scene_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_maps"
    raw_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples, sampling_audit = select_samples(
        args.dataset_root, args.scene_root, args.seed, args.min_gt
    )
    device = resolve_device(args.device)
    print(f"Loading checkpoint: {args.weights}")
    model = attempt_load(str(args.weights), map_location=device).to(device).float().eval()
    layer = find_rgca_layer(model, args.layer)
    layer.export_reliability_maps = True

    items: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    for sample in samples:
        image_id = str(sample["image_id"])
        scene = str(sample["scene"])
        rgb_path, ir_path, label_path = paired_paths(args.dataset_root, image_id)
        labels = read_yolo_labels(label_path)
        rgb, ir, tensor, boxes = prepare_pair(
            rgb_path, ir_path, labels, args.img_size
        )
        tensor = tensor.to(device, non_blocking=True)
        with torch.inference_mode():
            _ = model(tensor[:, :3], tensor[:, 3:])
        if not hasattr(layer, "last_reliability_maps"):
            raise RuntimeError(f"No reliability maps were exported by {args.layer}")
        exported = layer.last_reliability_maps
        native_first = exported["r_ir_to_rgb"][0, 0].numpy().astype(np.float32)
        native_second = exported["r_rgb_to_ir"][0, 0].numpy().astype(np.float32)
        if not np.isfinite(native_first).all() or not np.isfinite(native_second).all():
            raise RuntimeError(f"Non-finite reliability map for sample {image_id}")
        if native_first.min() < 0 or native_first.max() > 1:
            raise RuntimeError(f"R_IR->RGB is outside [0,1] for sample {image_id}")
        if native_second.min() < 0 or native_second.max() > 1:
            raise RuntimeError(f"R_RGB->IR is outside [0,1] for sample {image_id}")

        first = F.interpolate(
            torch.from_numpy(native_first)[None, None],
            size=(args.img_size, args.img_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy().astype(np.float32)
        second = F.interpolate(
            torch.from_numpy(native_second)[None, None],
            size=(args.img_size, args.img_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy().astype(np.float32)
        delta_native = (native_first - native_second).astype(np.float32)
        prefix = raw_dir / f"{scene}_{image_id}"
        np.save(str(prefix) + "_r_ir_to_rgb.npy", native_first, allow_pickle=False)
        np.save(str(prefix) + "_r_rgb_to_ir.npy", native_second, allow_pickle=False)
        np.save(str(prefix) + "_delta.npy", delta_native, allow_pickle=False)

        item = dict(sample)
        item.update(
            {
                "rgb": rgb,
                "ir": ir,
                "boxes": boxes,
                "r_ir_to_rgb": first,
                "r_rgb_to_ir": second,
                "native_map_height": native_first.shape[0],
                "native_map_width": native_first.shape[1],
            }
        )
        items.append(item)
        manifest_rows.append(
            {
                "selection_order": sample["selection_order"],
                "scene": scene,
                "scene_display": sample["scene_display"],
                "image_id": image_id,
                "rgb_path": str(rgb_path),
                "ir_path": str(ir_path),
                "label_path": str(label_path),
                "gt_count": len(labels),
                "seed": args.seed,
                "eligible_index_zero_based": sample["eligible_index"],
                "eligible_count": sample["eligible_count"],
                "unambiguous_scene_count": sample["unambiguous_scene_count"],
            }
        )
        stats_rows.append(calculate_stats(scene, image_id, first, second, boxes))
        print(
            f"[{sample['scene_display']}] ID={image_id}, GT={len(labels)}, "
            f"native map={native_first.shape}, mean |delta|={stats_rows[-1]['delta_abs_mean']:.6f}"
        )

    layer.export_reliability_maps = False
    write_csv(args.output_dir / "sample_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "panel_A_reliability_stats.csv", stats_rows)
    save_figures(items, args.output_dir)

    config = {
        "task": "RGCA Panel A bidirectional reliability visualization",
        "weights": str(args.weights),
        "weights_sha256": sha256sum(args.weights),
        "architecture": "pure RGCA + Mask + GFB (main M3FD experiment)",
        "layer": args.layer,
        "feature_level": "P3",
        "feature_stride": 8,
        "input_size": [args.img_size, args.img_size],
        "device": str(device),
        "seed": args.seed,
        "dataset_root": str(args.dataset_root),
        "scene_manifest_root": str(
            args.scene_root / "manifests/original_m3fd_split"
        ),
        "split": "canonical M3FD test",
        "sampling": sampling_audit,
        "direction_semantics": {
            "r_ir_to_rgb": "reliability_rgb gates IR-derived mixed_rgb injected into RGB",
            "r_rgb_to_ir": "reliability_ir gates RGB-derived mixed_ir injected into IR",
            "delta": "r_ir_to_rgb - r_rgb_to_ir",
        },
        "preprocessing": {
            "implementation": "matches LoadMultiModalImagesAndLabels validation path",
            "resize": "long side to img_size",
            "letterbox_auto": False,
            "letterbox_scaleup": False,
            "letterbox_value": 114,
            "normalization": "uint8 RGB / 255.0",
        },
        "visualization": {
            "reliability_range": [0.0, 1.0],
            "reliability_colormap": "magma",
            "reliability_overlay_alpha": 0.50,
            "delta_range": [-1.0, 1.0],
            "delta_colormap": "RdBu_r",
            "per_image_minmax_normalization": False,
            "upsampling": "bilinear, align_corners=False",
            "gt_color": "#ffea00",
            "gt_linewidth": 0.9,
            "prediction_boxes": False,
            "png_dpi": 600,
            "pdf_fonttype": 42,
        },
        "raw_maps": {
            "dtype": "float32",
            "resolution": [items[0]["native_map_height"], items[0]["native_map_width"]],
            "format": "NumPy .npy, allow_pickle=False",
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "opencv": str(cv2.__version__),
            "matplotlib": str(matplotlib.__version__),
            "git_commit": "unavailable_no_git_metadata",
        },
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    caption = (
        "Panel A. Bidirectional reliability fields learned by RGCA on M3FD test scenes. "
        "For each uniformly sampled test pair (seed 2026), RGB and IR images share the same "
        "ground-truth boxes. R_IR->RGB is the spatial gate applied to IR-derived information "
        "before injection into the RGB branch, whereas R_RGB->IR gates RGB-derived information "
        "before injection into the IR branch. Delta R=R_IR->RGB-R_RGB->IR visualizes directional "
        "asymmetry. Maps are exported from P3 without per-image normalization; reliability and "
        "difference ranges are fixed to [0,1] and [-1,1], respectively."
    )
    (args.output_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Panel A artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
