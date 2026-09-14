#!/usr/bin/env python3
"""Compare M3FD cross-modal focus with and without RGCA reliability gating.

The visualized quantity is identical for both trained models: channel-RMS
energy of the actual P3 RGB/IR cross-modal residual update.  The primary
figure divides each response by its spatial mean (not min-max normalization)
to compare where the two models allocate response.  GT is used only for
selection and quantitative foreground/background enrichment; boxes are not
drawn in the paper figure.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    foreground_mask,
    prepare_pair,
    read_yolo_labels,
    resolve_device,
    sha256sum,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gated-weights",
        type=Path,
        default=REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt",
    )
    parser.add_argument(
        "--no-gate-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/train/exp_rgca_no_reliability_mask_gfb_uniform_lr_m3fd_full200/weights/best.pt"
        ),
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
        default=REPO_ROOT / "paper/rgca_gate_focus_comparison_m3fd",
    )
    parser.add_argument("--layer", default="model.21.cross_modal_attention")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--display-quantile", type=float, default=0.995)
    return parser.parse_args()


def response_energy(exported: Dict[str, torch.Tensor]) -> np.ndarray:
    rgb = exported["rgb_update"]
    ir = exported["ir_update"]
    energy = torch.sqrt(
        0.5 * (rgb.square().mean(dim=1) + ir.square().mean(dim=1)) + 1e-12
    )
    return energy.numpy().astype(np.float32)


def run_response(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    batch: torch.Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        output = model(batch[:, :3], batch[:, 3:])
    del output
    if not hasattr(layer, "last_cross_modal_response_maps"):
        raise RuntimeError("Cross-modal response hook did not run")
    return response_energy(layer.last_cross_modal_response_maps)


def native_foreground_mask(
    boxes: np.ndarray, img_size: int, native_shape: Tuple[int, int]
) -> np.ndarray:
    mask = foreground_mask(boxes, img_size, img_size)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=native_shape, mode="nearest"
    )[0, 0].numpy()
    return resized > 0.5


def focus_metrics(
    gated_energy: np.ndarray,
    no_gate_energy: np.ndarray,
    foreground: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-12
    gated_density = gated_energy / max(float(gated_energy.mean()), eps)
    no_gate_density = no_gate_energy / max(float(no_gate_energy.mean()), eps)
    background = ~foreground
    if not foreground.any() or not background.any():
        raise ValueError("Foreground mask must contain foreground and background")

    gated_fg = float(gated_density[foreground].mean())
    gated_bg = float(gated_density[background].mean())
    no_gate_fg = float(no_gate_density[foreground].mean())
    no_gate_bg = float(no_gate_density[background].mean())
    gated_enrichment = gated_fg / max(gated_bg, eps)
    no_gate_enrichment = no_gate_fg / max(no_gate_bg, eps)
    log2_gain = float(np.log2((gated_enrichment + eps) / (no_gate_enrichment + eps)))
    return {
        "foreground_area_fraction": float(foreground.mean()),
        "gated_energy_mean": float(gated_energy.mean()),
        "no_gate_energy_mean": float(no_gate_energy.mean()),
        "gated_foreground_density": gated_fg,
        "gated_background_density": gated_bg,
        "no_gate_foreground_density": no_gate_fg,
        "no_gate_background_density": no_gate_bg,
        "gated_foreground_enrichment": gated_enrichment,
        "no_gate_foreground_enrichment": no_gate_enrichment,
        "foreground_enrichment_difference": gated_enrichment - no_gate_enrichment,
        "foreground_enrichment_log2_gain": log2_gain,
    }


def scan_test(
    gated_model: torch.nn.Module,
    gated_layer: torch.nn.Module,
    no_gate_model: torch.nn.Module,
    no_gate_layer: torch.nn.Module,
    candidates: List[Dict[str, object]],
    device: torch.device,
    img_size: int,
    batch_size: int,
) -> Tuple[
    List[Dict[str, object]],
    Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    all_gated_energy: List[np.ndarray] = []
    all_no_gate_energy: List[np.ndarray] = []
    all_density_delta: List[np.ndarray] = []

    for start in tqdm(
        range(0, len(candidates), batch_size),
        desc="Scanning gated vs no-gate focus",
        unit="batch",
    ):
        batch_candidates = candidates[start:start + batch_size]
        tensors: List[torch.Tensor] = []
        boxes_list: List[np.ndarray] = []
        for candidate in batch_candidates:
            labels = read_yolo_labels(Path(str(candidate["label_path"])))
            _, _, tensor, boxes = prepare_pair(
                Path(str(candidate["rgb_path"])),
                Path(str(candidate["ir_path"])),
                labels,
                img_size,
            )
            tensors.append(tensor)
            boxes_list.append(boxes)
        batch = torch.cat(tensors, dim=0).to(device, non_blocking=True)

        gated_energy_batch = run_response(gated_model, gated_layer, batch)
        no_gate_energy_batch = run_response(no_gate_model, no_gate_layer, batch)
        if len(gated_energy_batch) != len(batch_candidates):
            raise RuntimeError("Gated response batch size mismatch")

        for index, candidate in enumerate(batch_candidates):
            gated_energy = gated_energy_batch[index]
            no_gate_energy = no_gate_energy_batch[index]
            foreground = native_foreground_mask(
                boxes_list[index], img_size, gated_energy.shape)
            metrics = focus_metrics(gated_energy, no_gate_energy, foreground)
            key = (str(candidate["scene"]), str(candidate["image_id"]))
            cache[key] = (gated_energy.copy(), no_gate_energy.copy())
            gated_density = gated_energy / max(float(gated_energy.mean()), 1e-12)
            no_gate_density = no_gate_energy / max(float(no_gate_energy.mean()), 1e-12)
            all_gated_energy.append(gated_energy.ravel())
            all_no_gate_energy.append(no_gate_energy.ravel())
            all_density_delta.append((gated_density - no_gate_density).ravel())
            rows.append({**candidate, **metrics})
        del batch

    return (
        rows,
        cache,
        np.concatenate(all_gated_energy),
        np.concatenate(all_no_gate_energy),
        np.concatenate(all_density_delta),
    )


def rank_and_select(
    rows: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ranked_all: List[Dict[str, object]] = []
    selected: List[Dict[str, object]] = []
    for scene, _ in SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        scene_rows.sort(
            key=lambda row: (
                -float(row["foreground_enrichment_log2_gain"]),
                -float(row["foreground_enrichment_difference"]),
                str(row["image_id"]),
            )
        )
        if not scene_rows:
            raise RuntimeError(f"No comparison samples for {scene}")
        for rank, row in enumerate(scene_rows, start=1):
            ranked = dict(row)
            ranked["scene_rank_by_focus_gain"] = rank
            ranked["scene_percentile"] = (
                100.0 * (len(scene_rows) - rank + 1) / len(scene_rows)
            )
            ranked_all.append(ranked)
        selected.append(dict(scene_rows[0]))
    return ranked_all, selected


def aggregate_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    groups = [(scene, [r for r in rows if r["scene"] == scene]) for scene, _ in SCENES]
    groups.append(("all", list(rows)))
    for name, group in groups:
        gains = np.array(
            [float(row["foreground_enrichment_log2_gain"]) for row in group]
        )
        differences = np.array(
            [float(row["foreground_enrichment_difference"]) for row in group]
        )
        gated = np.array(
            [float(row["gated_foreground_enrichment"]) for row in group]
        )
        no_gate = np.array(
            [float(row["no_gate_foreground_enrichment"]) for row in group]
        )
        output.append(
            {
                "scene": name,
                "n": len(group),
                "positive_focus_gain_fraction": float((gains > 0).mean()),
                "log2_focus_gain_mean": float(gains.mean()),
                "log2_focus_gain_median": float(np.median(gains)),
                "enrichment_difference_mean": float(differences.mean()),
                "enrichment_difference_median": float(np.median(differences)),
                "gated_enrichment_mean": float(gated.mean()),
                "no_gate_enrichment_mean": float(no_gate.mean()),
            }
        )
    return output


def upsample_map(array: np.ndarray, img_size: int) -> np.ndarray:
    return torch.nn.functional.interpolate(
        torch.from_numpy(array)[None, None],
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy().astype(np.float32)


def crop_largest_target(
    image: np.ndarray,
    first_map: np.ndarray,
    second_map: np.ndarray,
    boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(boxes):
        return (
            np.concatenate((image, image), axis=1),
            np.concatenate((first_map, second_map), axis=1),
            np.array([image.shape[1]], dtype=np.int32),
        )
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    _, x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(float(x2 - x1), float(y2 - y1)) * 2.0
    side = max(side, 96.0)
    xa = max(0, int(round(cx - side / 2)))
    ya = max(0, int(round(cy - side / 2)))
    xb = min(image.shape[1], int(round(cx + side / 2)))
    yb = min(image.shape[0], int(round(cy + side / 2)))
    base_crop = image[ya:yb, xa:xb]
    first_crop = first_map[ya:yb, xa:xb]
    second_crop = second_map[ya:yb, xa:xb]
    return (
        np.concatenate((base_crop, base_crop), axis=1),
        np.concatenate((first_crop, second_crop), axis=1),
        np.array([base_crop.shape[1]], dtype=np.int32),
    )


def grayscale_base(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    rgb_gray = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    )
    ir_gray = ir.astype(np.float32).mean(axis=2)
    return (0.5 * rgb_gray + 0.5 * ir_gray) / 255.0


def build_items(
    selected: Sequence[Dict[str, object]],
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    img_size: int,
    raw_dir: Path,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    display_lookup = dict(SCENES)
    for row in selected:
        labels = read_yolo_labels(Path(str(row["label_path"])))
        rgb, ir, _, boxes = prepare_pair(
            Path(str(row["rgb_path"])), Path(str(row["ir_path"])),
            labels, img_size)
        key = (str(row["scene"]), str(row["image_id"]))
        gated_energy, no_gate_energy = cache[key]
        gated_density = gated_energy / max(float(gated_energy.mean()), 1e-12)
        no_gate_density = no_gate_energy / max(float(no_gate_energy.mean()), 1e-12)
        prefix = raw_dir / f"{row['scene']}_{row['image_id']}"
        for suffix, array in (
            ("gated_energy", gated_energy),
            ("no_gate_energy", no_gate_energy),
            ("gated_density", gated_density),
            ("no_gate_density", no_gate_density),
            ("density_difference", gated_density - no_gate_density),
        ):
            np.save(str(prefix) + f"_{suffix}.npy", array.astype(np.float32), allow_pickle=False)
        items.append(
            {
                **row,
                "scene_display": display_lookup[str(row["scene"])],
                "rgb": rgb,
                "ir": ir,
                "base": grayscale_base(rgb, ir),
                "boxes": boxes,
                "gated_energy": upsample_map(gated_energy, img_size),
                "no_gate_energy": upsample_map(no_gate_energy, img_size),
                "gated_density": upsample_map(gated_density, img_size),
                "no_gate_density": upsample_map(no_gate_density, img_size),
            }
        )
    return items


def draw_comparison_row(
    axes: Sequence[plt.Axes],
    item: Dict[str, object],
    map_kind: str,
    response_range: Tuple[float, float],
    difference_range: Tuple[float, float],
    show_titles: bool,
) -> None:
    rgb = item["rgb"]
    ir = item["ir"]
    base = item["base"]
    no_gate = item[f"no_gate_{map_kind}"]
    gated = item[f"gated_{map_kind}"]
    difference = gated - no_gate

    axes[0].imshow(rgb)
    axes[1].imshow(ir)
    for ax, response in ((axes[2], no_gate), (axes[3], gated)):
        ax.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(
            response,
            cmap="magma",
            vmin=response_range[0],
            vmax=response_range[1],
            alpha=0.68,
        )
    axes[4].imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].imshow(
        difference,
        cmap="RdBu_r",
        vmin=difference_range[0],
        vmax=difference_range[1],
        alpha=0.76,
    )

    crop_base, crop_map, divider = crop_largest_target(
        base, no_gate, gated, item["boxes"])
    axes[5].imshow(crop_base, cmap="gray", vmin=0.0, vmax=1.0)
    axes[5].imshow(
        crop_map,
        cmap="magma",
        vmin=response_range[0],
        vmax=response_range[1],
        alpha=0.68,
    )
    axes[5].axvline(float(divider[0]) - 0.5, color="white", linewidth=0.7)
    for xpos, label in ((0.25, "w/o gate"), (0.75, "with gate")):
        axes[5].text(
            xpos,
            0.03,
            label,
            transform=axes[5].transAxes,
            ha="center",
            va="bottom",
            color="white",
            fontsize=7,
            bbox={
                "facecolor": "black",
                "alpha": 0.55,
                "pad": 1,
                "edgecolor": "none",
            },
        )

    titles = (
        "RGB",
        "IR",
        "RGCA w/o reliability gate",
        "RGCA with reliability gate",
        "With gate − without gate",
        "Largest-target zoom",
    )
    for index, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#777777")
        if show_titles:
            ax.set_title(titles[index], fontsize=8.5, pad=4)
    axes[0].set_ylabel(
        f"{item['scene_display']}\nID {item['image_id']}",
        fontsize=9,
        labelpad=6,
    )


def add_colorbars(
    fig: plt.Figure,
    response_range: Tuple[float, float],
    difference_range: Tuple[float, float],
    response_label: str,
) -> None:
    response_axis = fig.add_axes([0.383, 0.050, 0.185, 0.010])
    difference_axis = fig.add_axes([0.668, 0.050, 0.115, 0.010])
    response_mappable = plt.cm.ScalarMappable(
        norm=Normalize(*response_range), cmap="magma")
    difference_mappable = plt.cm.ScalarMappable(
        norm=Normalize(*difference_range), cmap="RdBu_r")
    response_bar = fig.colorbar(
        response_mappable, cax=response_axis, orientation="horizontal")
    response_bar.set_label(response_label, fontsize=8, labelpad=1)
    response_bar.ax.tick_params(labelsize=7)
    difference_bar = fig.colorbar(
        difference_mappable, cax=difference_axis, orientation="horizontal")
    difference_bar.set_label("Focus difference", fontsize=8, labelpad=1)
    difference_bar.ax.tick_params(labelsize=7)


def save_comparison(
    items: Sequence[Dict[str, object]],
    output_dir: Path,
    basename: str,
    map_kind: str,
    response_range: Tuple[float, float],
    difference_range: Tuple[float, float],
    title: str,
    footer: str,
    response_label: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(4, 6, figsize=(16.2, 10.2), squeeze=False)
    for row_index, item in enumerate(items):
        draw_comparison_row(
            axes[row_index], item, map_kind, response_range,
            difference_range, show_titles=row_index == 0)
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(0.5, 0.002, footer, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(
        left=0.055, right=0.975, top=0.945, bottom=0.090,
        wspace=0.045, hspace=0.08)
    add_colorbars(fig, response_range, difference_range, response_label)
    fig.savefig(output_dir / f"{basename}.png", dpi=600)
    fig.savefig(output_dir / f"{basename}.pdf", dpi=600)
    plt.close(fig)

    row_dir = output_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        row_fig, row_axes = plt.subplots(1, 6, figsize=(16.2, 2.8), squeeze=False)
        draw_comparison_row(
            row_axes[0], item, map_kind, response_range,
            difference_range, show_titles=True)
        row_fig.suptitle(
            f"{item['scene_display']} — M3FD test ID {item['image_id']}",
            fontsize=11,
            y=0.99,
        )
        row_fig.subplots_adjust(
            left=0.055, right=0.975, top=0.82, bottom=0.18, wspace=0.045)
        add_colorbars(row_fig, response_range, difference_range, response_label)
        row_fig.savefig(
            row_dir / f"{basename}_{item['scene']}.png", dpi=600)
        plt.close(row_fig)


def main() -> None:
    args = parse_args()
    if not 0.5 < args.display_quantile < 1.0:
        raise ValueError("--display-quantile must be between 0.5 and 1")
    for name in (
        "gated_weights", "no_gate_weights", "dataset_root",
        "scene_root", "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.gated_weights, args.no_gate_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_maps"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates, ambiguous, scene_counts = build_candidates(
        args.dataset_root, args.scene_root, args.min_gt)
    device = resolve_device(args.device)
    print(f"Loading gated checkpoint: {args.gated_weights}")
    gated_model = attempt_load(
        str(args.gated_weights), map_location=device).to(device).float().eval()
    print(f"Loading no-gate checkpoint: {args.no_gate_weights}")
    no_gate_model = attempt_load(
        str(args.no_gate_weights), map_location=device).to(device).float().eval()
    gated_layer = find_rgca_layer(gated_model, args.layer)
    no_gate_layer = find_rgca_layer(no_gate_model, args.layer)
    if getattr(gated_layer, "reliability_gate", None) is None:
        raise RuntimeError("The gated checkpoint has no reliability gate")
    if getattr(no_gate_layer, "reliability_gate", "missing") is not None:
        raise RuntimeError("The no-gate checkpoint still contains a gate")
    gated_layer.export_cross_modal_response_maps = True
    no_gate_layer.export_cross_modal_response_maps = True

    rows, cache, all_gated_energy, all_no_gate_energy, all_density_delta = scan_test(
        gated_model, gated_layer, no_gate_model, no_gate_layer,
        candidates, device, args.img_size, args.batch_size)
    gated_layer.export_cross_modal_response_maps = False
    no_gate_layer.export_cross_modal_response_maps = False

    ranked_rows, selected = rank_and_select(rows)
    aggregates = aggregate_metrics(rows)
    write_csv(args.output_dir / "all_test_focus_metrics.csv", ranked_rows)
    write_csv(args.output_dir / "selected_manifest.csv", selected)
    write_csv(args.output_dir / "aggregate_focus_summary.csv", aggregates)
    items = build_items(selected, cache, args.img_size, raw_dir)

    quantile = args.display_quantile
    density_values = []
    absolute_differences = []
    for gated_energy, no_gate_energy in cache.values():
        gated_density = gated_energy / max(float(gated_energy.mean()), 1e-12)
        no_gate_density = no_gate_energy / max(float(no_gate_energy.mean()), 1e-12)
        density_values.extend((gated_density.ravel(), no_gate_density.ravel()))
        absolute_differences.append(np.abs(gated_density - no_gate_density).ravel())
    density_limit = float(np.quantile(np.concatenate(density_values), quantile))
    density_difference_limit = float(
        np.quantile(np.concatenate(absolute_differences), quantile))
    density_difference_limit = max(
        density_difference_limit, np.finfo(np.float32).eps)

    all_energy = np.concatenate((all_gated_energy, all_no_gate_energy))
    energy_limit = float(np.quantile(all_energy, quantile))
    # Absolute energy differences need the signed distribution, not the
    # normalized-density cache used for the primary focus figure.
    absolute_energy_delta = []
    for gated_energy, no_gate_energy in cache.values():
        absolute_energy_delta.append(np.abs(gated_energy - no_gate_energy).ravel())
    energy_difference_limit = float(
        np.quantile(np.concatenate(absolute_energy_delta), quantile))
    energy_difference_limit = max(energy_difference_limit, np.finfo(np.float32).eps)

    save_comparison(
        items,
        args.output_dir,
        basename="m3fd_gate_focus_comparison_normalized",
        map_kind="density",
        response_range=(0.0, density_limit),
        difference_range=(-density_difference_limit, density_difference_limit),
        title=(
            "M3FD  |  Cross-modal spatial focus without vs. with "
            "RGCA reliability gating"),
        footer=(
            "P3 actual residual response; response divided by spatial mean "
            "only (no min-max); GT used for selection, boxes not drawn."),
        response_label="Relative response density",
    )
    save_comparison(
        items,
        args.output_dir,
        basename="m3fd_gate_focus_comparison_absolute_audit",
        map_kind="energy",
        response_range=(0.0, energy_limit),
        difference_range=(-energy_difference_limit, energy_difference_limit),
        title=(
            "Audit view  |  Absolute P3 cross-modal residual energy "
            "without vs. with gating"),
        footer=(
            "One global scale across both checkpoints and all eligible test "
            "samples; no per-image normalization; GT boxes not drawn."),
        response_label="Absolute residual RMS",
    )

    config = {
        "task": "M3FD RGCA reliability-gate focus comparison",
        "reference_layout": str(
            REPO_ROOT / "paper/屏幕截图 2026-08-22 171127.png"),
        "gated_weights": str(args.gated_weights),
        "gated_weights_sha256": sha256sum(args.gated_weights),
        "no_gate_weights": str(args.no_gate_weights),
        "no_gate_weights_sha256": sha256sum(args.no_gate_weights),
        "layer": args.layer,
        "feature_level": "P3",
        "response_definition": (
            "sqrt(0.5*(mean_channel(rgb_update^2)+mean_channel(ir_update^2)))"
        ),
        "primary_normalization": "response divided by its spatial mean; no min-max",
        "selection_disclosure": (
            "One top foreground-enrichment-gain exemplar per scene; selection-biased"
        ),
        "selection_metric": (
            "log2(gated foreground/background enrichment divided by no-gate enrichment)"
        ),
        "gt_usage": "selection and metrics only; no boxes drawn",
        "split": "canonical M3FD test only",
        "eligible_total": len(candidates),
        "eligible_scene_counts": scene_counts,
        "excluded_ambiguous_scene_ids": ambiguous,
        "display_quantile": quantile,
        "normalized_display_ranges": {
            "response": [0.0, density_limit],
            "difference": [-density_difference_limit, density_difference_limit],
        },
        "absolute_display_ranges": {
            "response": [0.0, energy_limit],
            "difference": [-energy_difference_limit, energy_difference_limit],
        },
        "selected": [
            {
                "scene": row["scene"],
                "image_id": row["image_id"],
                "foreground_enrichment_log2_gain": row[
                    "foreground_enrichment_log2_gain"],
                "gated_foreground_enrichment": row[
                    "gated_foreground_enrichment"],
                "no_gate_foreground_enrichment": row[
                    "no_gate_foreground_enrichment"],
            }
            for row in selected
        ],
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    caveat = (
        "The two columns compare independently trained best checkpoints under "
        "the same M3FD protocol. The primary map compares spatial allocation "
        "after dividing each response by its spatial mean; it does not compare "
        "absolute update magnitude. Samples are deliberately selected by the "
        "largest GT-based focus gain in each scene. Use aggregate_focus_summary.csv "
        "to determine whether the claim generalizes beyond selected examples.\n"
    )
    (args.output_dir / "SCIENTIFIC_CAVEAT.txt").write_text(caveat, encoding="utf-8")

    print("Selected focus-gain exemplars:")
    for row in selected:
        print(
            f"  {row['scene_display']}: {row['image_id']}  "
            f"no-gate={float(row['no_gate_foreground_enrichment']):.4f}  "
            f"gated={float(row['gated_foreground_enrichment']):.4f}  "
            f"log2-gain={float(row['foreground_enrichment_log2_gain']):+.4f}"
        )
    print("Aggregate test summary:")
    for row in aggregates:
        print(
            f"  {row['scene']}: n={row['n']}  "
            f"positive={float(row['positive_focus_gain_fraction']):.3f}  "
            f"median_log2_gain={float(row['log2_focus_gain_median']):+.4f}"
        )
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
