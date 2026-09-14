#!/usr/bin/env python3
"""Scan M3FD test and render the strongest natural RGCA gate contrasts.

This is a transparent, metric-based exemplar selection utility, not an
unbiased sampling utility.  Every eligible test pair is scored before one
example per scene is selected.  The complete ranking is always saved.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.visualize_rgca_panel_a import (  # noqa: E402
    SCENES,
    calculate_stats,
    find_rgca_layer,
    image_id_from_manifest_line,
    paired_paths,
    prepare_pair,
    read_nonempty_lines,
    read_yolo_labels,
    resolve_device,
    save_figures,
    sha256sum,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt",
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
        default=REPO_ROOT / "paper/rgca_reliability_visualization/panel_a_representative",
    )
    parser.add_argument("--layer", default="model.21.cross_modal_attention")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--diagnostic-quantile",
        type=float,
        default=0.995,
        help="Dataset-global quantile used only for the diagnostic figure.",
    )
    return parser.parse_args()


def build_candidates(
    dataset_root: Path, scene_root: Path, min_gt: int
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, int]]:
    manifest_dir = scene_root / "manifests/original_m3fd_split"
    canonical_ids = {
        path.stem for path in (dataset_root / "images/vis_test").glob("*.png")
    }
    scene_members: Dict[str, set] = {}
    for scene, _ in SCENES:
        manifest = manifest_dir / f"{scene}_test_vis.txt"
        scene_members[scene] = {
            image_id_from_manifest_line(line)
            for line in read_nonempty_lines(manifest)
        } & canonical_ids
    membership = Counter(
        image_id for members in scene_members.values() for image_id in members
    )
    ambiguous = sorted(
        image_id for image_id, count in membership.items() if count > 1
    )
    ambiguous_set = set(ambiguous)

    candidates: List[Dict[str, object]] = []
    scene_counts: Dict[str, int] = {}
    for scene, display_name in SCENES:
        eligible_count = 0
        for image_id in sorted(scene_members[scene] - ambiguous_set):
            try:
                rgb_path, ir_path, label_path = paired_paths(
                    dataset_root, image_id)
            except FileNotFoundError:
                continue
            labels = read_yolo_labels(label_path)
            if len(labels) < min_gt:
                continue
            candidates.append(
                {
                    "scene": scene,
                    "scene_display": display_name,
                    "image_id": image_id,
                    "gt_count": len(labels),
                    "rgb_path": str(rgb_path),
                    "ir_path": str(ir_path),
                    "label_path": str(label_path),
                }
            )
            eligible_count += 1
        scene_counts[scene] = eligible_count
    return candidates, ambiguous, scene_counts


def scan_candidates(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    candidates: List[Dict[str, object]],
    device: torch.device,
    img_size: int,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    scan_rows: List[Dict[str, object]] = []
    map_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    reliability_values: List[np.ndarray] = []
    absolute_delta_values: List[np.ndarray] = []

    for start in tqdm(
        range(0, len(candidates), batch_size),
        desc="Scanning M3FD test reliability",
        unit="batch",
    ):
        batch_candidates = candidates[start:start + batch_size]
        tensors = []
        for candidate in batch_candidates:
            labels = read_yolo_labels(Path(str(candidate["label_path"])))
            _, _, tensor, _ = prepare_pair(
                Path(str(candidate["rgb_path"])),
                Path(str(candidate["ir_path"])),
                labels,
                img_size,
            )
            tensors.append(tensor)
        batch = torch.cat(tensors, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            _ = model(batch[:, :3], batch[:, 3:])
        exported = layer.last_reliability_maps
        first_batch = exported["r_ir_to_rgb"][:, 0].numpy().astype(np.float32)
        second_batch = exported["r_rgb_to_ir"][:, 0].numpy().astype(np.float32)
        if len(first_batch) != len(batch_candidates):
            raise RuntimeError("Reliability export batch size mismatch")

        for index, candidate in enumerate(batch_candidates):
            first = first_batch[index]
            second = second_batch[index]
            delta = first - second
            key = (str(candidate["scene"]), str(candidate["image_id"]))
            map_cache[key] = (first.copy(), second.copy())
            reliability_values.extend((first.ravel(), second.ravel()))
            absolute_delta_values.append(np.abs(delta).ravel())
            scan_rows.append(
                {
                    **candidate,
                    "r_ir_to_rgb_mean": float(first.mean()),
                    "r_ir_to_rgb_std": float(first.std()),
                    "r_ir_to_rgb_min": float(first.min()),
                    "r_ir_to_rgb_max": float(first.max()),
                    "r_rgb_to_ir_mean": float(second.mean()),
                    "r_rgb_to_ir_std": float(second.std()),
                    "r_rgb_to_ir_min": float(second.min()),
                    "r_rgb_to_ir_max": float(second.max()),
                    "delta_mean": float(delta.mean()),
                    "delta_abs_mean": float(np.abs(delta).mean()),
                    "delta_abs_max": float(np.abs(delta).max()),
                    "direction_map_pearson_r": float(
                        np.corrcoef(first.ravel(), second.ravel())[0, 1]
                    ),
                }
            )
        del batch

    return (
        scan_rows,
        map_cache,
        np.concatenate(reliability_values),
        np.concatenate(absolute_delta_values),
    )


def rank_and_select(
    scan_rows: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ranked_all: List[Dict[str, object]] = []
    selected: List[Dict[str, object]] = []
    for scene, _ in SCENES:
        rows = [row for row in scan_rows if row["scene"] == scene]
        rows.sort(
            key=lambda row: (
                -float(row["delta_abs_mean"]),
                -float(row["delta_abs_max"]),
                str(row["image_id"]),
            )
        )
        for rank, row in enumerate(rows, start=1):
            ranked = dict(row)
            ranked["scene_rank_by_delta_abs_mean"] = rank
            ranked["scene_percentile"] = 100.0 * (len(rows) - rank + 1) / len(rows)
            ranked_all.append(ranked)
        selected.append(ranked_all[-len(rows)])
    return ranked_all, selected


def build_selected_items(
    selected: List[Dict[str, object]],
    map_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    img_size: int,
    raw_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    items: List[Dict[str, object]] = []
    stats: List[Dict[str, object]] = []
    for row in selected:
        labels = read_yolo_labels(Path(str(row["label_path"])))
        rgb, ir, _, boxes = prepare_pair(
            Path(str(row["rgb_path"])), Path(str(row["ir_path"])),
            labels, img_size)
        key = (str(row["scene"]), str(row["image_id"]))
        native_first, native_second = map_cache[key]
        first = F.interpolate(
            torch.from_numpy(native_first)[None, None],
            size=(img_size, img_size), mode="bilinear", align_corners=False,
        )[0, 0].numpy().astype(np.float32)
        second = F.interpolate(
            torch.from_numpy(native_second)[None, None],
            size=(img_size, img_size), mode="bilinear", align_corners=False,
        )[0, 0].numpy().astype(np.float32)
        prefix = raw_dir / f"{row['scene']}_{row['image_id']}"
        np.save(str(prefix) + "_r_ir_to_rgb.npy", native_first, allow_pickle=False)
        np.save(str(prefix) + "_r_rgb_to_ir.npy", native_second, allow_pickle=False)
        np.save(
            str(prefix) + "_delta.npy",
            (native_first - native_second).astype(np.float32),
            allow_pickle=False,
        )
        item = dict(row)
        item.update(
            {
                "rgb": rgb,
                "ir": ir,
                "boxes": boxes,
                "r_ir_to_rgb": first,
                "r_rgb_to_ir": second,
            }
        )
        items.append(item)
        stats.append(calculate_stats(
            str(row["scene"]), str(row["image_id"]),
            first, second, boxes))
    return items, stats


def main() -> None:
    args = parse_args()
    if not 0.5 < args.diagnostic_quantile < 1.0:
        raise ValueError("--diagnostic-quantile must be between 0.5 and 1")
    for name in ("weights", "dataset_root", "scene_root", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_maps"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates, ambiguous, scene_counts = build_candidates(
        args.dataset_root, args.scene_root, args.min_gt)
    device = resolve_device(args.device)
    print(f"Loading checkpoint: {args.weights}")
    model = attempt_load(str(args.weights), map_location=device).to(device).float().eval()
    layer = find_rgca_layer(model, args.layer)
    layer.export_reliability_maps = True
    scan_rows, map_cache, reliability_values, absolute_delta_values = scan_candidates(
        model, layer, candidates, device, args.img_size, args.batch_size)
    layer.export_reliability_maps = False

    ranked_rows, selected = rank_and_select(scan_rows)
    write_csv(args.output_dir / "all_test_reliability_scan.csv", ranked_rows)
    write_csv(args.output_dir / "selected_manifest.csv", selected)

    quantile = args.diagnostic_quantile
    tail = (1.0 - quantile) / 2.0
    reliability_range = (
        float(np.quantile(reliability_values, tail)),
        float(np.quantile(reliability_values, 1.0 - tail)),
    )
    delta_limit = float(np.quantile(absolute_delta_values, quantile))
    delta_limit = max(delta_limit, np.finfo(np.float32).eps)
    diagnostic_delta_range = (-delta_limit, delta_limit)

    items, selected_stats = build_selected_items(
        selected, map_cache, args.img_size, raw_dir)
    write_csv(args.output_dir / "selected_reliability_stats.csv", selected_stats)

    save_figures(
        items,
        args.output_dir,
        basename="panel_A_representative_fixed_scale",
        title=(
            "Panel A  |  Highest natural directional contrasts "
            "from the complete M3FD test scan"),
        footer=(
            "P3; selection: maximum mean |directional difference| per scene; "
            "paper scale reliability [0, 1], difference [-1, 1]."),
    )
    save_figures(
        items,
        args.output_dir,
        basename="panel_A_representative_diagnostic_global_scale",
        title=(
            "Diagnostic only  |  Same selected samples with one "
            "dataset-global contrast scale"),
        footer=(
            f"No per-image normalization; global reliability "
            f"[{reliability_range[0]:.6f}, {reliability_range[1]:.6f}], "
            f"difference [{-delta_limit:.6f}, {delta_limit:.6f}] "
            f"from q={quantile:.3f}."),
        reliability_range=reliability_range,
        delta_range=diagnostic_delta_range,
    )

    config = {
        "task": "M3FD complete-test RGCA reliability exemplar scan",
        "selection_disclosure": (
            "Selection-biased diagnostic exemplars, not representative random samples"
        ),
        "selection_metric": "descending native-P3 mean(abs(R_IR_to_RGB - R_RGB_to_IR))",
        "tie_breakers": ["descending delta_abs_max", "ascending image_id"],
        "weights": str(args.weights),
        "weights_sha256": sha256sum(args.weights),
        "layer": args.layer,
        "feature_level": "P3",
        "input_size": [args.img_size, args.img_size],
        "dataset_root": str(args.dataset_root),
        "split": "canonical M3FD test only",
        "minimum_gt": args.min_gt,
        "excluded_ambiguous_scene_ids": ambiguous,
        "eligible_scene_counts": scene_counts,
        "eligible_total": len(candidates),
        "paper_fixed_scale": {
            "reliability": [0.0, 1.0],
            "delta": [-1.0, 1.0],
        },
        "diagnostic_global_scale": {
            "quantile": quantile,
            "reliability": list(reliability_range),
            "delta": list(diagnostic_delta_range),
            "per_image_normalization": False,
        },
        "selected": [
            {
                "scene": row["scene"],
                "image_id": row["image_id"],
                "delta_abs_mean": row["delta_abs_mean"],
                "delta_abs_max": row["delta_abs_max"],
            }
            for row in selected
        ],
    }
    with (args.output_dir / "scan_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    caveat = (
        "The fixed-scale figure follows the original Panel A protocol. The "
        "diagnostic figure uses one global scale estimated from every eligible "
        "M3FD test reliability pixel; it never performs per-image min-max "
        "normalization. These top-ranked natural examples visualize where the "
        "two learned gates differ most, but selection and visualization alone "
        "do not establish calibration or causal reliability. Controlled "
        "modality corruptions and detection-response tests are required for "
        "that claim.\n"
    )
    (args.output_dir / "SCIENTIFIC_CAVEAT.txt").write_text(caveat, encoding="utf-8")

    print("Selected highest-contrast natural test samples:")
    for row in selected:
        print(
            f"  {row['scene_display']}: {row['image_id']}  "
            f"mean|delta|={float(row['delta_abs_mean']):.8f}  "
            f"max|delta|={float(row['delta_abs_max']):.8f}"
        )
    print(
        f"Dataset-global diagnostic ranges: reliability={reliability_range}, "
        f"delta={diagnostic_delta_range}"
    )
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
