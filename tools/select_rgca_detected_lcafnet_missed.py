#!/usr/bin/env python3
"""Render M3FD instances detected by full RGCA but missed by LCAFNet."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_rgca_better_detection_pairs import (
    CLASS_NAMES,
    evaluate_image,
    read_yolo,
    scene_memberships,
    xywh_to_xyxy,
)


# image ID -> GT row index.  Every selected GT is matched by RGCA and unmatched
# by LCAFNet under the same class-aware IoU >= 0.50 protocol.
DEFAULT_TARGETS = {
    "02720": 13,
    "02576": 3,
    "03065": 4,
    "02503": 6,
    "02721": 4,
    "00524": 0,
    "00129": 12,
    "00172": 11,
    "01948": 5,
    "01958": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("paper/m3fd_detection_results"))
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD"))
    parser.add_argument(
        "--scene-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024"))
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "paper/m3fd_detection_results/rgca_detected_lcafnet_missed_top10"))
    return parser.parse_args()


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[1:3], second[1:3])
    right_bottom = np.minimum(first[3:5], second[3:5])
    intersection = float(np.maximum(right_bottom - left_top, 0.0).prod())
    first_area = float(np.maximum(first[3:5] - first[1:3], 0.0).prod())
    second_area = float(np.maximum(second[3:5] - second[1:3], 0.0).prod())
    return intersection / max(first_area + second_area - intersection, 1e-12)


def match_instances(
    ground_truth: np.ndarray, predictions: np.ndarray, threshold: float
) -> Dict[int, Tuple[int, float]]:
    gt_xyxy = xywh_to_xyxy(ground_truth)
    pred_xyxy = xywh_to_xyxy(predictions)
    matched_gt = set()
    matches: Dict[int, Tuple[int, float]] = {}
    order = np.argsort(-predictions[:, 5]).tolist() if len(predictions) else []
    for prediction_index in order:
        candidates = [
            (box_iou(pred_xyxy[prediction_index], gt_xyxy[gt_index]), gt_index)
            for gt_index in range(len(gt_xyxy))
            if gt_index not in matched_gt
            and int(predictions[prediction_index, 0])
            == int(ground_truth[gt_index, 0])
        ]
        if not candidates:
            continue
        overlap, gt_index = max(candidates)
        if overlap >= threshold:
            matched_gt.add(gt_index)
            matches[gt_index] = (prediction_index, overlap)
    return matches


def best_same_class_candidate(
    target: np.ndarray, predictions: np.ndarray
) -> Tuple[int, float, float]:
    if not len(predictions):
        return -1, 0.0, 0.0
    target_xyxy = xywh_to_xyxy(target.reshape(1, -1))[0]
    prediction_xyxy = xywh_to_xyxy(predictions)
    candidates = [
        (index, box_iou(prediction_xyxy[index], target_xyxy),
         float(predictions[index, 5]))
        for index in range(len(predictions))
        if int(predictions[index, 0]) == int(target[0])
    ]
    if not candidates:
        return -1, 0.0, 0.0
    best = max(candidates, key=lambda item: item[1])
    # A same-class box elsewhere in the image is not evidence for this target.
    # Report zero confidence when no candidate has any spatial overlap.
    return best if best[1] > 0.0 else (-1, 0.0, 0.0)


def collect_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    memberships = scene_memberships(args.scene_root)
    rows: List[Dict[str, object]] = []
    for label_path in sorted((args.dataset_root / "labels/vis_test").glob("*.txt")):
        image_id = label_path.stem
        ground_truth = read_yolo(label_path, 5)
        rgca = read_yolo(
            args.result_root / "rgca_full/labels" / f"{image_id}.txt", 6)
        lcafnet = read_yolo(
            args.result_root / "lcafnet_baseline/labels" / f"{image_id}.txt", 6)
        rgca_matches = match_instances(ground_truth, rgca, args.iou_threshold)
        lcafnet_matches = match_instances(ground_truth, lcafnet, args.iou_threshold)
        rgca_metrics = evaluate_image(
            ground_truth, rgca, args.iou_threshold)
        lcafnet_metrics = evaluate_image(
            ground_truth, lcafnet, args.iou_threshold)
        for target_index in sorted(set(rgca_matches) - set(lcafnet_matches)):
            prediction_index, overlap = rgca_matches[target_index]
            best_index, best_overlap, best_confidence = best_same_class_candidate(
                ground_truth[target_index], lcafnet)
            row: Dict[str, object] = {
                "image_id": image_id,
                "scene": "+".join(memberships.get(image_id, ["unassigned"])),
                "gt_count": len(ground_truth),
                "target_gt_index": target_index,
                "target_class_id": int(ground_truth[target_index, 0]),
                "target_class": CLASS_NAMES[int(ground_truth[target_index, 0])],
                "rgca_prediction_index": prediction_index,
                "rgca_target_confidence": float(rgca[prediction_index, 5]),
                "rgca_target_iou": overlap,
                "lcafnet_best_same_class_prediction_index": best_index,
                "lcafnet_best_same_class_confidence": best_confidence,
                "lcafnet_best_same_class_iou": best_overlap,
            }
            for model, metrics in (
                ("rgca", rgca_metrics), ("lcafnet", lcafnet_metrics)):
                for name, value in metrics.items():
                    row[f"{model}_image_{name}"] = value
            row["image_f1_gain"] = (
                rgca_metrics["f1"] - lcafnet_metrics["f1"])
            row["image_detection_accuracy_gain"] = (
                rgca_metrics["detection_accuracy"]
                - lcafnet_metrics["detection_accuracy"])
            rows.append(row)
    return rows


def box_pixels(box: np.ndarray, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    _, x, y, w, h = box[:5].tolist()
    return (
        int(round((x - w / 2.0) * width)),
        int(round((y - h / 2.0) * height)),
        int(round((x + w / 2.0) * width)),
        int(round((y + h / 2.0) * height)),
    )


def draw_labeled_box(
    image: np.ndarray, box: np.ndarray, label: str,
    color: Tuple[int, int, int], thickness: int = 4,
) -> None:
    x1, y1, x2, y2 = box_pixels(box, image.shape)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font_scale = 0.58
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    top = max(y1 - text_height - 8, 0)
    cv2.rectangle(image, (x1, top),
                  (min(x1 + text_width + 6, image.shape[1] - 1),
                   top + text_height + 8), color, -1)
    cv2.putText(image, label, (x1 + 3, top + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_gt(
    image: np.ndarray, ground_truth: np.ndarray, target_index: int
) -> np.ndarray:
    result = image.copy()
    for index, box in enumerate(ground_truth):
        is_target = index == target_index
        draw_labeled_box(
            result, box,
            (f"TARGET GT {CLASS_NAMES[int(box[0])]}" if is_target
             else f"GT {CLASS_NAMES[int(box[0])] }"),
            (210, 0, 210) if is_target else (38, 190, 38),
            5 if is_target else 2)
    return result


def crop_bounds(box: np.ndarray, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = box_pixels(box, shape)
    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    span_x = min(max((x2 - x1) * 4.0, 180.0), width)
    span_y = min(max((y2 - y1) * 4.0, 150.0), height)
    left = int(round(max(min(centre_x - span_x / 2.0, width - span_x), 0)))
    top = int(round(max(min(centre_y - span_y / 2.0, height - span_y), 0)))
    return left, top, int(round(left + span_x)), int(round(top + span_y))


def titled_panel(
    image: np.ndarray, title: str, width: int = 720, height: int = 500
) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    strip = np.full((48, width, 3), 247, dtype=np.uint8)
    cv2.putText(strip, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (25, 25, 25), 2, cv2.LINE_AA)
    return np.vstack((strip, resized))


def crop_panel(image: np.ndarray, bounds: Tuple[int, int, int, int], title: str) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    return titled_panel(image[y1:y2, x1:x2], title, height=360)


def render_selected(
    args: argparse.Namespace, row: Dict[str, object], output_path: Path
) -> None:
    image_id = str(row["image_id"])
    ground_truth = read_yolo(
        args.dataset_root / "labels/vis_test" / f"{image_id}.txt", 5)
    target_index = int(row["target_gt_index"])
    target = ground_truth[target_index]
    rgca_predictions = read_yolo(
        args.result_root / "rgca_full/labels" / f"{image_id}.txt", 6)
    rgca_prediction = rgca_predictions[int(row["rgca_prediction_index"])]

    sources = []
    rgca_images = []
    lcafnet_images = []
    for modality, folder in (("rgb", "vis_test"), ("ir", "Ir_test")):
        source = cv2.imread(str(
            args.dataset_root / "images" / folder / f"{image_id}.png"))
        rgca = cv2.imread(str(
            args.result_root / "rgca_full" / f"{image_id}_{modality}.png"))
        lcafnet = cv2.imread(str(
            args.result_root / "lcafnet_baseline" / f"{image_id}_{modality}.png"))
        if source is None or rgca is None or lcafnet is None:
            raise FileNotFoundError(f"Incomplete image results for {image_id}")
        source = draw_gt(source, ground_truth, target_index)
        draw_labeled_box(
            rgca, rgca_prediction,
            (f"RGCA HIT {row['target_class']} "
             f"{float(row['rgca_target_confidence']):.2f}"),
            (0, 210, 255), 5)
        draw_labeled_box(
            lcafnet, target, f"LCAFNet MISSED GT {row['target_class']}",
            (210, 0, 210), 5)
        sources.append(source)
        rgca_images.append(rgca)
        lcafnet_images.append(lcafnet)

    bounds = crop_bounds(target, sources[0].shape)
    full_rgb = np.hstack((
        titled_panel(sources[0], "Ground truth | RGB"),
        titled_panel(rgca_images[0], "Full RGCA | RGB"),
        titled_panel(lcafnet_images[0], "LCAFNet | RGB"),
    ))
    full_ir = np.hstack((
        titled_panel(sources[1], "Ground truth | IR"),
        titled_panel(rgca_images[1], "Full RGCA | IR"),
        titled_panel(lcafnet_images[1], "LCAFNet | IR"),
    ))
    crop_rgb = np.hstack((
        crop_panel(sources[0], bounds, "Target crop | GT RGB"),
        crop_panel(rgca_images[0], bounds, "Target crop | RGCA RGB"),
        crop_panel(lcafnet_images[0], bounds, "Target crop | LCAFNet RGB"),
    ))
    crop_ir = np.hstack((
        crop_panel(sources[1], bounds, "Target crop | GT IR"),
        crop_panel(rgca_images[1], bounds, "Target crop | RGCA IR"),
        crop_panel(lcafnet_images[1], bounds, "Target crop | LCAFNet IR"),
    ))

    header = np.full((176, full_rgb.shape[1], 3), 255, dtype=np.uint8)
    lines = (
        (f"M3FD test ID {image_id} | scene={row['scene']} | target="
         f"{row['target_class']} (GT index {target_index}) | "
         f"conf>={args.confidence_threshold:.2f}, IoU>={args.iou_threshold:.2f}",
         (20, 20, 20)),
        (f"RGCA HIT: confidence={float(row['rgca_target_confidence']):.3f}, "
         f"IoU={float(row['rgca_target_iou']):.3f}", (20, 125, 20)),
        (f"LCAFNet MISS: best same-class candidate IoU="
         f"{float(row['lcafnet_best_same_class_iou']):.3f}, confidence="
         f"{float(row['lcafnet_best_same_class_confidence']):.3f}",
         (30, 95, 190)),
        (f"Whole image F1 RGCA/LCAFNet="
         f"{float(row['rgca_image_f1']):.3f}/"
         f"{float(row['lcafnet_image_f1']):.3f} | "
         f"TP/FP/FN RGCA={int(row['rgca_image_tp'])}/"
         f"{int(row['rgca_image_fp'])}/{int(row['rgca_image_fn'])}, "
         f"LCAFNet={int(row['lcafnet_image_tp'])}/"
         f"{int(row['lcafnet_image_fp'])}/{int(row['lcafnet_image_fn'])}",
         (20, 20, 20)),
    )
    for index, (line, color) in enumerate(lines):
        cv2.putText(header, line, (18, 34 + index * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.69, color, 2, cv2.LINE_AA)
    result = np.vstack((header, full_rgb, full_ir, crop_rgb, crop_ir))
    if not cv2.imwrite(str(output_path), result,
                       [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise IOError(output_path)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gallery(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    cards = []
    for row in rows:
        image_id = html.escape(str(row["image_id"]))
        cards.append(
            f'<figure><a href="comparisons/{image_id}.jpg">'
            f'<img loading="lazy" src="comparisons/{image_id}.jpg"></a>'
            f'<figcaption>ID {image_id} · {html.escape(str(row["scene"]))} · '
            f'{html.escape(str(row["target_class"]))} · RGCA conf/IoU '
            f'{float(row["rgca_target_confidence"]):.3f}/'
            f'{float(row["rgca_target_iou"]):.3f}</figcaption></figure>')
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RGCA hits missed by LCAFNet</title><style>
body{font-family:Arial,sans-serif;margin:20px;background:#f3f4f6;color:#111827}
figure{background:white;padding:10px;margin:18px 0;border-radius:8px}
img{width:100%;height:auto;display:block}figcaption{text-align:center;font-weight:600;padding:8px}
</style></head><body><h1>M3FD test：RGCA检出、LCAFNet漏检的10组目标</h1>
<p>同类别、置信度≥0.25，一对一IoU≥0.50。每组同时提供全图和目标裁剪。</p>"""
    path.write_text(document + "".join(cards) + "</body></html>")


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.scene_root = args.scene_root.resolve()
    args.output_dir = args.output_dir.resolve()
    comparison_dir = args.output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args)
    lookup = {
        (str(row["image_id"]), int(row["target_gt_index"])): row
        for row in candidates
    }
    selected = []
    for image_id, target_index in DEFAULT_TARGETS.items():
        key = (image_id, target_index)
        if key not in lookup:
            raise ValueError(f"Selection no longer satisfies RGCA-only match: {key}")
        selected.append(lookup[key])
    for row in selected:
        render_selected(
            args, row,
            comparison_dir / f"{row['image_id']}_gt{row['target_gt_index']}.jpg")

    write_csv(args.output_dir / "all_rgca_hit_lcafnet_miss_candidates.csv", candidates)
    write_csv(args.output_dir / "selected_10_instances.csv", selected)
    write_gallery(args.output_dir / "gallery.html", selected)
    print(
        f"Found {len(candidates)} RGCA-only matched objects in "
        f"{len(set(str(row['image_id']) for row in candidates))} images")
    print(f"Rendered {len(selected)} selected comparisons")
    for row in selected:
        print(
            f"{row['image_id']} gt={row['target_gt_index']} "
            f"{row['target_class']}: RGCA conf={row['rgca_target_confidence']:.3f}, "
            f"IoU={row['rgca_target_iou']:.3f}; LCAF best IoU="
            f"{row['lcafnet_best_same_class_iou']:.3f}")


if __name__ == "__main__":
    main()
