#!/usr/bin/env python3
"""Select and render M3FD test examples where full RGCA beats LCAFNet.

Predictions must have been exported by ``detect_twostream.py --save-txt
--save-conf`` with the same confidence and NMS settings.  Per-image metrics use
class-aware, confidence-ordered, one-to-one matching at IoU >= 0.50.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


CLASS_NAMES = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
SCENES = ("daytime", "night", "overcast", "challenge")
MODELS = ("rgca_full", "lcafnet_baseline")

# Representative, auditable examples selected from the complete per-image
# metric table.  The set covers all four scenes and both dominant improvement
# modes: additional true positives and fewer false positives.
DEFAULT_SELECTED_IDS = (
    "02720",  # daytime: +1 TP and -6 FP
    "03065",  # daytime: +2 TP
    "02576",  # daytime: -6 FP
    "02367",  # daytime: perfect RGCA matching, -2 FP
    "00694",  # night: perfect RGCA matching, -2 FP
    "03933",  # night: perfect RGCA matching, -1 FP
    "00397",  # night: perfect RGCA matching, -1 FP
    "00831",  # overcast: perfect RGCA matching, -1 FP
    "03832",  # overcast: perfect RGCA matching, -1 FP
    "01877",  # challenge: -1 FP and higher TP confidence
)


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
        "--selected-ids", nargs="+", default=list(DEFAULT_SELECTED_IDS))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/m3fd_detection_results/rgca_better_top10"))
    return parser.parse_args()


def read_yolo(path: Path, columns: int) -> np.ndarray:
    if not path.exists():
        return np.empty((0, columns), dtype=np.float64)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = [float(value) for value in line.split()]
        if len(row) != columns:
            raise ValueError(
                f"{path}:{line_number}: expected {columns} fields, got {len(row)}")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, columns)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 2] - boxes[:, 4] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    converted[:, 4] = boxes[:, 2] + boxes[:, 4] / 2.0
    return converted


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[1:3], second[1:3])
    right_bottom = np.minimum(first[3:5], second[3:5])
    intersection = float(np.maximum(right_bottom - left_top, 0.0).prod())
    first_area = float(np.maximum(first[3:5] - first[1:3], 0.0).prod())
    second_area = float(np.maximum(second[3:5] - second[1:3], 0.0).prod())
    return intersection / max(first_area + second_area - intersection, 1e-12)


def evaluate_image(
    ground_truth: np.ndarray, predictions: np.ndarray, iou_threshold: float
) -> Dict[str, float]:
    gt_xyxy = xywh_to_xyxy(ground_truth)
    pred_xyxy = xywh_to_xyxy(predictions)
    matched_gt = set()
    matches: List[Tuple[int, int, float]] = []

    # detect_twostream writes detections in reverse order, so explicitly restore
    # confidence ordering before standard one-to-one matching.
    order: Iterable[int] = (
        np.argsort(-pred_xyxy[:, 5]).tolist() if len(pred_xyxy) else [])
    for prediction_index in order:
        candidates = [
            (box_iou(pred_xyxy[prediction_index], gt_xyxy[gt_index]), gt_index)
            for gt_index in range(len(gt_xyxy))
            if gt_index not in matched_gt
            and int(pred_xyxy[prediction_index, 0]) == int(gt_xyxy[gt_index, 0])
        ]
        if not candidates:
            continue
        overlap, gt_index = max(candidates)
        if overlap >= iou_threshold:
            matched_gt.add(gt_index)
            matches.append((prediction_index, gt_index, overlap))

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    # Jaccard-style detection accuracy, explicitly named to avoid confusion
    # with classification accuracy.
    detection_accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    matched_confidences = [predictions[prediction_index, 5]
                           for prediction_index, _, _ in matches]
    matched_ious = [overlap for _, _, overlap in matches]
    return {
        "predictions": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_accuracy": detection_accuracy,
        "matched_tp_mean_confidence": (
            float(np.mean(matched_confidences)) if matched_confidences else 0.0),
        "matched_tp_mean_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0),
    }


def read_manifest_ids(path: Path) -> set:
    return {Path(line.strip()).stem for line in path.read_text().splitlines()
            if line.strip()}


def scene_memberships(scene_root: Path) -> Dict[str, List[str]]:
    manifest_root = scene_root / "manifests/original_m3fd_split"
    memberships: Dict[str, List[str]] = {}
    for scene in SCENES:
        ids = read_manifest_ids(manifest_root / f"{scene}_test_vis.txt")
        for image_id in ids:
            memberships.setdefault(image_id, []).append(scene)
    return memberships


def collect_metrics(args: argparse.Namespace) -> List[Dict[str, object]]:
    label_dir = args.dataset_root / "labels/vis_test"
    memberships = scene_memberships(args.scene_root)
    rows: List[Dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = label_path.stem
        ground_truth = read_yolo(label_path, 5)
        row: Dict[str, object] = {
            "image_id": image_id,
            "scene": "+".join(memberships.get(image_id, ["unassigned"])),
            "gt_count": len(ground_truth),
        }
        model_metrics = {}
        for model in MODELS:
            predictions = read_yolo(
                args.result_root / model / "labels" / f"{image_id}.txt", 6)
            metrics = evaluate_image(
                ground_truth, predictions, args.iou_threshold)
            model_metrics[model] = metrics
            for name, value in metrics.items():
                row[f"{model}_{name}"] = value
        rgca = model_metrics["rgca_full"]
        lcafnet = model_metrics["lcafnet_baseline"]
        row.update({
            "tp_gain": rgca["tp"] - lcafnet["tp"],
            "fp_reduction": lcafnet["fp"] - rgca["fp"],
            "fn_reduction": lcafnet["fn"] - rgca["fn"],
            "precision_gain": rgca["precision"] - lcafnet["precision"],
            "recall_gain": rgca["recall"] - lcafnet["recall"],
            "f1_gain": rgca["f1"] - lcafnet["f1"],
            "detection_accuracy_gain": (
                rgca["detection_accuracy"] - lcafnet["detection_accuracy"]),
            "matched_tp_mean_confidence_gain": (
                rgca["matched_tp_mean_confidence"]
                - lcafnet["matched_tp_mean_confidence"]),
        })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_ground_truth(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for item in labels:
        class_id, x, y, w, h = item.tolist()
        x1 = int(round((x - w / 2.0) * width))
        y1 = int(round((y - h / 2.0) * height))
        x2 = int(round((x + w / 2.0) * width))
        y2 = int(round((y + h / 2.0) * height))
        color = (38, 190, 38)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        caption = f"GT {CLASS_NAMES[int(class_id)]}"
        (text_width, text_height), _ = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(y1 - text_height - 6, 0)
        cv2.rectangle(result, (x1, label_top),
                      (x1 + text_width + 4, label_top + text_height + 6),
                      color, -1)
        cv2.putText(result, caption, (x1 + 2, label_top + text_height + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return result


def fit_panel(image: np.ndarray, width: int = 720, height: int = 540) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    image = fit_panel(image)
    strip = np.full((48, image.shape[1], 3), 247, dtype=np.uint8)
    cv2.putText(strip, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, (25, 25, 25), 2, cv2.LINE_AA)
    return np.vstack((strip, image))


def metric_line(label: str, row: Dict[str, object], prefix: str) -> str:
    return (
        f"{label}: TP/FP/FN={int(row[prefix + '_tp'])}/"
        f"{int(row[prefix + '_fp'])}/{int(row[prefix + '_fn'])} | "
        f"P={float(row[prefix + '_precision']):.3f}, "
        f"R={float(row[prefix + '_recall']):.3f}, "
        f"F1={float(row[prefix + '_f1']):.3f}, "
        f"DetAcc={float(row[prefix + '_detection_accuracy']):.3f} | "
        f"TP-conf={float(row[prefix + '_matched_tp_mean_confidence']):.3f}, "
        f"TP-IoU={float(row[prefix + '_matched_tp_mean_iou']):.3f}")


def render_comparison(
    args: argparse.Namespace, row: Dict[str, object], output_path: Path
) -> None:
    image_id = str(row["image_id"])
    rgb_source = cv2.imread(
        str(args.dataset_root / "images/vis_test" / f"{image_id}.png"))
    ir_source = cv2.imread(
        str(args.dataset_root / "images/Ir_test" / f"{image_id}.png"))
    labels = read_yolo(
        args.dataset_root / "labels/vis_test" / f"{image_id}.txt", 5)
    if rgb_source is None or ir_source is None:
        raise FileNotFoundError(f"Missing source pair for {image_id}")
    images = []
    for model in MODELS:
        rgb = cv2.imread(str(args.result_root / model / f"{image_id}_rgb.png"))
        ir = cv2.imread(str(args.result_root / model / f"{image_id}_ir.png"))
        if rgb is None or ir is None:
            raise FileNotFoundError(f"Missing {model} result pair for {image_id}")
        images.append((rgb, ir))

    top = np.hstack((
        titled_panel(draw_ground_truth(rgb_source, labels), "Ground truth | RGB"),
        titled_panel(images[0][0], "Full RGCA | RGB"),
        titled_panel(images[1][0], "LCAFNet | RGB"),
    ))
    bottom = np.hstack((
        titled_panel(draw_ground_truth(ir_source, labels), "Ground truth | IR"),
        titled_panel(images[0][1], "Full RGCA | IR"),
        titled_panel(images[1][1], "LCAFNet | IR"),
    ))
    header = np.full((176, top.shape[1], 3), 255, dtype=np.uint8)
    lines = (
        (f"M3FD test ID {image_id} | scene={row['scene']} | "
         f"GT={int(row['gt_count'])} | conf>={args.confidence_threshold:.2f}, "
         f"class-aware IoU>={args.iou_threshold:.2f}", (20, 20, 20)),
        (metric_line("Full RGCA", row, "rgca_full"), (20, 125, 20)),
        (metric_line("LCAFNet", row, "lcafnet_baseline"), (30, 95, 190)),
        (f"Gain: TP={int(row['tp_gain']):+d}, FP reduction={int(row['fp_reduction']):+d}, "
         f"F1={float(row['f1_gain']):+.3f}, "
         f"DetAcc={float(row['detection_accuracy_gain']):+.3f}", (20, 20, 20)),
    )
    for index, (line, color) in enumerate(lines):
        cv2.putText(header, line, (18, 34 + index * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
    comparison = np.vstack((header, top, bottom))
    if not cv2.imwrite(str(output_path), comparison,
                       [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise IOError(f"Could not write {output_path}")


def write_gallery(path: Path, selected: Sequence[Dict[str, object]]) -> None:
    cards = []
    for row in selected:
        image_id = html.escape(str(row["image_id"]))
        cards.append(
            f'<figure><a href="comparisons/{image_id}.jpg">'
            f'<img loading="lazy" src="comparisons/{image_id}.jpg"></a>'
            f'<figcaption>ID {image_id} · {html.escape(str(row["scene"]))} · '
            f'F1 gain {float(row["f1_gain"]):+.3f}</figcaption></figure>')
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full RGCA better than LCAFNet: M3FD top 10</title><style>
body{font-family:Arial,sans-serif;margin:20px;background:#f3f4f6;color:#111827}
figure{background:white;padding:10px;margin:18px 0;border-radius:8px}
img{width:100%;height:auto;display:block}figcaption{text-align:center;font-weight:600;padding:8px}
</style></head><body><h1>M3FD test：完整RGCA优于LCAFNet的10组代表性结果</h1>
<p>相同推理协议；同类别一对一匹配；IoU≥0.50。点击图片查看完整分辨率。</p>"""
    path.write_text(document + "".join(cards) + "</body></html>")


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.scene_root = args.scene_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_rows = collect_metrics(args)
    by_id = {str(row["image_id"]): row for row in all_rows}
    selected = []
    for image_id in args.selected_ids:
        image_id = str(image_id).zfill(5)
        if image_id not in by_id:
            raise KeyError(f"Unknown M3FD test ID: {image_id}")
        row = by_id[image_id]
        if float(row["f1_gain"]) <= 0:
            raise ValueError(f"RGCA does not have positive F1 gain for {image_id}")
        selected.append(row)
        render_comparison(args, row, comparison_dir / f"{image_id}.jpg")

    write_csv(args.output_dir / "all_test_per_image_metrics.csv", all_rows)
    write_csv(args.output_dir / "selected_top10_metrics.csv", selected)
    write_gallery(args.output_dir / "gallery.html", selected)
    print(f"Evaluated {len(all_rows)} M3FD test images")
    print(f"Rendered {len(selected)} selected RGCA-better comparisons")
    for row in selected:
        print(
            f"{row['image_id']} {row['scene']}: "
            f"F1 {float(row['rgca_full_f1']):.3f}/"
            f"{float(row['lcafnet_baseline_f1']):.3f}, "
            f"DetAcc {float(row['rgca_full_detection_accuracy']):.3f}/"
            f"{float(row['lcafnet_baseline_detection_accuracy']):.3f}")


if __name__ == "__main__":
    main()
