#!/usr/bin/env python3
"""Select plausible unlabeled M3FD objects detected only by full RGCA.

These are qualitative annotation-audit candidates, not ground-truth-verified
true positives.  The script conservatively requires a confident RGCA box to
have little overlap with every existing GT and every LCAFNet prediction, then
renders manually reviewed examples in paired RGB/IR views.
"""

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

from tools.select_rgca_better_detection_pairs import (  # noqa: E402
    CLASS_NAMES,
    read_yolo,
    scene_memberships,
    xywh_to_xyxy,
)


# image ID -> RGCA saved-prediction row index and manual review note.
DEFAULT_SELECTION = {
    "00635": (1, "night; partially occluded vehicle beside a truck"),
    "00484": (2, "night; very small distant vehicle among glare and traffic"),
    "01587": (1, "overcast; distant small vehicle"),
    "02043": (0, "daytime; distant vehicle partly hidden by roadside trees"),
    "02485": (0, "daytime; small distant vehicle in dense traffic"),
    "02503": (1, "daytime; parked vehicle heavily occluded by a hedge"),
    "02582": (2, "daytime; very small vehicle among trucks and vegetation"),
    "00227": (6, "overcast/rain; roadside lamp or signal housing"),
    "01277": (2, "night/rain; small roadside lamp or signal"),
    "03694": (6, "daytime; small overhead traffic lamp or signal"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("paper/m3fd_detection_results"))
    parser.add_argument("--rgca-name", default="rgca_full")
    parser.add_argument("--lcafnet-name", default="lcafnet_baseline")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD"))
    parser.add_argument(
        "--scene-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024"))
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--max-gt-iou", type=float, default=0.10)
    parser.add_argument("--max-lcafnet-iou", type=float, default=0.20)
    parser.add_argument("--max-gt-coverage", type=float, default=0.25)
    parser.add_argument("--max-lcafnet-coverage", type=float, default=0.25)
    parser.add_argument(
        "--max-other-rgca-coverage", type=float, default=1.01,
        help=("Reject a candidate substantially covered by another RGCA box; "
              "the permissive default preserves the original top-10 audit."))
    parser.add_argument("--min-area", type=float, default=0.00005)
    parser.add_argument("--max-area", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "paper/m3fd_detection_results/rgca_possible_unlabeled_top10"))
    parser.add_argument(
        "--candidates-only", action="store_true",
        help="Write the automatic candidate table without rendering DEFAULT_SELECTION.")
    return parser.parse_args()


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[1:3], second[1:3])
    right_bottom = np.minimum(first[3:5], second[3:5])
    intersection = float(np.maximum(right_bottom - left_top, 0.0).prod())
    first_area = float(np.maximum(first[3:5] - first[1:3], 0.0).prod())
    second_area = float(np.maximum(second[3:5] - second[1:3], 0.0).prod())
    return intersection / max(first_area + second_area - intersection, 1e-12)


def best_overlap(box: np.ndarray, others: np.ndarray) -> Tuple[int, float]:
    if not len(others):
        return -1, 0.0
    overlaps = [box_iou(box, other) for other in others]
    index = int(np.argmax(overlaps))
    return index, float(overlaps[index])


def best_candidate_coverage(box: np.ndarray, others: np.ndarray) -> float:
    """Return the largest fraction of *box* covered by another box.

    IoU alone is insufficient here: a tiny duplicate prediction inside a much
    larger GT/LCAFNet box can have low IoU even though that object was already
    annotated or detected.  Candidate coverage rejects that nested-box case.
    """
    if not len(others):
        return 0.0
    left_top = np.maximum(box[1:3], others[:, 1:3])
    right_bottom = np.minimum(box[3:5], others[:, 3:5])
    intersection = np.maximum(right_bottom - left_top, 0.0).prod(axis=1)
    candidate_area = float(np.maximum(box[3:5] - box[1:3], 0.0).prod())
    return float(intersection.max() / max(candidate_area, 1e-12))


def collect_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    memberships = scene_memberships(args.scene_root)
    rows: List[Dict[str, object]] = []
    label_root = args.dataset_root / "labels/vis_test"
    for label_path in sorted(label_root.glob("*.txt")):
        image_id = label_path.stem
        ground_truth = read_yolo(label_path, 5)
        rgca = read_yolo(
            args.result_root / args.rgca_name / "labels" / f"{image_id}.txt", 6)
        lcafnet = read_yolo(
            args.result_root / args.lcafnet_name / "labels" / f"{image_id}.txt", 6)
        gt_xyxy = xywh_to_xyxy(ground_truth)
        rgca_xyxy = xywh_to_xyxy(rgca)
        lcafnet_xyxy = xywh_to_xyxy(lcafnet)
        for prediction_index, (prediction, prediction_xyxy) in enumerate(
                zip(rgca, rgca_xyxy)):
            confidence = float(prediction[5])
            area = float(prediction[3] * prediction[4])
            gt_index, gt_iou = best_overlap(prediction_xyxy, gt_xyxy)
            lcafnet_index, lcafnet_iou = best_overlap(
                prediction_xyxy, lcafnet_xyxy)
            gt_coverage = best_candidate_coverage(prediction_xyxy, gt_xyxy)
            lcafnet_coverage = best_candidate_coverage(
                prediction_xyxy, lcafnet_xyxy)
            other_rgca_xyxy = np.delete(rgca_xyxy, prediction_index, axis=0)
            other_rgca_coverage = best_candidate_coverage(
                prediction_xyxy, other_rgca_xyxy)
            if not (
                confidence >= args.min_confidence
                and gt_iou < args.max_gt_iou
                and lcafnet_iou < args.max_lcafnet_iou
                and gt_coverage < args.max_gt_coverage
                and lcafnet_coverage < args.max_lcafnet_coverage
                and other_rgca_coverage < args.max_other_rgca_coverage
                and args.min_area <= area <= args.max_area
            ):
                continue
            rows.append({
                "image_id": image_id,
                "scene": "+".join(memberships.get(image_id, ["unassigned"])),
                "gt_count": len(ground_truth),
                "rgca_prediction_index": prediction_index,
                "class_id": int(prediction[0]),
                "class_name": CLASS_NAMES[int(prediction[0])],
                "rgca_confidence": confidence,
                "normalized_box_area": area,
                "nearest_gt_index": gt_index,
                "max_iou_to_any_gt": gt_iou,
                "max_candidate_coverage_by_any_gt": gt_coverage,
                "nearest_lcafnet_prediction_index": lcafnet_index,
                "max_iou_to_any_lcafnet_prediction": lcafnet_iou,
                "max_candidate_coverage_by_any_lcafnet_prediction": (
                    lcafnet_coverage),
                "max_candidate_coverage_by_other_rgca_prediction": (
                    other_rgca_coverage),
                "manual_status": "not_reviewed",
                "interpretation": (
                    "candidate only; unmatched RGCA predictions are false positives "
                    "under the current annotation"),
            })
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
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    top = max(y1 - text_height - 8, 0)
    cv2.rectangle(image, (x1, top),
                  (min(x1 + text_width + 6, image.shape[1] - 1),
                   top + text_height + 8), color, -1)
    cv2.putText(image, label, (x1 + 3, top + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_annotations(
    image: np.ndarray, ground_truth: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    result = image.copy()
    for gt in ground_truth:
        draw_labeled_box(
            result, gt, f"GT {CLASS_NAMES[int(gt[0])]}", (38, 190, 38), 2)
    draw_labeled_box(
        result, candidate,
        f"POSSIBLE UNLABELED {CLASS_NAMES[int(candidate[0])]}",
        (0, 210, 255), 5)
    return result


def crop_bounds(box: np.ndarray, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = box_pixels(box, shape)
    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    span_x = min(max((x2 - x1) * 5.0, 190.0), width)
    span_y = min(max((y2 - y1) * 5.0, 170.0), height)
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
    rgca_predictions = read_yolo(
        args.result_root / args.rgca_name / "labels" / f"{image_id}.txt", 6)
    candidate = rgca_predictions[int(row["rgca_prediction_index"])]

    annotated_sources = []
    rgca_images = []
    lcafnet_images = []
    for modality, source_folder in (("rgb", "vis_test"), ("ir", "Ir_test")):
        source = cv2.imread(str(
            args.dataset_root / "images" / source_folder / f"{image_id}.png"))
        rgca = cv2.imread(str(
            args.result_root / args.rgca_name / f"{image_id}_{modality}.png"))
        lcafnet = cv2.imread(str(
            args.result_root / args.lcafnet_name / f"{image_id}_{modality}.png"))
        if source is None or rgca is None or lcafnet is None:
            raise FileNotFoundError(f"Incomplete result pair for {image_id}")
        annotated_sources.append(draw_annotations(source, ground_truth, candidate))
        draw_labeled_box(
            rgca, candidate,
            f"RGCA {row['class_name']} {float(row['rgca_confidence']):.2f}",
            (0, 210, 255), 5)
        draw_labeled_box(
            lcafnet, candidate, "SAME REGION: NO LCAFNet MATCH",
            (210, 0, 210), 5)
        rgca_images.append(rgca)
        lcafnet_images.append(lcafnet)

    bounds = crop_bounds(candidate, annotated_sources[0].shape)
    full_rgb = np.hstack((
        titled_panel(annotated_sources[0], "Existing GT + candidate | RGB"),
        titled_panel(rgca_images[0], "Full RGCA | RGB"),
        titled_panel(lcafnet_images[0], "LCAFNet | RGB"),
    ))
    full_ir = np.hstack((
        titled_panel(annotated_sources[1], "Existing GT + candidate | IR"),
        titled_panel(rgca_images[1], "Full RGCA | IR"),
        titled_panel(lcafnet_images[1], "LCAFNet | IR"),
    ))
    crop_rgb = np.hstack((
        crop_panel(annotated_sources[0], bounds, "Candidate crop | Raw RGB"),
        crop_panel(rgca_images[0], bounds, "Candidate crop | RGCA RGB"),
        crop_panel(lcafnet_images[0], bounds, "Candidate crop | LCAFNet RGB"),
    ))
    crop_ir = np.hstack((
        crop_panel(annotated_sources[1], bounds, "Candidate crop | Raw IR"),
        crop_panel(rgca_images[1], bounds, "Candidate crop | RGCA IR"),
        crop_panel(lcafnet_images[1], bounds, "Candidate crop | LCAFNet IR"),
    ))

    header = np.full((188, full_rgb.shape[1], 3), 255, dtype=np.uint8)
    lines = (
        (f"M3FD test ID {image_id} | scene={row['scene']} | possible unlabeled "
         f"{row['class_name']} | manual plausibility review only", (20, 20, 20)),
        (f"RGCA confidence={float(row['rgca_confidence']):.3f}; maximum IoU "
         f"to any existing GT={float(row['max_iou_to_any_gt']):.3f}",
         (20, 125, 20)),
        (f"LCAFNet: maximum IoU="
         f"{float(row['max_iou_to_any_lcafnet_prediction']):.3f}; candidate "
         f"coverage={float(row['max_candidate_coverage_by_any_lcafnet_prediction']):.3f}",
         (30, 95, 190)),
        (f"Visual note: {row['manual_note']}", (20, 20, 20)),
        ("Caution: no GT means this is NOT a verified true positive; re-annotation is required.",
         (20, 20, 180)),
    )
    for index, (line, color) in enumerate(lines):
        cv2.putText(header, line, (18, 32 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
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
        prediction_index = int(row["rgca_prediction_index"])
        source = f"comparisons/{image_id}_pred{prediction_index}.jpg"
        cards.append(
            f'<figure><a href="{source}"><img loading="lazy" src="{source}"></a>'
            f'<figcaption>ID {image_id} · {html.escape(str(row["scene"]))} · '
            f'possible {html.escape(str(row["class_name"]))} · RGCA confidence '
            f'{float(row["rgca_confidence"]):.3f}</figcaption></figure>')
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Possible unlabeled objects detected only by RGCA</title><style>
body{font-family:Arial,sans-serif;margin:20px;background:#f3f4f6;color:#111827}
figure{background:white;padding:10px;margin:18px 0;border-radius:8px}
img{width:100%;height:auto;display:block}figcaption{text-align:center;font-weight:600;padding:8px}
</style></head><body><h1>M3FD test：RGCA独有检出的疑似漏标目标</h1>
<p>这些是人工筛选的标注审计候选，不是经GT验证的真阳性。点击查看全图与RGB/IR裁剪。</p>"""
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
    if args.candidates_only:
        if candidates:
            write_csv(args.output_dir / "all_conservative_candidates.csv", candidates)
        else:
            (args.output_dir / "all_conservative_candidates.csv").write_text("")
        print(
            f"Found {len(candidates)} conservative candidates in "
            f"{len(set(str(row['image_id']) for row in candidates))} images")
        return
    lookup = {
        (str(row["image_id"]), int(row["rgca_prediction_index"])): row
        for row in candidates
    }
    selected = []
    for image_id, (prediction_index, note) in DEFAULT_SELECTION.items():
        key = (image_id, prediction_index)
        if key not in lookup:
            raise ValueError(f"Selected candidate no longer passes filters: {key}")
        row = dict(lookup[key])
        row["manual_status"] = "plausible_unlabeled_needs_reannotation"
        row["manual_note"] = note
        row["interpretation"] = (
            "qualitative annotation-audit candidate; not a verified true positive")
        selected.append(row)
        render_selected(
            args, row,
            comparison_dir / f"{image_id}_pred{prediction_index}.jpg")

    write_csv(args.output_dir / "all_conservative_candidates.csv", candidates)
    write_csv(args.output_dir / "selected_10_possible_unlabeled.csv", selected)
    write_gallery(args.output_dir / "gallery.html", selected)
    print(
        f"Found {len(candidates)} conservative candidates in "
        f"{len(set(str(row['image_id']) for row in candidates))} images")
    print(f"Rendered {len(selected)} manually reviewed plausible candidates")
    for row in selected:
        print(
            f"{row['image_id']} pred={row['rgca_prediction_index']} "
            f"{row['class_name']}: conf={row['rgca_confidence']:.3f}, "
            f"GT IoU={row['max_iou_to_any_gt']:.3f}, "
            f"GT coverage={row['max_candidate_coverage_by_any_gt']:.3f}, "
            f"LCAF IoU={row['max_iou_to_any_lcafnet_prediction']:.3f}, "
            f"LCAF coverage="
            f"{row['max_candidate_coverage_by_any_lcafnet_prediction']:.3f}")


if __name__ == "__main__":
    main()
