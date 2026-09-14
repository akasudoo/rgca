#!/usr/bin/env python3
"""Build compact RGB/IR review sheets for possible M3FD missing labels."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_rgca_better_detection_pairs import read_yolo  # noqa: E402
from tools.select_rgca_possible_unlabeled_detections import (  # noqa: E402
    box_pixels,
    crop_bounds,
    draw_labeled_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--exclude-csv", type=Path,
        help="Skip candidates already present in this CSV (image ID + prediction index).")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--rgca-name", required=True)
    parser.add_argument("--lcafnet-name", required=True)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-sheet", type=int, default=4)
    return parser.parse_args()


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def panel(image: np.ndarray, title: str, width: int = 350, height: int = 245) -> np.ndarray:
    body = fit(image, width, height)
    header = np.full((36, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        header, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (25, 25, 25), 1, cv2.LINE_AA)
    return np.vstack((header, body))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.exclude_csv:
        with args.exclude_csv.open(newline="") as handle:
            excluded = {
                (row["image_id"], row["rgca_prediction_index"])
                for row in csv.DictReader(handle)
            }
        rows = [
            row for row in rows
            if (row["image_id"], row["rgca_prediction_index"]) not in excluded
        ]
    rows.sort(key=lambda row: float(row["rgca_confidence"]), reverse=True)

    rendered = []
    for rank, row in enumerate(rows, 1):
        image_id = row["image_id"]
        prediction_index = int(row["rgca_prediction_index"])
        predictions = read_yolo(
            args.result_root / args.rgca_name / "labels" / f"{image_id}.txt", 6)
        candidate = predictions[prediction_index]
        rgb = load_image(args.dataset_root / "images/vis_test" / f"{image_id}.png")
        ir = load_image(args.dataset_root / "images/Ir_test" / f"{image_id}.png")
        rgca = load_image(
            args.result_root / args.rgca_name / f"{image_id}_rgb.png")
        lcafnet = load_image(
            args.result_root / args.lcafnet_name / f"{image_id}_rgb.png")

        full = rgb.copy()
        label = (
            f"#{rank} {image_id} p{prediction_index} {row['class_name']} "
            f"c={float(row['rgca_confidence']):.3f}")
        draw_labeled_box(full, candidate, label, (0, 210, 255), 5)
        bounds = crop_bounds(candidate, rgb.shape)
        x1, y1, x2, y2 = bounds
        for target, text, color in (
            (rgb, "candidate", (0, 210, 255)),
            (ir, "candidate", (0, 210, 255)),
            (rgca, "RGCA", (0, 210, 255)),
            (lcafnet, "same region", (210, 0, 210)),
        ):
            draw_labeled_box(target, candidate, text, color, 4)
        rendered.append(np.hstack((
            panel(full, label),
            panel(rgb[y1:y2, x1:x2], "Raw RGB crop"),
            panel(ir[y1:y2, x1:x2], "Raw IR crop"),
            panel(rgca[y1:y2, x1:x2], "RGCA result crop"),
            panel(lcafnet[y1:y2, x1:x2], "LCAFNet result crop"),
        )))

    for index in range(math.ceil(len(rendered) / args.rows_per_sheet)):
        chunk = rendered[index * args.rows_per_sheet:(index + 1) * args.rows_per_sheet]
        sheet = np.vstack(chunk)
        output = args.output_dir / f"sheet_{index + 1:02d}.jpg"
        cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"Rendered {len(rendered)} candidates into {math.ceil(len(rendered) / args.rows_per_sheet)} sheets")


if __name__ == "__main__":
    main()
