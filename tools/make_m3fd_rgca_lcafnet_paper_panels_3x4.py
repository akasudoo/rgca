#!/usr/bin/env python3
"""Create one publication-ready 3x4 M3FD panel per selected image pair.

Rows: Ground Truth, LCAFNet, Full RGCA.
Columns: RGB result, RGB detail, IR result, IR detail.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_m3fd_rgca_lcafnet_paper_panels import (  # noqa: E402
    CLASS_NAMES,
    TARGET_COLOR,
    crop_image,
    dashed_rectangle,
    fit_panel,
    load_image,
    modality_method_image,
    pil_font,
    rotate_row_label,
    save_figure,
    target_crop_bounds,
    text_center,
)
from tools.select_rgca_better_detection_pairs import (  # noqa: E402
    read_yolo,
    xywh_to_xyxy,
)
from tools.select_rgca_possible_unlabeled_detections import box_iou  # noqa: E402


TARGETS: Dict[str, Dict[str, object]] = {
    "00484": {
        "prediction_index": 2,
        "status": "RGCA-only car candidate; missing annotation is unverified",
        "paper_status": "audit_candidate",
    },
    "00635": {
        "prediction_index": 1,
        "status": "RGCA-only occluded-car candidate; re-annotation required",
        "paper_status": "audit_candidate",
    },
    "03065": {
        "prediction_index": 10,
        "status": "RGCA-only prediction; visual audit suggests a false positive",
        "paper_status": "likely_fp",
    },
    "02043": {
        "prediction_index": 0,
        "status": "RGCA-only distant-car candidate; re-annotation required",
        "paper_status": "audit_candidate",
    },
    "03694": {
        "prediction_index": 6,
        "status": "RGCA-only overhead-lamp candidate; re-annotation required",
        "paper_status": "audit_candidate",
    },
    "01277": {
        "prediction_index": 2,
        "status": "RGCA-only lamp candidate; re-annotation required",
        "paper_status": "audit_candidate",
    },
    "00219": {
        "prediction_index": 0,
        "status": "RGCA-only occluded-car candidate; re-annotation required",
        "paper_status": "audit_candidate",
    },
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
        "--output-dir", type=Path,
        default=Path(
            "paper/m3fd_detection_results/"
            "paper_visualizations_rgca_vs_lcafnet_selected7_3x4"))
    return parser.parse_args()


def load_case(args: argparse.Namespace, image_id: str) -> Dict[str, object]:
    rgb = load_image(args.dataset_root / "images/vis_test" / f"{image_id}.png")
    ir = load_image(args.dataset_root / "images/Ir_test" / f"{image_id}.png")
    gt = read_yolo(args.dataset_root / "labels/vis_test" / f"{image_id}.txt", 5)
    rgca = read_yolo(args.result_root / "rgca_full/labels" / f"{image_id}.txt", 6)
    lcafnet = read_yolo(
        args.result_root / "lcafnet_baseline/labels" / f"{image_id}.txt", 6)
    candidate_index = int(TARGETS[image_id]["prediction_index"])
    if candidate_index >= len(rgca):
        raise IndexError(f"Missing RGCA prediction {candidate_index} for {image_id}")
    candidate = rgca[candidate_index]
    return {
        "image_id": image_id,
        "rgb": rgb,
        "ir": ir,
        "gt": gt,
        "rgca": rgca,
        "lcafnet": lcafnet,
        "candidate_index": candidate_index,
        "candidate": candidate,
        "detail_bounds": target_crop_bounds(
            candidate, rgb.shape, multiplier=4.5, min_width=190),
    }


def make_figure(case: Dict[str, object], output_dir: Path) -> None:
    cell_w, cell_h, gap = 780, 585, 16
    left, top_header, title_h, bottom = 190, 105, 78, 70
    width = left + 4 * cell_w + 3 * gap + 30
    height = top_header + title_h + 3 * cell_h + 2 * gap + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    image_id = str(case["image_id"])
    candidate = case["candidate"]
    class_name = CLASS_NAMES[int(candidate[0])]
    confidence = float(candidate[5])
    status = str(TARGETS[image_id]["status"])

    draw.text(
        (left, 16),
        f"M3FD {image_id} | RGCA-only {class_name} {confidence:.3f}",
        font=pil_font(36, True), fill=(10, 10, 10))
    status_color = (
        (180, 35, 35)
        if TARGETS[image_id]["paper_status"] == "likely_fp"
        else (55, 55, 55))
    draw.text((left, 61), status, font=pil_font(23), fill=status_color)

    headers = ("RGB Result", "RGB Detail", "IR Result", "IR Detail")
    for col, header in enumerate(headers):
        x = left + col * (cell_w + gap)
        text_center(
            draw, (x, top_header, x + cell_w, top_header + title_h),
            header, pil_font(28, True))

    rows = (
        ("Ground Truth", "gt"),
        ("LCAFNet", "lcafnet"),
        ("Full RGCA", "rgca"),
    )
    for row, (row_name, method) in enumerate(rows):
        y = top_header + title_h + row * (cell_h + gap)
        rotate_row_label(
            canvas, row_name, (35, y, 145, y + cell_h), pil_font(31, True))
        panels = []
        for modality in ("rgb", "ir"):
            full = modality_method_image(case, modality, method, show_labels=True)
            dashed_rectangle(full, case["detail_bounds"])
            detail_source = modality_method_image(
                case, modality, method, show_labels=False)
            detail = crop_image(detail_source, case["detail_bounds"])
            panels.extend((full, detail))

        for col, panel_image in enumerate(panels):
            panel = fit_panel(panel_image, cell_w, cell_h)
            panel_rgb = Image.fromarray(
                cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)).convert("RGBA")
            x = left + col * (cell_w + gap)
            canvas.alpha_composite(panel_rgb, (x, y))
            ImageDraw.Draw(canvas).rectangle(
                (x, y, x + cell_w - 1, y + cell_h - 1),
                outline=(82, 82, 82), width=2)

    draw.text(
        (left, height - 49),
        "Red: existing GT   |   Magenta dashed: detail ROI   |   Yellow: selected RGCA-only prediction",
        font=pil_font(21), fill=(45, 45, 45))
    save_figure(canvas, output_dir / f"M3FD_{image_id}_GT_LCAFNet_RGCA_3x4")


def same_class_iou(candidate: np.ndarray, boxes: np.ndarray) -> float:
    if not len(boxes):
        return 0.0
    candidate_xyxy = xywh_to_xyxy(np.asarray([candidate]))[0]
    boxes_xyxy = xywh_to_xyxy(boxes)
    class_id = int(candidate[0])
    return max(
        (box_iou(candidate_xyxy, box)
         for source, box in zip(boxes, boxes_xyxy)
         if int(source[0]) == class_id),
        default=0.0)


def table_row(case: Dict[str, object]) -> str:
    candidate = case["candidate"]
    image_id = str(case["image_id"])
    return (
        f"| {image_id} | p{case['candidate_index']} | "
        f"{CLASS_NAMES[int(candidate[0])]} | {float(candidate[5]):.3f} | "
        f"{same_class_iou(candidate, case['gt']):.3f} | "
        f"{same_class_iou(candidate, case['lcafnet']):.3f} | "
        f"{TARGETS[image_id]['paper_status']} |")


def write_gallery(output_dir: Path, paths: Iterable[Path]) -> None:
    cards = []
    for path in paths:
        name = html.escape(path.name)
        cards.append(
            f'<figure><a href="{name}"><img loading="lazy" src="{name}"></a>'
            f'<figcaption>{name}</figcaption></figure>')
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M3FD 3x4 RGCA vs LCAFNet</title><style>
body{font-family:Arial,sans-serif;margin:22px;background:#eef0f3;color:#111}
figure{background:#fff;padding:10px;margin:18px 0;border-radius:8px}
img{display:block;width:100%;height:auto}figcaption{text-align:center;padding:8px;font-weight:600}
</style></head><body><h1>M3FD：3行×4列可视化</h1>"""
    (output_dir / "gallery.html").write_text(
        page + "".join(cards) + "</body></html>")


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {image_id: load_case(args, image_id) for image_id in TARGETS}
    for case in cases.values():
        make_figure(case, args.output_dir)

    pngs = sorted(args.output_dir.glob("*.png"))
    write_gallery(args.output_dir, pngs)
    table = "\n".join(table_row(case) for case in cases.values())
    readme = f"""# M3FD 3行x4列论文可视化

## 固定布局

| | Column 1 | Column 2 | Column 3 | Column 4 |
|---|---|---|---|---|
| Row 1 | GT RGB result | GT RGB detail | GT IR result | GT IR detail |
| Row 2 | LCAFNet RGB result | LCAFNet RGB detail | LCAFNet IR result | LCAFNet IR detail |
| Row 3 | Full RGCA RGB result | Full RGCA RGB detail | Full RGCA IR result | Full RGCA IR detail |

- 每个编号单独一张图，共7张PNG和7张PDF。
- PNG与PDF均按300 DPI输出。
- 推理结果统一来自 `img=640, conf=0.25, NMS IoU=0.50`。
- 没有使用生成式图像处理，只绘制原始数据、GT与保存的预测框。

## 候选核查

| Image | RGCA row | Class | Confidence | same-class IoU to GT | same-class IoU to LCAFNet | Status |
|---|---:|---|---:|---:|---:|---|
{table}

`audit_candidate` 表示现有GT未确认，需要人工复标；`likely_fp` 表示视觉上更像
误检。`03065` 不建议作为论文中的性能提升证据。

## 复现

```bash
cd /home/dell/lcp/LCAFNet-main
/home/dell/anaconda3/envs/MOD/bin/python \\
  tools/make_m3fd_rgca_lcafnet_paper_panels_3x4.py
```
"""
    (args.output_dir / "README_CN.md").write_text(readme)
    print(
        f"Saved {len(pngs)} PNG and {len(list(args.output_dir.glob('*.pdf')))} "
        f"PDF 3x4 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
