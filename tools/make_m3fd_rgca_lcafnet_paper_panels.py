#!/usr/bin/env python3
"""Create publication-ready M3FD RGCA-vs-LCAFNet qualitative panels.

The figures are deterministic composites of source images, existing labels,
and saved model predictions.  No generative image processing is used.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_rgca_better_detection_pairs import (  # noqa: E402
    CLASS_NAMES,
    read_yolo,
    xywh_to_xyxy,
)
from tools.select_rgca_possible_unlabeled_detections import (  # noqa: E402
    box_iou,
    box_pixels,
)


TARGETS: Dict[str, Dict[str, object]] = {
    # prediction_index refers to conf=0.25 saved RGCA labels.
    "02576": {
        "prediction_index": 0,
        "status": "GT-confirmed Car; LCAFNet classifies the region as Truck",
        "paper_status": "confirmed",
    },
    "03065": {
        "prediction_index": 10,
        "status": "RGCA-only prediction; visual audit suggests a false positive",
        "paper_status": "likely_fp",
    },
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
    "02043": {
        "prediction_index": 0,
        "status": "RGCA-only distant-car candidate; re-annotation required",
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

PAIRINGS = (("02576", "03065"), ("00484", "00635"), ("02043", "01277"))

CLASS_COLORS = {
    0: (54, 128, 255),   # People, BGR
    1: (0, 151, 255),    # Car
    2: (181, 87, 46),    # Motorcycle
    3: (44, 180, 44),    # Bus
    4: (42, 42, 205),    # Lamp
    5: (80, 80, 160),    # Truck
}
GT_COLOR = (40, 40, 220)
TARGET_COLOR = (0, 220, 255)
ROI_COLOR = (210, 30, 180)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")


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
            "paper_visualizations_rgca_vs_lcafnet_selected7"))
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return color[2], color[1], color[0]


def draw_label(
    image: np.ndarray,
    box: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 3,
    scale: float = 0.50,
) -> None:
    x1, y1, x2, y2 = box_pixels(box, image.shape)
    height, width = image.shape[:2]
    x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not label:
        return
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(thickness - 1, 1))
    top = max(y1 - th - baseline - 5, 0)
    right = min(x1 + tw + 6, width - 1)
    cv2.rectangle(image, (x1, top), (right, min(top + th + baseline + 5, height - 1)), color, -1)
    cv2.putText(
        image, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255), max(thickness - 1, 1), cv2.LINE_AA)


def dashed_rectangle(
    image: np.ndarray,
    bounds: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = ROI_COLOR,
    thickness: int = 3,
    dash: int = 12,
) -> None:
    x1, y1, x2, y2 = bounds
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), color, thickness)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), color, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), color, thickness)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), color, thickness)


def target_crop_bounds(
    box: np.ndarray,
    shape: Tuple[int, int, int],
    multiplier: float,
    min_width: int,
    aspect: float = 4.0 / 3.0,
) -> Tuple[int, int, int, int]:
    image_h, image_w = shape[:2]
    x1, y1, x2, y2 = box_pixels(box, shape)
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    width = max((x2 - x1) * multiplier, min_width)
    height = max((y2 - y1) * multiplier, width / aspect)
    width = max(width, height * aspect)
    width = min(width, image_w)
    height = min(width / aspect, image_h)
    if height == image_h:
        width = min(height * aspect, image_w)
    left = int(round(max(min(centre_x - width / 2, image_w - width), 0)))
    top = int(round(max(min(centre_y - height / 2, image_h - height), 0)))
    return left, top, int(round(left + width)), int(round(top + height))


def expand_bounds(
    bounds: Tuple[int, int, int, int], shape: Tuple[int, int, int], factor: float = 1.12
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    height, width = shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    span_x, span_y = (x2 - x1) * factor, (y2 - y1) * factor
    left = int(max(cx - span_x / 2, 0))
    right = int(min(cx + span_x / 2, width - 1))
    top = int(max(cy - span_y / 2, 0))
    bottom = int(min(cy + span_y / 2, height - 1))
    return left, top, right, bottom


def render_annotations(
    source: np.ndarray,
    boxes: np.ndarray,
    mode: str,
    candidate_index: Optional[int] = None,
    show_labels: bool = True,
) -> np.ndarray:
    output = source.copy()
    for index, box in enumerate(boxes):
        class_id = int(box[0])
        if mode == "gt":
            label = ""
            color = GT_COLOR
            thickness = 3
        else:
            confidence = float(box[5])
            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}" if show_labels else ""
            color = CLASS_COLORS.get(class_id, (180, 180, 180))
            thickness = 3
        draw_label(output, box, label, color, thickness)
        if mode == "rgca" and index == candidate_index:
            draw_label(
                output, box,
                (f"RGCA-only {CLASS_NAMES[class_id]} {float(box[5]):.2f}"
                 if show_labels else ""),
                TARGET_COLOR, 5, 0.56)
    return output


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    x1, y1, x2, y2 = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2), text,
        font=font, fill=fill, spacing=3, align="center")


def rotate_row_label(
    canvas: Image.Image,
    text: str,
    box: Tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    x1, y1, x2, y2 = box
    layer = Image.new("RGBA", (y2 - y1, x2 - x1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((layer.width - (bbox[2] - bbox[0])) / 2,
         (layer.height - (bbox[3] - bbox[1])) / 2),
        text, font=font, fill=(10, 10, 10, 255))
    rotated = layer.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (x1, y1))


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
        "context_bounds": target_crop_bounds(candidate, rgb.shape, 7.0, 260),
        "detail_bounds": target_crop_bounds(candidate, rgb.shape, 3.4, 145),
    }


def modality_method_image(
    case: Dict[str, object], modality: str, method: str, show_labels: bool = True,
) -> np.ndarray:
    source = case[modality]
    if method == "gt":
        return render_annotations(source, case["gt"], "gt", show_labels=False)
    if method == "lcafnet":
        return render_annotations(
            source, case["lcafnet"], "lcafnet", show_labels=show_labels)
    return render_annotations(
        source, case["rgca"], "rgca", int(case["candidate_index"]),
        show_labels=show_labels)


def crop_image(image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2]


def save_figure(canvas: Image.Image, stem: Path) -> None:
    rgb = canvas.convert("RGB")
    rgb.save(stem.with_suffix(".png"), dpi=(300, 300), compress_level=3)
    rgb.save(stem.with_suffix(".pdf"), resolution=300.0)


def make_single_case_figure(
    args: argparse.Namespace, case: Dict[str, object], output_dir: Path
) -> None:
    cell_w, cell_h, gap = 640, 480, 14
    left, top_header, title_h, bottom = 190, 105, 78, 72
    width = left + 6 * cell_w + 5 * gap + 30
    height = top_header + title_h + 3 * cell_h + 2 * gap + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    image_id = str(case["image_id"])
    candidate = case["candidate"]
    class_name = CLASS_NAMES[int(candidate[0])]
    confidence = float(candidate[5])
    status = str(TARGETS[image_id]["status"])
    draw.text(
        (left, 18),
        f"M3FD {image_id} | RGCA-only {class_name} {confidence:.3f}",
        font=pil_font(34, True), fill=(10, 10, 10))
    status_color = (180, 35, 35) if TARGETS[image_id]["paper_status"] == "likely_fp" else (55, 55, 55)
    draw.text((left, 60), status, font=pil_font(22), fill=status_color)
    headers = (
        "VIS\nFull image", "IR\nFull image",
        "VIS\nContext crop", "IR\nContext crop",
        "VIS\nDetail crop", "IR\nDetail crop",
    )
    for col, header in enumerate(headers):
        x1 = left + col * (cell_w + gap)
        text_center(draw, (x1, top_header, x1 + cell_w, top_header + title_h),
                    header, pil_font(25, True))

    row_names = (("Ground Truth", "gt"), ("LCAFNet", "lcafnet"), ("Full RGCA", "rgca"))
    for row, (row_name, method) in enumerate(row_names):
        y = top_header + title_h + row * (cell_h + gap)
        rotate_row_label(canvas, row_name, (35, y, 145, y + cell_h), pil_font(29, True))
        panels: List[np.ndarray] = []
        for modality in ("rgb", "ir"):
            annotated = modality_method_image(case, modality, method, show_labels=True)
            full = annotated.copy()
            dashed_rectangle(full, case["context_bounds"])
            panels.append(full)
        for bounds_name in ("context_bounds", "detail_bounds"):
            for modality in ("rgb", "ir"):
                annotated = modality_method_image(case, modality, method, show_labels=False)
                panels.append(crop_image(annotated, case[bounds_name]))
        for col, panel_image in enumerate(panels):
            panel = fit_panel(panel_image, cell_w, cell_h)
            panel_rgb = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)).convert("RGBA")
            x = left + col * (cell_w + gap)
            canvas.alpha_composite(panel_rgb, (x, y))
            ImageDraw.Draw(canvas).rectangle((x, y, x + cell_w - 1, y + cell_h - 1),
                                             outline=(85, 85, 85), width=2)
    legend_y = height - 52
    draw.text(
        (left, legend_y),
        "Red: existing GT   |   Magenta dashed: zoom ROI   |   Yellow: selected RGCA-only prediction",
        font=pil_font(21), fill=(45, 45, 45))
    save_figure(canvas, output_dir / f"M3FD_{image_id}_GT_LCAFNet_RGCA_3x6")


def detail_stack(case: Dict[str, object], method: str, cell_w: int, cell_h: int) -> np.ndarray:
    half_h = (cell_h - 28) // 2
    canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
    for index, modality in enumerate(("rgb", "ir")):
        annotated = modality_method_image(case, modality, method, show_labels=False)
        crop = crop_image(annotated, case["detail_bounds"])
        fitted = fit_panel(crop, cell_w, half_h)
        y = index * (half_h + 28)
        canvas[y:y + half_h] = fitted
        cv2.putText(
            canvas, "VIS detail" if modality == "rgb" else "IR detail",
            (10, y + half_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def make_pair_figure(
    args: argparse.Namespace,
    first: Dict[str, object],
    second: Dict[str, object],
    output_dir: Path,
) -> None:
    cell_w, cell_h, gap = 610, 458, 14
    left, header_h, title_h, bottom = 185, 78, 78, 65
    width = left + 6 * cell_w + 5 * gap + 30
    height = header_h + title_h + 3 * cell_h + 2 * gap + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    first_id, second_id = str(first["image_id"]), str(second["image_id"])
    draw.text(
        (left, 18), f"M3FD qualitative comparison: {first_id} and {second_id}",
        font=pil_font(31, True), fill=(10, 10, 10))
    headers = (
        f"{first_id}\nVIS", f"{first_id}\nIR",
        f"{second_id}\nVIS", f"{second_id}\nIR",
        f"{first_id}\nDetail (VIS / IR)", f"{second_id}\nDetail (VIS / IR)",
    )
    for col, header in enumerate(headers):
        x = left + col * (cell_w + gap)
        text_center(draw, (x, header_h, x + cell_w, header_h + title_h),
                    header, pil_font(23, True))

    row_names = (("Ground Truth", "gt"), ("LCAFNet", "lcafnet"), ("Full RGCA", "rgca"))
    for row, (row_name, method) in enumerate(row_names):
        y = header_h + title_h + row * (cell_h + gap)
        rotate_row_label(canvas, row_name, (35, y, 140, y + cell_h), pil_font(28, True))
        panels = []
        for case in (first, second):
            for modality in ("rgb", "ir"):
                annotated = modality_method_image(case, modality, method, show_labels=True)
                dashed_rectangle(annotated, case["context_bounds"])
                panels.append(annotated)
        panels.extend((
            detail_stack(first, method, cell_w, cell_h),
            detail_stack(second, method, cell_w, cell_h),
        ))
        for col, panel_image in enumerate(panels):
            panel = fit_panel(panel_image, cell_w, cell_h)
            panel_rgb = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)).convert("RGBA")
            x = left + col * (cell_w + gap)
            canvas.alpha_composite(panel_rgb, (x, y))
            ImageDraw.Draw(canvas).rectangle((x, y, x + cell_w - 1, y + cell_h - 1),
                                             outline=(85, 85, 85), width=2)
    draw.text(
        (left, height - 47),
        "Magenta dashed: zoom ROI   |   Yellow: selected RGCA-only prediction   |   Details show paired VIS/IR",
        font=pil_font(20), fill=(45, 45, 45))
    save_figure(
        canvas,
        output_dir / f"M3FD_{first_id}_{second_id}_paired_GT_LCAFNet_RGCA_3x6")


def write_gallery(output_dir: Path, pngs: Iterable[Path]) -> None:
    cards = []
    for path in pngs:
        name = html.escape(path.name)
        cards.append(
            f'<figure><a href="{name}"><img loading="lazy" src="{name}"></a>'
            f'<figcaption>{name}</figcaption></figure>')
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M3FD RGCA vs LCAFNet paper panels</title><style>
body{font-family:Arial,sans-serif;margin:22px;background:#eef0f3;color:#111}
figure{background:#fff;padding:10px;margin:18px 0;border-radius:8px}
img{display:block;width:100%;height:auto}figcaption{text-align:center;padding:8px;font-weight:600}
</style></head><body><h1>M3FD：Ground Truth / LCAFNet / Full RGCA</h1>"""
    (output_dir / "gallery.html").write_text(
        document + "".join(cards) + "</body></html>")


def validation_summary(case: Dict[str, object]) -> str:
    candidate = case["candidate"]
    candidate_index = int(case["candidate_index"])
    gt = case["gt"]
    lcafnet = case["lcafnet"]
    candidate_xyxy = xywh_to_xyxy(np.asarray([candidate]))[0]
    gt_xyxy = xywh_to_xyxy(gt)
    lcafnet_xyxy = xywh_to_xyxy(lcafnet)
    class_id = int(candidate[0])
    best_gt = max(
        (box_iou(candidate_xyxy, box) for source, box in zip(gt, gt_xyxy)
         if int(source[0]) == class_id),
        default=0.0)
    best_lcafnet = max(
        (box_iou(candidate_xyxy, box)
         for source, box in zip(lcafnet, lcafnet_xyxy)
         if int(source[0]) == class_id),
        default=0.0)
    return (
        f"| {case['image_id']} | p{candidate_index} | "
        f"{CLASS_NAMES[int(candidate[0])]} | {float(candidate[5]):.3f} | "
        f"{best_gt:.3f} | {best_lcafnet:.3f} | "
        f"{TARGETS[str(case['image_id'])]['paper_status']} |")


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {image_id: load_case(args, image_id) for image_id in TARGETS}

    for case in cases.values():
        make_single_case_figure(args, case, args.output_dir)
    for first_id, second_id in PAIRINGS:
        make_pair_figure(args, cases[first_id], cases[second_id], args.output_dir)

    pngs = sorted(args.output_dir.glob("*.png"))
    write_gallery(args.output_dir, pngs)
    table = "\n".join(validation_summary(case) for case in cases.values())
    readme = f"""# M3FD 论文可视化：GT / LCAFNet / Full RGCA

## 输出

- 7 张单样本 `3x6` 图：完整 VIS/IR、上下文放大 VIS/IR、细节放大 VIS/IR。
- 3 张双样本紧凑 `3x6` 图：前四列为两组完整 VIS/IR，第5、6列分别为
  两组目标的 VIS/IR 组合细节。
- 每张图同时输出 PNG（300 DPI）和 PDF。
- 第二行固定为 LCAFNet，第三行固定为完整 RGCA。

所有图均由原始图像、M3FD现有标签和保存的模型预测确定性绘制，没有使用
生成式图像处理。推理结果来自统一的 `img=640, conf=0.25, NMS IoU=0.50`。

## 候选核查

| Image | RGCA row | Class | RGCA conf | same-class IoU to GT | same-class IoU to LCAFNet | Status |
|---|---:|---|---:|---:|---:|---|
{table}

状态含义：

- `confirmed`：已有GT确认。`02576` 中RGCA正确预测为Car，而LCAFNet在同一区域
  预测成Truck；因此属于Car类别的漏检/误分类，不是“完全没有候选框”。
- `audit_candidate`：RGCA独有且现有GT未标，必须人工复标后才能称为真阳性。
- `likely_fp`：视觉检查更像误检，不应在论文中作为性能提升证据。`03065`
  属于此类，但依照指定编号仍生成了图。

## 图例

- 红色框：现有GT。
- 洋红虚线：细节放大的ROI，不代表检测框。
- 黄色粗框：选定的RGCA独有预测。
- 其他颜色框：相应类别的模型预测。

## 复现命令

```bash
cd /home/dell/lcp/LCAFNet-main
/home/dell/anaconda3/envs/MOD/bin/python \\
  tools/make_m3fd_rgca_lcafnet_paper_panels.py
```
"""
    (args.output_dir / "README_CN.md").write_text(readme)
    print(
        f"Saved {len(pngs)} PNG and {len(list(args.output_dir.glob('*.pdf')))} PDF figures "
        f"to {args.output_dir}")


if __name__ == "__main__":
    main()
