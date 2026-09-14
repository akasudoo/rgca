#!/usr/bin/env python3
"""Build an auditable RGCA/LCAFNet M3FD test detection gallery.

The inference images stay lossless in their model-specific directories.  The
HTML page uses lazy loading to place paired RGB/IR outputs from both models in
one row without creating another large set of recompressed image files.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List


CLASS_NAMES = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
MODELS = ("rgca_full", "lcafnet_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("paper/m3fd_detection_results"),
        help="Directory containing rgca_full and lcafnet_baseline outputs.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> List[List[float]]:
    if not path.exists():
        return []
    rows: List[List[float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected 6 fields")
        values = [float(value) for value in fields]
        class_id = int(values[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"{path}:{line_number}: value outside [0, 1]")
        rows.append(values)
    return rows


def image_ids(model_dir: Path) -> List[str]:
    return sorted(path.name[:-8] for path in model_dir.glob("*_rgb.png"))


def verify_outputs(root: Path) -> List[str]:
    ids_by_model: Dict[str, List[str]] = {}
    for model in MODELS:
        model_dir = root / model
        ids = image_ids(model_dir)
        if not ids:
            raise FileNotFoundError(f"No RGB result images under {model_dir}")
        for image_id in ids:
            for modality in ("rgb", "ir"):
                path = model_dir / f"{image_id}_{modality}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
        ids_by_model[model] = ids
    if ids_by_model[MODELS[0]] != ids_by_model[MODELS[1]]:
        raise ValueError("RGCA and LCAFNet result image IDs do not match")
    return ids_by_model[MODELS[0]]


def collect_rows(root: Path, ids: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for image_id in ids:
        row: Dict[str, object] = {"image_id": image_id}
        for model in MODELS:
            predictions = read_predictions(root / model / "labels" / f"{image_id}.txt")
            row[f"{model}_detections"] = len(predictions)
            row[f"{model}_mean_confidence"] = (
                sum(item[5] for item in predictions) / len(predictions)
                if predictions else 0.0
            )
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"{model}_{class_name}"] = sum(
                    int(item[0]) == class_id for item in predictions
                )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gallery(path: Path, rows: List[Dict[str, object]]) -> None:
    cards = []
    for row in rows:
        image_id = str(row["image_id"])
        captions = (
            ("RGCA · RGB", f"rgca_full/{image_id}_rgb.png"),
            ("RGCA · IR", f"rgca_full/{image_id}_ir.png"),
            ("LCAFNet · RGB", f"lcafnet_baseline/{image_id}_rgb.png"),
            ("LCAFNet · IR", f"lcafnet_baseline/{image_id}_ir.png"),
        )
        panels = "\n".join(
            f'<figure><a href="{src}"><img loading="lazy" src="{src}" '
            f'alt="M3FD {html.escape(image_id)} {caption}"></a>'
            f'<figcaption>{caption}</figcaption></figure>'
            for caption, src in captions
        )
        cards.append(
            f'<section class="sample" data-id="{html.escape(image_id)}">'
            f'<h2>ID {html.escape(image_id)} · detections RGCA/LCAFNet: '
            f'{row["rgca_full_detections"]}/{row["lcafnet_baseline_detections"]}</h2>'
            f'<div class="panels">{panels}</div></section>'
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M3FD test detection results: RGCA vs LCAFNet</title>
<style>
body{{font-family:Arial,sans-serif;margin:18px;background:#f3f4f6;color:#111827}}
.toolbar{{position:sticky;top:0;background:#f3f4f6;padding:10px 0;z-index:2}}
input{{font-size:16px;padding:8px;width:240px}}
.sample{{background:white;margin:16px 0;padding:12px;border-radius:8px}}
h1{{margin-bottom:6px}} h2{{font-size:16px;margin:0 0 10px}}
.note{{color:#4b5563}} .panels{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
figure{{margin:0}} img{{display:block;width:100%;height:auto}}
figcaption{{text-align:center;font-weight:600;margin-top:4px}}
@media(max-width:1000px){{.panels{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<h1>M3FD test：完整RGCA与LCAFNet检测结果</h1>
<p class="note">420对配准输入；640×640推理；confidence ≥ 0.25；NMS IoU = 0.50。点击图片查看原始分辨率。</p>
<div class="toolbar"><label>按图像ID筛选： <input id="filter" placeholder="例如 03979"></label></div>
{''.join(cards)}
<script>
const input=document.getElementById('filter');
input.addEventListener('input',()=>{{const q=input.value.trim();document.querySelectorAll('.sample').forEach(x=>x.hidden=q&&!x.dataset.id.includes(q));}});
</script></body></html>"""
    path.write_text(document)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    ids = verify_outputs(root)
    rows = collect_rows(root, ids)
    write_csv(root / "detection_counts.csv", rows)
    write_gallery(root / "gallery.html", rows)
    print(f"Verified and indexed {len(ids)} paired M3FD test results under {root}")


if __name__ == "__main__":
    main()
