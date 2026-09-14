#!/usr/bin/env python3
"""Render 40 manually screened M3FD missing-annotation audit candidates.

The selected regions are detected by RGCA and unmatched by LCAFNet and the
existing annotation.  They are not verified true positives until re-annotated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_rgca_possible_unlabeled_detections import (  # noqa: E402
    collect_candidates,
    render_selected,
    write_csv,
    write_gallery,
)


# image ID, saved RGCA prediction-row index, review tier, visual note
# Exactly one candidate is selected per image to provide 40 distinct pairs.
SELECTION = (
    ("03694", 6, "A_plausible", "small overhead traffic signal module"),
    ("00635", 1, "A_plausible", "partially occluded vehicle beside parked traffic"),
    ("00227", 8, "A_plausible", "roadside traffic signal housing in rain"),
    ("00146", 11, "A_plausible", "small distant traffic signal or lamp"),
    ("01277", 3, "A_plausible", "small roadside signal under night glare"),
    ("02043", 0, "A_plausible", "distant vehicle partly hidden by roadside trees"),
    ("02012", 0, "A_plausible", "vehicle-like target strongly occluded by trees"),
    ("00484", 3, "A_plausible", "very small distant vehicle in a rainy night scene"),
    ("01587", 2, "A_plausible", "distant small vehicle in an overcast scene"),
    ("03562", 1, "A_plausible", "small vehicle in dense urban traffic"),
    ("02503", 9, "A_plausible", "parked vehicle heavily occluded by a hedge"),
    ("02721", 6, "A_plausible", "small roadside traffic signal or lamp"),
    ("02231", 1, "A_plausible", "vehicle mostly hidden behind a hedge"),
    ("02504", 8, "A_plausible", "very small distant vehicle near an intersection"),
    ("01196", 1, "A_plausible", "distant vehicle amid rain and headlight glare"),
    ("00219", 2, "A_plausible", "small vehicle partly occluded by a pedestrian"),
    ("02577", 1, "A_plausible", "small distant vehicle among larger vehicles"),
    ("00072", 0, "A_plausible", "small vehicle under rain and reflection clutter"),
    ("02665", 0, "A_plausible", "small blue truck-like vehicle in traffic"),
    ("02587", 0, "A_plausible", "distant vehicle partly hidden by vegetation"),
    ("00543", 0, "A_plausible", "night vehicle visible in both RGB and IR"),
    ("02401", 1, "A_plausible", "partly occluded vehicle in dense traffic"),
    ("00629", 1, "A_plausible", "small thermally salient pedestrian-like target"),
    ("00073", 2, "A_plausible", "small vehicle in rainy low-visibility traffic"),
    ("03154", 0, "A_plausible", "distant vehicle partly hidden by another car"),
    ("03271", 0, "A_plausible", "vehicle-like target behind the leading car"),
    ("02075", 3, "A_plausible", "small roadside lamp or signal near a van"),
    ("00542", 0, "A_plausible", "night vehicle between two annotated cars"),
    ("01609", 1, "A_plausible", "small distant vehicle near the hedge"),
    ("01581", 1, "A_plausible", "small vehicle in the distant lane"),
    ("00545", 0, "A_plausible", "night vehicle between parked traffic"),
    ("00094", 1, "A_plausible", "very small distant vehicle in rain"),
    ("03250", 0, "A_plausible", "separate traffic-light module beside labeled lamps"),
    ("02591", 0, "A_plausible", "small vehicle partly occluded by a truck"),
    ("02316", 1, "A_plausible", "standing pedestrian visible in RGB and IR"),
    ("02581", 6, "B_ambiguous", "real vehicle-like target; bus class is uncertain"),
    ("00126", 1, "B_ambiguous", "tiny thermally salient pedestrian-like target"),
    ("00105", 2, "B_ambiguous", "distant pedestrian-like target under rain"),
    ("00145", 0, "B_ambiguous", "possible pedestrian between parked vehicles"),
    ("00990", 0, "B_ambiguous", "possible additional pedestrian in a crowded plaza"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("paper/m3fd_detection_results_lowconf_010"))
    parser.add_argument("--rgca-name", default="rgca_full_conf010")
    parser.add_argument("--lcafnet-name", default="lcafnet_baseline_conf010")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD"))
    parser.add_argument(
        "--scene-root", type=Path,
        default=Path("/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "paper/m3fd_detection_results_lowconf_010/"
            "rgca_possible_unlabeled_top40"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.scene_root = args.scene_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Relax GT coverage only enough to retain distinct, partly occluded objects.
    # LCAFNet matching and nested-RGCA duplicate rejection remain conservative.
    args.min_confidence = 0.10
    args.max_gt_iou = 0.15
    args.max_lcafnet_iou = 0.20
    args.max_gt_coverage = 0.60
    args.max_lcafnet_coverage = 0.35
    args.max_other_rgca_coverage = 0.50
    args.min_area = 0.00005
    args.max_area = 0.15

    candidates = collect_candidates(args)
    lookup = {
        (str(row["image_id"]), int(row["rgca_prediction_index"])): row
        for row in candidates
    }
    selected = []
    for image_id, prediction_index, tier, note in SELECTION:
        key = (image_id, prediction_index)
        if key not in lookup:
            raise ValueError(f"Selected candidate no longer passes filters: {key}")
        row = dict(lookup[key])
        row["review_tier"] = tier
        row["manual_status"] = (
            "plausible_needs_reannotation" if tier == "A_plausible"
            else "ambiguous_needs_reannotation")
        row["manual_note"] = note
        row["interpretation"] = (
            "annotation-audit candidate; not a verified true positive")
        selected.append(row)
        render_selected(
            args, row,
            comparison_dir / f"{image_id}_pred{prediction_index}.jpg")

    if len(selected) != 40 or len({row["image_id"] for row in selected}) != 40:
        raise AssertionError("The audit must contain 40 distinct image pairs")
    write_csv(args.output_dir / "all_automatic_candidates.csv", candidates)
    write_csv(args.output_dir / "selected_40_candidates.csv", selected)
    write_gallery(args.output_dir / "gallery.html", selected)

    tier_a = sum(row["review_tier"] == "A_plausible" for row in selected)
    tier_b = len(selected) - tier_a
    confidences = [float(row["rgca_confidence"]) for row in selected]
    readme = f"""# M3FD test：RGCA 独有检出的 40 组疑似漏标候选

## 结论边界

这里的 40 组是**标注审计候选**，不是已经证实的漏标或真阳性。现有 GT
没有这些框，因此在官方定量评测中它们仍被计为 FP。只有人工复标后，才能把
其中确认的目标用于论文中的漏标统计或模型优劣结论。

## 生成条件

- 数据：M3FD test，共 420 对严格配对 RGB/IR 图像。
- RGCA：`runs/train/m3fd_rgca_mask_gfb_best/weights/best.pt`。
- LCAFNet：`LCAFNet_M3FD.pt`。
- 两模型统一推理参数：640x640，confidence=0.10，NMS IoU=0.50。
- 自动规则：RGCA confidence >= 0.10；与任意 GT IoU < 0.15；与任意
  LCAFNet 框 IoU < 0.20；候选框被 GT/LCAFNet/其他 RGCA 框覆盖的比例
  分别 < 0.60/0.35/0.50。
- 自动筛出 {len(candidates)} 个候选，位于
  {len({str(row['image_id']) for row in candidates})} 张图像；人工视觉审查后
  选取 40 张互不重复的图像。
- A 档（视觉上较可信）：{tier_a}；B 档（目标/类别/边界仍有歧义）：{tier_b}。
- 本次 40 个 RGCA 置信度范围：{min(confidences):.3f}--{max(confidences):.3f}。

## 文件

- `comparisons/`：每组包含 RGB/IR 原图+现有 GT、RGCA 结果、LCAFNet
  结果以及候选区域裁剪。
- `selected_40_candidates.csv`：40 组候选的类别、置信度、框索引、重叠率、
  证据等级和人工备注。
- `all_automatic_candidates.csv`：完整自动候选池，便于继续人工筛选。
- `gallery.html`：浏览器画廊。

## 建议的论文使用流程

1. 由两名标注者独立检查 RGB 与 IR；判断“目标是否存在、类别、边界框”。
2. 对分歧项由第三人裁决，并报告一致率与最终确认数量。
3. 建立修正后的 audit 标签，同时重新评估 RGCA 和 LCAFNet；不能只给 RGCA
   补标签后直接比较。
4. 论文中使用“annotation-audit candidates / confirmed missing annotations”，
   不要在人工确认前写成“RGCA true positives”。

## 复现

```bash
cd /home/dell/lcp/LCAFNet-main
/home/dell/anaconda3/envs/MOD/bin/python \\
  tools/select_rgca_possible_unlabeled_top40.py
```
"""
    (args.output_dir / "README_CN.md").write_text(readme)
    print(
        f"Rendered {len(selected)} distinct candidates: "
        f"tier A={tier_a}, tier B={tier_b}; automatic pool={len(candidates)}")


if __name__ == "__main__":
    main()
