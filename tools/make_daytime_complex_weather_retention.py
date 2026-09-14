#!/usr/bin/env python3
"""Build the strict RGCA-vs-LCAF weather plot on Daytime test pairs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_complex_weather_degradation_retention import (  # noqa: E402
    WEATHER_MODALITIES,
    read_rows,
    write_weather_preview,
)
from tools.evaluate_modality_degradation_retention import (  # noqa: E402
    IR_CONTRAST_COLLAPSE,
    RGB_COMPLEX_WEATHER,
    ImageStats,
    arrays_to_records,
    bootstrap_summary,
    expected_keys,
    load_data_config,
    paired_difference_rows,
    plot_figure,
    write_summary_csv,
)
from tools.visualize_rgca_panel_a import sha256sum  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/multispectral/M3FD_Daytime_OriginalTest.yaml",
    )
    parser.add_argument(
        "--rgca-weights",
        type=Path,
        default=REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt",
    )
    parser.add_argument(
        "--lcaf-weights",
        type=Path,
        default=REPO_ROOT / "runs/train/exp_m3fd_lcaf_ca_mask/weights/best.pt",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_modality_degradation_m3fd",
    )
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_complex_weather_m3fd",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_complex_weather_m3fd_daytime_test136",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--preview-scenes", type=int, default=4)
    return parser.parse_args()


def read_manifest_ids(path: Path) -> List[str]:
    ids = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    return ids


def subset_records(
    records: Sequence[ImageStats], ordered_ids: Sequence[str], key: str
) -> List[ImageStats]:
    by_id = {record.image_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate cached image IDs in {key}")
    missing = [image_id for image_id in ordered_ids if image_id not in by_id]
    if missing:
        raise ValueError(f"{key} is missing Daytime IDs: {missing[:10]}")
    return [by_id[image_id] for image_id in ordered_ids]


def weather_keys() -> List[str]:
    return [
        f"{model}:{modality}:{severity}"
        for model in ("lcaf", "rgca")
        for modality in ("rgb_weather", "joint_weather")
        for severity in range(1, 5)
    ]


def write_readme(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    samples: Mapping[Tuple[str, str], np.ndarray],
    ids: Sequence[str],
    args: argparse.Namespace,
) -> None:
    lookup = {
        (str(row["model"]).lower(), str(row["degraded_modality"]).lower(), int(row["severity"])): row
        for row in rows
    }
    summaries = []
    for modality, display in (
        ("rgb_weather", "RGB 复合天气"),
        ("ir", "IR 退化"),
        ("joint_weather", "RGB 天气+IR 同时退化"),
    ):
        rgca = lookup[("rgca", modality, 4)]
        lcaf = lookup[("lcaf", modality, 4)]
        boot = 100 * (samples[("rgca", modality)][4] - samples[("lcaf", modality)][4])
        ci = np.nanpercentile(boot, [2.5, 97.5])
        point = 100 * (float(rgca["retention"]) - float(lcaf["retention"]))
        summaries.append(
            f"- Extreme {display}：RGCA={100*float(rgca['retention']):.2f}%，"
            f"LCAF={100*float(lcaf['retention']):.2f}%，差值={point:+.2f}pp，"
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]pp；绝对 mAP="
            f"{float(rgca['map50_95']):.4f} vs {float(lcaf['map50_95']):.4f}。"
        )
    mean_point, mean_boot = {}, {}
    for model in ("lcaf", "rgca"):
        mean_point[model] = 100 * np.mean([
            float(row["retention"]) for row in rows
            if str(row["model"]).lower() == model and int(row["severity"]) > 0
        ])
        mean_boot[model] = 100 * np.mean(
            np.concatenate([samples[(model, modality)][1:] for modality in WEATHER_MODALITIES], axis=0),
            axis=0,
        )
    mean_ci = np.nanpercentile(mean_boot["rgca"] - mean_boot["lcaf"], [2.5, 97.5])
    clean_rgca = float(lookup[("rgca", "rgb_weather", 0)]["clean_map50_95"])
    clean_lcaf = float(lookup[("lcaf", "rgb_weather", 0)]["clean_map50_95"])
    text = f"""# M3FD Daytime test136 复合天气退化实验

## Daytime 数据边界

- 只使用 `{args.data.name}` 中的 test 清单，共 {len(ids)} 对 RGB/IR。
- 场景成员来自 WACV 2024 RGBXFusion 元数据；train/val/test 成员关系仍保持原始本地 M3FD 3360/420/420 划分。
- 这 {len(ids)} 对是原 held-out test420 的严格子集；没有使用 Night、Overcast 或 Challenge 图像，也没有根据模型结果筛图。
- 两模型、全部退化等级和 bootstrap 使用完全相同的 Daytime 图像 ID 顺序。

## 三个面板

- A：Daytime RGB 施加雾+雨丝+颜色衰减+湿镜头模糊，IR 保持干净。
- B：Daytime IR 施加对比度塌缩+模糊+热噪声，RGB 保持干净。
- C：A 和 B 在相同等级同时施加。
- Clean mAP@0.5:0.95：RGCA={clean_rgca:.4f}，LCAF={clean_lcaf:.4f}。

## 结果

{chr(10).join(summaries)}
- 12 个非 Clean 条件平均保持率：RGCA={mean_point['rgca']:.2f}%，LCAF={mean_point['lcaf']:.2f}%，差值={mean_point['rgca']-mean_point['lcaf']:+.2f}pp，95% CI [{mean_ci[0]:+.2f}, {mean_ci[1]:+.2f}]pp。

## 复用缓存说明

全 test420 实验已经保存逐图像预测与匹配统计。本次先按官方 Daytime 清单过滤到 {len(ids)} 对，再重新计算所有 AP、保持率及 {args.bootstrap} 次成对 bootstrap；这与使用相同确定性输入重新跑模型后再按 Daytime 汇总在数值上等价，同时避免重复推理。原 test420 图和数据未被覆盖。
"""
    (output_dir / "README_DAYTIME_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 50:
        raise ValueError("--bootstrap must be at least 50")
    for path in (args.data, args.rgca_weights, args.lcaf_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data_config(args.data)
    rgb_ids = read_manifest_ids(Path(str(data["test_rgb"])))
    ir_ids = read_manifest_ids(Path(str(data["test_ir"])))
    if rgb_ids != ir_ids:
        raise ValueError("Daytime RGB and IR manifests are not pair-aligned")
    if len(rgb_ids) != 136:
        raise ValueError(f"Expected Daytime test136, found {len(rgb_ids)} pairs")

    preview_path = write_weather_preview(
        args.output_dir, data, args.seed, args.preview_scenes, args.dpi
    )
    base_cache = args.base_dir / "per_image_detection_stats.npz"
    weather_cache = args.weather_dir / "complex_weather_detection_stats.npz"
    for path in (base_cache, weather_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(base_cache, allow_pickle=False) as archive:
        base_signature = json.loads(str(archive["signature_json"]))
        if base_signature["rgca_sha256"] != sha256sum(args.rgca_weights):
            raise ValueError("RGCA weight hash differs from base cache")
        if base_signature["lcaf_sha256"] != sha256sum(args.lcaf_weights):
            raise ValueError("LCAF weight hash differs from base cache")
        base_all = arrays_to_records(archive, expected_keys())
    with np.load(weather_cache, allow_pickle=False) as archive:
        weather_signature = json.loads(str(archive["signature_json"]))
        if weather_signature["rgca_sha256"] != sha256sum(args.rgca_weights):
            raise ValueError("RGCA weight hash differs from weather cache")
        if weather_signature["lcaf_sha256"] != sha256sum(args.lcaf_weights):
            raise ValueError("LCAF weight hash differs from weather cache")
        if weather_signature["rgb_weather_schedule"] != json.loads(json.dumps(RGB_COMPLEX_WEATHER)):
            raise ValueError("Weather schedule differs from cached evaluation")
        weather_all = arrays_to_records(archive, weather_keys())

    records: Dict[str, List[ImageStats]] = {}
    for model in ("lcaf", "rgca"):
        clean_key = f"{model}:clean:0"
        records[clean_key] = subset_records(base_all[clean_key], rgb_ids, clean_key)
        for severity in range(1, 5):
            ir_key = f"{model}:ir:{severity}"
            records[ir_key] = subset_records(base_all[ir_key], rgb_ids, ir_key)
            for modality in ("rgb_weather", "joint_weather"):
                key = f"{model}:{modality}:{severity}"
                records[key] = subset_records(weather_all[key], rgb_ids, key)
    print(f"Validated and selected {len(rgb_ids)} paired Daytime test images")
    print(f"Computing {args.bootstrap} paired Daytime bootstrap replicates...")
    rows, samples = bootstrap_summary(
        records, args.bootstrap, args.seed, modalities=WEATHER_MODALITIES
    )

    metrics_path = args.output_dir / "daytime_complex_weather_degradation_metrics.csv"
    difference_path = args.output_dir / "daytime_paired_retention_differences.csv"
    bootstrap_path = args.output_dir / "daytime_bootstrap_retention_samples.npz"
    write_summary_csv(metrics_path, rows)
    write_summary_csv(
        difference_path,
        paired_difference_rows(rows, samples, modalities=WEATHER_MODALITIES),
    )
    np.savez_compressed(
        bootstrap_path,
        lcaf_rgb_weather=samples[("lcaf", "rgb_weather")],
        rgca_rgb_weather=samples[("rgca", "rgb_weather")],
        lcaf_ir=samples[("lcaf", "ir")], rgca_ir=samples[("rgca", "ir")],
        lcaf_joint_weather=samples[("lcaf", "joint_weather")],
        rgca_joint_weather=samples[("rgca", "joint_weather")],
    )
    stem = "m3fd_daytime_test136_rgca_vs_lcaf_complex_weather_degradation_retention"
    png_path, pdf_path = plot_figure(
        rows, samples, args.output_dir, args.dpi, len(rgb_ids),
        lcaf_label="LCAF attention",
        figure_title="Daytime-only cross-attention robustness under compound adverse weather",
        modalities=WEATHER_MODALITIES,
        output_stem=stem,
    )
    config = {
        "task": "Daytime-only strict RGCA-vs-LCAF compound-weather robustness",
        "scene": "daytime",
        "scene_source": "WACV 2024 RGBXFusion metadata",
        "split_source": "original local M3FD held-out test420 subset",
        "data_yaml": str(args.data.resolve()),
        "image_count": len(rgb_ids),
        "image_ids_sha256": hashlib.sha256("\n".join(rgb_ids).encode()).hexdigest(),
        "image_ids": rgb_ids,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "rgb_weather_schedule": list(RGB_COMPLEX_WEATHER),
        "ir_schedule": list(IR_CONTRAST_COLLAPSE),
        "rgca_weights": str(args.rgca_weights.resolve()),
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_weights": str(args.lcaf_weights.resolve()),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "source_caches": {"base": str(base_cache.resolve()), "weather": str(weather_cache.resolve())},
        "outputs": {
            "preview": str(preview_path.resolve()), "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()), "metrics": str(metrics_path.resolve()),
            "paired_differences": str(difference_path.resolve()),
            "bootstrap_samples": str(bootstrap_path.resolve()),
        },
    }
    with (args.output_dir / "run_config_daytime.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_readme(args.output_dir, rows, samples, rgb_ids, args)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
