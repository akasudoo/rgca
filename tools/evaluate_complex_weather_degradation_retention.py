#!/usr/bin/env python3
"""Evaluate strict RGCA-vs-LCAF robustness to compound adverse weather."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.add_joint_modality_degradation_panel import (  # noqa: E402
    read_rows,
    validate_base_signature,
    validate_model_structure,
)
from tools.compare_strict_rgca_lcaf_scene_cross_attention import (  # noqa: E402
    restore_strict_lcaf_layers,
)
from tools.evaluate_modality_degradation_retention import (  # noqa: E402
    IR_CONTRAST_COLLAPSE,
    RGB_COMPLEX_WEATHER,
    ImageStats,
    arrays_to_records,
    bootstrap_summary,
    calculate_map,
    degrade_rgb_complex_weather,
    evaluate_condition,
    expected_keys,
    load_data_config,
    make_dataloader,
    paired_difference_rows,
    plot_figure,
    records_to_arrays,
    resolve_device,
    validate_class_names,
    write_summary_csv,
)
from tools.visualize_rgca_panel_a import sha256sum  # noqa: E402
from utils.general import keep_force_fp32_modules  # noqa: E402


OUTPUT_STEM = "m3fd_rgca_vs_lcaf_complex_weather_degradation_retention"
WEATHER_MODALITIES = ("rgb_weather", "ir", "joint_weather")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--data", type=Path, default=REPO_ROOT / "data/multispectral/M3FD.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_complex_weather_m3fd",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--preview-scenes", type=int, default=4)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(max_images=0, lcaf_variant="strict")
    return parser.parse_args()


def weather_keys() -> List[str]:
    return [
        f"{model}:{modality}:{severity}"
        for model in ("lcaf", "rgca")
        for modality in ("rgb_weather", "joint_weather")
        for severity in range(1, 5)
    ]


def weather_signature(
    args: argparse.Namespace, data: Mapping[str, object]
) -> Dict[str, object]:
    return {
        "version": 1,
        "task": "compound adverse RGB weather and joint RGB-weather+IR degradation",
        "weather_model": (
            "spatial atmospheric scattering + directional rain streaks + "
            "desaturation + local wet-lens blur"
        ),
        "depth_map_used": False,
        "exposure_or_pixel_noise_used": False,
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "data": str(args.data.resolve()),
        "test_rgb": str(data["test_rgb"]),
        "test_ir": str(data["test_ir"]),
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "seed": args.seed,
        "rgb_weather_schedule": RGB_COMPLEX_WEATHER,
        "ir_schedule": IR_CONTRAST_COLLAPSE,
    }


def write_weather_preview(
    output_dir: Path,
    data: Mapping[str, object],
    seed: int,
    scene_count: int,
    dpi: int,
) -> Path:
    rgb_root = Path(str(data["test_rgb"]))
    if rgb_root.is_file():
        paths = [
            Path(line.strip())
            for line in rgb_root.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        paths = sorted(
            path
            for path in rgb_root.iterdir()
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
    if not paths:
        raise FileNotFoundError(f"No RGB test images in {rgb_root}")
    rng = np.random.default_rng(seed + 71)
    selected = [paths[int(index)] for index in rng.choice(len(paths), scene_count, replace=False)]
    fig, axes = plt.subplots(scene_count, 5, figsize=(15.5, 2.6 * scene_count))
    axes = np.atleast_2d(axes)
    for row_index, path in enumerate(selected):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        shape = ((rgb.shape[0], rgb.shape[1]), ((1.0, 1.0), (0.0, 0.0)))
        for severity in range(5):
            degraded = degrade_rgb_complex_weather(
                tensor, [str(path)], [shape], severity, seed
            )[0].permute(1, 2, 0).cpu().numpy()
            axis = axes[row_index, severity]
            axis.imshow(degraded)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(("Clean", "Mild", "Moderate", "Severe", "Extreme")[severity])
            if severity == 0:
                axis.text(
                    0.01, 0.03, path.stem, transform=axis.transAxes, color="white",
                    fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
    fig.suptitle(
        "M3FD RGB compound adverse-weather protocol: fog + rain + wet-lens blur",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.15, h_pad=0.25)
    path = output_dir / "complex_weather_protocol_preview.png"
    fig.savefig(path, dpi=min(dpi, 300), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run_weather_inference(
    args: argparse.Namespace,
    data: Mapping[str, object],
    device: torch.device,
    reference_ids: Sequence[str],
) -> Tuple[Dict[str, List[ImageStats]], Dict[str, object], Dict[str, object]]:
    all_records: Dict[str, List[ImageStats]] = {}
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    for model_name, weight_path in (("lcaf", args.lcaf_weights), ("rgca", args.rgca_weights)):
        print(f"Loading {model_name.upper()}: {weight_path}")
        model = attempt_load(str(weight_path), map_location=device).to(device).float().eval()
        if model_name == "lcaf":
            restore_audit["lcaf"] = restore_strict_lcaf_layers(model, device)
        class_metadata = validate_class_names(model, data["names"])
        validate_model_structure(model_name, model)
        dataloader, dataset = make_dataloader(data, args, max(int(model.stride.max()), 32))
        if len(dataset) != 420:
            raise ValueError(f"Expected M3FD test420, found {len(dataset)} images")
        if device.type == "cuda":
            model.half()
            keep_force_fp32_modules(model)
        for modality in ("rgb_weather", "joint_weather"):
            for severity in range(1, 5):
                key = f"{model_name}:{modality}:{severity}"
                start = time.time()
                records = evaluate_condition(
                    model, dataloader, device, modality, severity, args,
                    f"{model_name.upper()} {modality} S{severity}",
                )
                if [record.image_id for record in records] != list(reference_ids):
                    raise ValueError(f"Paired image/order audit failed for {key}")
                all_records[key] = records
                map50, map_value, _ = calculate_map(records)
                print(
                    f"{key}: images={len(records)}, mAP50={map50:.4f}, "
                    f"mAP50:95={map_value:.4f}, elapsed={time.time()-start:.1f}s"
                )
        model_audit[model_name] = {
            "class": type(model).__name__,
            "fusion_classes": [type(model.model[i]).__name__ for i in (20, 21, 22, 23)],
            "class_metadata": class_metadata,
            "images": len(reference_ids),
        }
        del model, dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_records, restore_audit, model_audit


def write_readme(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    samples: Mapping[Tuple[str, str], np.ndarray],
    args: argparse.Namespace,
) -> None:
    lookup = {
        (str(r["model"]).lower(), str(r["degraded_modality"]).lower(), int(r["severity"])): r
        for r in rows
    }
    lines = []
    for modality, label in (
        ("rgb_weather", "RGB 复合天气"),
        ("ir", "IR 退化"),
        ("joint_weather", "RGB 天气+IR 同时退化"),
    ):
        rgca, lcaf = lookup[("rgca", modality, 4)], lookup[("lcaf", modality, 4)]
        diff = 100 * (samples[("rgca", modality)][4] - samples[("lcaf", modality)][4])
        ci = np.nanpercentile(diff, [2.5, 97.5])
        point = 100 * (float(rgca["retention"]) - float(lcaf["retention"]))
        lines.append(
            f"- Extreme {label}：RGCA={100*float(rgca['retention']):.2f}%，"
            f"LCAF={100*float(lcaf['retention']):.2f}%，差值={point:+.2f}pp，"
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]pp。"
        )
    mean_point, mean_boot = {}, {}
    for model in ("lcaf", "rgca"):
        mean_point[model] = 100 * np.mean([
            float(r["retention"]) for r in rows
            if str(r["model"]).lower() == model and int(r["severity"]) > 0
        ])
        mean_boot[model] = 100 * np.mean(np.concatenate([
            samples[(model, modality)][1:] for modality in WEATHER_MODALITIES
        ], axis=0), axis=0)
    mean_ci = np.nanpercentile(mean_boot["rgca"] - mean_boot["lcaf"], [2.5, 97.5])
    text = f"""# M3FD 复合恶劣天气模态退化实验

## 协议

- A：只退化 RGB，IR 保持干净。RGB 同时包含空间变化的大气散射雾、方向性雨丝、颜色/对比度衰减和局部湿镜头模糊。
- B：只退化 IR，RGB 保持干净；沿用原来的 IR 对比度塌缩、模糊和热噪声协议。
- C：A 的 RGB 复合天气和 B 的 IR 退化在相同等级同时施加。
- RGB 不再使用 gamma、曝光增益或独立像素高斯噪声。四个等级保持同一种雾雨复合天气，仅同步提高雾密度、雨丝密度/长度和湿镜头影响。
- 雾使用 `I=J*t(x)+A*(1-t(x))`。M3FD 没有深度图，因此 `t(x)` 是平滑空间代理，不是基于真实距离的精确雾仿真。

## 结果

{chr(10).join(lines)}
- 12 个非 Clean 条件平均保持率：RGCA={mean_point['rgca']:.2f}%，LCAF={mean_point['lcaf']:.2f}%，差值={mean_point['rgca']-mean_point['lcaf']:+.2f}pp，95% CI [{mean_ci[0]:+.2f}, {mean_ci[1]:+.2f}]pp。

## 科学边界

这是物理启发的可复现合成复合天气，比单纯 gamma+噪声更贴近雾雨环境，但不等价于真实采集的配对恶劣天气数据。复合因素同时变化，因此本图证明的是整体天气鲁棒性，不能把性能差异单独归因于雾、雨丝或湿镜头中的某一项。

参考依据：Sakaridis et al., ECCV 2018（合成雾与真实雾）；Hu et al., CVPR 2019（雨丝与雾的联合成像）；Li et al., CVPR 2019（rain streak / raindrop / rain-and-mist 基准）。
"""
    (output_dir / "README_COMPLEX_WEATHER_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 50:
        raise ValueError("--bootstrap must be at least 50")
    for path in (args.rgca_weights, args.lcaf_weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data_config(args.data)
    preview_path = write_weather_preview(
        args.output_dir, data, args.seed, args.preview_scenes, args.dpi
    )
    print(f"Saved protocol preview: {preview_path}")
    if args.preview_only:
        return

    base_dir = REPO_ROOT / "paper/rgca_lcaf_modality_degradation_m3fd"
    base_cache_path = base_dir / "per_image_detection_stats.npz"
    base_metrics_path = base_dir / "modality_degradation_metrics.csv"
    base_bootstrap_path = base_dir / "bootstrap_retention_samples.npz"
    for path in (base_cache_path, base_metrics_path, base_bootstrap_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(base_cache_path, allow_pickle=False) as archive:
        validate_base_signature(json.loads(str(archive["signature_json"])), args, data)
        base_records = arrays_to_records(archive, expected_keys())
    reference_ids = [r.image_id for r in base_records["lcaf:clean:0"]]
    if len(reference_ids) != 420:
        raise ValueError(f"Expected 420 base records, found {len(reference_ids)}")

    signature = weather_signature(args, data)
    cache_path = args.output_dir / "complex_weather_detection_stats.npz"
    weather_records = None
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    if cache_path.is_file() and not args.force:
        with np.load(cache_path, allow_pickle=False) as archive:
            if json.loads(str(archive["signature_json"])) == json.loads(json.dumps(signature)):
                weather_records = arrays_to_records(archive, weather_keys())
                print(f"Reusing compatible weather cache: {cache_path}")
    if weather_records is None:
        weather_records, restore_audit, model_audit = run_weather_inference(
            args, data, resolve_device(args.device), reference_ids
        )
        np.savez_compressed(cache_path, **records_to_arrays(weather_records, signature))
        print(f"Saved weather prediction cache: {cache_path}")

    bootstrap_records: Dict[str, Sequence[ImageStats]] = {}
    for model in ("lcaf", "rgca"):
        bootstrap_records[f"{model}:clean:0"] = base_records[f"{model}:clean:0"]
        for modality in ("rgb_weather", "joint_weather"):
            for severity in range(1, 5):
                key = f"{model}:{modality}:{severity}"
                bootstrap_records[key] = weather_records[key]
    print(f"Computing {args.bootstrap} paired bootstrap replicates...")
    weather_rows, samples = bootstrap_summary(
        bootstrap_records, args.bootstrap, args.seed,
        modalities=("rgb_weather", "joint_weather"),
    )
    base_rows = read_rows(base_metrics_path)
    ir_rows = [row for row in base_rows if row["degraded_modality"] == "IR"]
    with np.load(base_bootstrap_path, allow_pickle=False) as archive:
        if archive["lcaf_ir"].shape != (5, args.bootstrap):
            raise ValueError("Base bootstrap replicate count differs")
        samples[("lcaf", "ir")] = archive["lcaf_ir"].copy()
        samples[("rgca", "ir")] = archive["rgca_ir"].copy()
    rows: List[Mapping[str, object]] = weather_rows + ir_rows

    metrics_path = args.output_dir / "complex_weather_degradation_metrics.csv"
    difference_path = args.output_dir / "paired_complex_weather_retention_differences.csv"
    bootstrap_path = args.output_dir / "complex_weather_bootstrap_retention_samples.npz"
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
    png_path, pdf_path = plot_figure(
        rows, samples, args.output_dir, args.dpi, len(reference_ids),
        lcaf_label="LCAF attention",
        figure_title="Cross-attention robustness under compound adverse weather",
        modalities=WEATHER_MODALITIES,
        output_stem=OUTPUT_STEM,
    )
    config = {
        **signature,
        "protocol": "M3FD local held-out test420",
        "image_count": len(reference_ids),
        "image_ids_sha256": hashlib.sha256("\n".join(reference_ids).encode()).hexdigest(),
        "bootstrap_replicates": args.bootstrap,
        "same_weather_realization_across_models": True,
        "corrupt_letterbox_padding": False,
        "comparison_scope": "attention control with identical Foreground Mask and GFB",
        "rgca_weights": str(args.rgca_weights.resolve()),
        "lcaf_weights": str(args.lcaf_weights.resolve()),
        "model_audit": model_audit,
        "lcaf_runtime_compatibility_audit": restore_audit,
        "outputs": {
            "preview": str(preview_path.resolve()), "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()), "metrics_csv": str(metrics_path.resolve()),
            "paired_difference_csv": str(difference_path.resolve()),
            "bootstrap_samples": str(bootstrap_path.resolve()), "cache": str(cache_path.resolve()),
        },
    }
    with (args.output_dir / "run_config_complex_weather.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_readme(args.output_dir, rows, samples, args)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
