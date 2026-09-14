#!/usr/bin/env python3
"""Add simultaneous RGB+IR degradation to the strict M3FD robustness figure.

This companion script deliberately reuses the already-computed clean and
single-modality test420 predictions.  It evaluates only the eight new joint
conditions (two models x four non-clean severity levels), then writes a new
three-panel figure without overwriting the original two-panel result.
"""

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

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.compare_strict_rgca_lcaf_scene_cross_attention import (  # noqa: E402
    restore_strict_lcaf_layers,
)
from tools.evaluate_modality_degradation_retention import (  # noqa: E402
    IR_CONTRAST_COLLAPSE,
    RGB_LOW_LIGHT,
    ImageStats,
    arrays_to_records,
    bootstrap_summary,
    calculate_map,
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


OUTPUT_STEM = "m3fd_rgca_vs_lcaf_modality_degradation_retention_with_joint"


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
        default=REPO_ROOT / "paper/rgca_lcaf_modality_degradation_m3fd",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the eight joint-degradation inference conditions.",
    )
    # Shared evaluation helpers expect these two attributes.
    parser.set_defaults(max_images=0, lcaf_variant="strict")
    return parser.parse_args()


def joint_keys() -> List[str]:
    return [
        f"{model_name}:joint:{severity}"
        for model_name in ("lcaf", "rgca")
        for severity in range(1, 5)
    ]


def joint_signature(
    args: argparse.Namespace, data: Mapping[str, object]
) -> Dict[str, object]:
    return {
        "version": 1,
        "task": "simultaneous RGB+IR degradation on M3FD test420",
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "lcaf_variant": "strict",
        "data": str(args.data.resolve()),
        "test_rgb": str(data["test_rgb"]),
        "test_ir": str(data["test_ir"]),
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "seed": args.seed,
        "rgb_exposure_regime": "underexposure_only_no_overexposure",
        "overexposure_included": False,
        "rgb_schedule": RGB_LOW_LIGHT,
        "ir_schedule": IR_CONTRAST_COLLAPSE,
    }


def validate_base_signature(
    cached: Mapping[str, object], args: argparse.Namespace, data: Mapping[str, object]
) -> None:
    expected = {
        "version": 1,
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "data": str(args.data.resolve()),
        "test_rgb": str(data["test_rgb"]),
        "test_ir": str(data["test_ir"]),
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "seed": args.seed,
        "max_images": 0,
        "rgb_schedule": list(RGB_LOW_LIGHT),
        "ir_schedule": list(IR_CONTRAST_COLLAPSE),
    }
    # The original strict cache predates the explicit lcaf_variant field.  All
    # substantive protocol fields and both checkpoint hashes must still match.
    mismatches = {
        key: (cached.get(key), value)
        for key, value in expected.items()
        if cached.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Existing strict single-modality cache is incompatible:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    if cached.get("lcaf_variant", "strict") != "strict":
        raise ValueError("Base cache is not the strict LCAF attention control")


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def validate_model_structure(model_name: str, model: torch.nn.Module) -> None:
    for layer_index in (20, 21, 22, 23):
        layer = model.model[layer_index]
        if model_name == "rgca":
            attention = getattr(layer, "cross_modal_attention", None)
            if attention is None or getattr(attention, "reliability_gate", None) is None:
                raise TypeError(f"RGCA layer {layer_index} lacks reliability_gate")
            if getattr(attention, "channel_prior", None) is not None:
                raise TypeError(f"RGCA layer {layer_index} unexpectedly has channel_prior")
            required = ("foreground_mask", "fusion")
        else:
            required = ("mhca_rgb", "mhca_ir", "foreground_mask", "fusion")
        if not all(hasattr(layer, name) for name in required):
            raise TypeError(f"{model_name.upper()} layer {layer_index} failed validation")


def run_joint_inference(
    args: argparse.Namespace,
    data: Mapping[str, object],
    device: torch.device,
    reference_ids: Sequence[str],
) -> Tuple[Dict[str, List[ImageStats]], Dict[str, object], Dict[str, object]]:
    all_records: Dict[str, List[ImageStats]] = {}
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    for model_name, weight_path in (
        ("lcaf", args.lcaf_weights),
        ("rgca", args.rgca_weights),
    ):
        print(f"Loading {model_name.upper()}: {weight_path}")
        model = attempt_load(str(weight_path), map_location=device).to(device).float().eval()
        if model_name == "lcaf":
            restore_audit["lcaf"] = restore_strict_lcaf_layers(model, device)
        class_metadata = validate_class_names(model, data["names"])
        validate_model_structure(model_name, model)
        stride = max(int(model.stride.max()), 32)
        dataloader, dataset = make_dataloader(data, args, stride)
        if len(dataset) != 420:
            raise ValueError(f"Expected M3FD test420, found {len(dataset)} images")
        if device.type == "cuda":
            model.half()
            keep_force_fp32_modules(model)
        for severity in range(1, 5):
            key = f"{model_name}:joint:{severity}"
            start = time.time()
            records = evaluate_condition(
                model,
                dataloader,
                device,
                "joint",
                severity,
                args,
                f"{model_name.upper()} RGB+IR S{severity}",
            )
            image_ids = [record.image_id for record in records]
            if image_ids != list(reference_ids):
                raise ValueError(f"Paired image/order audit failed for {key}")
            all_records[key] = records
            map50, map_value, _ = calculate_map(records)
            print(
                f"{key}: images={len(records)}, mAP50={map50:.4f}, "
                f"mAP50:95={map_value:.4f}, elapsed={time.time() - start:.1f}s"
            )
        model_audit[model_name] = {
            "class": type(model).__name__,
            "fusion_classes": [
                type(model.model[index]).__name__ for index in (20, 21, 22, 23)
            ],
            "class_metadata": class_metadata,
            "images": len(reference_ids),
        }
        del model
        del dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_records, restore_audit, model_audit


def write_joint_readme(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    samples: Mapping[Tuple[str, str], np.ndarray],
    args: argparse.Namespace,
) -> None:
    lookup = {
        (
            str(row["model"]).lower(),
            str(row["degraded_modality"]).lower(),
            int(row["severity"]),
        ): row
        for row in rows
    }
    mean_point: Dict[str, float] = {}
    mean_boot: Dict[str, np.ndarray] = {}
    for model_name in ("lcaf", "rgca"):
        degraded = [
            float(row["retention"])
            for row in rows
            if str(row["model"]).lower() == model_name
            and int(row["severity"]) > 0
        ]
        mean_point[model_name] = 100 * float(np.mean(degraded))
        mean_boot[model_name] = 100 * np.mean(
            np.concatenate(
                [samples[(model_name, modality)][1:] for modality in ("rgb", "ir", "joint")],
                axis=0,
            ),
            axis=0,
        )
    mean_diff_ci = np.nanpercentile(mean_boot["rgca"] - mean_boot["lcaf"], [2.5, 97.5])
    joint_rgca = lookup[("rgca", "joint", 4)]
    joint_lcaf = lookup[("lcaf", "joint", 4)]
    joint_diff_point = 100 * (
        float(joint_rgca["retention"]) - float(joint_lcaf["retention"])
    )
    joint_diff_boot = 100 * (
        samples[("rgca", "joint")][4] - samples[("lcaf", "joint")][4]
    )
    joint_diff_ci = np.nanpercentile(joint_diff_boot, [2.5, 97.5])
    clean_rgca = float(joint_rgca["clean_map50_95"])
    clean_lcaf = float(joint_lcaf["clean_map50_95"])
    text = f"""# M3FD 单模态与双模态同时退化对比

## 三幅曲线分别回答什么

- A（RGB）：只对 RGB 施加欠曝/低照度和传感器噪声，IR 保持干净，主要检验模型能否利用 IR 补偿 RGB 信息损失；不包含过曝。
- B（IR）：只退化 IR，RGB 保持干净，主要检验模型能否利用 RGB 补偿 IR 信息损失。
- C（RGB+IR）：RGB 与 IR 在同一等级同时退化，检验两路信息一起受损时的整体鲁棒性。它更严格，但不能单独区分模型究竟依赖哪一路模态。
- 纵轴为 `退化后 mAP@0.5:0.95 / 各模型自身 clean mAP@0.5:0.95`；阴影为同一批图像的成对 bootstrap 95% CI（{args.bootstrap} 次）。

## 本次结果

- Clean mAP@0.5:0.95：RGCA={clean_rgca:.4f}，LCAF={clean_lcaf:.4f}。
- Extreme RGB+IR 同时退化：RGCA 绝对 mAP={float(joint_rgca['map50_95']):.4f}、保持率={100*float(joint_rgca['retention']):.2f}%；LCAF 绝对 mAP={float(joint_lcaf['map50_95']):.4f}、保持率={100*float(joint_lcaf['retention']):.2f}%。
- Extreme RGB+IR 的 RGCA−LCAF 保持率差={joint_diff_point:+.2f}pp，成对 95% CI [{joint_diff_ci[0]:+.2f}, {joint_diff_ci[1]:+.2f}]pp。
- 12 个非 clean 条件（RGB、IR、RGB+IR 各 4 级）的平均保持率：RGCA={mean_point['rgca']:.2f}%，LCAF={mean_point['lcaf']:.2f}%，差值={mean_point['rgca']-mean_point['lcaf']:+.2f}pp，成对 95% CI [{mean_diff_ci[0]:+.2f}, {mean_diff_ci[1]:+.2f}]pp。

## 实验协议与边界

- 数据为 M3FD held-out test420，共 420 对严格对齐的 RGB/IR 图像；两模型的图像顺序、退化参数和确定性噪声完全相同。
- RGB 协议仅包含欠曝：所有非 Clean 等级均为 `gamma>1` 且 `gain<1`，没有 `gamma<1`、`gain>1` 或高亮饱和操作，因此不包含过曝。
- 同时退化并不是把两路直接置零：每一级分别使用 A/B 面板已经预设的 RGB 欠曝与 IR 退化参数，并在 C 面板中同时施加。
- 比较对象为纯 RGCA+Mask+GFB 与原始 LCAF attention+相同 Mask+GFB，是注意力路径的严格控制对比。
- 这张图只证明模型对这些预定义合成退化的稳健性；不能外推为所有真实传感器故障，也不能代替多训练随机种子实验。

数值文件：`modality_degradation_metrics_with_joint.csv`、`paired_retention_differences_with_joint.csv`；复现实验配置：`run_config_joint.yaml`。
"""
    (output_dir / "README_JOINT_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 50:
        raise ValueError("--bootstrap must be at least 50")
    for path in (args.rgca_weights, args.lcaf_weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_cache_path = args.output_dir / "per_image_detection_stats.npz"
    base_metrics_path = args.output_dir / "modality_degradation_metrics.csv"
    base_bootstrap_path = args.output_dir / "bootstrap_retention_samples.npz"
    for path in (base_cache_path, base_metrics_path, base_bootstrap_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required strict single-modality result is missing: {path}"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data_config(args.data)
    with np.load(base_cache_path, allow_pickle=False) as archive:
        base_signature = json.loads(str(archive["signature_json"]))
        validate_base_signature(base_signature, args, data)
        base_records = arrays_to_records(archive, expected_keys())
    reference_ids = [record.image_id for record in base_records["lcaf:clean:0"]]
    if len(reference_ids) != 420:
        raise ValueError(f"Expected 420 cached image pairs, found {len(reference_ids)}")
    for key, records in base_records.items():
        if [record.image_id for record in records] != reference_ids:
            raise ValueError(f"Base paired image/order audit failed for {key}")
    print(f"Validated strict base cache: {len(reference_ids)} aligned test pairs")

    signature = joint_signature(args, data)
    joint_cache_path = args.output_dir / "joint_modality_detection_stats.npz"
    joint_records = None
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    if joint_cache_path.is_file() and not args.force:
        with np.load(joint_cache_path, allow_pickle=False) as archive:
            cached_signature = json.loads(str(archive["signature_json"]))
            if cached_signature == json.loads(json.dumps(signature)):
                joint_records = arrays_to_records(archive, joint_keys())
                print(f"Reusing compatible joint inference cache: {joint_cache_path}")
            else:
                print("Joint inference cache signature differs; rerunning inference")
    if joint_records is None:
        device = resolve_device(args.device)
        joint_records, restore_audit, model_audit = run_joint_inference(
            args, data, device, reference_ids
        )
        np.savez_compressed(
            joint_cache_path, **records_to_arrays(joint_records, signature)
        )
        print(f"Saved joint per-image prediction cache: {joint_cache_path}")
    for key, records in joint_records.items():
        if [record.image_id for record in records] != reference_ids:
            raise ValueError(f"Joint paired image/order audit failed for {key}")

    # Recompute only clean+joint bootstrap statistics.  The bootstrap RNG and
    # image count match the saved single-modality samples exactly, so paired
    # differences and the 12-condition aggregate remain image-aligned.
    joint_bootstrap_records: Dict[str, Sequence[ImageStats]] = {}
    for model_name in ("lcaf", "rgca"):
        joint_bootstrap_records[f"{model_name}:clean:0"] = base_records[
            f"{model_name}:clean:0"
        ]
        for severity in range(1, 5):
            key = f"{model_name}:joint:{severity}"
            joint_bootstrap_records[key] = joint_records[key]
    print(f"Computing {args.bootstrap} paired bootstrap replicates for joint conditions...")
    joint_rows, joint_samples = bootstrap_summary(
        joint_bootstrap_records, args.bootstrap, args.seed, modalities=("joint",)
    )

    base_rows = read_rows(base_metrics_path)
    with np.load(base_bootstrap_path, allow_pickle=False) as archive:
        required = ("lcaf_rgb", "rgca_rgb", "lcaf_ir", "rgca_ir")
        for key in required:
            if key not in archive:
                raise KeyError(f"Missing base bootstrap array: {key}")
            if archive[key].shape != (5, args.bootstrap):
                raise ValueError(
                    f"Base bootstrap shape for {key} is {archive[key].shape}; "
                    f"expected (5, {args.bootstrap})"
                )
        retention_samples = {
            ("lcaf", "rgb"): archive["lcaf_rgb"].copy(),
            ("rgca", "rgb"): archive["rgca_rgb"].copy(),
            ("lcaf", "ir"): archive["lcaf_ir"].copy(),
            ("rgca", "ir"): archive["rgca_ir"].copy(),
        }
    retention_samples.update(joint_samples)
    rows: List[Mapping[str, object]] = base_rows + joint_rows
    modalities = ("rgb", "ir", "joint")

    metrics_path = args.output_dir / "modality_degradation_metrics_with_joint.csv"
    write_summary_csv(metrics_path, rows)
    differences_path = args.output_dir / "paired_retention_differences_with_joint.csv"
    write_summary_csv(
        differences_path,
        paired_difference_rows(rows, retention_samples, modalities=modalities),
    )
    bootstrap_path = args.output_dir / "bootstrap_retention_samples_with_joint.npz"
    np.savez_compressed(
        bootstrap_path,
        lcaf_rgb=retention_samples[("lcaf", "rgb")],
        rgca_rgb=retention_samples[("rgca", "rgb")],
        lcaf_ir=retention_samples[("lcaf", "ir")],
        rgca_ir=retention_samples[("rgca", "ir")],
        lcaf_joint=retention_samples[("lcaf", "joint")],
        rgca_joint=retention_samples[("rgca", "joint")],
    )
    png_path, pdf_path = plot_figure(
        rows,
        retention_samples,
        args.output_dir,
        args.dpi,
        len(reference_ids),
        lcaf_label="LCAF attention",
        figure_title=(
            "Cross-attention robustness under single- and dual-modality degradation"
        ),
        modalities=modalities,
        output_stem=OUTPUT_STEM,
    )

    config = {
        "task": "strict RGCA-vs-LCAF single- and simultaneous dual-modality degradation robustness",
        "comparison_scope": "attention control with identical Foreground Mask and GFB",
        "protocol": "M3FD local held-out test420",
        "image_count": len(reference_ids),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(reference_ids).encode("utf-8")
        ).hexdigest(),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "paired_bootstrap": True,
        "joint_condition": (
            "At each severity, apply the RGB and IR schedules simultaneously; "
            "clean is shared with the single-modality evaluation"
        ),
        "rgb_exposure_regime": "underexposure only",
        "overexposure_included": False,
        "rgb_schedule": list(RGB_LOW_LIGHT),
        "ir_schedule": list(IR_CONTRAST_COLLAPSE),
        "same_corruption_realization_across_models": True,
        "corrupt_letterbox_padding": False,
        "rgca_weights": str(args.rgca_weights.resolve()),
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_weights": str(args.lcaf_weights.resolve()),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "lcaf_identity": "original LCAF attention + identical Foreground Mask + GFB",
        "rgca_identity": "pure prior-free RGCA + Foreground Mask + GFB",
        "data_yaml": str(args.data.resolve()),
        "evaluation": {
            "image_size": args.img_size,
            "batch_size": args.batch_size,
            "confidence_threshold": args.conf_thres,
            "nms_iou": args.nms_iou,
            "iou_thresholds": "0.50:0.05:0.95",
        },
        "base_cache": str(base_cache_path.resolve()),
        "joint_cache": str(joint_cache_path.resolve()),
        "model_audit": model_audit,
        "lcaf_runtime_compatibility_audit": restore_audit,
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "metrics_csv": str(metrics_path.resolve()),
            "paired_difference_csv": str(differences_path.resolve()),
            "bootstrap_samples": str(bootstrap_path.resolve()),
        },
    }
    with (args.output_dir / "run_config_joint.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_joint_readme(args.output_dir, rows, retention_samples, args)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
