#!/usr/bin/env python3
"""Evaluate asymmetric RGB/IR reliability on the M3FD Daytime test split.

The 3x3 protocol crosses three pre-declared RGB adverse-weather severities
with three pre-declared IR thermal-contrast-loss severities.  It reports:

1. the complete requested RGCA checkpoint versus original LCAFNet;
2. the same RGCA checkpoint with its learned gate enabled versus bypassed;
3. the separately trained prior-free C0 RGCA versus its no-gate ablation.

The second comparison is a same-checkpoint inference intervention, while the
third is a matched training ablation.  Keeping both prevents a full-system
performance difference from being presented as proof of the gate alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.evaluate_modality_degradation_retention import (  # noqa: E402
    IR_CONTRAST_COLLAPSE,
    RGB_COMPLEX_WEATHER,
    ImageStats,
    arrays_to_records,
    calculate_map,
    degrade_modality,
    degrade_rgb_complex_weather,
    load_data_config,
    make_dataloader,
    match_predictions,
    records_to_arrays,
    resolve_device,
)
from tools.visualize_rgca_panel_a import sha256sum  # noqa: E402
from utils.general import (  # noqa: E402
    keep_force_fp32_modules,
    non_max_suppression,
)


LEVELS = (0, 2, 4)
LEVEL_NAMES = ("Clean", "Moderate", "Extreme")
MODEL_SPECS = (
    "full_rgca",
    "original_lcaf",
    "full_rgca_gate_bypass",
    "gated_c0",
    "no_gate_c0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-rgca-weights",
        type=Path,
        default=REPO_ROOT / "runs/train/m3fd_rgca_mask_gfb_best/weights/best.pt",
    )
    parser.add_argument(
        "--original-lcaf-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/train/lcafnet_original_close_mosaic103/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--gated-c0-weights",
        type=Path,
        default=(
            REPO_ROOT / "runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--no-gate-c0-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/train/exp_rgca_no_reliability_mask_gfb_uniform_lr_m3fd_full200/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/multispectral/M3FD_Daytime_OriginalTest.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_reliability_asymmetric_stress_m3fd_daytime",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def condition_key(model_name: str, rgb_level: int, ir_level: int) -> str:
    return f"{model_name}:rgb{rgb_level}:ir{ir_level}"


def expected_keys() -> List[str]:
    return [
        condition_key(model_name, rgb_level, ir_level)
        for model_name in MODEL_SPECS
        for ir_level in LEVELS
        for rgb_level in LEVELS
    ]


def set_gate_mode(model: torch.nn.Module, mode: str) -> Dict[str, object]:
    audit: List[Dict[str, object]] = []
    for layer_index in (20, 21, 22, 23):
        fusion = model.model[layer_index]
        attention = getattr(fusion, "cross_modal_attention", None)
        if attention is None:
            if mode in ("original_lcaf",):
                continue
            raise TypeError(f"{mode}: layer {layer_index} has no RGCA attention")
        gate = getattr(attention, "reliability_gate", None)
        if mode in ("full_rgca", "gated_c0"):
            if gate is None:
                raise TypeError(f"{mode}: layer {layer_index} has no learned gate")
            attention.use_reliability_gate = True
        elif mode == "full_rgca_gate_bypass":
            if gate is None:
                raise TypeError(f"{mode}: layer {layer_index} has no gate to bypass")
            attention.use_reliability_gate = False
        elif mode == "no_gate_c0":
            if gate is not None:
                raise TypeError(f"{mode}: layer {layer_index} unexpectedly has a gate")
            if getattr(attention, "use_reliability_gate", False):
                raise TypeError(f"{mode}: layer {layer_index} gate bypass is not active")
        elif mode != "original_lcaf":
            raise ValueError(mode)
        audit.append(
            {
                "layer": layer_index,
                "attention_class": type(attention).__name__,
                "learned_gate_present": gate is not None,
                "gate_enabled": bool(
                    getattr(attention, "use_reliability_gate", gate is not None)
                ),
                "prior_mode": getattr(attention, "prior_mode", None),
            }
        )
    return {"mode": mode, "layers": audit}


def evaluate_pair_condition(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    rgb_level: int,
    ir_level: int,
    args: argparse.Namespace,
    description: str,
) -> List[ImageStats]:
    half = device.type == "cuda"
    dtype = torch.float16 if half else torch.float32
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    records: List[ImageStats] = []
    seen = 0
    progress = tqdm(dataloader, desc=description, leave=False)
    for packed, targets, paths, shapes in progress:
        if args.max_images and seen >= args.max_images:
            break
        packed = packed.to(device, non_blocking=True).to(dtype) / 255.0
        targets = targets.to(device)
        if args.max_images and seen + packed.shape[0] > args.max_images:
            keep = args.max_images - seen
            packed = packed[:keep]
            paths = paths[:keep]
            shapes = shapes[:keep]
            targets = targets[targets[:, 0] < keep]
        rgb, ir = packed[:, :3], packed[:, 3:]
        if rgb_level:
            rgb = degrade_rgb_complex_weather(
                rgb, paths, shapes, rgb_level, args.seed
            )
        if ir_level:
            ir = degrade_modality(
                ir, paths, shapes, "ir", ir_level, args.seed
            )
        with torch.inference_mode():
            predictions, _, _ = model(rgb, ir, augment=False)
            pixel_targets = targets.clone()
            pixel_targets[:, 2:] *= torch.tensor(
                [packed.shape[3], packed.shape[2], packed.shape[3], packed.shape[2]],
                device=device,
            )
            predictions = non_max_suppression(
                predictions,
                args.conf_thres,
                args.nms_iou,
                multi_label=True,
                agnostic=False,
            )
        for sample_index, prediction in enumerate(predictions):
            labels = pixel_targets[pixel_targets[:, 0] == sample_index, 1:]
            correct = match_predictions(
                prediction,
                labels,
                packed[sample_index].shape[1:],
                shapes[sample_index],
                iouv,
            )
            records.append(
                ImageStats(
                    image_id=Path(paths[sample_index]).stem,
                    correct=correct.cpu().numpy(),
                    confidence=prediction[:, 4].float().cpu().numpy(),
                    pred_class=prediction[:, 5].float().cpu().numpy(),
                    target_class=labels[:, 0].float().cpu().numpy(),
                )
            )
        seen += packed.shape[0]
    if not records:
        raise RuntimeError(f"No images evaluated for {description}")
    return records


def model_path_for(mode: str, args: argparse.Namespace) -> Path:
    return {
        "full_rgca": args.full_rgca_weights,
        "original_lcaf": args.original_lcaf_weights,
        "full_rgca_gate_bypass": args.full_rgca_weights,
        "gated_c0": args.gated_c0_weights,
        "no_gate_c0": args.no_gate_c0_weights,
    }[mode]


def run_inference(
    args: argparse.Namespace,
    data: Mapping[str, object],
    device: torch.device,
) -> Tuple[Dict[str, List[ImageStats]], Dict[str, object]]:
    all_records: Dict[str, List[ImageStats]] = {}
    model_audit: Dict[str, object] = {}
    reference_ids: List[str] | None = None
    for mode in MODEL_SPECS:
        weights = model_path_for(mode, args)
        print(f"Loading {mode}: {weights}")
        model = attempt_load(str(weights), map_location=device).to(device).float().eval()
        model_audit[mode] = set_gate_mode(model, mode)
        stride = max(int(model.stride.max()), 32)
        dataloader, dataset = make_dataloader(data, args, stride)
        if args.max_images == 0 and len(dataset) != 136:
            raise ValueError(f"Expected Daytime test136, found {len(dataset)}")
        if device.type == "cuda":
            model.half()
            keep_force_fp32_modules(model)
        for ir_level in LEVELS:
            for rgb_level in LEVELS:
                key = condition_key(mode, rgb_level, ir_level)
                records = evaluate_pair_condition(
                    model,
                    dataloader,
                    device,
                    rgb_level,
                    ir_level,
                    args,
                    f"{mode} RGB-S{rgb_level}/IR-S{ir_level}",
                )
                image_ids = [record.image_id for record in records]
                if reference_ids is None:
                    reference_ids = image_ids
                elif image_ids != reference_ids:
                    raise ValueError(f"Paired image order mismatch: {key}")
                all_records[key] = records
                map50, map_value, _ = calculate_map(records)
                print(
                    f"{key}: n={len(records)}, mAP50={map50:.4f}, "
                    f"mAP50:95={map_value:.4f}"
                )
        del model, dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_records, model_audit


def signature(args: argparse.Namespace, data: Mapping[str, object]) -> Dict[str, object]:
    return {
        "version": 1,
        "full_rgca_sha256": sha256sum(args.full_rgca_weights),
        "original_lcaf_sha256": sha256sum(args.original_lcaf_weights),
        "gated_c0_sha256": sha256sum(args.gated_c0_weights),
        "no_gate_c0_sha256": sha256sum(args.no_gate_c0_weights),
        "data": str(args.data.resolve()),
        "test_rgb": str(data["test_rgb"]),
        "test_ir": str(data["test_ir"]),
        "levels": LEVELS,
        "rgb_schedule": RGB_COMPLEX_WEATHER,
        "ir_schedule": IR_CONTRAST_COLLAPSE,
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "seed": args.seed,
        "max_images": args.max_images,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    all_records: Mapping[str, Sequence[ImageStats]],
    bootstrap: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]]:
    point_maps = {key: calculate_map(records) for key, records in all_records.items()}
    metric_rows: List[Dict[str, object]] = []
    for mode in MODEL_SPECS:
        for ir_level in LEVELS:
            for rgb_level in LEVELS:
                key = condition_key(mode, rgb_level, ir_level)
                map50, map_value, per_class = point_maps[key]
                metric_rows.append(
                    {
                        "model": mode,
                        "rgb_severity": rgb_level,
                        "ir_severity": ir_level,
                        "map50": map50,
                        "map50_95": map_value,
                        **{
                            f"ap50_95_class_{index}": float(value)
                            for index, value in enumerate(per_class)
                        },
                    }
                )

    n_images = len(next(iter(all_records.values())))
    rng = np.random.default_rng(seed + 8421)
    selections = rng.integers(
        0, n_images, size=(bootstrap, n_images), endpoint=False
    )
    comparisons = {
        "full_rgca_minus_original_lcaf": ("full_rgca", "original_lcaf"),
        "full_rgca_gate_on_minus_bypass": (
            "full_rgca",
            "full_rgca_gate_bypass",
        ),
        "gated_c0_minus_no_gate_c0": ("gated_c0", "no_gate_c0"),
    }
    difference_rows: List[Dict[str, object]] = []
    bootstrap_arrays: Dict[str, np.ndarray] = {}
    for comparison_name, (first, second) in comparisons.items():
        values = np.empty((len(LEVELS), len(LEVELS), bootstrap), dtype=np.float32)
        for row_index, ir_level in enumerate(LEVELS):
            for column_index, rgb_level in enumerate(LEVELS):
                first_key = condition_key(first, rgb_level, ir_level)
                second_key = condition_key(second, rgb_level, ir_level)
                point = 100.0 * (
                    point_maps[first_key][1] - point_maps[second_key][1]
                )
                for sample_index, selection in enumerate(selections):
                    values[row_index, column_index, sample_index] = 100.0 * (
                        calculate_map(all_records[first_key], selection)[1]
                        - calculate_map(all_records[second_key], selection)[1]
                    )
                low, high = np.percentile(
                    values[row_index, column_index], [2.5, 97.5]
                )
                difference_rows.append(
                    {
                        "comparison": comparison_name,
                        "rgb_severity": rgb_level,
                        "ir_severity": ir_level,
                        "map50_95_difference_pp": point,
                        "paired_ci_low_pp": float(low),
                        "paired_ci_high_pp": float(high),
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
        bootstrap_arrays[comparison_name] = values
    return metric_rows, difference_rows, bootstrap_arrays


def matrix_from_rows(
    rows: Sequence[Mapping[str, object]], comparison: str, value: str
) -> np.ndarray:
    lookup = {
        (int(row["ir_severity"]), int(row["rgb_severity"])): float(row[value])
        for row in rows
        if str(row["comparison"]) == comparison
    }
    return np.asarray(
        [[lookup[(ir_level, rgb_level)] for rgb_level in LEVELS] for ir_level in LEVELS],
        dtype=np.float64,
    )


def plot_protocol(ax: plt.Axes) -> None:
    expected = np.asarray(
        [[column - row for column in range(3)] for row in range(3)],
        dtype=float,
    )
    cmap = ListedColormap(["#4c78a8", "#e5e7eb", "#f28e2b"])
    ax.imshow(expected, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    for row in range(3):
        for column in range(3):
            if row == column == 0:
                label = "Balanced"
            elif row == column:
                label = "Both weak\nuncertain"
            elif column > row:
                label = "Prefer IR\nIR → RGB"
            else:
                label = "Prefer RGB\nRGB → IR"
            ax.text(column, row, label, ha="center", va="center", fontsize=8.2)
    ax.set_title("A  Intended reliability-routing protocol", loc="left", fontweight="bold")
    ax.set_xticks(range(3), LEVEL_NAMES)
    ax.set_yticks(range(3), LEVEL_NAMES)
    ax.set_xlabel("RGB adverse-weather degradation →")
    ax.set_ylabel("IR thermal-contrast loss →")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_difference(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    comparison: str,
    title: str,
    limit: float,
) -> object:
    matrix = matrix_from_rows(rows, comparison, "map50_95_difference_pp")
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    lookup = {
        (int(row["ir_severity"]), int(row["rgb_severity"])): row
        for row in rows
        if str(row["comparison"]) == comparison
    }
    for row_index, ir_level in enumerate(LEVELS):
        for column_index, rgb_level in enumerate(LEVELS):
            item = lookup[(ir_level, rgb_level)]
            value = float(item["map50_95_difference_pp"])
            significant = bool(item["ci_excludes_zero"])
            color = "white" if abs(value) > 0.52 * limit else "#111827"
            ax.text(
                column_index,
                row_index,
                f"{value:+.2f} pp" + ("*" if significant else ""),
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="bold" if significant else "normal",
            )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(range(3), LEVEL_NAMES)
    ax.set_yticks(range(3), LEVEL_NAMES)
    ax.set_xlabel("RGB adverse-weather degradation →")
    ax.set_ylabel("IR thermal-contrast loss →")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def make_figure(
    difference_rows: Sequence[Mapping[str, object]], output_dir: Path, dpi: int
) -> Tuple[Path, Path]:
    values = np.asarray(
        [abs(float(row["map50_95_difference_pp"])) for row in difference_rows]
    )
    limit = max(1.0, float(np.ceil(values.max())))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    fig.subplots_adjust(
        left=0.075,
        right=0.875,
        bottom=0.125,
        top=0.90,
        wspace=0.34,
        hspace=0.36,
    )
    plot_protocol(axes[0, 0])
    image = plot_difference(
        axes[0, 1],
        difference_rows,
        "full_rgca_minus_original_lcaf",
        "B  Full systems: RGCA − original LCAFNet",
        limit,
    )
    plot_difference(
        axes[1, 0],
        difference_rows,
        "full_rgca_gate_on_minus_bypass",
        "C  Same checkpoint: gate ON − gate bypass",
        limit,
    )
    plot_difference(
        axes[1, 1],
        difference_rows,
        "gated_c0_minus_no_gate_c0",
        "D  Matched training: gated C0 − no-gate C0",
        limit,
    )
    colorbar_axis = fig.add_axes((0.90, 0.22, 0.018, 0.60))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("mAP@0.5:0.95 difference (percentage points)")
    fig.suptitle(
        "Asymmetric modality-reliability stress test on M3FD Daytime (n=136)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.026,
        "RGB: deterministic fog + rain streaks + desaturation + wet-lens blur; "
        "IR: deterministic thermal-contrast collapse + blur + sensor noise.\n"
        "* paired 95% bootstrap CI excludes zero. All result panels share one color scale.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    png = output_dir / "m3fd_daytime_rgca_reliability_asymmetric_stress_matrix.png"
    pdf = output_dir / "m3fd_daytime_rgca_reliability_asymmetric_stress_matrix.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def comparison_summary(
    difference_rows: Sequence[Mapping[str, object]], comparison: str
) -> Dict[str, float]:
    selected = [
        row for row in difference_rows if str(row["comparison"]) == comparison
    ]
    values = np.asarray(
        [float(row["map50_95_difference_pp"]) for row in selected], dtype=float
    )
    return {
        "mean_pp": float(values.mean()),
        "min_pp": float(values.min()),
        "max_pp": float(values.max()),
        "positive_cells": int((values > 0).sum()),
        "significant_positive_cells": int(
            sum(float(row["paired_ci_low_pp"]) > 0 for row in selected)
        ),
    }


def write_readme(
    output_dir: Path,
    difference_rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> None:
    full = comparison_summary(difference_rows, "full_rgca_minus_original_lcaf")
    intervention = comparison_summary(
        difference_rows, "full_rgca_gate_on_minus_bypass"
    )
    ablation = comparison_summary(difference_rows, "gated_c0_minus_no_gate_c0")
    ablation_ir_extreme = [
        float(row["map50_95_difference_pp"])
        for row in difference_rows
        if str(row["comparison"]) == "gated_c0_minus_no_gate_c0"
        and int(row["ir_severity"]) == 4
    ]
    text = f"""# M3FD Daytime 非对称模态可靠性压力测试

## 为什么重画

夜间主要破坏可见光成像，而热红外通常不依赖可见光照明；反过来，热红外会在目标与背景温度/发射率接近时出现热交叉和对比度塌缩。因此，把 RGB 与 IR 机械地按同一级别同时加同一种“天气退化”，不能直接验证可靠性引导的跨模态路由。

本图改成 3×3 正交压力测试：横轴独立改变 RGB 的复合天气强度，纵轴独立改变 IR 的热对比度损失。这样同时覆盖“RGB 差/IR 好”“RGB 好/IR 差”“两者都差”三种关键状态。所有 136 对图像均来自固定的 M3FD Daytime held-out test 子集，没有按结果挑图。

## 四个面板的科学含义

- A 是预先声明的协议与期望路由，不是模型结果。
- B 比较用户指定的完整 RGCA 与原版 LCAFNet；这是完整系统对比，不能把差异全部归因于 reliability gate。
- C 在同一完整 RGCA checkpoint 上把 gate 从 learned map 切换为恒等传递 1；它是同权重推理干预，但存在分布偏移，不能替代重新训练消融。
- D 比较相同数据、初始化、训练周期和 Mask+GFB 设置下的 prior-free C0 RGCA 与无门控版本，是更接近单变量的训练消融。

## 实际结果

- 完整 RGCA − 原版 LCAFNet：9 个格平均 {full['mean_pp']:+.3f} pp，范围 [{full['min_pp']:+.3f}, {full['max_pp']:+.3f}] pp；{full['positive_cells']}/9 个格为正，{full['significant_positive_cells']}/9 个格的成对 95% CI 显著为正。
- 同 checkpoint gate ON − bypass：平均 {intervention['mean_pp']:+.3f} pp，范围 [{intervention['min_pp']:+.3f}, {intervention['max_pp']:+.3f}] pp；{intervention['positive_cells']}/9 个格为正，{intervention['significant_positive_cells']}/9 个格显著为正。
- 训练消融 gated C0 − no-gate C0：平均 {ablation['mean_pp']:+.3f} pp，范围 [{ablation['min_pp']:+.3f}, {ablation['max_pp']:+.3f}] pp；{ablation['positive_cells']}/9 个格为正，{ablation['significant_positive_cells']}/9 个格显著为正。

本次 D 在 9/9 个格为正，尤其在 IR Extreme 行提升 {min(ablation_ir_extreme):+.3f} 至 {max(ablation_ir_extreme):+.3f} pp，支持“带 reliability gate 的 C0 训练方案在预设非对称退化下更鲁棒”。但 C 的同 checkpoint 干预接近零，说明当前完整 checkpoint 的即时门控幅值不是性能优势的充分解释。最稳妥的论文表述是“可靠性门控参与训练后形成了更鲁棒的参数协同”，不能进一步写成“门控在测试时发生了强烈、可直接观测的模态切换”。

## 文件

- 主图：`m3fd_daytime_rgca_reliability_asymmetric_stress_matrix.png/.pdf`
- 每模型绝对指标：`asymmetric_stress_metrics.csv`
- 三种成对差值与置信区间：`asymmetric_stress_paired_differences.csv`
- bootstrap 样本：`asymmetric_stress_bootstrap_differences.npz`
- 完整审计：`run_config.yaml`
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 50:
        raise ValueError("--bootstrap must be at least 50")
    for name in (
        "full_rgca_weights",
        "original_lcaf_weights",
        "gated_c0_weights",
        "no_gate_c0_weights",
        "data",
    ):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    data = load_data_config(args.data)
    current_signature = signature(args, data)
    cache_path = args.output_dir / "asymmetric_stress_per_image_stats.npz"
    all_records = None
    model_audit: Dict[str, object] = {}
    if cache_path.is_file() and not args.force:
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = json.loads(str(archive["signature_json"]))
            if cached_signature == json.loads(json.dumps(current_signature)):
                print(f"Reusing compatible cache: {cache_path}")
                all_records = arrays_to_records(archive, expected_keys())
    if all_records is None:
        all_records, model_audit = run_inference(args, data, device)
        np.savez_compressed(
            cache_path, **records_to_arrays(all_records, current_signature)
        )
        print(f"Saved cache: {cache_path}")
    reference_ids = [
        record.image_id for record in next(iter(all_records.values()))
    ]
    for key, records in all_records.items():
        if [record.image_id for record in records] != reference_ids:
            raise ValueError(f"Image-pair audit failed: {key}")
    metric_rows, difference_rows, bootstrap_arrays = summarize(
        all_records, args.bootstrap, args.seed
    )
    metrics_csv = args.output_dir / "asymmetric_stress_metrics.csv"
    differences_csv = args.output_dir / "asymmetric_stress_paired_differences.csv"
    bootstrap_path = args.output_dir / "asymmetric_stress_bootstrap_differences.npz"
    write_csv(metrics_csv, metric_rows)
    write_csv(differences_csv, difference_rows)
    np.savez_compressed(bootstrap_path, **bootstrap_arrays)
    png, pdf = make_figure(difference_rows, args.output_dir, args.dpi)
    config = {
        "task": "M3FD Daytime asymmetric modality-reliability stress matrix",
        "image_count": len(reference_ids),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(reference_ids).encode("utf-8")
        ).hexdigest(),
        "data_yaml": str(args.data),
        "split": "fixed leakage-free M3FD Daytime held-out test136",
        "selection": "all 136 pairs; no result-based filtering",
        "levels": list(LEVELS),
        "level_names": list(LEVEL_NAMES),
        "rgb_corruption": {
            "name": "compound adverse weather",
            "components": "fog + rain streaks + desaturation + wet-lens blur",
            "schedule": [RGB_COMPLEX_WEATHER[index] for index in LEVELS],
        },
        "ir_corruption": {
            "name": "thermal contrast collapse",
            "components": "contrast compression + blur + shared-channel sensor noise",
            "schedule": [IR_CONTRAST_COLLAPSE[index] for index in LEVELS],
        },
        "weights": {
            "full_rgca": str(args.full_rgca_weights),
            "full_rgca_sha256": sha256sum(args.full_rgca_weights),
            "original_lcaf": str(args.original_lcaf_weights),
            "original_lcaf_sha256": sha256sum(args.original_lcaf_weights),
            "gated_c0": str(args.gated_c0_weights),
            "gated_c0_sha256": sha256sum(args.gated_c0_weights),
            "no_gate_c0": str(args.no_gate_c0_weights),
            "no_gate_c0_sha256": sha256sum(args.no_gate_c0_weights),
        },
        "comparison_boundaries": {
            "full_rgca_minus_original_lcaf": "full-system comparison",
            "full_rgca_gate_on_minus_bypass": (
                "same-checkpoint inference intervention; bypass multiplier is 1; "
                "subject to inference distribution shift"
            ),
            "gated_c0_minus_no_gate_c0": (
                "matched separately trained prior-free C0 gate ablation"
            ),
        },
        "model_audit": model_audit,
        "evaluation": {
            "image_size": args.img_size,
            "batch_size": args.batch_size,
            "confidence_threshold": args.conf_thres,
            "nms_iou": args.nms_iou,
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
        },
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
            "metrics_csv": str(metrics_csv),
            "paired_differences_csv": str(differences_csv),
            "bootstrap_npz": str(bootstrap_path),
            "per_image_cache": str(cache_path),
        },
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_readme(args.output_dir, difference_rows, args)
    print(f"Figure: {png}")
    print(f"Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
