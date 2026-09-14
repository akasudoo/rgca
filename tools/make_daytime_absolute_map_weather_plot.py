#!/usr/bin/env python3
"""Plot absolute mAP for all Daytime test136 pairs without sample selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_modality_degradation_retention import (  # noqa: E402
    ImageStats,
    arrays_to_records,
    calculate_map,
    expected_keys,
)
from tools.make_daytime_complex_weather_retention import (  # noqa: E402
    read_manifest_ids,
    subset_records,
    weather_keys,
)


MODALITIES = ("rgb_weather", "ir", "joint_weather")
SEVERITY_NAMES = ("Clean", "Mild", "Moderate", "Severe", "Extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daytime-rgb-manifest",
        type=Path,
        default=Path(
            "/home/dell/lcp/dataset/M3FD_SceneSplit_WACV2024/manifests/"
            "original_m3fd_split/daytime_test_vis.txt"
        ),
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_modality_degradation_m3fd/per_image_detection_stats.npz",
    )
    parser.add_argument(
        "--weather-cache",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_complex_weather_m3fd/complex_weather_detection_stats.npz",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / (
            "paper/rgca_lcaf_complex_weather_m3fd_daytime_test136/"
            "daytime_complex_weather_degradation_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/rgca_lcaf_complex_weather_m3fd_daytime_test136",
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_daytime_records(args: argparse.Namespace) -> Tuple[Dict[str, List[ImageStats]], List[str]]:
    ids = read_manifest_ids(args.daytime_rgb_manifest)
    if len(ids) != 136:
        raise ValueError(f"Expected Daytime test136, found {len(ids)}")
    with np.load(args.base_cache, allow_pickle=False) as archive:
        base = arrays_to_records(archive, expected_keys())
    with np.load(args.weather_cache, allow_pickle=False) as archive:
        weather = arrays_to_records(archive, weather_keys())
    output: Dict[str, List[ImageStats]] = {}
    for model in ("lcaf", "rgca"):
        clean = f"{model}:clean:0"
        output[clean] = subset_records(base[clean], ids, clean)
        for severity in range(1, 5):
            ir_key = f"{model}:ir:{severity}"
            output[ir_key] = subset_records(base[ir_key], ids, ir_key)
            for modality in ("rgb_weather", "joint_weather"):
                key = f"{model}:{modality}:{severity}"
                output[key] = subset_records(weather[key], ids, key)
    return output, ids


def absolute_bootstrap(
    records: Mapping[str, Sequence[ImageStats]], bootstrap: int, seed: int
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    n_images = len(next(iter(records.values())))
    if any(len(value) != n_images for value in records.values()):
        raise ValueError("Condition image counts differ")
    rng = np.random.default_rng(seed + 1907)
    selections = rng.integers(0, n_images, size=(bootstrap, n_images), endpoint=False)
    points = {key: calculate_map(value)[1] for key, value in records.items()}
    samples: Dict[str, np.ndarray] = {}
    for key, value in records.items():
        sample = np.empty(bootstrap, dtype=np.float64)
        for index, selection in enumerate(selections):
            sample[index] = calculate_map(value, selection)[1]
        samples[key] = sample
    return points, samples


def condition_key(model: str, modality: str, severity: int) -> str:
    return f"{model}:clean:0" if severity == 0 else f"{model}:{modality}:{severity}"


def plot_absolute_map(
    points: Mapping[str, float],
    samples: Mapping[str, np.ndarray],
    output_dir: Path,
    dpi: int,
) -> Tuple[Path, Path, List[Dict[str, object]]]:
    colours = {"lcaf": "#D55E00", "rgca": "#0072B2"}
    markers = {"lcaf": "s", "rgca": "o"}
    labels = {"lcaf": "LCAF attention", "rgca": "RGCA (ours)"}
    titles = {
        "rgb_weather": "RGB compound adverse weather",
        "ir": "IR contrast collapse + sensor noise",
        "joint_weather": "RGB weather + IR degradation",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18.9, 5.15), sharey=True)
    difference_rows: List[Dict[str, object]] = []
    all_upper = []
    for modality in MODALITIES:
        for model in ("lcaf", "rgca"):
            for severity in range(5):
                key = condition_key(model, modality, severity)
                all_upper.append(np.percentile(samples[key], 97.5))
    y_max = max(0.55, 0.05 * np.ceil((max(all_upper) + 0.02) / 0.05))
    for panel_index, (axis, modality) in enumerate(zip(axes, MODALITIES)):
        x = np.arange(5)
        for model in ("lcaf", "rgca"):
            keys = [condition_key(model, modality, severity) for severity in range(5)]
            y = np.asarray([points[key] for key in keys])
            boot = np.stack([samples[key] for key in keys])
            low, high = np.percentile(boot, [2.5, 97.5], axis=1)
            axis.plot(
                x, y, color=colours[model], marker=markers[model], linewidth=2.25,
                markersize=6.5, label=labels[model], zorder=3,
            )
            axis.fill_between(x, low, high, color=colours[model], alpha=0.15, linewidth=0)
        for severity in range(5):
            rgca_key = condition_key("rgca", modality, severity)
            lcaf_key = condition_key("lcaf", modality, severity)
            diff_boot = samples[rgca_key] - samples[lcaf_key]
            ci = np.percentile(diff_boot, [2.5, 97.5])
            difference_rows.append({
                "condition": modality.upper(), "severity": severity,
                "severity_name": SEVERITY_NAMES[severity],
                "rgca_map50_95": points[rgca_key], "lcaf_map50_95": points[lcaf_key],
                "rgca_minus_lcaf_map50_95": points[rgca_key] - points[lcaf_key],
                "paired_ci_low": float(ci[0]), "paired_ci_high": float(ci[1]),
            })
        rgca_extreme = condition_key("rgca", modality, 4)
        lcaf_extreme = condition_key("lcaf", modality, 4)
        diff = samples[rgca_extreme] - samples[lcaf_extreme]
        ci = np.percentile(diff, [2.5, 97.5])
        point = points[rgca_extreme] - points[lcaf_extreme]
        axis.text(
            0.72, 0.20,
            f"Extreme: RGCA − LCAF = {point:+.3f}\npaired 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.92},
        )
        axis.set_xticks(x, SEVERITY_NAMES)
        axis.set_xlim(-0.15, 4.15)
        axis.set_ylim(0, y_max)
        axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.grid(False, axis="x")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Degradation severity")
        axis.legend(loc="lower left", frameon=True, fontsize=8.8)
        axis.set_title(f"{chr(65+panel_index)}  {titles[modality]}", loc="left", fontweight="bold")
    axes[0].set_ylabel("Absolute mAP@0.5:0.95")

    degraded_keys = {
        model: [f"{model}:{modality}:{severity}" for modality in MODALITIES for severity in range(1, 5)]
        for model in ("lcaf", "rgca")
    }
    mean_point = {
        model: np.mean([points[key] for key in keys]) for model, keys in degraded_keys.items()
    }
    mean_boot = {
        model: np.mean(np.stack([samples[key] for key in keys]), axis=0)
        for model, keys in degraded_keys.items()
    }
    mean_ci = np.percentile(mean_boot["rgca"] - mean_boot["lcaf"], [2.5, 97.5])
    fig.suptitle(
        "Daytime-only absolute detection performance under compound adverse weather",
        fontsize=14, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.915,
        f"Mean absolute mAP over 12 degraded conditions: RGCA {mean_point['rgca']:.3f} · "
        f"LCAF {mean_point['lcaf']:.3f} · paired Δ {mean_point['rgca']-mean_point['lcaf']:+.3f} "
        f"(95% CI [{mean_ci[0]:+.3f}, {mean_ci[1]:+.3f}])",
        ha="center", va="top", fontsize=9.2, color="#333333",
    )
    fig.text(
        0.5, 0.006,
        "M3FD Daytime-only held-out test (136 aligned pairs) · all samples retained · shaded: paired image-bootstrap 95% CI",
        ha="center", va="bottom", fontsize=8.8, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.995, 0.86), w_pad=2.0)
    stem = "m3fd_daytime_test136_rgca_vs_lcaf_complex_weather_absolute_map"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf, difference_rows


def main() -> None:
    args = parse_args()
    for path in (args.daytime_rgb_manifest, args.base_cache, args.weather_cache, args.metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, ids = load_daytime_records(args)
    print(f"Using all {len(ids)} Daytime pairs; no result-based selection")
    print(f"Computing {args.bootstrap} absolute-mAP bootstrap replicates...")
    points, samples = absolute_bootstrap(records, args.bootstrap, args.seed)
    png, pdf, difference_rows = plot_absolute_map(points, samples, args.output_dir, args.dpi)
    csv_path = args.output_dir / "daytime_paired_absolute_map_differences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(difference_rows[0].keys()))
        writer.writeheader()
        writer.writerows(difference_rows)
    np.savez_compressed(
        args.output_dir / "daytime_absolute_map_bootstrap_samples.npz",
        **{key.replace(":", "__"): value for key, value in samples.items()},
    )
    audit = {
        "selection": "all 136 Daytime test pairs; no performance-based filtering",
        "image_count": len(ids),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "metric": "absolute mAP@0.5:0.95",
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "paired_differences": str(csv_path.resolve()),
    }
    (args.output_dir / "absolute_map_plot_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved PNG: {png}")
    print(f"Saved PDF: {pdf}")


if __name__ == "__main__":
    main()
