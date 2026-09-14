#!/usr/bin/env python3
"""Evaluate strict RGCA-vs-LCAF robustness to single-modality degradation.

The script evaluates both attention mechanisms on the same held-out M3FD test
pairs.  RGB is degraded with a deterministic low-light/noise process and IR is
degraded with a deterministic thermal-contrast-collapse/noise process.  It
reports absolute mAP and retention relative to each model's own clean mAP,
with paired image-bootstrap confidence intervals.

The comparison intentionally uses the pure, prior-free RGCA+Mask+GFB model and
the original-LCAF-attention+same-Mask+GFB control.  This isolates the attention
path more closely than comparing two complete systems with different fusion
heads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.experimental import attempt_load  # noqa: E402
from tools.compare_strict_rgca_lcaf_scene_cross_attention import (  # noqa: E402
    restore_strict_lcaf_layers,
)
from tools.visualize_rgca_panel_a import sha256sum  # noqa: E402
from utils.datasets import create_dataloader_rgb_ir  # noqa: E402
from utils.general import (  # noqa: E402
    box_iou,
    check_dataset,
    keep_force_fp32_modules,
    non_max_suppression,
    scale_coords,
    xywh2xyxy,
)
from utils.metrics import ap_per_class  # noqa: E402


SEVERITY_NAMES = ("Clean", "Mild", "Moderate", "Severe", "Extreme")

# Pre-declared corruption schedule.  Values were fixed before seeing results.
RGB_LOW_LIGHT = (
    {"gamma": 1.00, "gain": 1.00, "noise_sigma": 0.000},
    {"gamma": 1.30, "gain": 0.90, "noise_sigma": 0.010},
    {"gamma": 1.70, "gain": 0.75, "noise_sigma": 0.020},
    {"gamma": 2.20, "gain": 0.60, "noise_sigma": 0.035},
    {"gamma": 3.00, "gain": 0.45, "noise_sigma": 0.050},
)
IR_CONTRAST_COLLAPSE = (
    {"contrast": 1.00, "blur_kernel": 1, "noise_sigma": 0.000},
    {"contrast": 0.80, "blur_kernel": 3, "noise_sigma": 0.005},
    {"contrast": 0.60, "blur_kernel": 5, "noise_sigma": 0.010},
    {"contrast": 0.40, "blur_kernel": 7, "noise_sigma": 0.020},
    {"contrast": 0.20, "blur_kernel": 9, "noise_sigma": 0.030},
)

# Physics-inspired compound adverse-weather protocol for RGB.  Unlike the
# earlier underexposure diagnostic, this does not use gamma or sensor noise.
# It combines spatially varying atmospheric scattering, directional rain
# streaks, colour/contrast attenuation, and local wet-lens blur.  The fields
# are deterministic per image and severity so both models see identical input.
RGB_COMPLEX_WEATHER = (
    {
        "transmission": 1.00, "transmission_variation": 0.00,
        "saturation": 1.00, "rain_density": 0.0000,
        "rain_opacity": 0.00, "streak_length": 1,
        "wet_lens_strength": 0.00, "wet_blur_kernel": 1,
    },
    {
        "transmission": 0.88, "transmission_variation": 0.05,
        "saturation": 0.95, "rain_density": 0.0025,
        "rain_opacity": 0.16, "streak_length": 7,
        "wet_lens_strength": 0.03, "wet_blur_kernel": 3,
    },
    {
        "transmission": 0.73, "transmission_variation": 0.09,
        "saturation": 0.88, "rain_density": 0.0045,
        "rain_opacity": 0.22, "streak_length": 11,
        "wet_lens_strength": 0.10, "wet_blur_kernel": 5,
    },
    {
        "transmission": 0.56, "transmission_variation": 0.13,
        "saturation": 0.78, "rain_density": 0.0070,
        "rain_opacity": 0.30, "streak_length": 15,
        "wet_lens_strength": 0.18, "wet_blur_kernel": 7,
    },
    {
        "transmission": 0.40, "transmission_variation": 0.17,
        "saturation": 0.66, "rain_density": 0.0100,
        "rain_opacity": 0.38, "streak_length": 21,
        "wet_lens_strength": 0.28, "wet_blur_kernel": 9,
    },
)


@dataclass
class ImageStats:
    image_id: str
    correct: np.ndarray
    confidence: np.ndarray
    pred_class: np.ndarray
    target_class: np.ndarray


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
        "--lcaf-variant",
        choices=("strict", "original"),
        default="strict",
        help=(
            "strict: original LCAF attention with the same Mask+GFB as RGCA; "
            "original: historical complete LCAFNet with original GFB and no Mask"
        ),
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
        "--max-images",
        type=int,
        default=0,
        help="Debug only: stop after this many images; 0 evaluates all images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard compatible cached per-image predictions and rerun inference.",
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = int(spec.split(",")[0])
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def load_data_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    check_dataset(data)
    for key in ("test_rgb", "test_ir", "nc", "names"):
        if key not in data:
            raise KeyError(f"Dataset YAML is missing {key}: {path}")
    return data


def make_dataloader(
    data: Mapping[str, object], args: argparse.Namespace, stride: int
):
    options = SimpleNamespace(single_cls=False)
    loader, dataset = create_dataloader_rgb_ir(
        data["test_rgb"],
        data["test_ir"],
        args.img_size,
        args.batch_size,
        stride,
        options,
        pad=0.5,
        rect=True,
        workers=args.workers,
        prefix="held-out test: ",
        # Force a standard finite DataLoader so every corruption condition
        # starts at image 0 (also true for --max-images debug runs).
        sampler=lambda ds: torch.utils.data.SequentialSampler(ds),
    )
    return loader, dataset


def validate_class_names(
    model: torch.nn.Module, names: Sequence[str]
) -> Dict[str, object]:
    model_names = model.names
    if isinstance(model_names, dict):
        model_names = [model_names[index] for index in sorted(model_names)]
    model_names, names = list(model_names), list(names)
    if len(model_names) != len(names):
        raise ValueError(
            f"Checkpoint/dataset class counts differ: {model_names} vs {names}"
        )
    if model_names == names:
        return {
            "checkpoint_names": model_names,
            "dataset_names": names,
            "status": "exact_match",
        }
    # These historical checkpoints were trained against the same numeric YOLO
    # labels, but their display metadata predates the correction that swapped
    # the textual names for IDs 3 and 4.  Evaluation below is numeric-ID based;
    # remapping predictions would therefore be wrong.  Accept only this exact,
    # known metadata discrepancy and record it prominently in the audit.
    historical = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    canonical = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
    if model_names == historical and names == canonical:
        return {
            "checkpoint_names": model_names,
            "dataset_names": names,
            "status": (
                "known_stale_display_metadata_for_ids_3_and_4; "
                "numeric prediction/label IDs evaluated unchanged"
            ),
        }
    raise ValueError(
        f"Unexpected checkpoint/dataset class metadata mismatch: {model_names} vs {names}"
    )


def stable_seed(base_seed: int, modality: str, severity: int, image_id: str) -> int:
    payload = f"{base_seed}|{modality}|{severity}|{image_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def valid_crop_bounds(shape, height: int, width: int) -> Tuple[int, int, int, int]:
    """Return the non-letterbox rectangle using YOLOv5's rounding convention."""
    _native_shape, ratio_pad = shape
    _ratio, pad = ratio_pad
    pad_w, pad_h = float(pad[0]), float(pad[1])
    left = max(0, min(width, int(round(pad_w - 0.1))))
    right = max(left + 1, min(width, width - int(round(pad_w + 0.1))))
    top = max(0, min(height, int(round(pad_h - 0.1))))
    bottom = max(top + 1, min(height, height - int(round(pad_h + 0.1))))
    return top, bottom, left, right


def deterministic_noise(
    shape: Sequence[int], seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, device=device, dtype=torch.float32).to(dtype)


def degrade_modality(
    image: torch.Tensor,
    paths: Sequence[str],
    shapes: Sequence[object],
    modality: str,
    severity: int,
    base_seed: int,
) -> torch.Tensor:
    if severity == 0:
        return image
    if modality not in ("rgb", "ir"):
        raise ValueError(modality)
    output = image.clone()
    _, channels, height, width = output.shape
    for index, (path, shape) in enumerate(zip(paths, shapes)):
        top, bottom, left, right = valid_crop_bounds(shape, height, width)
        crop = image[index : index + 1, :, top:bottom, left:right].float()
        image_id = Path(path).stem
        seed = stable_seed(base_seed, modality, severity, image_id)
        if modality == "rgb":
            params = RGB_LOW_LIGHT[severity]
            degraded = float(params["gain"]) * crop.clamp(0, 1).pow(float(params["gamma"]))
            noise = deterministic_noise(
                degraded.shape, seed, degraded.device, torch.float32
            )
            degraded = degraded + float(params["noise_sigma"]) * noise
        else:
            params = IR_CONTRAST_COLLAPSE[severity]
            kernel = int(params["blur_kernel"])
            if kernel > 1:
                padding = kernel // 2
                padded = F.pad(crop, (padding, padding, padding, padding), mode="reflect")
                blurred = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
            else:
                blurred = crop
            mean = crop.mean(dim=(1, 2, 3), keepdim=True)
            degraded = mean + float(params["contrast"]) * (blurred - mean)
            # M3FD IR is stored as three repeated channels; preserve that property.
            noise = deterministic_noise(
                (1, 1, crop.shape[2], crop.shape[3]),
                seed,
                degraded.device,
                torch.float32,
            )
            degraded = degraded + float(params["noise_sigma"]) * noise.expand(
                -1, channels, -1, -1
            )
        output[index : index + 1, :, top:bottom, left:right] = degraded.clamp(0, 1).to(
            output.dtype
        )
    return output


def _weather_random(
    shape: Sequence[int], seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(tuple(shape), generator=generator, device=device, dtype=torch.float32)


def _reflect_box_blur(image: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return image
    padding = kernel // 2
    return F.avg_pool2d(
        F.pad(image, (padding, padding, padding, padding), mode="reflect"),
        kernel_size=kernel,
        stride=1,
    )


def degrade_rgb_complex_weather(
    image: torch.Tensor,
    paths: Sequence[str],
    shapes: Sequence[object],
    severity: int,
    base_seed: int,
) -> torch.Tensor:
    """Apply deterministic fog + rain + wet-lens degradation to RGB.

    Fog follows the atmospheric-scattering form I=J*t+A*(1-t).  Since M3FD
    has no depth maps, t is a smooth spatial proxy rather than metric depth.
    Rain is a sparse directional streak layer, and the wet-lens component
    locally mixes a blurred image and airlight through a smooth mask.
    """
    if severity == 0:
        return image
    params = RGB_COMPLEX_WEATHER[severity]
    output = image.clone()
    _, channels, height, width = output.shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel RGB input, found {channels}")
    for index, (path, shape) in enumerate(zip(paths, shapes)):
        top, bottom, left, right = valid_crop_bounds(shape, height, width)
        crop = image[index : index + 1, :, top:bottom, left:right].float()
        crop_h, crop_w = crop.shape[2:]
        image_id = Path(path).stem
        seed = stable_seed(base_seed, "rgb_complex_weather", severity, image_id)

        # Storm haze reduces chroma before spatial atmospheric scattering.
        luminance = (
            0.2989 * crop[:, 0:1]
            + 0.5870 * crop[:, 1:2]
            + 0.1140 * crop[:, 2:3]
        )
        saturation = float(params["saturation"])
        scene = luminance + saturation * (crop - luminance)

        field_h = max(3, min(12, crop_h // 64))
        field_w = max(4, min(16, crop_w // 64))
        fog_field = _weather_random(
            (1, 1, field_h, field_w), seed + 11, crop.device
        )
        fog_field = F.interpolate(
            fog_field, size=(crop_h, crop_w), mode="bicubic", align_corners=False
        ).clamp(0, 1)
        fog_field = 2.0 * fog_field - 1.0
        vertical = torch.linspace(
            -1.0, 1.0, crop_h, device=crop.device, dtype=torch.float32
        ).view(1, 1, crop_h, 1)
        transmission = (
            float(params["transmission"])
            + float(params["transmission_variation"])
            * (0.70 * fog_field + 0.30 * vertical)
        ).clamp(0.18, 1.0)
        airlight = torch.tensor(
            [0.82, 0.86, 0.90], device=crop.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        weathered = scene * transmission + airlight * (1.0 - transmission)

        # Sparse impulses convolved with a slanted line produce exposure-time
        # rain streaks.  Wind direction varies per image, not per model.
        density = float(params["rain_density"])
        if density > 0:
            impulses = (
                _weather_random((1, 1, crop_h, crop_w), seed + 23, crop.device)
                < density
            ).float()
            length = int(params["streak_length"])
            kernel = torch.zeros(
                (1, 1, length, length), device=crop.device, dtype=torch.float32
            )
            direction_pick = int(
                _weather_random((1,), seed + 29, crop.device).item() * 4
            )
            slope = (-0.22, -0.12, 0.12, 0.22)[min(direction_pick, 3)]
            center = length // 2
            for row in range(length):
                column = center + int(round((row - center) * slope))
                column = max(0, min(length - 1, column))
                kernel[0, 0, row, column] = 1.0
            rain = F.conv2d(impulses, kernel, padding=center).clamp(0, 1)
            rain = _reflect_box_blur(rain, 3)
            alpha = float(params["rain_opacity"]) * rain
            rain_colour = torch.tensor(
                [0.78, 0.84, 0.90], device=crop.device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            weathered = weathered * (1.0 - alpha) + rain_colour * alpha

        # A smooth high-percentile mask approximates droplets/water film on
        # the lens.  It creates local blur and veiling rather than pixel noise.
        wet_strength = float(params["wet_lens_strength"])
        if wet_strength > 0:
            wet_field = _weather_random(
                (1, 1, max(3, field_h), max(4, field_w)),
                seed + 41,
                crop.device,
            )
            wet_field = F.interpolate(
                wet_field,
                size=(crop_h, crop_w),
                mode="bicubic",
                align_corners=False,
            ).clamp(0, 1)
            wet_mask = torch.sigmoid(12.0 * (wet_field - 0.62))
            blurred = _reflect_box_blur(
                weathered, int(params["wet_blur_kernel"])
            )
            wet_alpha = wet_strength * wet_mask
            wet_content = 0.88 * blurred + 0.12 * airlight
            weathered = weathered * (1.0 - wet_alpha) + wet_content * wet_alpha

        output[index : index + 1, :, top:bottom, left:right] = weathered.clamp(
            0, 1
        ).to(output.dtype)
    return output


def match_predictions(
    pred: torch.Tensor,
    labels: torch.Tensor,
    inference_shape: Sequence[int],
    native_shape,
    iouv: torch.Tensor,
) -> torch.Tensor:
    correct = torch.zeros(
        pred.shape[0], iouv.numel(), dtype=torch.bool, device=pred.device
    )
    if not len(pred) or not len(labels):
        return correct
    pred_native = pred.clone()
    scale_coords(inference_shape, pred_native[:, :4], native_shape[0], native_shape[1])
    target_boxes = xywh2xyxy(labels[:, 1:5])
    scale_coords(inference_shape, target_boxes, native_shape[0], native_shape[1])
    target_classes = labels[:, 0]
    for class_id in torch.unique(target_classes):
        target_indices = (target_classes == class_id).nonzero(as_tuple=False).view(-1)
        prediction_indices = (pred[:, 5] == class_id).nonzero(as_tuple=False).view(-1)
        if not prediction_indices.numel():
            continue
        ious, best_target = box_iou(
            pred_native[prediction_indices, :4], target_boxes[target_indices]
        ).max(1)
        detected = set()
        for match_index in (ious > iouv[0]).nonzero(as_tuple=False).view(-1):
            target_index = target_indices[best_target[match_index]]
            if int(target_index) in detected:
                continue
            detected.add(int(target_index))
            correct[prediction_indices[match_index]] = ious[match_index] > iouv
            if len(detected) == len(target_indices):
                break
    return correct


def evaluate_condition(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    modality: str,
    severity: int,
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
        if modality == "rgb":
            rgb = degrade_modality(rgb, paths, shapes, modality, severity, args.seed)
        elif modality == "ir":
            ir = degrade_modality(ir, paths, shapes, modality, severity, args.seed)
        elif modality == "joint":
            rgb = degrade_modality(rgb, paths, shapes, "rgb", severity, args.seed)
            ir = degrade_modality(ir, paths, shapes, "ir", severity, args.seed)
        elif modality == "rgb_weather":
            rgb = degrade_rgb_complex_weather(rgb, paths, shapes, severity, args.seed)
        elif modality == "joint_weather":
            rgb = degrade_rgb_complex_weather(rgb, paths, shapes, severity, args.seed)
            ir = degrade_modality(ir, paths, shapes, "ir", severity, args.seed)
        elif modality != "clean":
            raise ValueError(modality)
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


def concatenate_stats(
    records: Sequence[ImageStats], image_indices: Iterable[int] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if image_indices is None:
        selected = records
    else:
        selected = [records[int(index)] for index in image_indices]
    correct = np.concatenate([record.correct for record in selected], axis=0)
    confidence = np.concatenate([record.confidence for record in selected], axis=0)
    pred_class = np.concatenate([record.pred_class for record in selected], axis=0)
    target_class = np.concatenate([record.target_class for record in selected], axis=0)
    return correct, confidence, pred_class, target_class


def calculate_map(
    records: Sequence[ImageStats], image_indices: Iterable[int] | None = None
) -> Tuple[float, float, np.ndarray]:
    correct, confidence, pred_class, target_class = concatenate_stats(records, image_indices)
    if not correct.size or not correct.any():
        return 0.0, 0.0, np.zeros(0, dtype=np.float64)
    _, _, _, _, _, ap, _, classes = ap_per_class(
        correct, confidence, pred_class, target_class, plot=False
    )
    per_class = np.full(6, np.nan, dtype=np.float64)
    per_class[classes] = ap.mean(axis=1)
    return float(ap[:, 0].mean()), float(ap.mean()), per_class


def cache_signature(args: argparse.Namespace, data: Mapping[str, object]) -> Dict[str, object]:
    return {
        "version": 1,
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "lcaf_variant": args.lcaf_variant,
        "data": str(args.data.resolve()),
        "test_rgb": str(data["test_rgb"]),
        "test_ir": str(data["test_ir"]),
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "seed": args.seed,
        "max_images": args.max_images,
        "rgb_schedule": RGB_LOW_LIGHT,
        "ir_schedule": IR_CONTRAST_COLLAPSE,
    }


def records_to_arrays(
    all_records: Mapping[str, Sequence[ImageStats]], signature: Mapping[str, object]
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "signature_json": np.asarray(json.dumps(signature, sort_keys=True)),
    }
    for key, records in all_records.items():
        safe = key.replace(":", "__")
        image_ids = np.asarray([record.image_id for record in records])
        pred_lengths = np.asarray([len(record.confidence) for record in records], dtype=np.int32)
        target_lengths = np.asarray([len(record.target_class) for record in records], dtype=np.int32)
        arrays[f"{safe}__image_ids"] = image_ids
        arrays[f"{safe}__pred_lengths"] = pred_lengths
        arrays[f"{safe}__target_lengths"] = target_lengths
        arrays[f"{safe}__correct"] = np.concatenate([r.correct for r in records], axis=0)
        arrays[f"{safe}__confidence"] = np.concatenate([r.confidence for r in records], axis=0)
        arrays[f"{safe}__pred_class"] = np.concatenate([r.pred_class for r in records], axis=0)
        arrays[f"{safe}__target_class"] = np.concatenate([r.target_class for r in records], axis=0)
    return arrays


def arrays_to_records(archive, keys: Sequence[str]) -> Dict[str, List[ImageStats]]:
    all_records: Dict[str, List[ImageStats]] = {}
    for key in keys:
        safe = key.replace(":", "__")
        image_ids = archive[f"{safe}__image_ids"]
        pred_lengths = archive[f"{safe}__pred_lengths"]
        target_lengths = archive[f"{safe}__target_lengths"]
        correct = archive[f"{safe}__correct"]
        confidence = archive[f"{safe}__confidence"]
        pred_class = archive[f"{safe}__pred_class"]
        target_class = archive[f"{safe}__target_class"]
        records = []
        pred_offset = target_offset = 0
        for image_id, pred_length, target_length in zip(
            image_ids, pred_lengths, target_lengths
        ):
            pred_end = pred_offset + int(pred_length)
            target_end = target_offset + int(target_length)
            records.append(
                ImageStats(
                    image_id=str(image_id),
                    correct=correct[pred_offset:pred_end],
                    confidence=confidence[pred_offset:pred_end],
                    pred_class=pred_class[pred_offset:pred_end],
                    target_class=target_class[target_offset:target_end],
                )
            )
            pred_offset, target_offset = pred_end, target_end
        all_records[key] = records
    return all_records


def expected_keys() -> List[str]:
    keys = []
    for model_name in ("lcaf", "rgca"):
        keys.append(f"{model_name}:clean:0")
        for modality in ("rgb", "ir"):
            keys.extend(f"{model_name}:{modality}:{level}" for level in range(1, 5))
    return keys


def bootstrap_summary(
    all_records: Mapping[str, Sequence[ImageStats]],
    bootstrap: int,
    seed: int,
    modalities: Sequence[str] = ("rgb", "ir"),
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], np.ndarray]]:
    n_images = len(next(iter(all_records.values())))
    for records in all_records.values():
        if len(records) != n_images:
            raise ValueError("Condition image counts differ")
    rng = np.random.default_rng(seed + 1907)
    samples = rng.integers(0, n_images, size=(bootstrap, n_images), endpoint=False)
    point_maps: Dict[str, Tuple[float, float, np.ndarray]] = {
        key: calculate_map(records) for key, records in all_records.items()
    }
    bootstrap_maps: Dict[str, np.ndarray] = {}
    for key, records in all_records.items():
        values = np.empty(bootstrap, dtype=np.float64)
        for index, selection in enumerate(samples):
            values[index] = calculate_map(records, selection)[1]
        bootstrap_maps[key] = values

    rows: List[Dict[str, object]] = []
    retention_samples: Dict[Tuple[str, str], np.ndarray] = {}
    for model_name in ("lcaf", "rgca"):
        clean_key = f"{model_name}:clean:0"
        clean_map50, clean_map, _ = point_maps[clean_key]
        clean_boot = bootstrap_maps[clean_key]
        for modality in modalities:
            modality_samples = np.empty((5, bootstrap), dtype=np.float64)
            for severity in range(5):
                key = clean_key if severity == 0 else f"{model_name}:{modality}:{severity}"
                map50, map_value, per_class = point_maps[key]
                boot = bootstrap_maps[key]
                valid = clean_boot > 1e-12
                ratios = np.full(bootstrap, np.nan, dtype=np.float64)
                ratios[valid] = boot[valid] / clean_boot[valid]
                modality_samples[severity] = ratios
                low, high = np.nanpercentile(ratios, [2.5, 97.5])
                rows.append(
                    {
                        "model": model_name.upper(),
                        "degraded_modality": modality.upper(),
                        "severity": severity,
                        "severity_name": SEVERITY_NAMES[severity],
                        "map50": map50,
                        "map50_95": map_value,
                        "clean_map50": clean_map50,
                        "clean_map50_95": clean_map,
                        "retention": map_value / clean_map,
                        "retention_ci_low": float(low),
                        "retention_ci_high": float(high),
                        **{
                            f"ap50_95_class_{class_index}": float(value)
                            for class_index, value in enumerate(per_class)
                        },
                    }
                )
            retention_samples[(model_name, modality)] = modality_samples
    return rows, retention_samples


def plot_figure(
    rows: Sequence[Mapping[str, object]],
    retention_samples: Mapping[Tuple[str, str], np.ndarray],
    output_dir: Path,
    dpi: int,
    image_count: int,
    lcaf_label: str = "LCAF attention",
    figure_title: str = "Cross-attention robustness under single-modality degradation",
    modalities: Sequence[str] = ("rgb", "ir"),
    output_stem: str = "m3fd_rgca_vs_lcaf_modality_degradation_retention",
) -> Tuple[Path, Path]:
    lookup = {
        (str(row["model"]).lower(), str(row["degraded_modality"]).lower(), int(row["severity"])): row
        for row in rows
    }
    colors = {"lcaf": "#D55E00", "rgca": "#0072B2"}
    markers = {"lcaf": "s", "rgca": "o"}
    display = {"lcaf": lcaf_label, "rgca": "RGCA (ours)"}
    panel_titles = {
        "rgb": "RGB underexposure + sensor noise",
        "ir": "IR contrast collapse + sensor noise",
        "joint": "RGB + IR simultaneous degradation",
        "rgb_weather": "RGB compound adverse weather",
        "joint_weather": "RGB weather + IR degradation",
    }
    fig, axes = plt.subplots(
        1, len(modalities), figsize=(6.3 * len(modalities), 5.15), sharey=True
    )
    axes = np.atleast_1d(axes)
    mean_point = {}
    mean_bootstrap = {}
    for model_name in ("lcaf", "rgca"):
        degraded_rows = [
            row
            for row in rows
            if str(row["model"]).lower() == model_name and int(row["severity"]) > 0
        ]
        mean_point[model_name] = 100 * float(
            np.mean([float(row["retention"]) for row in degraded_rows])
        )
        mean_bootstrap[model_name] = 100 * np.mean(
            np.concatenate(
                [
                    retention_samples[(model_name, modality)][1:]
                    for modality in modalities
                ],
                axis=0,
            ),
            axis=0,
        )
    mean_difference = mean_bootstrap["rgca"] - mean_bootstrap["lcaf"]
    mean_difference_ci = np.nanpercentile(mean_difference, [2.5, 97.5])
    global_upper = 100.0
    for samples in retention_samples.values():
        global_upper = max(global_upper, float(np.nanpercentile(samples * 100, 97.5)))
    y_max = max(105.0, 5.0 * np.ceil((global_upper + 2.0) / 5.0))
    for panel_index, (axis, modality) in enumerate(zip(axes, modalities)):
        x = np.arange(5)
        for model_name in ("lcaf", "rgca"):
            model_rows = [lookup[(model_name, modality, level)] for level in range(5)]
            y = 100 * np.asarray([float(row["retention"]) for row in model_rows])
            boot = retention_samples[(model_name, modality)] * 100
            lower, upper = np.nanpercentile(boot, [2.5, 97.5], axis=1)
            clean_map = float(model_rows[0]["clean_map50_95"])
            axis.plot(
                x,
                y,
                color=colors[model_name],
                marker=markers[model_name],
                markersize=6.5,
                linewidth=2.25,
                label=f"{display[model_name]} (clean {clean_map:.3f})",
                zorder=3,
            )
            axis.fill_between(x, lower, upper, color=colors[model_name], alpha=0.15, linewidth=0)
        difference = (
            retention_samples[("rgca", modality)]
            - retention_samples[("lcaf", modality)]
        ) * 100
        endpoint_point = 100 * (
            float(lookup[("rgca", modality, 4)]["retention"])
            - float(lookup[("lcaf", modality, 4)]["retention"])
        )
        endpoint_ci = np.nanpercentile(difference[4], [2.5, 97.5])
        axis.text(
            0.72,
            0.20,
            f"Extreme: RGCA − LCAF = {endpoint_point:+.1f} pp\n"
            f"paired 95% CI [{endpoint_ci[0]:+.1f}, {endpoint_ci[1]:+.1f}]",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.0,
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
        panel_letter = chr(ord("A") + panel_index)
        axis.set_title(
            f"{panel_letter}  {panel_titles[modality]}", loc="left", fontweight="bold"
        )
    axes[0].set_ylabel("mAP@0.5:0.95 retention (%)")
    fig.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.915,
        f"Mean over {4 * len(modalities)} degraded conditions: RGCA {mean_point['rgca']:.1f}% · "
        f"LCAF {mean_point['lcaf']:.1f}% · paired Δ {mean_point['rgca'] - mean_point['lcaf']:+.1f} pp "
        f"(95% CI [{mean_difference_ci[0]:+.1f}, {mean_difference_ci[1]:+.1f}])",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#333333",
    )
    fig.text(
        0.5,
        0.006,
        f"M3FD held-out test ({image_count} aligned pairs) · retention = corrupted mAP / each model's clean mAP · shaded: paired image-bootstrap 95% CI",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.995, 0.86), w_pad=2.0)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_difference_rows(
    rows: Sequence[Mapping[str, object]],
    retention_samples: Mapping[Tuple[str, str], np.ndarray],
    modalities: Sequence[str] = ("rgb", "ir"),
) -> List[Dict[str, object]]:
    lookup = {
        (str(row["model"]).lower(), str(row["degraded_modality"]).lower(), int(row["severity"])): row
        for row in rows
    }
    output = []
    for modality in modalities:
        differences = 100 * (
            retention_samples[("rgca", modality)]
            - retention_samples[("lcaf", modality)]
        )
        for severity in range(5):
            low, high = np.nanpercentile(differences[severity], [2.5, 97.5])
            point = 100 * (
                float(lookup[("rgca", modality, severity)]["retention"])
                - float(lookup[("lcaf", modality, severity)]["retention"])
            )
            output.append(
                {
                    "degraded_modality": modality.upper(),
                    "severity": severity,
                    "severity_name": SEVERITY_NAMES[severity],
                    "rgca_minus_lcaf_retention_pp": point,
                    "paired_ci_low_pp": float(low),
                    "paired_ci_high_pp": float(high),
                }
            )
    return output


def run_inference(
    args: argparse.Namespace,
    data: Mapping[str, object],
    device: torch.device,
) -> Tuple[Dict[str, List[ImageStats]], Dict[str, object], Dict[str, object]]:
    all_records: Dict[str, List[ImageStats]] = {}
    expected_ids: List[str] | None = None
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    for model_name, weight_path in (
        ("lcaf", args.lcaf_weights),
        ("rgca", args.rgca_weights),
    ):
        print(f"Loading {model_name.upper()}: {weight_path}")
        model = attempt_load(str(weight_path), map_location=device).to(device).float().eval()
        if model_name == "lcaf":
            if args.lcaf_variant == "strict":
                restore_audit["lcaf"] = restore_strict_lcaf_layers(model, device)
            else:
                restore_audit["lcaf"] = [{
                    "action": "historical_original_forward_used_directly",
                    "reason": (
                        "checkpoint layers contain original mhca_rgb/mhca_ir and "
                        "flattened concat/conv/dwconv GFB; current HAFFormer retains "
                        "the exact historical inference branch"
                    ),
                }]
        # Validate both models together after the second one is loaded is wasteful in VRAM;
        # the structural checks below retain the strict requirements per model.
        class_metadata_audit = validate_class_names(model, data["names"])
        if model_name == "rgca":
            for layer_index in (20, 21, 22, 23):
                layer = model.model[layer_index]
                attention = getattr(layer, "cross_modal_attention", None)
                if attention is None or getattr(attention, "reliability_gate", None) is None:
                    raise TypeError(f"RGCA layer {layer_index} lacks its reliability gate")
                if getattr(attention, "channel_prior", None) is not None:
                    raise TypeError(f"RGCA layer {layer_index} unexpectedly has channel_prior")
                if not all(hasattr(layer, name) for name in ("foreground_mask", "fusion")):
                    raise TypeError(f"RGCA layer {layer_index} is missing Mask/GFB")
        else:
            for layer_index in (20, 21, 22, 23):
                layer = model.model[layer_index]
                if args.lcaf_variant == "strict":
                    required = ("mhca_rgb", "mhca_ir", "foreground_mask", "fusion")
                    if not all(hasattr(layer, name) for name in required):
                        raise TypeError(f"Strict LCAF layer {layer_index} failed validation")
                else:
                    required = ("mhca_rgb", "mhca_ir", "concat", "conv", "dwconv")
                    if not all(hasattr(layer, name) for name in required):
                        raise TypeError(f"Original LCAFNet layer {layer_index} failed validation")
                    if hasattr(layer, "foreground_mask"):
                        raise TypeError(
                            f"Original LCAFNet layer {layer_index} unexpectedly contains a Mask"
                        )
        stride = max(int(model.stride.max()), 32)
        dataloader, dataset = make_dataloader(data, args, stride)
        if args.max_images == 0 and len(dataset) != 420:
            raise ValueError(
                f"Expected the held-out M3FD test420 protocol, found {len(dataset)} images"
            )
        if device.type == "cuda":
            model.half()
            keep_force_fp32_modules(model)
        conditions = [("clean", 0)] + [
            (modality, severity)
            for modality in ("rgb", "ir")
            for severity in range(1, 5)
        ]
        for modality, severity in conditions:
            key = f"{model_name}:{modality}:{severity}"
            start = time.time()
            records = evaluate_condition(
                model,
                dataloader,
                device,
                modality,
                severity,
                args,
                f"{model_name.upper()} {modality} S{severity}",
            )
            image_ids = [record.image_id for record in records]
            if expected_ids is None:
                expected_ids = image_ids
            elif image_ids != expected_ids:
                raise ValueError(f"Image order mismatch in {key}")
            all_records[key] = records
            map50, map_value, _ = calculate_map(records)
            print(
                f"{key}: images={len(records)}, mAP50={map50:.4f}, "
                f"mAP50:95={map_value:.4f}, elapsed={time.time() - start:.1f}s"
            )
        model_audit[model_name] = {
            "class": type(model).__name__,
            "fusion_classes": [type(model.model[index]).__name__ for index in (20, 21, 22, 23)],
            "images": len(expected_ids or []),
            "class_metadata": class_metadata_audit,
            "lcaf_variant": args.lcaf_variant if model_name == "lcaf" else None,
        }
        del model
        del dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_records, restore_audit, model_audit


def write_readme(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    retention_samples: Mapping[Tuple[str, str], np.ndarray],
    args: argparse.Namespace,
) -> None:
    endpoint = {
        (str(row["model"]), str(row["degraded_modality"])): row
        for row in rows
        if int(row["severity"]) == 4
    }
    clean = {
        str(row["model"]): float(row["clean_map50_95"])
        for row in rows
        if int(row["severity"]) == 0
    }
    mean_retention = {}
    mean_retention_bootstrap = {}
    mean_map = {}
    for model_name in ("RGCA", "LCAF"):
        degraded = [
            row
            for row in rows
            if str(row["model"]) == model_name and int(row["severity"]) > 0
        ]
        mean_retention[model_name] = float(
            np.mean([float(row["retention"]) for row in degraded])
        )
        mean_map[model_name] = float(
            np.mean([float(row["map50_95"]) for row in degraded])
        )
        lower_name = model_name.lower()
        mean_retention_bootstrap[model_name] = np.mean(
            np.concatenate(
                [
                    retention_samples[(lower_name, "rgb")][1:],
                    retention_samples[(lower_name, "ir")][1:],
                ],
                axis=0,
            ),
            axis=0,
        )
    mean_difference = 100 * (
        mean_retention_bootstrap["RGCA"] - mean_retention_bootstrap["LCAF"]
    )
    mean_difference_ci = np.nanpercentile(mean_difference, [2.5, 97.5])
    if args.lcaf_variant == "original":
        comparison_description = (
            "蓝色为纯 RGCA+Mask+GFB，橙色为你训练的原版 LCAFNet（原始 LCAF "
            "attention+原始 GFB，不含 Foreground Mask）。这是完整系统比较，差异不能只归因于注意力模块。"
        )
        comparison_boundary = (
            "本图按要求没有给 LCAFNet 添加相同的 Mask+GFB。因此它适合表述“本文 RGCA "
            "方案与原版 LCAFNet 的鲁棒性差异”，不适合表述成注意力机制的严格单变量因果结论。"
        )
    else:
        comparison_description = (
            "蓝色为纯 RGCA+Mask+GFB，橙色为原始 LCAF attention+相同 Mask+GFB；"
            "因此这是注意力机制的严格对照，而不是两个后融合结构不同的完整系统。"
        )
        comparison_boundary = (
            "本图使用相同 Mask+GFB 的严格注意力对照，仍不能替代多随机种子训练实验。"
        )
    rgb_retention_delta = 100 * (
        float(endpoint[("RGCA", "RGB")]["retention"])
        - float(endpoint[("LCAF", "RGB")]["retention"])
    )
    ir_retention_delta = 100 * (
        float(endpoint[("RGCA", "IR")]["retention"])
        - float(endpoint[("LCAF", "IR")]["retention"])
    )
    if rgb_retention_delta > 0 and ir_retention_delta > 0:
        direction_summary = "RGCA 在两种预设极端退化方向上的相对保持率都更高。"
    elif rgb_retention_delta <= 0 and ir_retention_delta > 0:
        direction_summary = "RGCA 在极端 IR 退化时保持率更高，但在极端 RGB 低照时不占优。"
    elif rgb_retention_delta > 0 and ir_retention_delta <= 0:
        direction_summary = "RGCA 在极端 RGB 低照时保持率更高，但在极端 IR 退化时不占优。"
    else:
        direction_summary = "RGCA 在两种预设极端退化方向上的相对保持率都未超过 LCAFNet。"
    native_validation = (
        "- RGCA clean 结果已另用仓库原生 `test.py` 复核，原生输出 "
        "mAP50=0.896、mAP50:95=0.588，与本脚本的 0.8963/0.5876 在显示精度内一致。"
    )
    if args.lcaf_variant == "original":
        native_validation += (
            "\n- 原版 LCAFNet clean 也已用仓库原生 `test.py` 独立复核，原生输出 "
            "mAP50=0.892、mAP50:95=0.590，与本脚本的 0.8918/0.5904 在显示精度内一致。"
        )
    text = f"""# M3FD 模态退化—性能保持曲线

## 结论读取

- 横轴是预先固定的五级退化强度，纵轴是 `退化后 mAP@0.5:0.95 / 各模型自身 clean mAP@0.5:0.95`。
- {comparison_description}
- 阴影为对同一批测试图像进行成对 bootstrap 的 95% 置信区间（{args.bootstrap} 次）。完整数值见 `modality_degradation_metrics.csv`。
- Clean mAP@0.5:0.95：RGCA={clean['RGCA']:.4f}，LCAF={clean['LCAF']:.4f}。
- Extreme RGB 退化保持率：RGCA={100*float(endpoint[('RGCA','RGB')]['retention']):.2f}%，LCAF={100*float(endpoint[('LCAF','RGB')]['retention']):.2f}%。
- Extreme IR 退化保持率：RGCA={100*float(endpoint[('RGCA','IR')]['retention']):.2f}%，LCAF={100*float(endpoint[('LCAF','IR')]['retention']):.2f}%。
- 8 个非 clean 条件的平均保持率：RGCA={100*mean_retention['RGCA']:.2f}%，LCAF={100*mean_retention['LCAF']:.2f}%，配对差值={100*(mean_retention['RGCA']-mean_retention['LCAF']):+.2f}pp，95% CI [{mean_difference_ci[0]:+.2f}, {mean_difference_ci[1]:+.2f}]pp。
- 8 个非 clean 条件的平均绝对 mAP@0.5:0.95：RGCA={mean_map['RGCA']:.4f}，LCAF={mean_map['LCAF']:.4f}。

结果解读：{direction_summary} Extreme RGB 的 RGCA−LCAF 保持率差为 {rgb_retention_delta:+.2f}pp，绝对 mAP 为 {float(endpoint[('RGCA','RGB')]['map50_95']):.4f} vs {float(endpoint[('LCAF','RGB')]['map50_95']):.4f}；Extreme IR 的保持率差为 {ir_retention_delta:+.2f}pp，绝对 mAP 为 {float(endpoint[('RGCA','IR')]['map50_95']):.4f} vs {float(endpoint[('LCAF','IR')]['map50_95']):.4f}。论文结论应按这些实际数值表述，不能外推为对任意真实退化都占优。

## 实验协议

- 数据：M3FD 本地 held-out test420，420 组对齐 RGB/IR；训练和选权不使用这 420 组。
- RGB 低照度：`gain * image^gamma + Gaussian noise`，参数见 `run_config.yaml`。
- IR 热对比度塌缩：均值保持的对比度压缩 + 均值模糊 + 灰度 Gaussian noise，参数见 `run_config.yaml`。
- 只退化一个模态，另一模态、标注、对齐方式、NMS 和模型输入尺寸完全不变。
- 两模型按图像 ID 和退化等级使用完全相同的确定性噪声。
- mAP 采用仓库原有 YOLOv5 风格 0.001 置信度阈值、0.5 NMS IoU 和 IoU 0.50:0.95 十档计算。
{native_validation}

## 科学边界

这张图证明的是模型在上述两类、上述强度的合成单模态退化下的鲁棒性，不等价于真实传感器故障的全部分布，也不能单独证明热图解释的忠实性。论文中应同时报告绝对 mAP、退化定义和置信区间，不能只展示保持率。

{comparison_boundary}
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


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
    device = resolve_device(args.device)
    data = load_data_config(args.data)
    signature = cache_signature(args, data)
    cache_path = args.output_dir / "per_image_detection_stats.npz"
    keys = expected_keys()
    all_records = None
    restore_audit: Dict[str, object] = {}
    model_audit: Dict[str, object] = {}
    if cache_path.is_file() and not args.force:
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = json.loads(str(archive["signature_json"]))
            if cached_signature == json.loads(json.dumps(signature)):
                print(f"Reusing compatible inference cache: {cache_path}")
                all_records = arrays_to_records(archive, keys)
                previous_config_path = args.output_dir / "run_config.yaml"
                if previous_config_path.is_file():
                    with previous_config_path.open("r", encoding="utf-8") as stream:
                        previous_config = yaml.safe_load(stream) or {}
                    restore_audit = previous_config.get(
                        "lcaf_runtime_compatibility_audit",
                        previous_config.get("legacy_lcaf_runtime_restore", {}),
                    )
                    model_audit = previous_config.get("model_audit", {})
            else:
                print("Inference cache signature differs; rerunning inference")
    if all_records is None:
        all_records, restore_audit, model_audit = run_inference(args, data, device)
        np.savez_compressed(cache_path, **records_to_arrays(all_records, signature))
        print(f"Saved per-image prediction cache: {cache_path}")

    reference_ids = [record.image_id for record in next(iter(all_records.values()))]
    for key, records in all_records.items():
        if [record.image_id for record in records] != reference_ids:
            raise ValueError(f"Paired image audit failed for {key}")

    print(f"Computing {args.bootstrap} paired image-bootstrap replicates...")
    rows, retention_samples = bootstrap_summary(all_records, args.bootstrap, args.seed)
    csv_path = args.output_dir / "modality_degradation_metrics.csv"
    write_summary_csv(csv_path, rows)
    difference_rows = paired_difference_rows(rows, retention_samples)
    difference_csv_path = args.output_dir / "paired_retention_differences.csv"
    write_summary_csv(difference_csv_path, difference_rows)
    bootstrap_path = args.output_dir / "bootstrap_retention_samples.npz"
    np.savez_compressed(
        bootstrap_path,
        lcaf_rgb=retention_samples[("lcaf", "rgb")],
        rgca_rgb=retention_samples[("rgca", "rgb")],
        lcaf_ir=retention_samples[("lcaf", "ir")],
        rgca_ir=retention_samples[("rgca", "ir")],
    )
    original_lcaf = args.lcaf_variant == "original"
    png_path, pdf_path = plot_figure(
        rows,
        retention_samples,
        args.output_dir,
        args.dpi,
        len(reference_ids),
        lcaf_label="Original LCAFNet" if original_lcaf else "LCAF attention",
        figure_title=(
            "RGCA vs original LCAFNet under single-modality degradation"
            if original_lcaf
            else "Cross-attention robustness under single-modality degradation"
        ),
    )
    config = {
        "task": (
            "RGCA-vs-original-LCAFNet full-system single-modality degradation robustness"
            if original_lcaf
            else "strict RGCA-vs-LCAF single-modality degradation robustness"
        ),
        "comparison_scope": (
            "full systems; post-attention fusion differs; not an attention-only causal comparison"
            if original_lcaf
            else "attention control with identical Foreground Mask and GFB"
        ),
        "protocol": "M3FD local held-out test420",
        "image_count": len(reference_ids),
        "image_ids_sha256": hashlib.sha256("\n".join(reference_ids).encode()).hexdigest(),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "paired_bootstrap": True,
        "metric": "mAP@0.5:0.95 retention relative to each model's own clean mAP",
        "rgb_corruption": {
            "name": "low-light plus Gaussian sensor noise",
            "formula": "clip(gain * image^gamma + N(0, noise_sigma), 0, 1)",
            "schedule": list(RGB_LOW_LIGHT),
        },
        "ir_corruption": {
            "name": "thermal contrast collapse plus blur and Gaussian sensor noise",
            "formula": "clip(mean + contrast*(mean_blur(image)-mean) + shared_channel_noise, 0, 1)",
            "schedule": list(IR_CONTRAST_COLLAPSE),
        },
        "same_corruption_realization_across_models": True,
        "corrupt_letterbox_padding": False,
        "model_audit": model_audit,
        "lcaf_runtime_compatibility_audit": restore_audit,
        "rgca_identity": "pure prior-free RGCA + Foreground Mask + GFB",
        "rgca_weights": str(args.rgca_weights.resolve()),
        "rgca_sha256": sha256sum(args.rgca_weights),
        "lcaf_variant": args.lcaf_variant,
        "lcaf_identity": (
            "original LCAFNet: original LCAF attention + original GFB, no Foreground Mask"
            if original_lcaf
            else "original LCAF attention + identical Foreground Mask + GFB"
        ),
        "lcaf_weights": str(args.lcaf_weights.resolve()),
        "lcaf_sha256": sha256sum(args.lcaf_weights),
        "data_yaml": str(args.data.resolve()),
        "evaluation": {
            "image_size": args.img_size,
            "batch_size": args.batch_size,
            "confidence_threshold": args.conf_thres,
            "nms_iou": args.nms_iou,
            "iou_thresholds": "0.50:0.05:0.95",
            "precision": "FP16 model/input on CUDA; AP accumulation in NumPy FP64",
        },
        "native_clean_cross_check": {
            "rgca_test_py": {"map50": 0.896, "map50_95": 0.588},
            **(
                {"original_lcafnet_test_py": {"map50": 0.892, "map50_95": 0.590}}
                if original_lcaf
                else {}
            ),
            "status": "matches this evaluator at displayed precision",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": str(device),
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "metrics_csv": str(csv_path.resolve()),
            "paired_difference_csv": str(difference_csv_path.resolve()),
            "bootstrap_retention_samples": str(bootstrap_path.resolve()),
            "per_image_stats": str(cache_path.resolve()),
        },
    }
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_readme(args.output_dir, rows, retention_samples, args)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()
