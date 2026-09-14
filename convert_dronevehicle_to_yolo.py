#!/usr/bin/env python3
"""Convert DroneVehicleNoBorder COCO annotations to paired YOLO labels.

The generated layout matches the multispectral datasets used by this project::

    images/{vis,Ir}_{train,val,test}
    labels/{vis,Ir}_{train,val,test}

Images are hard-linked by default, so the reorganized dataset does not duplicate
the image payload. COCO boxes are clipped to the 640x512 image boundary before
being converted to normalized YOLO ``class cx cy width height`` records.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = ("car", "freight_car", "truck", "bus", "van")
SPLITS = ("train", "val", "test")
MODALITIES = {"vis": "rgb", "Ir": "ir"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dell/lcp/dataset/DroneVehicle"),
        help="DroneVehicle root containing raw/ and coco_annotations_noborder/",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to populate the reorganized images directories",
    )
    parser.add_argument(
        "--force-labels",
        action="store_true",
        help="Replace an existing label if its content differs",
    )
    return parser.parse_args()


def install_image(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Refusing to replace existing image: {destination}")

    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        try:
            os.link(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)


def write_label(path: Path, content: str, force: bool) -> None:
    if path.exists():
        if path.read_text() == content:
            return
        if not force:
            raise FileExistsError(
                f"Label differs from generated content: {path}; "
                "use --force-labels to replace it"
            )
    path.write_text(content)


def convert_partition(root: Path, split: str, output_modality: str, source_modality: str,
                      image_mode: str, force_labels: bool) -> dict[str, int]:
    source_images = root / "raw" / split / source_modality / "images"
    annotation_file = (
        root / "coco_annotations_noborder" / f"DV_{split}_{source_modality}.json"
    )
    if not source_images.is_dir():
        raise FileNotFoundError(source_images)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)

    with annotation_file.open() as handle:
        coco = json.load(handle)

    category_to_class = {}
    for category in coco["categories"]:
        name = category["name"]
        if name not in CLASS_NAMES:
            raise ValueError(f"Unknown category in {annotation_file}: {name}")
        category_to_class[category["id"]] = CLASS_NAMES.index(name)
    if set(category_to_class.values()) != set(range(len(CLASS_NAMES))):
        raise ValueError(f"Incomplete category mapping in {annotation_file}")

    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_output = root / "images" / f"{output_modality}_{split}"
    label_output = root / "labels" / f"{output_modality}_{split}"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "clipped": 0,
        "dropped": 0,
        "duplicates": 0,
    }
    known_image_ids = set()

    for image in sorted(coco["images"], key=lambda item: item["file_name"]):
        image_id = image["id"]
        known_image_ids.add(image_id)
        filename = Path(image["file_name"]).name
        width = float(image["width"])
        height = float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {filename}: {width}x{height}")

        source = source_images / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        install_image(source, image_output / filename, image_mode)

        records = []
        record_set = set()
        for annotation in sorted(
            annotations_by_image.get(image_id, []), key=lambda item: item["id"]
        ):
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1 = min(width, max(0.0, x))
            y1 = min(height, max(0.0, y))
            x2 = min(width, max(0.0, x + box_width))
            y2 = min(height, max(0.0, y + box_height))
            if (x1, y1, x2, y2) != (x, y, x + box_width, y + box_height):
                stats["clipped"] += 1
            if x2 <= x1 or y2 <= y1:
                stats["dropped"] += 1
                continue

            class_id = category_to_class[annotation["category_id"]]
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            record = (
                f"{class_id} {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )
            if record in record_set:
                stats["duplicates"] += 1
                continue
            record_set.add(record)
            records.append(record)

        label_content = "\n".join(records)
        if label_content:
            label_content += "\n"
        write_label(label_output / f"{Path(filename).stem}.txt", label_content, force_labels)

        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += len(records)

    orphan_ids = set(annotations_by_image) - known_image_ids
    if orphan_ids:
        raise ValueError(f"Annotations reference unknown image IDs: {sorted(orphan_ids)[:10]}")

    output_image_names = {path.stem for path in image_output.glob("*.jpg")}
    output_label_names = {path.stem for path in label_output.glob("*.txt")}
    if output_image_names != output_label_names:
        raise RuntimeError(f"Image/label mismatch in {output_modality}_{split}")

    return stats


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grand_total = defaultdict(int)

    print(f"Dataset root: {root}")
    print("YOLO classes: " + ", ".join(f"{i}={name}" for i, name in enumerate(CLASS_NAMES)))
    for split in SPLITS:
        for output_modality, source_modality in MODALITIES.items():
            stats = convert_partition(
                root,
                split,
                output_modality,
                source_modality,
                args.image_mode,
                args.force_labels,
            )
            for key, value in stats.items():
                grand_total[key] += value
            print(
                f"{output_modality}_{split}: images={stats['images']}, "
                f"labels={stats['labels']}, boxes={stats['boxes']}, "
                f"clipped={stats['clipped']}, dropped={stats['dropped']}, "
                f"duplicates={stats['duplicates']}"
            )

    print(
        "Total: "
        + ", ".join(f"{key}={grand_total[key]}" for key in grand_total)
    )


if __name__ == "__main__":
    main()
