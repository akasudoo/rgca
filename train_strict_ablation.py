"""Launch a paired, multi-seed strict-fair M3FD ablation study."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from models.yolo_test import Model
from train import (
    STRICT_ABLATION_FUSIONS,
    build_ablation_model_config,
    build_dual_stream_backbone_state_dict,
)
from utils.torch_utils import torch_load


ROOT = Path(__file__).resolve().parent
DEFAULT_VARIANTS = tuple(STRICT_ABLATION_FUSIONS)
FREQUENCY_VARIANTS = {'frequency_gfb', 'frequency_mask'}


def inspect_yolov5s_weight(weight_path):
    """Fail fast unless the checkpoint is a classic anchor-based YOLOv5s."""
    weight_path = Path(weight_path).resolve()
    checkpoint = torch_load(weight_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or checkpoint.get('model') is None:
        raise RuntimeError("checkpoint must contain a non-empty 'model' entry")
    source_model = checkpoint['model'].float()
    model_yaml = getattr(source_model, 'yaml', {})
    state = source_model.state_dict()
    anchors = model_yaml.get('anchors', [])
    depth = float(model_yaml.get('depth_multiple', -1))
    width = float(model_yaml.get('width_multiple', -1))
    stride = [float(x) for x in getattr(source_model, 'stride', [])]
    detect = getattr(getattr(source_model, 'model', [None])[-1], 'm', [])
    detect_scales = len(detect)
    parameter_count = sum(parameter.numel() for parameter in source_model.parameters())
    is_s = abs(depth - 0.33) < 1e-6 and abs(width - 0.50) < 1e-6
    is_anchor_based = len(anchors) == 3 and detect_scales == 3 and stride == [8.0, 16.0, 32.0]
    if not is_s or not is_anchor_based:
        raise RuntimeError(
            'weight is not the expected classic anchor-based YOLOv5s: '
            f'depth={depth}, width={width}, anchors={len(anchors)}, '
            f'detect_scales={detect_scales}, stride={stride}')

    target_cfg = build_ablation_model_config(
        ROOT / 'models/transformer/yolov5s_LCAFNet_M3FD.yaml',
        fusion_variant='frequency_mask',
        deterministic_init=True,
        seed=1)
    target_model = Model(target_cfg, ch=3, nc=6)
    _, transfer = build_dual_stream_backbone_state_dict(
        state, target_model.state_dict())
    if transfer['missing']:
        raise RuntimeError(
            f"checkpoint migration misses {len(transfer['missing'])} backbone tensors")

    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    return {
        'path': str(weight_path),
        'sha256': digest,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'best_fitness': checkpoint.get('best_fitness'),
        'model_class': (
            f'{source_model.__class__.__module__}.{source_model.__class__.__name__}'),
        'depth_multiple': depth,
        'width_multiple': width,
        'parameter_count': parameter_count,
        'anchors': anchors,
        'stride': stride,
        'format': transfer['format'],
        'focus_to_conv6': transfer['converted_focus'],
        'transferred_items_per_modal_branch': transfer['expected_per_branch'],
    }


def validate_frequency_routes(run_dir, expected_epochs):
    path = Path(run_dir) / 'route_diagnostics.csv'
    if not path.is_file():
        raise RuntimeError(f'missing route diagnostics: {path}')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    epochs = {}
    for row in rows:
        epochs.setdefault(int(row['epoch']), []).append(float(row['confidence_std']))
    if set(epochs) != set(range(expected_epochs)):
        raise RuntimeError(
            f'{path} covers epochs {sorted(epochs)}, expected 0..{expected_epochs - 1}')
    for epoch, values in epochs.items():
        if len(values) != 8:
            raise RuntimeError(
                f'{path}: epoch {epoch} has {len(values)} routes, expected 8')
        if min(values) <= 1e-8:
            raise RuntimeError(
                f'{path}: collapsed frequency route detected at epoch {epoch}')


def read_best_metrics(run_dir):
    path = Path(run_dir) / 'results.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f'no result rows in {path}')
    best = max(rows, key=lambda row: float(row['metrics/mAP_0.5:0.95']))
    return {
        'best_epoch': int(float(best['epoch'])),
        'precision': float(best['metrics/precision']),
        'recall': float(best['metrics/recall']),
        'map50': float(best['metrics/mAP_0.5']),
        'map50_95': float(best['metrics/mAP_0.5:0.95']),
    }


def write_summary(project, variants, seeds):
    run_rows = []
    for seed in seeds:
        for variant in variants:
            run_dir = project / f'{variant}_seed{seed}'
            metrics = read_best_metrics(run_dir)
            run_rows.append({'variant': variant, 'seed': seed, **metrics})

    run_csv = project / 'ablation_runs.csv'
    with run_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows = []
    for variant in variants:
        selected = [row for row in run_rows if row['variant'] == variant]
        summary = {'variant': variant, 'runs': len(selected)}
        for metric in ('precision', 'recall', 'map50', 'map50_95'):
            values = [row[metric] for row in selected]
            summary[f'{metric}_mean'] = statistics.mean(values)
            summary[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    summary_csv = project / 'ablation_summary.csv'
    with summary_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    return run_csv, summary_csv


def build_command(args, variant, seed, run_name):
    return [
        sys.executable,
        str(ROOT / 'archives/train_experiments_legacy_20260805.py'),
        '--weights', str(Path(args.weights).resolve()),
        '--pretrained-scope', 'backbone',
        '--cfg', str(Path(args.cfg).resolve()),
        '--ablation-fusion', variant,
        '--deterministic-ablation-init',
        '--strict-determinism',
        '--data', str(Path(args.data).resolve()),
        '--hyp', str(Path(args.hyp).resolve()),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--img-size', str(args.img_size), str(args.img_size),
        '--device', args.device,
        '--workers', str(args.workers),
        '--seed', str(seed),
        '--optimizer', 'SGD',
        '--no-differential-lr',
        '--prior-lr-ratio', '1.0',
        '--reset-frequency-beta', '0.01',
        '--warmup-min-iters', '0',
        '--warmup-max-fraction', '0.2',
        '--grad-clip-norm', '10.0',
        '--close-mosaic', str(args.close_mosaic),
        '--no-afss',
        '--noautoanchor',
        '--amp',
        '--project', str(Path(args.project).resolve()),
        '--name', run_name,
        '--exist-ok',
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(ROOT / 'yolov5s.pt'))
    parser.add_argument(
        '--cfg', default=str(ROOT / 'models/transformer/yolov5s_LCAFNet_M3FD.yaml'))
    parser.add_argument('--data', default=str(ROOT / 'data/multispectral/M3FD.yaml'))
    parser.add_argument('--hyp', default=str(ROOT / 'data/hyp.strict_fair_ablation.yaml'))
    parser.add_argument('--project', default=str(ROOT / 'runs/strict_ablation'))
    parser.add_argument('--variants', nargs='+', choices=DEFAULT_VARIANTS,
                        default=list(DEFAULT_VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--close-mosaic', type=int, default=10)
    parser.add_argument(
        '--execute', action='store_true',
        help='run all experiments; without this flag only validate and print commands')
    return parser.parse_args()


def main():
    args = parse_args()
    report = inspect_yolov5s_weight(args.weights)
    print(json.dumps({'weight_validation': report}, ensure_ascii=False, indent=2, default=str))

    project = Path(args.project).resolve()
    commands = []
    for seed in args.seeds:
        for variant in args.variants:
            run_name = f'{variant}_seed{seed}'
            run_dir = project / run_name
            if args.execute and run_dir.exists():
                raise FileExistsError(
                    f'refusing to overwrite strict-ablation run: {run_dir}')
            commands.append((variant, seed, run_dir, build_command(
                args, variant, seed, run_name)))

    for variant, seed, _, command in commands:
        print(f'[{variant}, seed={seed}]', ' '.join(command))
    if not args.execute:
        print('Validation/dry-run complete. Add --execute to launch the full matrix.')
        return

    project.mkdir(parents=True, exist_ok=False)
    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'weight': report,
        'variants': args.variants,
        'seeds': args.seeds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.img_size,
        'strategy': {
            'optimizer': 'SGD',
            'new_module_lr': 0.01,
            'pretrained_backbone_lr': 0.01,
            'differential_lr': False,
            'scheduler': 'cosine one-cycle',
            'warmup_epochs': 3.0,
            'mosaic_close_epochs': args.close_mosaic,
            'afss': False,
            'autoanchor': False,
            'gradient_clip_norm': 10.0,
        },
        'commands': [command for _, _, _, command in commands],
    }
    (project / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + '\n')

    for variant, seed, run_dir, command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
        if variant in FREQUENCY_VARIANTS:
            validate_frequency_routes(run_dir, args.epochs)

    run_csv, summary_csv = write_summary(project, args.variants, args.seeds)
    print(f'Completed strict ablation. Per-run metrics: {run_csv}')
    print(f'Mean/std summary: {summary_csv}')


if __name__ == '__main__':
    main()
