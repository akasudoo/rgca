"""Two-stage training launcher for the entropy-based frequency-route fix."""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def validate_route_diagnostics(output_dir, expected_epochs, expected_branches=8):
    diagnostics = output_dir / 'route_diagnostics.csv'
    if not diagnostics.is_file():
        raise FileNotFoundError(f'Missing frequency-route diagnostics: {diagnostics}')
    by_epoch = {}
    with diagnostics.open(newline='') as f:
        for row in csv.DictReader(f):
            epoch = int(row['epoch'])
            confidence_std = float(row['confidence_std'])
            if not math.isfinite(confidence_std) or confidence_std <= 1e-8:
                raise RuntimeError(
                    f'Frequency route collapsed at epoch {epoch}: '
                    f'confidence_std={confidence_std}')
            by_epoch.setdefault(epoch, 0)
            by_epoch[epoch] += 1
    missing = [
        epoch for epoch in range(expected_epochs)
        if by_epoch.get(epoch, 0) < expected_branches
    ]
    if missing:
        raise RuntimeError(
            f'Incomplete frequency-route diagnostics for epochs: {missing}')


def run_stage(args, name, weights, epochs, hyp, extra):
    output_dir = Path(args.project) / name
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {output_dir}")

    command = [
        sys.executable,
        str(ROOT / 'archives/train_experiments_legacy_20260805.py'),
        '--weights', str(weights),
        '--cfg', str(ROOT / 'models/transformer/yolov5s_LCAFNet_M3FD.yaml'),
        '--data', str(ROOT / 'data/multispectral/M3FD.yaml'),
        '--hyp', str(hyp),
        '--epochs', str(epochs),
        '--batch-size', str(args.batch_size),
        '--img-size', str(args.img_size), str(args.img_size),
        '--device', args.device,
        '--workers', str(args.workers),
        '--project', str(args.project),
        '--name', name,
        '--seed', str(args.seed),
        '--optimizer', 'SGD',
        '--warmup-min-iters', '0',
        '--warmup-max-fraction', '0.2',
        '--no-afss',
        '--prefer-ema-weights',
        *extra,
    ]
    if args.dry_run:
        print(' '.join(str(part) for part in command))
        return output_dir / 'weights' / 'best.pt'
    subprocess.run(command, cwd=ROOT, check=True)
    best = output_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise FileNotFoundError(f'Stage did not produce a best checkpoint: {best}')
    validate_route_diagnostics(output_dir, epochs)
    return best


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adapt the repaired frequency route, then jointly fine-tune LCAFNet.')
    parser.add_argument(
        '--source-weights',
        type=Path,
        default=ROOT / 'runs/train/exp_freq_mask_gate_no_gfb_sgd_full_afss/weights/last.pt',
        help='compatible pre-fix checkpoint used to initialize route adaptation')
    parser.add_argument('--adapt-epochs', type=int, default=5)
    parser.add_argument('--joint-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--project', type=Path, default=Path('runs/train'))
    parser.add_argument('--run-name', default='freq_route_entropy_fix')
    parser.add_argument('--dry-run', action='store_true',
                        help='print both stage commands without starting training')
    return parser.parse_args()


def main():
    args = parse_args()
    source_weights = args.source_weights.expanduser().resolve()
    if not source_weights.is_file():
        raise FileNotFoundError(f'Compatible source checkpoint not found: {source_weights}')
    if args.adapt_epochs <= 0 or args.joint_epochs <= 0:
        raise ValueError('Both --adapt-epochs and --joint-epochs must be positive')

    adapt_name = f'{args.run_name}_adapt'
    adapt_best = run_stage(
        args,
        adapt_name,
        source_weights,
        args.adapt_epochs,
        ROOT / 'data/hyp.freq_route_adapt.yaml',
        ['--finetune-fusion-head', '--reset-frequency-beta', '0.01',
         '--close-mosaic', '0'],
    )

    run_stage(
        args,
        f'{args.run_name}_joint',
        adapt_best,
        args.joint_epochs,
        ROOT / 'data/hyp.freq_route_joint.yaml',
        ['--train-all-layers', '--close-mosaic', '5'],
    )


if __name__ == '__main__':
    main()
