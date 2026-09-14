import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


PRESETS = {
    "llvip_baseline_finetune": {
        "optimizer": "adam",
        "weights": "LCAFNet_LLVIP.pt",
        "pretrained_scope": "all",
        "cfg": "models/transformer/yolov5s_LCAFNet_LLVIP_RGCA.yaml",
        "data": "data/multispectral/LLVIP.yaml",
        "hyp": "data/hyp.rgca_llvip_baseline_finetune.yaml",
        "epochs": 50,
        "batch_size": 8,
        "img_size": [1024, 1024],
        "close_mosaic": 0,
        "lr0": 0.0005,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0005,
        "differential_lr": True,
        "pretrained_lr_ratio": 0.1,
        "prior_adapt_epochs": 3,
        "finetune_step_lr": True,
        "freeze_loaded_bn": True,
        "default_name": "exp_rgca_mask_gfb_llvip_lcafnet_baseline_ft50_adam_1024",
        "description": (
            "Two-stage 50-epoch Adam fine-tuning from the LCAFNet LLVIP baseline."
        ),
    },
    "baseline_finetune": {
        "optimizer": "adam",
        "weights": str(ROOT / "LCAFNet_FLIR.pt"),
        "pretrained_scope": "all",
        "hyp": "data/hyp.rgca_flir_baseline_finetune.yaml",
        "epochs": 50,
        "batch_size": 32,
        "lr0": 0.0005,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0005,
        "differential_lr": True,
        "pretrained_lr_ratio": 0.1,
        "prior_adapt_epochs": 7,
        "finetune_step_lr": True,
        "freeze_loaded_bn": True,
        "default_name": "exp_rgca_mask_gfb_flir_lcafnet_baseline_ft50_adam",
        "description": (
            "Two-stage 50-epoch Adam fine-tuning from the LCAFNet FLIR baseline."
        ),
    },
    "adam": {
        "optimizer": "adam",
        "hyp": "data/hyp.rgca_flir_adam.yaml",
        "lr0": 0.001,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.1,
        "warmup_bias_lr": 0.01,
        "description": "Adam setting from the current FLIR experiment.",
    },
    "adam_stable": {
        "optimizer": "adam",
        "hyp": "data/hyp.rgca_flir_adam.yaml",
        "lr0": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.005,
        "description": "Stable strict-fair Adam setting for RGCA FLIR.",
    },
    "sgd": {
        "optimizer": "sgd",
        "hyp": "data/hyp.rgca_flir_pretrained.yaml",
        "lr0": 0.01,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "description": "Strict-fair RGCA FLIR setting with uniform-LR SGD.",
    },
}


def parser_args():
    parser = argparse.ArgumentParser(
        description="Convenient FLIR training launcher for LCAFNet."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="sgd",
        help="training parameter preset",
    )
    parser.add_argument(
        "--weights", default=None,
        help="pretrained weights; defaults to the file selected by --preset")
    parser.add_argument(
        "--cfg", default=None,
        help="model yaml; defaults to the file selected by --preset")
    parser.add_argument(
        "--data", default=None,
        help="dataset yaml; defaults to the file selected by --preset")
    parser.add_argument(
        "--hyp", default=None,
        help="hyperparameter yaml; defaults to the file selected by --preset")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="number of epochs; defaults to the value selected by --preset")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="batch size; defaults to the value selected by --preset")
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=None,
        help="train/test image sizes; defaults to the value selected by --preset")
    parser.add_argument("--device", default="0", help="cuda device, e.g. 0 or 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--project", default="runs/train", help="save root")
    parser.add_argument("--name", default=None, help="experiment name; auto-generated if omitted")
    parser.add_argument(
        "--close-mosaic", type=int, default=None,
        help="disable mosaic for final N epochs; defaults to the preset value")
    parser.add_argument("--lr0", type=float, default=None, help="override preset initial learning rate")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="override preset warmup epochs")
    parser.add_argument("--warmup-momentum", type=float, default=None, help="override preset warmup momentum")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="override preset warmup bias lr")
    parser.add_argument("--resume", default=None, help="resume path or True-like value passed to train.py")
    parser.add_argument("--exist-ok", action="store_true", help="allow existing project/name")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--cache-images", action="store_true", help="cache images")
    parser.add_argument("--multi-scale", action="store_true", help="multi-scale training")
    parser.add_argument("--noautoanchor", action="store_true", help="disable autoanchor check")
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="disable AMP")
    parser.add_argument("--dry-run", action="store_true", help="print command without starting training")
    parser.add_argument(
        "--background",
        action="store_true",
        help="start training in background and write stdout/stderr to a log file",
    )
    parser.add_argument("--log", default=None, help="background log file path")
    return parser.parse_known_args()


def value_from(args, preset, key):
    cli_value = getattr(args, key)
    return preset[key] if cli_value is None else cli_value


def add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def add_option(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def ensure_exists(path_like, label):
    path = ROOT / path_like
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_command(args, extra_args):
    preset = PRESETS[args.preset]
    optimizer = preset["optimizer"]
    weights = args.weights if args.weights is not None else preset.get("weights", "yolov5s.pt")
    epochs = args.epochs if args.epochs is not None else preset.get("epochs", 150)
    batch_size = args.batch_size if args.batch_size is not None else preset.get("batch_size", 32)
    cfg = args.cfg if args.cfg is not None else preset.get(
        "cfg", "models/transformer/yolov5s_LCAFNet_FLIR.yaml")
    data = args.data if args.data is not None else preset.get(
        "data", "data/multispectral/FLIR.yaml")
    img_size = args.img_size if args.img_size is not None else preset.get(
        "img_size", [640, 640])
    close_mosaic = (
        args.close_mosaic if args.close_mosaic is not None
        else preset.get("close_mosaic", 10)
    )
    pretrained_scope = preset.get("pretrained_scope", "backbone")
    hyp = args.hyp if args.hyp is not None else preset["hyp"]
    lr0 = value_from(args, preset, "lr0")
    warmup_epochs = value_from(args, preset, "warmup_epochs")
    warmup_momentum = value_from(args, preset, "warmup_momentum")
    warmup_bias_lr = value_from(args, preset, "warmup_bias_lr")
    name = args.name
    if name is None:
        name = preset.get("default_name")
        if name is None:
            name = (
                "exp_rgca_mask_gfb_flir_uniform_lr"
                if args.preset == "sgd"
                else f"flir_{args.preset}_close_mosaic{close_mosaic}"
            )

    ensure_exists(cfg, "model cfg")
    ensure_exists(data, "data yaml")
    ensure_exists(hyp, "hyp yaml")
    if weights:
        ensure_exists(weights, "weights")

    cmd = [
        sys.executable,
        "archives/train_experiments_legacy_20260805.py",
        "--weights",
        weights,
        "--pretrained-scope",
        pretrained_scope,
        "--cfg",
        cfg,
        "--data",
        data,
        "--hyp",
        hyp,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--img-size",
        *[str(v) for v in img_size],
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--project",
        args.project,
        "--name",
        name,
        "--close-mosaic",
        str(close_mosaic),
        "--lr0",
        str(lr0),
        "--warmup-epochs",
        str(warmup_epochs),
        "--warmup-momentum",
        str(warmup_momentum),
        "--warmup-bias-lr",
        str(warmup_bias_lr),
        "--no-afss",
    ]

    if preset.get("differential_lr", False):
        cmd.extend([
            "--differential-lr",
            "--pretrained-lr-ratio",
            str(preset.get("pretrained_lr_ratio", 0.1)),
        ])
    else:
        cmd.append("--no-differential-lr")

    add_option(cmd, "--prior-adapt-epochs", preset.get("prior_adapt_epochs"))
    add_option(cmd, "--prior-lr-ratio", preset.get("prior_lr_ratio", 0.1))
    add_flag(cmd, "--finetune-step-lr", preset.get("finetune_step_lr", False))
    add_flag(cmd, "--freeze-loaded-bn", preset.get("freeze_loaded_bn", False))

    cmd.extend(["--optimizer", "Adam" if optimizer == "adam" else "SGD"])

    add_option(cmd, "--resume", args.resume)
    add_flag(cmd, "--exist-ok", args.exist_ok)
    add_flag(cmd, "--rect", args.rect)
    add_flag(cmd, "--cache-images", args.cache_images)
    add_flag(cmd, "--multi-scale", args.multi_scale)
    add_flag(cmd, "--noautoanchor", args.noautoanchor)
    add_flag(cmd, "--notest", args.notest)
    add_flag(cmd, "--nosave", args.nosave)
    add_flag(cmd, "--no-amp", args.no_amp)
    cmd.extend(extra_args)

    return cmd, name, preset


def main():
    args, extra_args = parser_args()
    cmd, name, preset = build_command(args, extra_args)

    print(f"Preset: {args.preset} - {preset['description']}")
    print(f"Experiment name: {name}")
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    if args.background:
        log_path = Path(args.log) if args.log else ROOT / "runs" / "train" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started background training: pid={process.pid}")
        print(f"Log file: {log_path}")
        return

    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
