# Group D-FT — Full BECP fine-tuning from the no-prior optimum

This is the current experiment launched by `python train.py`.

## Initialization

- Source: `runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt`
- Transfer scope: all compatible tensors
- Shared backbone, fusion, GFB, foreground-mask, and detection weights are loaded
- Historical per-channel residual scales are migrated to the corrected bounded bidirectional scalars
- Only the four lightweight BECP evidence controllers are newly initialized (304 parameters total)

The loader rejects the run if any unexpected non-BECP tensors are missing.

## Schedule

- Epochs 0–4: evidence-controller-only calibration; evidence loss on, prior exactly zero
- Epochs 5–14: joint fine-tuning with the BECP prior ramped from 0.1 to 1.0
- Epochs 15–49: full BECP joint fine-tuning
- Optimizer: SGD; AFSS disabled
- Base LR: 0.002
- Loaded backbone LR: 0.0002
- Loaded fusion/head LR: 0.0004
- BECP controller LR: 0.01 during calibration, 0.002 during joint tuning
- Batch/image size: 16 / 640
- Mosaic closes for the final 10 epochs
- Seed: 1

Expected output directory:
`runs/train/exp_group_D_becp_finetune50_from_no_prior_best_m3fd`.

This experiment tests BECP as a post-training plug-in. A publication-grade
causal comparison additionally requires a matched A-FT run from the same
checkpoint with `--becp-off-control` and otherwise identical settings.
