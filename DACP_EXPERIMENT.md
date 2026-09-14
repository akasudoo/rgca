# DACP M3FD fine-tuning experiment

This is the active experiment launched by `python train.py`. It replaces the
failed fixed diagonal prior and BECP controller in the active model with a
Detection-Aligned Foreground-Background Contrastive Prior (DACP). Historical
classes remain in `models/common.py` only so old checkpoints can still be read.

## Active model

- Base: corrected frequency-free RGCA + existing foreground fusion mask + GFB.
- New supervision head per fusion scale: `4 -> 8 -> 1` spatial convolutional
  head, trained from the augmented YOLO boxes.
- Head input: RGB mean, IR mean, absolute RGB/IR difference, and normalized
  cross-modal agreement. Each statistic is spatially standardized.
- The supervised DACP mask is separate from the existing post-attention fusion
  mask. This prevents one mask from receiving two incompatible semantics.
- Added parameters: 1,252 over four scales; total model parameters: 13,742,776.

For objectness map `m`, detached channel queries/keys `Q,K`, foreground and
background normalized weights are

```text
w_fg = m / sum(m),  w_bg = (1-m) / sum(1-m),  w = w_fg - w_bg
R = center_rows((Q * w) K^T).
```

The dynamic prior added to content logits `L` is

```text
P = rho * RMS(L) * R / max(RMS(R), eps),
rho = 0.1 * warmup * tanh(raw_rho).
```

`raw_rho` represents an initial full-ramp ratio of 0.01, but the explicit
warmup multiplier is zero for epochs 0-4. Thus the calibration phase is still
exactly the loaded no-prior attention without relying on a zero-gradient cold
start. The signed coefficient is learned independently for both directions and
every head, while `|P|_RMS <= 0.1 |L|_RMS` prevents the prior from overwhelming
content evidence.

## Default 50-epoch schedule

| Epochs | Trainable path | Prior scale | Mask-loss gain |
|---|---|---:|---:|
| 0-4 | DACP objectness head only | 0 | 0.05 |
| 5-14 | Complete DACP adapter only | 0.1 -> 1.0 | 0.05 |
| 15-49* | DACP + fusion/detection layers 20+ | 1.0 | 0.01 |

`*` Epoch 15 is only the earliest joint start. Loaded fusion/detection layers
remain frozen until the previous epoch has mean foreground gap >= 0.005,
positive-gap fraction >= 0.75, and observed prior/content RMS >= 0.001.

The dual-stream backbone (layers 0-19) stays frozen for all 50 epochs. BatchNorm
statistics are frozen everywhere before the health gate passes, then remain
frozen only in the backbone. The auxiliary loss independently normalizes
positive Gaussian mass and negative background mass before focal weighting,
then adds `0.5 * Dice`. This prevents the background-only collapse observed in
the retained DACP-v1 run.

Default optimization is SGD, batch 16, image size 640, `lr0=0.001`, cosine final
factor 0.1, one warmup epoch, and weight decay 0.0005. Initial parameter-group
LRs are 0.0001 for loaded backbone tensors (frozen), 0.0001 for loaded
fusion/head tensors, and 0.001 for the new DACP tensors. Mosaic is disabled for
the final 10 epochs. AFSS and every BECP loss/path are disabled.

## Diagnostics

Each run writes `dacp_diagnostics.csv` and `dacp_health.csv`. The primary checks are:

- `foreground_gap = foreground_mean - background_mean` should become positive;
- `effective_rho_*` should leave zero after epoch 5 without saturating;
- `prior_content_ratio_max` must remain at or below 0.1;
- validation metrics should be compared with
  `exp_rgca_mask_gfb_uniform_lr` at the same epoch budget and seed.
