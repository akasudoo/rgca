# Group A — matched BECP-off control

This completed 200-epoch run is the structure-matched BECP-off control for the
from-scratch A/D experiment. It is retained as an experimental record but is
no longer the current `python train.py` default.

## Model combination

- Corrected reliability-guided bidirectional cross-attention (RGCA)
- Foreground-mask gated fusion
- GFB fusion
- Complete BECP evidence controller retained at all four fusion scales (304 parameters total)
- Corrected bounded bidirectional residual scaling retained (`residual_init=0.1`)

## Controlled treatment

`becp_off_control=true` forces both causal BECP contributions to exactly zero for every epoch:

- BECP channel-log-prior scale: `0`
- Auxiliary BECP evidence-loss scale: `0`

The BECP controller, matched/mismatched evidence computation, parameter count, optimizer parameter group, and diagnostics remain present. The controller receives no gradient and does not update. This is therefore a structure-matched BECP-off control, not the historical model in which the prior/controller class is absent.

## Training recipe shared with Group D

- Dataset: M3FD
- Initialization: classic `yolov5s.pt`, transferred to both modal backbones
- Epochs: 200
- Image size: 640
- Batch size: 16
- Optimizer: SGD
- Initial learning rate: 0.01
- Final LR factor: 0.1, cosine schedule
- Warmup: 3 epochs
- Weight decay: 0.0005 before nominal-batch scaling
- Mosaic: enabled, then closed for the final 10 epochs
- Seed: 1
- AFSS: disabled
- Differential learning rate: disabled

Paired treatment run: `runs/train/exp_rgca_mask_gfb_becp_yolov5s_fulltrain200_m3fd` (Group D — Full BECP, stopped after 121 complete epochs).

Expected output directory: `runs/train/exp_group_A_becp_off_rgca_mask_gfb_yolov5s_fulltrain200_m3fd`.
