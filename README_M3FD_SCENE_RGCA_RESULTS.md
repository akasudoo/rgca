# M3FD四场景完整RGCA实验记录

本文记录C0双骨干完整RGCA（Reliability-Guided Cross-modal Attention）模型在
M3FD WACV 2024四个场景划分上的专项训练与对应测试结果。每个场景均从
`yolov5s.pt`独立训练，只使用该场景的train集训练、val集选择`best.pt`，最后
仅在对应场景的test集评测。四个场景内部的train/val/test文件名交集均为0。

## 固定实验协议

- 模型：C0完整双骨干 + 四尺度RGCA + foreground Mask + GFB + 密集C3 PAN + P2-P5 Detect
- RGCA：无先验、双向交叉注意力、局部/全局混合、可靠性门控、`legacy_channel`残差尺度
- 输入：对齐RGB/IR，640×640
- 训练：200 epochs，physical batch 16，梯度累积4次，effective batch 64
- 优化器：SGD，`lr0=0.01`，`lrf=0.1`，momentum 0.937，weight decay 0.0005
- 调度：3 epochs warmup + cosine decay；最后10 epochs关闭Mosaic
- 选优：val集YOLO composite fitness
- 测试：batch 16，conf 0.001，NMS IoU 0.5，不启用TTA
- 复核环境：RTX 3090，PyTorch 1.13.1+cu117，2026-08-21
- 参数量：训练态13,743,436；Conv-BN融合验证态13,727,116

## Test汇总结果

| 场景 | Test图像 | Labels | Precision | Recall | mAP50 | mAP75 | mAP50:95 | 推理FPS | 含NMS FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daytime | 320 | 2,702 | 0.839 | 0.820 | 0.845 | 0.535 | 0.506 | 164.99 | 133.01 |
| Night | 140 | 1,141 | 0.935 | 0.964 | 0.973 | 0.673 | 0.625 | 87.42 | 76.43 |
| Overcast | 205 | 1,491 | 0.928 | 0.879 | 0.920 | 0.584 | 0.568 | 107.07 | 85.08 |
| Challenge | 156 | 1,726 | 0.916 | 0.828 | 0.896 | 0.495 | 0.504 | 94.06 | 79.99 |
| 四场景宏平均 | — | — | 0.9045 | 0.8728 | 0.9085 | 0.5718 | 0.5508 | — | — |

FPS为单次场景复核值，受数据扫描、GPU预热和测试集大小影响，不应代替统一
warmup/repeat协议下的正式速度对比。

## 各类别AP50

| 场景 | People | Car | Bus | Lamp | Motorcycle | Truck |
|---|---:|---:|---:|---:|---:|---:|
| Daytime | 0.748 | 0.927 | 0.901 | 0.888 | 0.704 | 0.903 |
| Night | 0.925 | 0.971 | 0.995 | 0.960 | 0.995 | 0.995 |
| Overcast | 0.921 | 0.950 | 0.852 | 0.961 | 0.840 | 0.995 |
| Challenge | 0.869 | 0.888 | 0.884 | 0.961 | 0.963 | 0.814 |

## 权重和复核目录

| 场景 | 专项训练权重 | SHA-256 | 复核输出目录 |
|---|---|---|---|
| Daytime | `runs/train/exp_rgca_mask_gfb_uniform_lr_m3fd_daytime_wacv2024/weights/best.pt` | `1b4cbca380630f2faf44485b4245441fb2b3eecfa48642e572a74835de6cba79` | `runs/test/m3fd_rgca_daytime_retrained_test320_recheck` |
| Night | `runs/train/exp_rgca_mask_gfb_uniform_lr_m3fd_night_wacv2024/weights/best.pt` | `508e90906364b03e8e53aa3140bf60d5982e62e2ceda70a87a2174607b478055` | `runs/test/m3fd_rgca_night_retrained_test140_recheck` |
| Overcast | `runs/train/exp_rgca_mask_gfb_uniform_lr_m3fd_overcast_wacv2024/weights/best.pt` | `0943750767055df240dd8b23b1c93fd8a69d7e4bd5a5fdcbb53e5e3709f7690f` | `runs/test/m3fd_rgca_overcast_retrained_test205_recheck` |
| Challenge | `runs/train/exp_rgca_mask_gfb_uniform_lr_m3fd_challenge_wacv2024/weights/best.pt` | `9d7975bac54afd5877b616bc18020e0d63fc4f91d23be2e89df3bb13220a19d8` | `runs/test/m3fd_rgca_challenge_retrained_test156_recheck` |

以上结果是“每个场景独立训练、对应场景独立测试”的场景专项结果，不是同一个
完整M3FD模型直接切分测试所得结果。后续LCAFNet原始注意力消融应复用完全相同
的数据划分、训练策略和测试阈值，才能与本表进行单变量比较。
