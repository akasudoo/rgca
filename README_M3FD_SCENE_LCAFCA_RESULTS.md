# M3FD四场景LCAF注意力实验验证结果

本文记录C0双骨干LCAF注意力模型在M3FD WACV 2024四个场景划分上的
专项训练和对应测试结果。四个场景各自使用独立模型：只使用对应场景的
train集训练，在对应val集上选择`best.pt`，最后在对应test集上评测。

## 模型与评测协议

- 模型：C0完整双骨干 + 原版LCAFNet双向交叉注意力 + foreground Mask + GFB
  + 密集C3 PAN + P2-P5 Detect
- 消融条件：移除RGCA及其可靠性门控，其余结构和训练策略不变
- 输入：对齐RGB/IR，640×640
- 测试：batch 16，confidence 0.001，NMS IoU 0.5，不使用TTA
- 类别：People、Car、Bus、Lamp、Motorcycle、Truck
- 验证态参数量（Conv-BN融合后）：15,660,264（15.660M）
- 复杂度：23.358 GMACs / 46.716 GFLOPs（batch=1，双模态640×640，
  按1 MAC=2 FLOPs；THOP可能漏计未注册的函数式张量操作）
- 复核环境：RTX 3090，PyTorch 1.13.1+cu117，2026-08-21

## Test集汇总结果

| 场景 | Test图像 | Labels | Precision | Recall | mAP50 | mAP75 | mAP50:95 | 推理FPS | 含NMS FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daytime | 320 | 2,702 | 0.860 | 0.770 | 0.828 | 0.490 | 0.487 | 247.11 | 172.69 |
| Night | 140 | 1,141 | 0.890 | 0.902 | 0.912 | 0.664 | 0.614 | 91.27 | 77.27 |
| Overcast | 205 | 1,491 | 0.928 | 0.838 | 0.889 | 0.539 | 0.543 | 133.06 | 110.37 |
| Challenge | 156 | 1,726 | 0.913 | 0.824 | 0.906 | 0.511 | 0.520 | 100.57 | 80.23 |
| 四场景宏平均 | — | — | 0.8978 | 0.8335 | 0.8838 | 0.5510 | 0.5410 | — | — |

宏平均是四个场景指标的等权算术平均，并不是把四个test集预测合并后重新计算的
全局指标。FPS是本次单次复核输出，受首批预热、场景规模和I/O影响；论文中的
正式速度比较应使用相同权重、固定warmup/repeat和相同机器重新测试。

## 各类别AP50

| 场景 | People | Car | Bus | Lamp | Motorcycle | Truck |
|---|---:|---:|---:|---:|---:|---:|
| Daytime | 0.786 | 0.921 | 0.889 | 0.866 | 0.617 | 0.890 |
| Night | 0.900 | 0.964 | 0.995 | 0.875 | 0.863 | 0.876 |
| Overcast | 0.922 | 0.947 | 0.785 | 0.953 | 0.747 | 0.984 |
| Challenge | 0.876 | 0.883 | 0.930 | 0.971 | 0.967 | 0.810 |

## 与完整RGCA模型的单变量对比

下表使用相同的场景划分、训练轮数、输入尺寸及test评测参数。差值为
`LCAF注意力 - 完整RGCA`；正数代表本次LCAF注意力模型更高。

| 场景 | RGCA mAP50 | LCAF mAP50 | 差值 | RGCA mAP50:95 | LCAF mAP50:95 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Daytime | 0.845 | 0.828 | -0.017 | 0.506 | 0.487 | -0.019 |
| Night | 0.973 | 0.912 | -0.061 | 0.625 | 0.614 | -0.011 |
| Overcast | 0.920 | 0.889 | -0.031 | 0.568 | 0.543 | -0.025 |
| Challenge | 0.896 | 0.906 | +0.010 | 0.504 | 0.520 | +0.016 |
| 四场景宏平均 | 0.9085 | 0.8838 | -0.0247 | 0.5508 | 0.5410 | -0.0098 |

完整RGCA在四场景宏平均上更优，尤其明显改善Night和Overcast的mAP50；原版
LCAF注意力只在Challenge上取得更高结果。该结果支持可靠性引导机制对多数场景
具有整体收益，但正式论文结论仍应结合多随机种子均值和标准差。

## 权重和复核输出

| 场景 | 专项训练权重 | SHA-256 | 复核输出目录 |
|---|---|---|---|
| Daytime | `runs/train/exp_lcafca_mask_gfb_uniform_lr_m3fd_daytime_wacv2024/weights/best.pt` | `40431992c8f716cc5fb065cbbe3a327a00bf1e5e10c42320263a4ca422f45f10` | `runs/test/m3fd_lcafca_daytime_retrained_test320_final` |
| Night | `runs/train/exp_lcafca_mask_gfb_uniform_lr_m3fd_night_wacv2024/weights/best.pt` | `4e8ce5e5c465be0c704d2a428dad9fa64b0dc9c38a54754e5f5b49987f1289ec` | `runs/test/m3fd_lcafca_night_retrained_test140_final` |
| Overcast | `runs/train/exp_lcafca_mask_gfb_uniform_lr_m3fd_overcast_wacv2024/weights/best.pt` | `7e4ddd0c8a5e2b3dcc59423c946ec690c87ff0b1f08cf14614345dc5afeb90b8` | `runs/test/m3fd_lcafca_overcast_retrained_test205_final` |
| Challenge | `runs/train/exp_lcafca_mask_gfb_uniform_lr_m3fd_challenge_wacv2024/weights/best.pt` | `2d28863a3a609c4353788ab0738d080067f0fdf784ccc155858618260e63c666` | `runs/test/m3fd_lcafca_challenge_retrained_test156_final` |

每个复核目录均保存PR、P、R、F1曲线、混淆矩阵以及标签/预测可视化图。
这些结果属于“每个场景独立训练、对应场景独立测试”的专项模型结果，不能表述为
同一个模型在完整M3FD test集上的统一泛化结果。
