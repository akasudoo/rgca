# LCAFNet 双模态模型与 AKC-Mamba-Lite YOLO 改进说明

> 用途：记录当前项目中已经验证的原版模型、针对 YOLO Neck 的实验性改进、训练策略、权重迁移方法以及新项目复现步骤。
>
> 记录日期：2026-08-04
>
> 项目路径：`/home/dell/lcp/LCAFNet-main`（与 `/mydata/lcp/LCAFNet-main` 指向同一项目）

## 1. 版本结论与使用建议

当前项目包含两个需要严格区分的版本。

| 版本 | 结构组合 | 配置文件 | 当前结论 |
|---|---|---|---|
| 原版最优模型 | RGCA + 前景 Mask + GFB，无 prior、无 BECP、无 AFSS | `models/transformer/yolov5s_LCAFNet_M3FD.yaml` | 已完成训练验证，是当前论文和后续实验的可靠基线 |
| YOLO 改进实验版 | 原版最优结构 + 3 个 `C3AKCMambaLite` PAN 模块 | `models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml` | 可正常训练，但尚未证明最终指标优于原版，不应提前表述为有效改进 |

原版最优实验为：

```text
runs/train/exp_rgca_mask_gfb_uniform_lr
```

其最优权重为：

```text
runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt
```

论文主模型在没有得到更强的完整实验结果前，应继续定义为：

```text
双流 YOLOv5s + RGCA + 前景 Mask + GFB
```

其中不包含 prior、BECP、DACP、AFSS 或 AKC-Mamba-Lite。

---

## 2. 原版模型总体结构

### 2.1 输入与双流主干

模型接收严格配准的 RGB 与红外图像：

\[
I_r\in\mathbb{R}^{3\times H\times W},\qquad
I_i\in\mathbb{R}^{3\times H\times W}.
\]

RGB 和红外图像分别进入一套 YOLOv5s 主干网络。两条主干结构相同，但参数独立，因此能够学习各自模态的特征分布：

\[
F_r^l=B_r^l(I_r),\qquad F_i^l=B_i^l(I_i).
\]

在当前 YAML 中：

- 第 0～9 层为 RGB Backbone；
- 第 10～19 层为 IR Backbone；
- 第 20～23 层在四个尺度进行双模态融合；
- 检测头使用 P2、P3、P4、P5 四个检测尺度。

四组融合输入分别为：

```text
[2, 12]   -> P2/4
[4, 14]   -> P3/8
[6, 16]   -> P4/16
[9, 19]   -> P5/32
```

每个尺度均使用 `HAFFormerRGCAMaskGFB`。其内部顺序为：

```text
RGB/IR 特征
    ↓
RGCA 双向跨模态交互
    ↓
残差回注入两种模态
    ↓
前景稀疏融合 Mask
    ↓
GFB 自适应模态门控
    ↓
单路融合特征
```

### 2.2 检测尺度和锚框

原版采用四尺度检测，对较小目标更加友好：

| 检测层 | 步长 | Anchors |
|---|---:|---|
| P2 | 4 | `(5,6), (8,14), (15,11)` |
| P3 | 8 | `(10,13), (16,30), (33,23)` |
| P4 | 16 | `(30,61), (62,45), (59,119)` |
| P5 | 32 | `(116,90), (156,198), (373,326)` |

后续移植时不要遗漏 P2 检测层，也不要直接换成标准三尺度 YOLOv5s Detect，否则不再是结构严格相同的模型。

---

## 3. 原版双模态融合模块

原版融合模块位于 `models/common.py`，核心类如下：

```text
ReliabilityGuidedBidirectionalCrossAttention
ForegroundSparseFusionMask
GatedModalFusion
HAFFormerRGCAMaskGFB
```

类所在行号会随代码修改而变化，因此迁移时应通过类名搜索，而不要只依赖旧行号。

### 3.1 RGCA：可靠性引导的双向交叉注意力

RGCA 首先对两种模态进行归一化，并使用共享的 `1×1 QKV` 投影和深度可分离 `3×3` 卷积生成查询、键和值：

\[
(Q_r,K_r,V_r)=\phi(F_r),\qquad
(Q_i,K_i,V_i)=\phi(F_i).
\]

查询和键在空间维度上进行归一化，然后计算两个方向的通道交叉协方差注意力：

\[
A_{r\leftarrow i}
=\operatorname{Softmax}\left(\tau\widehat Q_r\widehat K_i^{\mathsf T}\right),
\]

\[
A_{i\leftarrow r}
=\operatorname{Softmax}\left(\tau\widehat Q_i\widehat K_r^{\mathsf T}\right).
\]

这里的注意力矩阵大小是每个 head 内的 `head_dim × head_dim`，而不是 `HW × HW`。因此它比全空间交叉注意力更节省显存和计算量。

跨模态全局值与局部深度卷积值按 head 进行混合：

\[
C_{r\leftarrow i}
=\lambda A_{r\leftarrow i}V_i
+(1-\lambda)L(V_i),
\]

\[
C_{i\leftarrow r}
=\lambda A_{i\leftarrow r}V_r
+(1-\lambda)L(V_r).
\]

其中 `L(·)` 为局部深度卷积，`λ` 为可学习的跨模态混合系数。

模块进一步根据主模态、补充模态、绝对差异和一致性构造可靠性门控：

\[
g_r=f_g([F_r,F_i,|F_r-F_i|,F_r\odot F_i]),
\]

\[
g_i=f_g([F_i,F_r,|F_i-F_r|,F_i\odot F_r]).
\]

最终输出的是待回注入的增量，而不是直接覆盖原特征：

\[
\Delta_r=s_r\odot P(g_r\odot C_{r\leftarrow i}),
\]

\[
\Delta_i=s_i\odot P(g_i\odot C_{i\leftarrow r}),
\]

\[
F'_r=F_r+\Delta_r,\qquad F'_i=F_i+\Delta_i.
\]

当前最优版使用 `legacy_channel` 残差缩放，即两个方向各自具有逐通道可学习缩放，初始值为 0.1。

#### 无 prior 的严格定义

最优配置明确使用：

```text
prior_mode = none
residual_scale_mode = legacy_channel
```

因此该模型不存在显式 Fourier prior、对角通道 prior、BECP 证据 prior 或 DACP prior。未来移植原版时必须保留这一设置，否则无法构成严格公平的原版复现。

### 3.2 前景 Mask

RGCA 更新后的两种模态特征用于预测一个空间前景门控：

\[
M=\sigma\left(f_M([F'_r,F'_i,|F'_r-F'_i|])\right).
\]

实现结构为：

```text
Concat -> 1×1 Conv -> DW 3×3 Conv -> 1×1 Conv -> Sigmoid
```

该 Mask 没有单独的像素级分割标签，而是通过最终检测损失端到端学习。它表达的是有利于检测任务的空间区域，不应直接解释成严格的语义分割结果。

### 3.3 GFB：门控模态融合

GFB 先根据两种模态的联合特征预测 RGB 门控 logits：

\[
\ell=DWConv\left(GELU\left(Conv_{1\times1}([F'_r,F'_i])\right)\right).
\]

随后使用前景 Mask 对 logits 进行修正：

\[
w_r=\sigma\left(\ell-\gamma(2M-1)\right),
\]

\[
w_i=1-w_r.
\]

融合输出为：

\[
F=w_r\odot F'_r+w_i\odot F'_i.
\]

需要注意，按照当前代码中的减号，当 `M` 较大时会降低 RGB 权重、相对增强红外特征。这是现有最优模型的真实行为，迁移时不要在未做消融的情况下擅自改成加号。

---

## 4. 原版实验记录

### 4.1 最优指标

实验目录：

```text
runs/train/exp_rgca_mask_gfb_uniform_lr
```

关键结果：

| 指标 | 数值 | Epoch |
|---|---:|---:|
| 最佳 mAP@0.5 | 0.914466 | 177 |
| 最佳 mAP@0.5:0.95 | 0.593820 | 192 |
| Epoch 192 对应 mAP@0.5 | 0.911689 | 192 |

模型参数量：

```text
13,743,436 parameters
```

在 RTX 3090、PyTorch FP32、640×640、batch size 1、随机输入、仅统计模型前向的历史测试中：

```text
平均延迟：47.559 ms
吞吐量：21.03 FPS
```

该 FPS 是模型级基准，不包含图像读取、预处理、NMS、结果保存和显示，因此不能直接当作完整部署流水线 FPS。

### 4.2 最优权重校验

```text
文件：runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt
大小：28,074,954 bytes
SHA256：12896c3b47017197d4ab1ea86e515332458dcea45f5e6742c380798e8039f882
```

如果迁移后得到的文件哈希不同并不一定代表权重错误，因为重新保存会改变 checkpoint 内容；但从当前项目复制原文件时，可使用哈希确认复制完整性。

---

## 5. 本次 YOLO Neck 改进：AKC-Mamba-Lite

### 5.1 改进位置

本次改动没有改变以下部分：

- RGB/IR 双流 Backbone；
- 四尺度 RGCA + Mask + GFB；
- Top-down FPN；
- P2 检测分支；
- Detect 层、anchors 和检测损失；
- 无 prior 设置。

仅将 PAN 自底向上的三个输出 C3 模块替换为 `C3AKCMambaLite`：

| YAML 层 | 输出尺度 | 原模块 | 新模块 |
|---:|---|---|---|
| 38 | P3/8 | `C3` | `C3AKCMambaLite` |
| 41 | P4/16 | `C3` | `C3AKCMambaLite` |
| 44 | P5/32 | `C3` | `C3AKCMambaLite` |

配置参数为：

```text
[out_channels, True, 5, 0.5, 0.25]
```

新配置文件：

```text
models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml
```

### 5.2 模块组成

相关实现都位于 `models/common.py`：

```text
AdaptiveKernelConv2d
AKCResidualBlock
SelectiveAxisStateSpace2D
AKCChannelRecalibration
C3AKCMambaLite
```

模块整体结构为：

```text
CSP 输入
 ├─ 变换分支：AKCResidualBlock -> SelectiveAxisStateSpace2D
 │                             -> AKCChannelRecalibration
 └─ 旁路分支：轻量卷积
                ↓
             Concat + Conv
```

CSP 旁路保留原始和局部信息，避免所有通道都经过状态空间递推而破坏 YOLO 的局部检测特征。

### 5.3 自适应核卷积 AKC

AKC 对每个位置预测一组有界偏移，并通过双线性插值采样邻域特征。第 `n` 个采样点为：

\[
p_n(u)=u+p_n^0+2\tanh\left(\frac{o_n(u)}{2}\right),
\]

其中：

- `p_n^0` 为固定的初始采样位置；
- `o_n(u)` 为可学习偏移；
- `2·tanh(·/2)` 将偏移限制在约 ±2 像素；
- 当前使用 5 个对称采样点。

采样结果沿通道拼接，再使用 `1×1` 卷积聚合：

\[
y(u)=W_a[\widetilde x(p_1(u)),\ldots,\widetilde x(p_N(u))].
\]

偏移层采用零初始化，因此初始状态接近规则采样；偏移分支的梯度缩放为 0.1，用于减小训练早期采样位置剧烈漂移的风险。

当前实现基于 PyTorch `grid_sample`，无需额外 CUDA 扩展，但其 GPU 延迟可能高于普通深度卷积。

### 5.4 轻量选择性轴向状态空间模块

该模块沿水平方向和垂直方向分别执行前向与反向递推。输入相关的保留率定义为：

\[
a_t=0.90+0.09\sigma(g_t),
\]

状态更新为：

\[
h_t=a_t\odot h_{t-1}+(1-a_t)\odot v_t.
\]

水平方向与垂直方向的双向结果经融合后，以可学习残差尺度回注入。该尺度初始值为 0.1，以尽量保持初始化阶段接近普通 CSP/C3 行为。

AMP 下的累计乘积和累计求和使用 FP32 计算，以减小长序列半精度数值误差。

### 5.5 通道重标定

通道分支使用类似 SE 的全局通道建模。最终扩展层为零初始化，门控形式为：

\[
y=x\odot(0.5+\sigma(f_c(x))).
\]

由于零初始化时 `σ(0)=0.5`，初始有效乘数严格为 1，因而不会在刚开始训练时整体压低或放大特征。

### 5.6 对“Mamba”命名的严格说明

本项目的 `C3AKCMambaLite` 是受 Mamba/选择性状态空间思想启发的轻量实现，但它不是标准 Mamba/S6 Block，也不是 AKCMamba-YOLO 论文代码的逐行复现。

建议在代码或实验记录中称为：

```text
AKC-Mamba-Lite
或
Mamba-inspired lightweight selective state-space PAN block
```

不建议直接宣称“采用标准 Mamba Block”。

对参考项目 `/home/dell/lcp/AKCMamba_YOLO` 的检查表明：

- 其 `SS2D` 更接近 VMamba/VSS 风格的二维选择性扫描；
- 使用动态 `Δ/B/C/A/D/z` 和四方向 CrossScan；
- 依赖 `selective_scan_cuda_core` 自定义 CUDA 扩展；
- 当前提供的工程中没有发现论文完整 `3CAK/4CAK/AKCAttention` 与 AKCMamba YAML 的完整组合；
- 默认入口仍可能加载普通 YOLOv8 配置；
- 当前 `MOD` 环境没有安装 `mamba_ssm` 或 `selective_scan_cuda`。

因此，本项目选择了无自定义 CUDA 依赖的可训练轻量版本，以保证当前环境能直接运行。

---

## 6. 改进版当前性能与诊断结论

### 6.1 参数和速度

改进版参数量：

```text
13,626,973 parameters
```

与原版相比：

```text
减少 116,463 个参数，约 -0.85%
```

相同 RTX 3090、FP32、640×640、batch size 1 的历史模型级基准：

```text
平均延迟：56.799 ms
吞吐量：17.61 FPS
```

尽管参数更少，速度却比原版约慢 16.3%。原因是参数量不能代表实际硬件效率，`grid_sample`、轴向累计运算以及大量小算子的 kernel launch 成本都不利于 GPU 吞吐。

### 6.2 当前训练中的模块状态

当前实验的阶段性诊断显示：

- AKC 偏移均值已经离开 0，说明自适应采样分支确实在更新；
- 轴向 retention 的标准差仍很小，输入选择性较弱；
- 状态空间上下文残差尺度从约 0.1 降到约 0.044，说明优化器倾向于抑制该分支；
- 通道有效乘数仍接近 1，通道重标定接近恒等映射。

这说明“代码能训练”不等同于“所有新分支都提供了有效增益”。当前结果只能用于判断该实验正常运行，不能用于声称改进已经成立。

建议至少比较到 Epoch 80，并使用与原版相同 epoch 区间的 `mAP@0.5:0.95`、检测损失和模块诊断量共同判断。若最终没有超过原版，应将该模块记录为负结果或继续简化，而不是替换论文主模型。

诊断 CSV 的字段为：

```text
epoch
module
offset_abs_mean
offset_std
offset_abs_max
horizontal_retention_mean
horizontal_retention_std
vertical_retention_mean
vertical_retention_std
context_scale
channel_gate_mean
channel_gate_std
```

---

## 7. 训练策略

### 7.1 原版公平基线策略

原版最优实验的核心设置为：

| 参数 | 设置 |
|---|---|
| 初始化权重 | `yolov5s.pt` |
| 权重加载范围 | 双流 backbone |
| Epochs | 200 |
| Batch size | 16 |
| 输入尺寸 | 640×640 |
| 优化器 | SGD |
| 初始学习率 | 0.01 |
| 最终学习率系数 | 0.1 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup | 3 epochs |
| LR 调度 | cosine one-cycle |
| Seed | 1 |
| Workers | 8 |
| AMP | 开启 |
| Close mosaic | 最后 10 epochs |
| Differential LR | 关闭 |
| AFSS | 关闭 |
| Prior 分段训练 | 关闭 |

严格复现实验时，不要同时更改优化器、数据增强、epoch、batch size 或预训练策略，否则无法将差异归因到网络结构。

原版控制组命令示例：

```bash
cd /home/dell/lcp/LCAFNet-main
conda activate MOD
python train.py \
  --cfg models/transformer/yolov5s_LCAFNet_M3FD.yaml \
  --hyp data/hyp.rgca_mask_gfb_pretrained.yaml \
  --workers 8 \
  --name exp_rgca_mask_gfb_no_prior_control_m3fd
```

### 7.2 当前改进版默认策略

当前 `train.py` 默认配置已经指向 AKC-Mamba-Lite 版本：

```text
weights: /home/dell/lcp/LCAFNet-main/yolov5s.pt
cfg: models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml
hyp: data/hyp.rgca_mask_gfb_akcmamba_joint.yaml
epochs: 200
batch-size: 16
img-size: 640
optimizer: SGD
workers: 8
pretrained-scope: backbone
differential-lr: false
AFSS: false
prior staged training: false
```

运行方式：

```bash
cd /home/dell/lcp/LCAFNet-main
conda activate MOD
python train.py
```

该命令会训练实验版，不是原版最优结构。新项目中建议保留显式 `--cfg` 的运行脚本，避免以后因为 `train.py` 默认值变化而混淆实验组合。

### 7.3 速度和显存设置

当前已使用：

- `workers=8`；
- 固定输入尺寸下启用 cuDNN benchmark；
- AMP 混合精度；
- `optimizer.zero_grad(set_to_none=True)`；
- DataLoader pin memory 和 GPU non-blocking copy；
- OpenCV 内部线程关闭，避免和 DataLoader 抢占 CPU。

在 RTX 3090 上，640 输入、batch size 16 的完整前向反向已验证可运行。此前 batch size 16 的显存峰值约 16.1 GB，因此 24 GB 显存不建议未经实测直接升到 batch size 32。

当前主机内存条件也不适合盲目开启 `--cache-images`。PyTorch 1.13 环境不能依赖 PyTorch 2.x 的 `torch.compile` 加速。

---

## 8. YOLOv5s 权重迁移

### 8.1 为什么可以加载单流 `yolov5s.pt`

双模态模型不能把单流 YOLOv5s checkpoint 直接按完整模型严格加载，但可以将其 Backbone 权重分别复制到 RGB 和 IR 两条结构相同的主干：

\[
\theta_r^0\leftarrow\theta_{YOLO},\qquad
\theta_i^0\leftarrow\theta_{YOLO}.
\]

项目中的 `build_dual_stream_backbone_state_dict` 完成以下操作：

1. 读取标准 YOLOv5s Backbone；
2. 将同一套权重映射到 RGB Backbone；
3. 再映射到 IR Backbone；
4. 处理旧版 Focus 到 6 通道卷积的转换；
5. 融合模块、Neck 和 Detect 中无法对应的层保持随机初始化。

已验证加载结果为：

```text
RGB Backbone: 198 tensors
IR Backbone:  198 tensors
Total:        396 tensors
Missing in expected backbone mapping: 0
```

必须使用：

```text
--pretrained-scope backbone
```

不要对单流 `yolov5s.pt` 使用 `pretrained-scope=all`，因为单流检测头和当前双流四尺度模型并不结构等价。

标准权重记录：

```text
文件：yolov5s.pt
大小：14,795,158 bytes
SHA256：f1610cfd81f8cab94254b35f6b7da2981fa40f93ad1bd3dd1803c52e7f44753e
```

---

## 9. 新项目迁移清单

### 9.1 原版模型最小迁移范围

新项目要复现原版最优模型，至少需要迁移或重新实现：

```text
models/common.py
models/yolo_test.py
models/transformer/yolov5s_LCAFNet_M3FD.yaml
utils/datasets.py
train.py
data/multispectral/M3FD.yaml
data/hyp.rgca_mask_gfb_pretrained.yaml
```

需要从 `models/common.py` 保留的核心类：

```text
ReliabilityGuidedBidirectionalCrossAttention
ForegroundSparseFusionMask
GatedModalFusion
HAFFormerRGCAMaskGFB
```

同时必须保留：

- 双输入数据集读取与 RGB/IR 一一配对逻辑；
- `model(rgb, ir)` 双输入 forward 路径；
- YAML parser 对 `HAFFormerRGCAMaskGFB` 的识别；
- 单流 YOLOv5s 到双流 Backbone 的权重映射；
- P2/P3/P4/P5 四尺度 Detect；
- `prior_mode=none` 和 `legacy_channel`。

仅复制 YAML 而使用标准单输入 YOLOv5 代码，模型不会正常复现。

### 9.2 改进版额外迁移范围

如果还要复现 AKC-Mamba-Lite 实验版，需要额外迁移：

```text
models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml
data/hyp.rgca_mask_gfb_akcmamba_joint.yaml
tests/test_akcmamba_lite.py
tests/test_training_strategy.py
```

以及以下五个类：

```text
AdaptiveKernelConv2d
AKCResidualBlock
SelectiveAxisStateSpace2D
AKCChannelRecalibration
C3AKCMambaLite
```

还需在 YAML parser 中注册 `C3AKCMambaLite`，并保留 `train.py` 中的模块诊断记录逻辑。

### 9.3 数据集要求

数据加载器必须保证 RGB 和 IR 样本严格对应。建议新项目开始前先检查：

1. RGB 和 IR 文件数量一致；
2. 文件名或映射表一一对应；
3. 几何增强对两种模态使用完全相同的随机参数；
4. 标签坐标与两个模态保持一致；
5. 训练、验证和测试划分不交叉；
6. 推理时的 resize、letterbox 和坐标还原与验证阶段相同。

双模态任务中，如果两幅图像分别随机裁剪、翻转或缩放，注意力与门控模块会学习到错误对应关系，即使训练损失仍可能下降。

---

## 10. 新项目复现后的验收步骤

建议按以下顺序验收，不要直接开始长时间训练。

### 10.1 原版结构验收

1. 使用原版 YAML 构建模型；
2. 确认存在 4 个 `HAFFormerRGCAMaskGFB`；
3. 确认不存在 prior、BECP、DACP 和 `C3AKCMambaLite`；
4. 确认参数量为 `13,743,436`；
5. 用 RGB/IR 随机张量完成一次前向；
6. 用一个小 batch 完成 AMP 前向与反向；
7. 验证 Detect 输出包含 P2～P5 四尺度；
8. 加载 `yolov5s.pt` 后确认两条 Backbone 均成功迁移。

### 10.2 改进版结构验收

1. 使用 AKC-Mamba-Lite YAML 构建模型；
2. 确认存在 4 个无 prior 融合模块；
3. 确认存在 3 个 `C3AKCMambaLite`；
4. 确认参数量为 `13,626,973`；
5. 完成 FP32 和 AMP 前向、反向；
6. 检查偏移、retention、context scale 和 channel gate 能被记录；
7. 使用与原版完全相同的训练设置做结构公平比较。

当前项目已经完成的回归测试记录：

```text
unittest discover: 53 passed
CUDA AMP batch=16, img=640 forward/backward: passed
checkpoint save/load: passed
```

---

## 11. 推理和部署注意事项

原版 RGCA 使用通道交叉注意力，计算量相对可控，更适合继续做 PyTorch 推理和部署优化。

AKC-Mamba-Lite 当前使用：

- `grid_sample`；
- `cumprod/cumsum` 形式的轴向递推；
- AMP 中局部 FP32 计算。

这些操作在 PyTorch 中可正常运行，但导出 ONNX/TensorRT 前必须单独检查算子支持、动态尺寸和数值一致性。不能仅因为 PyTorch 推理成功，就默认 TensorRT 可以无修改部署。

正式报告 FPS 时应同时说明：

- GPU 型号；
- PyTorch、CUDA 和 cuDNN 版本；
- FP32/FP16；
- batch size；
- 输入分辨率；
- 是否包含预处理、NMS 和数据传输；
- warmup 次数和计时迭代次数。

---

## 12. 论文写作边界

### 12.1 原版可以稳定描述的贡献

原版模型可围绕以下逻辑组织：

1. 使用双流主干保留可见光纹理和红外热目标的模态专属性；
2. 使用轻量通道交叉协方差注意力进行双向跨模态信息交换；
3. 使用可靠性门控限制不可靠补充特征的注入；
4. 使用检测任务驱动的前景 Mask 聚焦目标相关区域；
5. 使用 GFB 在不同位置自适应分配 RGB/IR 权重；
6. 使用 P2～P5 四尺度检测兼顾小目标。

### 12.2 改进版在结果未确认前的表述

在完整实验没有超过原版前，只能表述为“针对 YOLO PAN 的候选结构探索”。不能直接写成：

```text
AKC-Mamba-Lite 显著提升了检测精度和速度
```

因为当前已知事实是参数略有下降，但推理更慢，而且阶段性精度尚未形成稳定优势。

如果最终结果没有提升，仍可将它保留为内部负实验，用于支持以下设计认识：复杂全局建模并不一定适合已经完成多尺度跨模态融合的 Neck，额外长程分支可能被优化器主动抑制。

---

## 13. 环境记录

当前已验证环境：

```text
OS: Linux
Python: 3.9.23
PyTorch: 1.13.1+cu117
GPU: NVIDIA RTX 3090
Environment: /home/dell/anaconda3/envs/MOD
```

当前 AKC-Mamba-Lite 不依赖：

```text
mamba_ssm
selective_scan_cuda
selective_scan_cuda_core
```

因此新环境只要能够运行原项目的 PyTorch/CUDA 依赖，即可构建和训练该实验版；不需要编译额外 Mamba CUDA 扩展。

---

## 14. 版本记录模板

以后新建项目或继续修改时，建议复制下面的模板记录每一次结构变化：

```text
版本名称：
日期：
父版本：
配置文件：
新增模块：
删除模块：
是否包含 prior：
是否包含 AFSS：
初始化权重及加载范围：
数据集与划分：
随机种子：
输入尺寸：
Batch size：
优化器和学习率：
参数量：
FLOPs：
FPS 测试条件：
最佳 mAP@0.5：
最佳 mAP@0.5:0.95：
权重路径：
代码提交或归档路径：
结论：
```

对所有消融实验，应一次只改变一个核心变量，并使用独立的实验名称。不要覆盖 `exp_rgca_mask_gfb_uniform_lr` 及其权重，它是目前所有后续研究的基准锚点。

