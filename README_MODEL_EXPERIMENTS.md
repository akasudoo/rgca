# LCAFNet 当前模型与实验探索记录

> 最后更新：2026-08-09  
> 用途：记录当前代码状态、关键实验结果和已经尝试过的改进，供后续实验及新的 Codex 对话快速接手。  
> 原项目说明仍见 `README.md`，本文档不替代原始 README。

> **2026-08-09 当前活动入口：FLIR C0 从 yolov5s.pt 正式训练200轮。** 论文主
> 架构仍是 C0 完整双流 RGCA+Mask+GFB；默认执行 `python train.py` 时，使用
> `yolov5s_LCAFNet_FLIR.yaml`、FLIR完整配对数据和 canonical `yolov5s.pt`。
> RGB/IR 两条 C2--C5 backbone 各加载 `198` 个状态，共 `396/396`；四尺度
> RGCA+Mask+GFB、密集 PAN 和三类四尺度 Detect 保持模型初始化。训练协议为
> SGD、统一 `lr0=0.01`、batch 16、640输入、200 epochs、3 epoch warmup，
> 最后10 epochs关闭 mosaic。默认输出目录为
> `runs/train/exp_rgca_mask_gfb_c0_full_dual_joint200_yolov5s_flir`。

> **论文主模型选择不变：** M3FD C0 使用 `yolov5s_LCAFNet_M3FD.yaml`，参数量
> `13,743,436`；历史控制实验 `exp_rgca_mask_gfb_uniform_lr` 独立最高
> mAP@0.5=`0.914466`、mAP@0.5:0.95=`0.593820`。本次仅临时切换数据集和初始化
> 协议，验证 C0 在 FLIR 上从 canonical backbone 完整训练的表现，没有回到
> C1/C1.5。

> **C1.5 正式实验已完成并退出活动入口：** 200轮独立最高
> mAP@0.5=`0.902993`（epoch 142）、mAP@0.5:0.95=`0.585030`（epoch 184），
> 均低于 C0 的 `0.914466/0.593820`，也低于参数更少的 C1。C1.5 说明仅保留
> 双流 C4、共享 C5 的折中没有形成有效精度—参数 Pareto 改善；代码、配置、
> 权重和结果完整保留为结构消融，不再作为论文主模型。

> **前一活动入口（C0→C1.5迁移筛选）：** 30轮实验已完成，最佳mAP@0.5
> `0.913595`、mAP@0.5:0.95 `0.591893`（epoch 29）。它证明C0可压缩为C1.5
> 后接近原精度，但因继承C0全部目标状态，只作为恢复能力筛选，不作为公平结构结论。

> **前一活动入口（LLVIP C1 结构迁移）：** 已从完整双流 `LCAFNet_LLVIP.pt`
> 迁移到一类 LLVIP C1 并完成 50 epochs。该轮实际保存配置为 SGD、
> `lr0=0.001`、warmup 3、batch 8、1024；最佳 mAP@0.5:0.95 为 `0.651328`
>（epoch 4），后续明显退化，因此只使用 `best.pt`，不以它替代公平结构对比。

> **前一活动入口（FLIR C1 结构迁移）：** 曾配置为从完整双流
> `LCAFNet_FLIR.pt` 迁移到三类 FLIR C1，使用 batch 16、640 输入微调 50
> epochs；该入口已被本次 LLVIP 任务替换，配置、权重和文档记录均完整保留。

> **更早活动入口（M3FD C1 同构微调控制）：** 曾配置为从已完成 200 epochs 的
> M3FD C1 `best.pt` 加载全部同构状态，以新建优化器继续低学习率训练 30 epochs。

> **2026-08-09 路线最终判断：** C1 将参数从完整双流 C0 的 `13,743,436`
> 降至 `8,045,752`（减少 `41.46%`），Best mAP@0.5:0.95 仅从 `0.593820`
> 降至 `0.591458`（下降 `0.236` 个百分点），因此 C1 是有效的轻量 Pareto
> 基线而不是失败模型。C2 私有低秩旁路、C3 GT 前景辅助监督和两版 D3C3 PAN
> 均未形成收益，后续不再沿低秩补偿、辅助损失、全 PAN 深度大核或 Mamba 堆叠
> 路线继续。C1.5 完整训练同样未超过 C0/C1，因此论文主模型和默认训练入口已
> 回到具有最高已验证精度的 **C0 完整双流 RGCA+Mask+GFB**；C1 可作为核心
> 轻量化消融和部署对照。详见本文第 8 节。

> **2026-08-06 C4 保守修订版历史入口：** 六层 D3C3 C4 最终在 epoch 64
> 停止，最佳/最后 10 轮平均 fitness 为 `0.694513/0.684480`，未达到预设的
> `0.695/0.690` 判定线。当时严格回到 C1，只将输出 P4/P5 的
> 模型层 34/37 换成 D3C3，保留 top-down、P2/P3 的密集 C3。参数量
> `7,418,424`，较 C1 减少 `7.80%`。正式 from-scratch 结论仍要求从同一个
> `yolov5s.pt` 重新训练；该配置现已退出默认入口。
> 详见 [`paper/C4_DEEP_P4P5_D3C3_20260806.md`](paper/C4_DEEP_P4P5_D3C3_20260806.md)。

> **历史 C4 30-epoch 筛选方式：** 当时 `python train.py` 加载 C1
> `best.pt`，完整迁移 `558/558` 个兼容状态（含层 34/37 外层 CSP 投影），
> 仅两个内部 D3 mixer 的 60 个新状态保持随机初始化；全部参数用 SGD
> `lr0=0.001` 联合微调。它用于
> 低成本验证结构手术后的恢复能力，不等同于 from-scratch 公平结论。

> **2026-08-06 C4 六层替换历史入口：** C3 完成 200 epochs 后，最佳
> `mAP@0.5:0.95=0.586240`，明确低于 C1/C2。C4 因此不继承私有旁路，而是
> 直接回到 C1，只将 PAN 的 6 个 C3 换成 YOLO-ULM 启发的 D3C3。参数量
> `7,118,776`，较 C1 再减少 `926,976`（11.52%），估算 MAC 降低 12.51%；
> P2/P3/P4/P5 全部保留，不含 RepDown、新损失、Mamba 或蒸馏。PyTorch eager
> 实测延迟尚未改善，因此当前只能称为参数/MAC 轻量候选。详见
> [`paper/C4_D3C3_SLIM_PAN_20260806.md`](paper/C4_D3C3_SLIM_PAN_20260806.md)。

> **2026-08-05 C3 历史入口：** C2 完成 200 epochs 后，最佳
> `mAP@0.5:0.95=0.590481`，没有超过 C1 的 `0.591458`；诊断显示私有旁路
> 确实参与训练，但 C4/C5 注入比例在后期持续下降。C3 因此保持 C2 推理图不变，
> 只在训练时用 GT 框生成类别无关的软前景热图，分别直接监督 RGB/IR 私有状态。
> C3 共 `8,075,330` 个 checkpoint 参数，仅比 C2 多 68 个训练头参数，部署时
> 不执行这些头；不使用教师、蒸馏或跨模态特征强制对齐。正式默认实验仍从同一个
> `yolov5s.pt` 训练 200 epochs。详见
> [`paper/C3_GT_PRIVATE_FOREGROUND_20260805.md`](paper/C3_GT_PRIVATE_FOREGROUND_20260805.md)。
> C3 完整代码和 200-epoch 结果已存档到
> `archives/c3_gt_private_foreground_completed_20260806.tar.gz`。

> **2026-08-05 C2 历史入口：** C1 完成 200 epochs 后，最佳
> `mAP@0.5:0.95=0.591458`，在参数量从 `13,743,436` 降到 `8,045,752`
> 的同时仅比完整双流控制组低 `0.236` 个百分点。现按原实验规划进入 C2：
> C1 主路完全保留，在 C4/C5 加入 RGB/IR 参数不共享的 rank-16 低秩旁路。
> C2 共 `8,075,262` 参数（仅比 C1 多 `29,510`），不含教师蒸馏或额外损失。
> 结构、公式、论文依据、诊断及判废标准见
> [`paper/C2_PRIVATE_LOW_RANK_BYPASS_20260805.md`](paper/C2_PRIVATE_LOW_RANK_BYPASS_20260805.md)。
> C2 完整代码与结果已存档到
> `archives/c2_private_lowrank_c4c5_completed_20260805.tar.gz`。

> **2026-08-04 C1 轻量主干：** AKCMambaLite 实验在完成 epoch 111 并写入
> `last.pt` 后正常中断。当时的默认入口改为“双流到 C3、之后单路 C4/C5”：
> C2 融合保留 P2 skip，C3 为主合流点，原 PAN/Detect 不变。C1 参数量
> `8,045,752`，较完整双流控制减少 `41.46%`。实现、迁移映射和命令见
> [`paper/C1_DUAL_TO_SHARED_C4C5_20260804.md`](paper/C1_DUAL_TO_SHARED_C4C5_20260804.md)。
> 未加入 AKCMambaLite 的控制代码存档为
> `archives/rgca_mask_gfb_uniform_lr_control_20260804_2331.tar.gz`。

> **2026-08-05 训练入口精简：** 活动 `train.py` 从 2853 行收敛为约 1000 行
>（其中包含集中、带注释的完整训练参数区），
> 只保留 C1/C4 需要的双模态数据、YOLOv5s backbone 映射、统一学习率 SGD、AMP、
> EMA、验证、诊断、checkpoint 和断点续训。旧 BECP/DACP/AFSS/AK/频域消融
> 训练器移入 `archives/train_experiments_legacy_20260805.py`，不会被
> `python train.py` 的正常训练路径导入。
> 常用参数和完整超参数现集中在 `train.py` 顶部的 `TRAINING_CONFIG` 与
> `HYPERPARAMETERS` 两个字典中；修改后仍直接运行 `python train.py`。

> **2026-08-04 YOLO neck 历史实验：** 曾在已恢复的
> **无 prior RGCA + 前景 Mask + GFB** 上，启用 AKCMamba-YOLO 启发的
> `C3AKCMambaLite` PAN 输出颈部；该实验已在 epoch 111 正常中断，不再是
> 默认入口。原最优结构 YAML 保持不变作为严格控制组。
> 新设计、实现边界、复杂度和判废条件见
> [`paper/AKCMAMBA_YOLOV5_ADAPTATION_20260804.md`](paper/AKCMAMBA_YOLOV5_ADAPTATION_20260804.md)。

> **2026-08-04 无 prior 恢复：** `exp_rgca_mask_gfb_uniform_lr` 对应的
> **无 prior RGCA + 前景 Mask + GFB** 已完整恢复并保留为控制结构；fixed prior、
> BECP、DACP、prior/evidence 辅助损失仍不进入新实验。恢复清单见
> [`paper/NO_PRIOR_RESTORATION_20260804.md`](paper/NO_PRIOR_RESTORATION_20260804.md)。
> 本文后续较早的 prior/频域描述仅是历史实验记录，不代表当前默认入口。

> **2026-08-01 结构替换说明：** 修正熵路由的统一学习率完整实验最佳
> `mAP@0.5:0.95=0.5852`，低于 `exp_mask=0.5919`，频域路由已从默认
> 活动路径完全移除。当前默认模块是无 FFT 的
> `HAFFormerRGCAMaskGFB`。本文第 3～8 节中关于频域模型的文字保留为
> 历史实验记录，不再表示当前默认实现。新模型的结构、论文依据、复杂度、
> 训练设置和判废标准统一见 [`paper/RGCA_DESIGN.md`](paper/RGCA_DESIGN.md)。

## 1. 当前结论速览

- 当前活动模型为 M3FD 6 类 C0：**RGB/IR 各自保留完整 C2--C5/SPPF
  backbone，并在 P2/P3/P4/P5 四尺度融合**；原始密集 C3 PAN 和四尺度 Detect
  均保持不变。它是论文主模型，也是当前最高已验证精度结构。C1 是主要轻量化
  对照；C1.5、C2/C3 私有补偿及两版 C4 D3C3 均已完成验证并退出活动路径。
- RGCA 使用真实跨模态 Q/K、depthwise Value 局部分支和显式可靠性拒绝门；
  注意力 logits 完全来自内容相关性，不叠加 fixed/BECP/DACP prior；为严格对应
  最优实验，残差恢复为逐方向、逐通道参数，不包含 FFT、复数滤波基、频域路由
  或频率残差 `beta`。
- 修正频域路由虽不再恒定，但统一学习率实验仍未超过 `exp_mask`，因此只保留
  为历史复现/严格消融类，不再进入默认前向。
- 历史 GOAT-style、ME-CAP、BECP 与 DACP prior 路线均未超过无 prior 最优实验，
  因而只保留为可复现的独立消融配置，不进入默认模型。
- Bayesian GNN、Bayesian-GFB、BNN、FD2 风格跨模态高低频增强等路线没有获得稳定收益，也不在当前训练调用路径中。
- 当前没有额外的 mask 监督损失、BNN KL loss 或 NWD loss。

## 2. 当前代码的真实调用路径

训练入口为：

```text
train.py
  -> models/yolo_test.py
  -> models/common.py::HAFFormerRGCAMaskGFB
  -> independent RGB/IR C2/C3/C4/C5/SPPF
  -> P2/P3/P4/P5 four-scale fusion
  -> dense C3 PAN at top-down/P2/P3/P4/P5
  -> original four-scale Detect
```

当前 M3FD C0 模型配置：

```text
models/transformer/yolov5s_LCAFNet_M3FD.yaml
```

该 YAML 在 C2、C3、C4、C5 四个尺度调用 `HAFFormerRGCAMaskGFB`：

```text
C2: [2,12] -> layer 20，保存给 P2 neck skip
C3: [4,14] -> layer 21，保存给 P3 neck skip
C4: [6,16] -> layer 22，保存给 P4 neck skip
C5: [9,19] -> layer 23，保存给 P5 neck skip
```

当前 C0 YAML 使用显式 `architecture_variant: c0_full_dual_rgca` 和
`pretrained_backbone_map`；这些元数据只用于严格初始化和实验身份校验，不改变
原 C0 网络计算图。

## 3. 当前模型结构

### 3.1 无频域可靠性引导双向跨模态注意力（RGCA）

当前 C0 的四个融合块均使用
`ReliabilityGuidedBidirectionalCrossAttention`。RGB 与 IR 先经过各自的
LayerNorm，再通过共享 QKV 和 depthwise 3×3 投影进入相同潜在空间。两个方向
使用真实的跨模态相关性，而不是同一模态自相关：

```text
A_rgb = softmax(Q_rgb @ K_ir^T * temperature)
A_ir  = softmax(Q_ir  @ K_rgb^T * temperature)
```

注意力在每个 head 的通道维建立交互，空间复杂度随 `H×W` 线性增长；Value
同时经过 5×5 depthwise 局部分支，并由每个 head 的可学习系数混合全局通道交互
与局部信息。可靠性门根据主模态、互补模态、绝对差和逐元素乘积生成单通道门控：

```text
R = sigmoid(DWConv([V_primary, V_complementary,
                    abs(V_primary - V_complementary),
                    V_primary * V_complementary]))
F_rgb = RGB + gamma_rgb * Proj(R_rgb * Mixed_rgb)
F_ir  = IR  + gamma_ir  * Proj(R_ir  * Mixed_ir)
```

`gamma_rgb/gamma_ir` 使用 C0 最优实验一致的逐方向、逐通道残差参数，初始值
为 `0.1`。活动配置显式设置 `prior_mode=none`，因此 attention logits 只来自
内容相关性；不包含 FFT、复数滤波、频率路由、固定 prior、BECP 或 DACP。

### 3.2 前景稀疏 mask

双向 RGCA 增强后，将下列三项拼接：

```text
[F_rgb, F_ir, abs(F_rgb - F_ir)]
```

然后通过轻量 mask 预测器：

```text
1x1 Conv
-> BN + SiLU
-> 3x3 Depthwise Conv
-> BN + SiLU
-> 1x1 Conv
-> Sigmoid
```

得到单通道空间门控图 `M`。原始 GFB 同时从双模态增强特征预测逐通道
RGB/IR 融合 logit `L_gfb`，mask 在 logit 空间对 GFB 进行有界空间调制：

```text
W_rgb = sigmoid(L_gfb - (2M - 1))
F_out = W_rgb * F_rgb + (1 - W_rgb) * F_ir
```

其中：

- `M=0.5` 时完全保持原 GFB 权重；
- `M` 较小时在 GFB 基础上偏向 RGB；
- `M` 较大时在 GFB 基础上偏向 IR；
- GFB 保留通道/局部选择，mask 提供单通道空间先验。

mask 最后一层的输出偏置当前初始化为 `0.0`，因此初始 mask 大致位于 `0.5` 附近，不会在训练开始时强行偏向某个模态。

### 3.3 当前 C0 与历史 exp_mask 的区别

历史 `exp_mask` 使用频域注意力；当前 C0 保留相同的 mask/GFB 后融合规则，
但已用上述无 FFT 的 RGCA 替换频域注意力。当前论文主结果应引用
`exp_rgca_mask_gfb_uniform_lr`，不能把 `exp_mask` 的指标写成 C0 指标。

### 3.4 关于“前景”命名的说明

当前 mask 没有使用分割标注，也没有单独的前景监督或稀疏正则项。它是由最终检测损失端到端学习得到的空间融合门控图。

因此在论文中，更严谨的称呼是：

```text
foreground-aware sparse fusion mask
```

不能仅凭模块名称断言它等价于真实前景分割。论文中应补充：

- mask 可视化；
- 目标框内外 mask 响应统计；
- mask 开启/关闭消融；
- 参数量、FLOPs 和速度对比。

## 4. 当前训练设置

当前默认训练直接运行：

```bash
cd /home/dell/lcp/LCAFNet-main
conda activate MOD
python train.py
```

当前命令对应 FLIR C0 从 `yolov5s.pt` 开始的200轮完整训练。

需要调整实验时，优先修改 `train.py` 顶部两个集中配置字典：

- `TRAINING_CONFIG`：模型/数据路径、epochs、batch size、图像尺寸、GPU、
  workers、优化器、AMP、mosaic 关闭轮数、输出目录和实验名；
- `HYPERPARAMETERS`：学习率、momentum、weight decay、warmup、检测损失权重
  和数据增强参数。

命令行参数仍可用于一次性覆盖常用设置，例如
`python train.py --batch-size 16 --name c0_flir_ft50_repeat`；正式对比实验应同步修改
`experiment_name`，避免把不同设置混入同一结果目录。

M3FD C0 从头训练时使用的本地 `yolov5s.pt` 是经典 anchor-based YOLOv5s，SHA256 为
`f1610cfd81f8cab94254b35f6b7da2981fa40f93ad1bd3dd1803c52e7f44753e`。
它的 depth/width multiplier 为 `0.33/0.50`、参数量 `7,276,605`，
含 3 个 anchor 检测尺度和 `[8,16,32]` stride。该文件属于旧
Focus+SPP 结构，而当前工程使用 Conv6+SPPF，因此不是逐层完全相同的
版本。训练入口现已把 Focus 核等价重排为 Conv6，并按模块类型把旧
C3/SPP 迁移到新 C3/SPPF。C0 通过 YAML 显式映射初始化 RGB C2--C5 `198`、
IR C2--C5 `198` 个状态，合计 `396/396`；四尺度融合、PAN 和四尺度 Detect
保持模型初始化。C0→C1.5 的 `676/676` 专用迁移仍保留，只供历史筛选复现。

当前默认入口训练 FLIR C0 **完整双流 C2--C5/SPPF + 四尺度 no-prior
RGCA/mask/GFB + 原密集 PAN/Detect**：从 `yolov5s.pt` 映射两条 backbone 后
训练200 epochs、batch 16、640输入、SGD、AMP、cosine one-cycle、3 epoch
warmup，最后10 epochs关闭mosaic。所有参数使用统一初始 LR `0.01`；融合、
PAN、Detect不加载不兼容的单流三尺度检测头。AFSS、差分学习率、AKCMambaLite、
D3C3、私有旁路、辅助头、BECP、DACP及蒸馏均不活动。

训练过程中每轮首个真实批次分别写入：

- `attention_diagnostics.csv`：C2/C3/C4/C5 四个尺度的 RGCA 注意力、可靠性门和残差统计；
- `fusion_diagnostics.csv`：C2/C3/C4/C5 四个尺度的 mask 与最终 GFB 门控。

`private_bypass_diagnostics.csv`、`private_foreground_diagnostics.csv` 和
`akcmamba_diagnostics.csv` 均只属于历史实验；当前 C0 不含
任何 AKConv 或 SSM 模块。

BECP/DACP 诊断文件只属于相应历史消融配置，不是默认模型的有效指标。

### 4.1 已移出的训练策略

AFSS、BECP、DACP、AKCMambaLite 诊断、差分学习率、prior 分阶段训练、单模态
消融和超参数进化均已从活动 `train.py` 删除。它们只保留在历史训练器
`archives/train_experiments_legacy_20260805.py` 中，用于复查旧结果；不得把旧
参数（例如 `--afss`、`--dacp-training` 或 `--differential-lr`）传给当前 C0
入口。当前 `python train.py` 使用完整 FLIR 配对训练集。

## 5. 已验证的模型和结果

以下结果均来自 M3FD。表中的 mAP 是各指标在完整训练过程中的独立最高值，因此两个最高值不一定出现在同一 epoch。

| 实验 | 主要结构 | Best mAP@0.5 | Best mAP@0.5:0.95 | 0.5/0.5 Fitness |
|---|---|---:|---:|---:|
| `lcafnet_original_close_mosaic103` | 原始 LCAFNet | 90.583% (159) | 58.033% (191) | 74.134% (191) |
| `exp21` | 频域 Transformer + 原始 GFB | 90.766% (180) | 58.650% (180) | 74.708% (180) |
| `exp_mask` | 频域 Transformer + 前景 mask + 原始 GFB | **91.066% (189)** | **59.188% (175)** | **75.031% (188)** |
| `exp_prior_sgd_full` | exp_mask + channel/spatial prior | 90.848% (181) | 58.849% (185) | 74.758% (182) |
| `exp_channel_prior_sgd_full` | exp_mask + channel prior | 90.730% (177) | 58.802% (176) | 74.743% (177) |
| `exp_rgca_mask_gfb_uniform_lr` | C0 完整双流 RGCA 控制 | 91.447% (177) | 59.382% (192) | 75.275% (192) |
| `exp_rgca_mask_gfb_c1_dual_to_c3_shared_c4c5_joint200_yolov5s_m3fd` | C1 双流到 C3、共享 C4/C5 | 91.188% (181) | 59.146% (195) | 75.041% (162) |
| `exp_rgca_mask_gfb_c1_5_dual_to_c4_shared_c5_joint200_yolov5s_m3fd` | C1.5 双流到 C4、共享 C5 | 90.299% (142) | 58.503% (184) | 74.323% (135) |
| `exp_rgca_mask_gfb_c2_c1_private_lowrank_c4c5_joint200_yolov5s_m3fd` | C2 + rank-16 模态私有旁路 | 91.107% (186) | 59.048% (199) | 75.055% (186) |
| `exp_rgca_mask_gfb_c3_c2_gt_private_fg_joint200_yolov5s_m3fd` | C3 + GT 私有前景监督 | 90.611% (182) | 58.624% (191) | 74.496% (186) |
| `exp_rgca_mask_gfb_c4_c1_d3c3_slimpan_joint200_yolov5s_m3fd` | 六层 D3C3 C4，判弱后于 epoch 64 停止 | 87.193% (64) | 51.709% (64) | 69.451% (64) |

相对原始 LCAFNet，`exp_mask` 的独立最高指标约提升：

```text
mAP@0.5:       +0.483 个百分点
mAP@0.5:0.95:  +1.155 个百分点
```

目前能够支持的结论是：

- 频域 Transformer 对严格定位指标 `mAP@0.5:0.95` 有稳定的正向迹象；
- 在频域 Transformer 基础上加入前景 mask 后，当前单次实验进一步提升；
- GOAT prior 的加入没有超过 `exp_mask`；
- 当前还不能仅凭单个随机种子把较小差值视为稳定结论。

## 6. 已探索路线记录

### 6.1 频域 Transformer

在 V 分支引入 FFT、动态路由和可学习复数滤波基，用较低的空间复杂度建模全局信息。`exp21` 相比原始 LCAFNet 有小幅提升，说明该方向具有实验意义，尤其对 mAP@0.5:0.95 更有帮助。

### 6.2 频域 Transformer + 前景 mask

该组合对应历史 `exp_mask`。它证明 mask/GFB 后融合具有价值，但其频域注意力
已被后续 C0 的无 FFT RGCA 替代；当前最高结果应引用
`exp_rgca_mask_gfb_uniform_lr`。

### 6.3 Bayesian GNN / Bayesian-GFB / BNN

尝试过贝叶斯 GNN、Bayesian-GFB、BNN 不确定性和 KL loss 等方案。主要问题为：

- 参数量和训练时间明显增加；
- 深拷贝、状态张量及训练稳定性曾出现问题；
- 没有获得稳定、可复现的精度提升；
- 贝叶斯残差或直接融合均未形成可靠优势。

这些模块已从当前调用路径移除，但历史类仍保留在：

```text
models/common.py::HAFFormerAdaptiveGNNArchive
models/common.py::HAFFormerBayesianGFBArchive
```

### 6.4 FD2 风格跨模态高低频互补增强

尝试在频域 Transformer 后加入跨模态高低频增强，并将普通卷积替换为深度可分离卷积以减小参数量。实验没有稳定超过单频域 Transformer，推测与已有频域分支功能重叠，并增加了优化难度，因此已移除。

### 6.5 GOAT-style attention prior

尝试过：

- channel prior；
- spatial routing prior；
- channel + spatial prior。

梯度、优化器状态和参数变化检查表明 prior 确实在学习，不是“死模块”。其中 channel prior 的参数均发生更新；但最终结果仍低于 `exp_mask`。spatial prior 的更新强度明显弱于 channel prior，且整个 prior 路线没有形成精度收益，因此当前已移除。

结论应表述为：

```text
prior 正常工作，但其归纳偏置没有改善当前任务，而不是 prior 没有学习。
```

### 6.6 NWD loss、Adam/AdamW 和分阶段微调

曾尝试 NWD loss、Adam/AdamW、差分学习率、冻结已有层、分阶段微调以及从 LCAFNet 数据集权重继续训练。整体没有比从相同 `yolov5s.pt` 使用 SGD 完整训练更可靠。

当前公平结构实验统一采用：

```text
同一 yolov5s.pt backbone 初始化
+ 同一数据划分
+ 同一 SGD 超参数
+ 200 epochs
+ 三个配对随机种子
```

## 7. 当前代码验证状态

最近一次检查结果：

- 当前默认 YAML 为 FLIR 3 类 C0，参数量 `13,734,184`；模型中没有 D3/D3C3、私有
  低秩旁路、GT 前景训练头、AKCMamba、旧频域块或 attention prior；
- canonical `yolov5s.pt` 到 FLIR C0 的迁移实测为 RGB backbone `198`、IR
  backbone `198`，合计 `396/396`，映射缺失数为0；
- 四个融合块、PAN和三类四尺度Detect保持模型初始化，加载和新增参数全部参与
  统一学习率 SGD 训练；
- 当前融合块包含无 FFT 的 RGCA、前景 mask 和原始 GFB；
- FLIR train/test RGB/IR 各 `4118/1010` 张，配对数量一致；
- 真实 FLIR batch 16、640×640、CUDA AMP 检测损失前向/反向通过，峰值已分配/
  保留显存约 `10.23/10.81 GiB`，loss 和四尺度输出均有限；
- RGB/IR C4、四尺度 RGCA 和 Detect 的梯度均实测为有限非零；
- RGCA 的共享 QKV、可靠性门、残差尺度、Mask 和 GFB 均获得有限非零梯度；
- 默认初始化的小尺寸检查中，双向 attention std 与 reliability std 均非零；
- 历史保守 C4 参数量虽降至 `7,418,424`，但 RTX 3090、batch=1、640、
  Conv-BN fuse 同进程 FP32 中位延迟为 `27.663 ms`，反而比 C1 的
  `27.041 ms` 慢约 `2.30%`；这是后续必须报告实际延迟而不能只报告参数/MAC
  的直接依据；
- 详细公式、测试边界和严格判废条件见 `paper/RGCA_DESIGN.md`。
- 2026-08-09 FLIR 入口修改后完整测试为 `97 passed`。

测试命令：

```bash
cd /home/dell/lcp/LCAFNet-main
PYTHONPATH=. pytest -q tests
```

## 8. 下一步实验决策

### 8.1 当前入口：FLIR C0 从 yolov5s.pt 完整训练200轮

运行：

```bash
python train.py
```

默认从 `yolov5s.pt` 初始化完整双主干，使用 SGD `lr0=0.01`、batch 16、
640输入训练200 epochs；新建 optimizer，不冻结任何层，warmup 3 epochs，
最后10 epochs关闭mosaic。训练过程应同步检查
`attention_diagnostics.csv`和
`fusion_diagnostics.csv`，确认 P2/P3/P4/P5 四处 RGCA/Mask 均正常工作。

该实验采用与 M3FD C0 相同的 canonical backbone 初始化和 SGD 协议，可作为
FLIR 上的正式完整训练结果；M3FD C0 历史主结果保持不变。

### 8.2 为什么下一步不再继续 C2/C3/D3C3/Mamba

当前关键结果为：

| 结构 | 参数量 | Best mAP@0.5 | Best mAP@0.5:0.95 | 判断 |
|---|---:|---:|---:|---|
| C0 完整双流 | 13.743M | 91.447% | 59.382% | 精度控制 |
| C1 双流到 C3、共享 C4/C5 | 8.046M | 91.188% | 59.146% | 主轻量基线 |
| C1.5 双流到 C4、共享 C5 | 9.323M | 90.299% | 58.503% | 参数更多但低于 C1 |
| C2 低秩私有旁路 | 8.075M | 91.107% | 59.048% | 旁路正常学习但无收益 |
| C3 + GT 私有前景监督 | 8.075M | 90.611% | 58.624% | 额外约束损害主任务 |
| C4 六层 D3C3 | 7.119M | 87.193% | 51.709% | epoch 64 判弱停止 |
| C4 仅 P4/P5 D3C3 pilot | 7.418M | 约 91.082% | 约 57.579% | 30 轮未恢复 C1 |

C1 相对 C0 少 `5.698M` 参数，只损失 `0.236` 个百分点 mAP@0.5:0.95。
在各自 mAP@0.5:0.95 最优 epoch，C1 Precision 高于 C0，而 Recall 低约
`0.88` 个百分点。现有证据更符合“C3 后过早共享导致一部分困难目标的深层模态
信息丢失”，而不是“检测头容量不足”或“融合模块没有工作”。因此：

- rank-16 低秩旁路容量不足以替代真实的独立 C4 空间特征提取；
- GT 前景辅助监督增加优化冲突，不能作为默认补偿；
- D3C3/大核深度卷积虽然减参，却已在当前 CUDA/PyTorch 环境表现出精度下降且
  eager 延迟没有改善；
- 当前 RGCA 已包含跨模态相关性、局部 Value、可靠性门、Mask 和 GFB，再叠加
  Mamba、频域或同类注意力会造成职责重叠和更大的优化不确定性。

小于约 `0.3--0.5` 个百分点的单随机种子差异不能直接视为稳定结构收益。正式论文
结论应使用相同 `yolov5s.pt`、相同训练设置和配对随机种子。

### 8.3 已完成消融：C1.5 双流到 C4、只共享 C5

C1.5 已实现并完成 200 epochs 正式训练，结构为：

```text
RGB: C2_rgb -> C3_rgb -> C4_rgb --\
                                   P4 RGCA fusion -> shared C5/SPPF -> P5
IR:  C2_ir  -> C3_ir  -> C4_ir  --/

P2 = RGCA(C2_rgb, C2_ir) -> PAN P2 skip
P3 = RGCA(C3_rgb, C3_ir) -> PAN P3 skip
P4 = RGCA(C4_rgb, C4_ir) -> shared C5 and PAN P4 skip
P2/P3/P4/P5 -> unchanged dense C3 PAN -> four-scale Detect
```

边界必须保持明确：

- RGB/IR 独立提取到 stride 16 的 C4；
- 只在 P2/P3/P4 使用现有 `HAFFormerRGCAMaskGFB`；
- P4 融合后只计算一次 C5/SPPF，不恢复昂贵的双 C5 和 P5 融合；
- 不加入低秩旁路、辅助损失、D3C3、Mamba、额外频域模块或新检测头；
- 原始密集 PAN、P2 小目标输出和 anchors 不变，以便单独判断合流位置收益。

实测参数量：

```text
C1                                           8,045,752
第二条独立 C4                               +920,576
恢复已在 C0 验证过的 P4 RGCA 融合           +356,234
-----------------------------------------------------
C1.5 实测                                    9,322,562
```

C1.5 比 C0 减少约 `32.17%` 参数，但正式结果仅达到独立最高
mAP@0.5=`90.299%`、mAP@0.5:0.95=`58.503%`，未达到 `59.30%` 验收线，
同时低于参数更少的 C1。因此它没有形成新的 Pareto 点，已退出默认入口；该结果
应作为“深层双流保留位置”消融在论文中报告。

以下曾规划为 C1.5 通过后的候选，因 C1.5 未通过，目前不进入默认主线：

1. **P4 样本级条件门控：** 用 RGB/IR P4 的全局统计产生接近等权、受限幅度的
   两模态权重，适应白天、夜间和恶劣天气；只调节现有融合输入，不再复制局部注意力。
2. **Compact Gather-Distribute Neck：** 把 P2--P5 压缩并汇聚到 P4，生成不超过
   `0.25--0.35M` 参数的上下文，再以残差回注 P2/P3，增强小目标的深层语义；
   不增加 P1，不再全量替换 PAN C3。

论文方向依据和可借鉴边界：

- [GM-DETR](https://openaccess.thecvf.com/content/CVPR2024W/JRDB/html/Xiao_GM-DETR_Generalized_Muiltispectral_DEtection_TRansformer_with_Efficient_Fusion_Encoder_for_CVPRW_2024_paper.html)：高层模态特定信息和跨模态尺度融合，支持优先验证更晚合流；
- [CVPR 2026 多传感器融合深度消融](https://openaccess.thecvf.com/content/CVPR2026/html/Iaboni_Tri-Modal_Fusion_Transformers_for_UAV-based_Object_Detection_CVPR_2026_paper.html)：融合位置对检测性能有显著影响；
- [ICCV 2025 条件感知融合](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)：复杂光照/天气下使用样本条件动态融合；
- [Gold-YOLO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html)：Gather-and-Distribute 多尺度语义聚合，可保守适配到 P2/P3 小目标路径；
- [PKINet](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)：尺度变化和复杂背景下的深层多尺度上下文；不应把大核直接铺满浅层/PAN。

### 8.4 Backbone 更换作为独立路线

Backbone 可以更换，但不能与 C1.5、neck 和融合改动同时进行，否则无法归因。
优先候选为：

1. **MobileOne-S1：** 纯卷积、部署时结构重参数化，当前环境已有非教师蒸馏的
   ImageNet 预训练权重；建议保持“双流到 stride 16、共享 stride 32”和现有
   RGCA/PAN。它适合作为强调实际设备延迟的首个新 backbone。
2. **EfficientViT-B1：** 当前环境有非蒸馏预训练权重，能输出 stride
   4/8/16/32 特征；适合需要轻量多尺度全局建模的 GPU 路线，但集成风险高于
   MobileOne。

暂不把 RepViT 作为第一候选，因为当前环境可直接使用的预训练权重均为
`.dist_*` 蒸馏版本，与“不采用教师蒸馏”的实验边界冲突。任何 backbone 实验都
必须重新统计 Params、MAC、显存、PyTorch eager、ONNX/TensorRT 实际延迟，不能
只用参数量或 FLOPs 判断轻量化。

### 8.5 做严格公平的复现实验

统一启动器为：

```bash
python train_strict_ablation.py
python train_strict_ablation.py --execute
```

第一条命令只验证权重并打印 18 个命令；第二条才依次执行 6 个结构 ×
3 个种子。启动器强制使用相同数据、anchors、增强、batch、SGD、
200 epochs 和配对 seed，并保存权重哈希与全部命令到 `manifest.json`。
建模后会重新设置数据加载器随机种子，各共享检测头层使用按层隔离的
随机流，因此结构参数量不同不会改变检测头初值或增强序列。

训练使用统一学习率：预训练双 backbone、新融合和检测头均为 `0.01`；
3 epoch warmup、cosine one-cycle、最后 10 epochs 关闭
mosaic、梯度裁剪 10、关闭 AFSS 和 autoanchor。完成后自动输出
`ablation_runs.csv` 和均值/标准差 `ablation_summary.csv`。

小于约 0.3 至 0.5 个百分点的单次差异应谨慎解释。

### 8.6 推荐的核心消融表

建议至少包含：

严格主表全部输出 C 通道并共用完全相同的检测头：

| 启动器名称 | 注意力 | 融合 | 目的 |
|---|---|---|---|
| `mean` | 无 | 均值 | 无注意力下限 |
| `original_gfb` | 原始空间 CA | GFB | 原始 LCAFNet 对照 |
| `frequency_gfb` | 修正频域 CA | GFB | 只替换注意力 |
| `original_mask` | 原始空间 CA | mask | mask 下的注意力对照 |
| `frequency_mask` | 修正频域 CA | mask | 历史频域对照 |
| `rgca_mask_gfb` | RGCA | mask + GFB | 当前默认候选结构 |

关键配对为 `original_gfb -> rgca_mask_gfb`。由于该配对同时包含 Mask，
论文最终还必须补充 RGCA+GFB、RGCA 去 local、RGCA 去 reliability 三项，
以分离注意力、局部分支和可靠性门的贡献。

补充实验建议：

- RGB-only、IR-only、直接 Add/Concat；
- RGCA 降维率、head 数和 3×3/5×5 local kernel；
- 四尺度与只在 C3/C4/C5 使用 RGCA；
- mask 可视化与框内外响应统计；
- Params、FLOPs、FPS、显存和训练时间；
- M3FD、LLVIP、FLIR/FLIR-aligned 跨数据集验证。

## 9. 重要文件索引

```text
train.py
    当前为 FLIR C0 从 yolov5s.pt 正式训练200 epochs的入口。

archives/train_experiments_legacy_20260805.py
    2026-08-05 前的多实验训练器，仅供旧实验复现，不在当前 C0 活动路径中。

models/yolo_test.py
    模型解析、双主干和 HAFFormer 调用链。

models/common.py
    当前 no-prior RGCA、D3/D3C3 及历史私有/AK/Mamba 类。

models/transformer/yolov5s_LCAFNet_M3FD.yaml
    M3FD C0 论文主结构：完整双主干、P2/P3/P4/P5 四尺度 RGCA+Mask+GFB。

models/transformer/yolov5s_LCAFNet_FLIR.yaml
    当前默认 FLIR C0，结构与 M3FD C0 对齐，类别数为3。

models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml
    已中断的历史 YOLO neck 改进实验配置。

models/transformer/yolov5s_LCAFNet_M3FD_C1_SharedC4C5.yaml
    已完成的 M3FD C1 轻量模型配置，是当前 C0 的核心效率对照。

models/transformer/yolov5s_LCAFNet_M3FD_C1_5_SharedC5.yaml
    已完成的 C1.5 消融：双流到 C4、P2/P3/P4 融合、仅共享 C5/SPPF。

models/transformer/yolov5s_LCAFNet_FLIR_C1_SharedC4C5.yaml
    保留的 3 类 FLIR C1 模型配置，当前 M3FD C0 入口不使用。

models/transformer/yolov5s_LCAFNet_LLVIP_C1_SharedC4C5.yaml
    已完成的一类 LLVIP C1 迁移模型配置，当前 M3FD 入口不使用。

models/transformer/yolov5s_LCAFNet_M3FD_C2_PrivateLowRank.yaml
    已完成的 C2 模型配置。

models/transformer/yolov5s_LCAFNet_M3FD_C3_GTPrivateForeground.yaml
    已完成的 C3 模型配置；推理主图与 C2 相同。

models/transformer/yolov5s_LCAFNet_M3FD_C4_D3C3SlimPAN.yaml
    已判弱的六层 D3C3 C4 配置，保留为历史消融。

models/transformer/yolov5s_LCAFNet_M3FD_C4_DeepP4P5D3C3.yaml
    历史保守 C4 配置：C1 + 仅层 34/37 的深层 P4/P5 D3C3；已退出默认入口。

paper/C2_PRIVATE_LOW_RANK_BYPASS_20260805.md
    C2 结构、论文依据、复杂度、诊断和判废标准。

paper/C3_GT_PRIVATE_FOREGROUND_20260805.md
    C3 动机、损失、参数、正式/微调边界、诊断与验收标准。

paper/C4_D3C3_SLIM_PAN_20260806.md
    历史六层 C4 结构、参数/MAC、实测延迟、训练与判废标准。

paper/C4_DEEP_P4P5_D3C3_20260806.md
    保守 C4 的结构边界、参数、训练入口和验证状态。

data/hyp.rgca_mask_gfb_pretrained.yaml
    当前 M3FD C0 正式200轮训练超参数，与历史 uniform-LR 控制一致。

data/hyp.c1_5_from_c0_finetune30.yaml
    已完成的C0->C1.5 30轮筛选超参数：SGD、lr0=0.001、warmup 1 epoch。

data/hyp.rgca_flir_baseline_finetune.yaml
    历史 FLIR 权重迁移微调超参数，当前入口不使用。

data/hyp.rgca_flir_pretrained.yaml
    当前 FLIR C0 正式训练超参数：SGD、lr0=0.01、warmup 3 epochs。

data/hyp.rgca_llvip_baseline_finetune.yaml
    保留的 LLVIP C1 微调超参数；已完成运行实际配置以其 opt/hyp.yaml 为准。

paper/C1_DUAL_TO_SHARED_C4C5_20260804.md
    C1 结构、迁移、复杂度、验证和训练命令。

archives/rgca_mask_gfb_uniform_lr_control_20260804_2331.tar.gz
    未加入 AKCMambaLite 的控制代码与运行配置存档。

archives/c3_gt_private_foreground_completed_20260806.tar.gz
    进入 C4 前的 C3 完整代码、配置及 200-epoch 结果存档。

paper/NO_PRIOR_RESTORATION_20260804.md
    无 prior 最优控制结构、指标和兼容性验收记录。

paper/AKCMAMBA_YOLOV5_ADAPTATION_20260804.md
    当前 YOLO neck 改进的论文依据、公式、实测和实验方案。

data/hyp.strict_fair_ablation.yaml
train_strict_ablation.py
    严格公平消融的统一超参数、权重校验、实验矩阵和结果汇总。

runs/strict_ablation_smoke/
    真实 M3FD 一轮启动验证；8 条路由 confidence std 均非零。

runs/train/exp_mask/
    历史频域 attention + mask + GFB 实验结果，不是当前 C0 主结果。

runs/train/lcafnet_original_close_mosaic103/
    原始 LCAFNet baseline 结果。

models/fretransformer.py
models/fretransformer2.py
    历史频域版本存档，不能据此判断当前实际调用结构。

freformer.py
    历史频域模块的同步导出入口，不进入当前 C0 调用路径。
```

## 10. 给后续 Codex 的交接要求

继续修改前请先完成以下检查：

1. 阅读本文档；
2. 检查 `train.py` 当前默认参数；
3. 沿 `train.py -> yolo_test.py -> common.py::HAFFormer` 确认真实调用路径；
4. 检查目标实验目录下的 `opt.yaml` 和 `hyp.yaml`，不要只根据目录名称推断；
5. 修改结构后进行完整模型前向、AMP 反向和关键参数梯度检查；
6. 结构消融应从相同 `yolov5s.pt` 和相同训练设置开始，避免用某个结构的 `best.pt` 初始化另一个结构后直接比较；
7. 不要覆盖历史实验结果或删除存档类；
8. 新实验完成后更新本文档中的当前结构、结果表和结论。

严格消融的正式 5×3×200 epoch 矩阵尚未启动；一轮真实数据 smoke
已完成，结果位于 `runs/strict_ablation_smoke`。
