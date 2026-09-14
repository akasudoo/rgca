# LCAFNet / RGCA 实验交接总览

更新时间：2026-08-24  
项目根目录：`/home/dell/lcp/LCAFNet-main`（实际挂载路径也可能显示为
`/mydata/lcp/LCAFNet-main`）

本文件用于开启新对话时快速恢复上下文。新对话应先阅读本文件，再按文末链接
查看各专项记录。所有数值均来自本地现有权重、`results.csv` 或已经保存的复核
记录；没有完成或没有统一复评的实验会明确标出，不能按已完成结果引用。

## 1. 最重要的模型身份

### 1.1 论文核心模型：纯 C0 完整 RGCA

核心模型应定义为：

```text
C0 完整 RGB/IR 双骨干
+ P2/P3/P4/P5 四尺度双向 RGCA
+ 互补模态局部 Value 分支和全局/局部混合
+ 双向逐位置可靠性门控
+ Foreground Mask
+ GFB
+ 原密集 C3 PAN
+ P2-P5 四尺度水平检测头
```

活动版本没有 BECP、DACP、固定 attention prior、SOD-ASF、C3CrossConv、
Mamba 或 OBB 检测头。两个模态在 C2-C5 均保留独立骨干，四个尺度分别融合。

- 训练目录：`runs/train/exp_rgca_mask_gfb_uniform_lr`
- 纯 RGCA 权重：
  `runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt`
- SHA-256：
  `12896c3b47017197d4ab1ea86e515332458dcea45f5e6742c380798e8039f882`
- 模型配置：`models/transformer/yolov5s_LCAFNet_M3FD.yaml`
- 独立创新代码说明：`models/lcafnet_innovations.py`
- 纯 RGCA 历史代码快照：
  `archives/rgca_mask_gfb_uniform_lr_control_20260804_2331.tar.gz`

### 1.2 原始 LCAFNet 基线

原始 LCAFNet 使用原版空间交叉注意力和 GFB，不包含本文提出的 RGCA 可靠性
门控。需要区分以下两个对照：

1. 原论文/原结构基线：原始 LCAF attention + 原始 GFB，不含 Foreground Mask。
   权重为 `LCAFNet_M3FD.pt`。
2. 更严格的注意力单变量对照：原始 LCAF attention + 与 RGCA 相同的
   Mask+GFB，实验目录为 `runs/train/exp_m3fd_lcaf_ca_mask`。

比较“RGCA 注意力是否优于 LCAF attention”时，应优先使用第2项，因为其后融合
模块与完整 RGCA 一致。比较“本文完整方案与论文原始 LCAFNet”时才使用第1项。

### 1.3 名称容易误导的 `m3fd_rgca_mask_gfb_best`

`runs/train/m3fd_rgca_mask_gfb_best` **不是纯 RGCA 从头训练实验**。其
`opt.yaml` 表明它从纯 RGCA best 权重出发，换用
`yolov5s_LCAFNet_M3FD_BECP.yaml` 又进行了50轮 BECP 微调。因此应标记为：

```text
RGCA + Mask + GFB + BECP fine-tune
```

- 权重：`runs/train/m3fd_rgca_mask_gfb_best/weights/best.pt`
- SHA-256：
  `7b1d92ca61872af8cdf08afcf4a080c44f120013737fcc809c38064dbe4ab9b8`
- 对应代码存档：`archives/c0_m3fd_becp_20260819`

此前完整 test 检测图和论文定性对比大量使用了这个权重，并在可视化目录中简称
为 `rgca_full`。写论文定量消融时不能把它误写成纯 RGCA。

## 2. RGCA 的设计动机与组成

RGCA（Reliability-Guided Cross-modal Attention）的目标是在 RGB 或 IR 受暗光、
眩光、热交叉、遮挡和背景噪声影响时，避免无条件把互补模态信息注入主模态。

核心步骤为：

1. RGB 与 IR 分别归一化，但共享低维 QKV 投影，使两个模态位于相同潜在空间；
2. RGB Query 与 IR Key/Value、IR Query 与 RGB Key/Value形成真正双向跨模态相关；
3. 用跨通道注意力建模全局关系，避免构造 `(H×W)²` 空间注意力矩阵；
4. 通过 5×5 depthwise Value 分支保留小目标边缘和局部热响应；
5. 根据主模态、互补模态、绝对差异和逐点一致性预测方向相关的可靠性门；
6. 可靠性门调制跨模态残差注入，再通过 Foreground Mask 和 GFB完成最终融合。

完整数学描述和设计依据见 `paper/RGCA_DESIGN.md`。

## 3. 完整 M3FD 的统一实验协议

- 本地划分：3360组 train、420组 val、420组 held-out test；RGB/IR配对输入。
- 原论文协议为3360 train + 840 official test。本地把这840张拆成了420 val和
  420 test。
- 输入尺寸：640×640。
- 训练：200 epochs，physical batch 16，梯度累积到 effective batch 64。
- 初始化：`yolov5s.pt` 对称映射到 RGB/IR 两个骨干。
- 优化器：SGD，`lr0=0.01`，`lrf=0.1`，momentum 0.937，
  weight decay 0.0005。
- 调度：3 epochs warmup + cosine decay。
- 增强：RGB/IR共享完全相同的 Mosaic、几何缩放和翻转参数；最后10轮只关闭
  Mosaic，其他增强保留。
- seed：1。
- 检测：水平框 HBB，P2-P5 四尺度 Detect。

下表统一从每个 `results.csv` 选择
`0.5 × mAP50 + 0.5 × mAP50:95` 最高的 epoch，然后报告该 epoch 的 val 指标。
这避免历史训练器选权重公式变化带来的读表差异，但不保证该 epoch 一定等于目录
中现存 `best.pt` 的保存标准。

参数量为 Conv-BN 融合后的验证态参数量。计算量为 batch=1、RGB和IR各一个
`3×640×640` 输入，THOP统计；表中采用 `1 MAC = 2 FLOPs`。函数式张量操作
可能未被 THOP 完整统计，尤其不能把频域模型的数值当作严格硬件代价。

## 4. 完整 M3FD 核心消融结果（统一 val 口径）

| 实验 | 注意力/输入 | 后融合 | epoch | P | R | mAP50 | mAP50:95 | Params(M) | GFLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `exp_m3fd_vis_single` | VIS-only | 单模态 | 198 | 0.9133 | 0.8205 | 0.8833 | 0.5606 | 7.687 | 26.929 |
| `exp_m3fd_ir_single` | IR-only | 单模态 | 185 | 0.9141 | 0.7948 | 0.8684 | 0.5358 | 7.687 | 26.929 |
| `exp_m3fd_concat` | 无注意力 | RGB/IR Concat | 180 | 0.8979 | 0.8700 | 0.9064 | 0.5836 | 7.691 | 27.637 |
| `lcafnet_original_close_mosaic103` | 原始 LCAF CA | 原始 GFB，无 Mask | 191 | 0.9006 | 0.8631 | 0.9023 | 0.5803 | 15.396 | 46.060 |
| `exp_m3fd_lcaf_ca_concat` | 原始 LCAF CA | Concat | 190 | 0.9003 | 0.8756 | 0.9076 | 0.5858 | 14.928 | 45.480 |
| `exp_m3fd_freq_ca_concat2` | 历史频域 CA | Concat | 185 | 0.8940 | 0.8758 | 0.9057 | 0.5863 | 18.814 | 45.480† |
| `exp_m3fd_lcaf_ca_mask` | 原始 LCAF CA | Mask+GFB | 191 | 0.9215 | 0.8589 | 0.9083 | 0.5875 | 15.660 | 46.716 |
| `exp_mask` | 历史频域 CA | Mask+GFB | 188 | 0.9209 | 0.8542 | 0.9104 | 0.5902 | 19.546 | 46.716† |
| `exp_freq_entropy_mask_gfb_uniform_lr` | 修正熵路由频域 CA | Mask+GFB | 153 | 0.9061 | 0.8528 | 0.9043 | 0.5809 | 19.546 | 53.527† |
| `exp_mask_gfb_only_uniform_lr_m3fd_full200` | 无注意力 | Mask+GFB | 178 | 0.8923 | 0.8572 | 0.9001 | 0.5852 | 12.823 | 39.674 |
| `exp_rgca_no_reliability_mask_gfb_uniform_lr_m3fd_full200` | RGCA，无可靠性门 | Mask+GFB | 185 | 0.9173 | 0.8631 | 0.9115 | 0.5932 | 13.550 | 43.349 |
| **`exp_rgca_mask_gfb_uniform_lr`** | **完整 RGCA，含可靠性门** | **Mask+GFB** | **192** | **0.9073** | **0.8694** | **0.9117** | **0.5938** | **13.727** | **44.218** |

† 频域模型的 FFT、复数运算和函数式操作可能未被 THOP完整统计。

### 4.1 最有价值的消融结论

| 对比（后者减前者） | ΔP | ΔR | ΔmAP50 | ΔmAP50:95 | ΔParams | ΔGFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Mask+GFB → RGCA无门控+Mask+GFB | +2.50pp | +0.59pp | +1.14pp | +0.80pp | +0.727M | +3.675G |
| RGCA无门控 → 完整RGCA | -1.00pp | +0.63pp | +0.02pp | +0.06pp | +0.177M | +0.870G |
| LCAF CA+Mask+GFB → 完整RGCA | -1.42pp | +1.04pp | +0.34pp | +0.63pp | -1.933M | -2.498G |
| Mask+GFB → 完整RGCA | +1.50pp | +1.21pp | +1.16pp | +0.86pp | +0.904M | +4.545G |

现有单 seed 结果支持：

- 整体 RGCA 模态增强相对无注意力 Mask+GFB 有明确收益；
- RGCA 相对同后融合的原始 LCAF attention，mAP50和mAP50:95更高，同时参数和
  计算量更低；
- RGCA 更偏向提高 Recall，单点 Precision不一定更高。

但可靠性门控本身只带来 `+0.02pp mAP50 / +0.06pp mAP50:95`，属于非常小的
单次差异。当前证据不足以声称门控在精度上有“显著提升”；论文应保守表述为
“以很小开销实现内容相关的跨模态抑制，并保持/略微改善总体AP”，并补做至少
3个随机种子的均值±标准差。

### 4.2 可靠性门控的结构开销（实测）

| RGCA版本 | 原始参数量 | 融合后参数量 | GMACs | GFLOPs |
|---|---:|---:|---:|---:|
| 无可靠性门 | 13,566,472 | 13,550,152 | 21.674 | 43.349 |
| 含可靠性门 | 13,743,436 | 13,727,116 | 22.109 | 44.218 |
| 增量 | +176,964 | +176,964 | +0.435 | +0.870 |

相对无门控版本，门控增加约1.30%参数和2.01%计算量。四个融合尺度均包含一个
可靠性门网络；两个权重之间的原始参数差与门控参数总数完全一致，说明这是严格
的门控单变量结构对比。

### 4.3 尚未完成的消融

- 单 GFB 配置已经存在：
  `models/transformer/yolov5s_LCAFNet_M3FD_GFBOnly.yaml`。
- 当前没有找到与该配置对应的完整训练目录和 `best.pt`，所以不能报告单 GFB
  精度。
- 仍建议补做：RGCA去局部分支、RGCA把真实跨模态Q/K改回同模态Q/K，以及核心
  模型3个以上随机种子。

## 5. M3FD 最终 test 结果与协议警告

当前已有论文汇总表中的 M3FD 最终 test 数值来自
`m3fd_rgca_mask_gfb_best`，即 RGCA+BECP 微调模型，而不是第4节的纯 RGCA：

| 模型/协议 | People AP50 | Car | Bus | Lamp | Motorcycle | Truck | mAP50 | mAP50:95 | Params(M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LCAFNet论文基线，official test840 | 0.902 | 0.943 | 0.924 | 0.942 | 0.860 | 0.895 | 0.911 | 0.593 | 15.4 |
| RGCA+BECP，本地 paper test840 | 0.898 | 0.947 | 0.946 | 0.920 | 0.875 | 0.836 | 0.904 | 0.590 | 13.742 |
| RGCA+BECP，本地 held-out test420 | 0.896 | 0.954 | 0.934 | 0.915 | 0.862 | 0.819 | 0.897 | 0.588 | 13.742 |

注意：

- `paper test840` 合并了本地 val420和test420，其中val420曾用于选择权重，存在
  选模偏差；held-out test420完全未参与选模，更适合表示本地泛化结果。
- LCAFNet论文的official test840与本地held-out test420不是同一协议，不能直接
  作严格优劣结论。
- 纯 RGCA、无门控RGCA和Mask+GFB-only目前最可靠的统一对比是第4节val日志表。
  若要形成正式test消融表，应固定各自best权重，在同一held-out test420上重新
  运行并保存机器可读CSV。
- 本地 `runs/test/lcafnet_original_m3fd_test420` 等旧复核目录保存了曲线和图片，
  但没有保存完整标量结果文件；引用精确数值前应重新验证并记录终端输出。

## 6. M3FD 四场景专项实验

Daytime、Night、Overcast和Challenge均为“只用该场景train训练、该场景val选优、
该场景test评测”的独立专家模型。它们不是同一个完整M3FD模型按场景切片测试。

| 场景 | RGCA P | RGCA R | RGCA mAP50 | RGCA mAP50:95 | LCAF P | LCAF R | LCAF mAP50 | LCAF mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Daytime | 0.839 | 0.820 | 0.845 | 0.506 | 0.860 | 0.770 | 0.828 | 0.487 |
| Night | 0.935 | 0.964 | 0.973 | 0.625 | 0.890 | 0.902 | 0.912 | 0.614 |
| Overcast | 0.928 | 0.879 | 0.920 | 0.568 | 0.928 | 0.838 | 0.889 | 0.543 |
| Challenge | 0.916 | 0.828 | 0.896 | 0.504 | 0.913 | 0.824 | 0.906 | 0.520 |
| 四场景宏平均 | 0.9045 | 0.8728 | 0.9085 | 0.5508 | 0.8978 | 0.8335 | 0.8838 | 0.5410 |

完整路径、类别AP50、权重SHA和复核目录见：

- `README_M3FD_SCENE_RGCA_RESULTS.md`
- `README_M3FD_SCENE_LCAFCA_RESULTS.md`

四场景宏平均上RGCA更高，主要提升来自Night和Overcast；LCAF attention在
Challenge专项训练中反而更高。因此论文不能写成RGCA在每个场景都更好。

## 7. 可视化资产与科学解释边界

### 7.1 检测结果与论文排版图

- 完整test检测结果：`paper/m3fd_detection_results`
- RGCA优于LCAFNet的筛选：
  `paper/m3fd_detection_results/rgca_better_top10`
- RGCA检测而LCAFNet漏检的GT实例：
  `paper/m3fd_detection_results/rgca_detected_lcafnet_missed_top10`
- 可能漏标候选：
  `paper/m3fd_detection_results_lowconf_010/rgca_possible_unlabeled_top40`
- 7组3×4论文排版图：
  `paper/m3fd_detection_results/paper_visualizations_rgca_vs_lcafnet_selected7_3x4`

上述检测可视化中的“RGCA”权重是 `m3fd_rgca_mask_gfb_best`，即RGCA+BECP。
可能漏标候选在现有标注口径下仍属于false positive；未经人工复核和补标，不能
当作准确检测用于定量证明。

### 7.2 可靠性图与热图

- Panel A可靠性图：`paper/rgca_reliability_visualization`
- 有/无门控Grad-CAM：`paper/rgca_gate_gradcam_comparison_m3fd`
- 四场景全部test的RGB/IR配对热图：
  `paper/rgca_gate_paired_modal_heatmaps_m3fd`
- 门控聚焦统计：`paper/rgca_gate_focus_comparison_m3fd`

每个目录中的 `SCIENTIFIC_CAVEAT.txt` 或 `README_CN.md` 应与图片一起阅读。
目前原始可靠性图的均值约在0.48附近，部分层/样本的空间标准差很小，RGB和IR
门控差异也可能很弱。这说明不能只挑图宣称门控产生了强烈的模态切换。更可靠的
证据组合应包括：严格无门控消融、多seed统计、可靠性分布/方差、目标与背景区域
统计以及配对RGB/IR热图。

Grad-CAM通常是针对某个预测目标或某个标量反向传播；一张图有多个目标而热图只
突出一个目标，不等价于模型只能检测一个主体。论文图必须说明target selection
规则，并优先为多个目标分别生成target-specific热图。

## 8. 其他数据集结果的当前结论

详细表格见 `paper/RGB_T_multidataset_comparison_tables.md`。现有汇总结论为：

| 数据集 | LCAFNet mAP50 / mAP50:95 | 本地模型 mAP50 / mAP50:95 | 本地模型参数量 |
|---|---:|---:|---:|
| M3FD paper-test840 | 0.911 / 0.593 | 0.904 / 0.590（RGCA+BECP） | 13.742M |
| MFAD | 0.798 / 0.533 | 0.804 / 0.535（C0-RGCA） | 13.743M |
| LLVIP | 0.977 / 0.650 | 0.9762 / 0.6526（C0-RGCA） | 13.728M |
| FLIR | 0.813 / 0.411 | 0.812 / 0.408（C0-SOD-ASF） | 13.925M |

这些行不是同一个最终模型变体：M3FD含BECP，FLIR含SOD-ASF，MFAD/LLVIP为
C0-RGCA。当前证据支持“参数减少约9.6%-10.9%时取得总体相当性能，并在MFAD
小幅提升”，不支持“在四个数据集全面超过LCAFNet”。MFAD、LLVIP和FLIR的本地
数据配置还存在val直接指向官方test并用于选模的问题，投稿前应重新建立独立val。

## 9. 当前代码状态

截至本文件生成时，直接运行：

```bash
cd /home/dell/lcp/LCAFNet-main
/home/dell/anaconda3/envs/MOD/bin/python train.py
```

会启动的是：

```text
完整M3FD + RGCA无可靠性门 + Mask+GFB
200 epochs / batch16 / effective batch64 / 640×640 / SGD / cosine
```

即 `train.py` 当前默认仍指向：

`models/transformer/yolov5s_LCAFNet_M3FD_RGCANoReliability_MaskGFB.yaml`

该实验已经训练完成；再次直接运行会因 `exist_ok=False` 创建递增的新目录，而不是
恢复或训练完整有门控RGCA。开始新实验前必须先确认 `TRAINING_CONFIG`，不要仅凭
历史对话假设默认模型。

现代可直接加载的核心配置：

- 完整RGCA：`models/transformer/yolov5s_LCAFNet_M3FD.yaml`
- 无可靠性门RGCA：
  `models/transformer/yolov5s_LCAFNet_M3FD_RGCANoReliability_MaskGFB.yaml`
- Mask+GFB only：
  `models/transformer/yolov5s_LCAFNet_M3FD_MaskGFBOnly.yaml`
- GFB only（未完成训练）：
  `models/transformer/yolov5s_LCAFNet_M3FD_GFBOnly.yaml`
- LCAF attention + Mask+GFB：
  `models/transformer/yolov5s_LCAFNet_M3FD_LCAFCA_MaskGFB.yaml`

旧checkpoint以pickle保存完整模型类。部分历史频域/LCAF实验使用过后来被重定义
的通用类名，直接用当前代码加载可能报错，甚至静默进入不同forward。正式复评旧
权重前必须恢复对应时期的代码；不要因为checkpoint能加载就默认结构正确。

## 10. 推荐的新对话起始指令

可以把下面内容直接发送给新的Codex对话：

```text
请先完整阅读
/home/dell/lcp/LCAFNet-main/README_RGCA_LCAFNET_EXPERIMENT_HANDOFF.md，
并以文件中记录的模型身份、权重路径、统一val口径和协议警告为准。
核心论文模型是纯C0完整RGCA：
runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt。
注意runs/train/m3fd_rgca_mask_gfb_best是RGCA+BECP微调，不是纯RGCA。
在修改代码或汇总论文表格前，请先核对当前train.py默认配置和目标checkpoint的
opt.yaml/模型结构，不要混用val、held-out test420、paper test840或场景专项结果。
```

## 11. 相关详细记录索引

- 完整M3FD消融审计：`paper/M3FD_RGCA_ABLATION_SUMMARY.md`（审计日期为
  2026-08-22，其中“无可靠性门尚未完成”等旧缺口已由本文件的最新结果覆盖）
- RGCA设计：`paper/RGCA_DESIGN.md`
- 四场景RGCA：`README_M3FD_SCENE_RGCA_RESULTS.md`
- 四场景LCAF attention：`README_M3FD_SCENE_LCAFCA_RESULTS.md`
- 多数据集论文对比：`paper/RGB_T_multidataset_comparison_tables.md`
- 检测可视化说明：`paper/m3fd_detection_results/README_CN.md`
- C0-BECP代码快照：`archives/c0_m3fd_becp_20260819/README.md`
- 历史代码存档索引：`archives/README.md`
