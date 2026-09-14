import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ConvGRU import ConvGRUCell
except ModuleNotFoundError:
    class ConvGRUCell(nn.Module):
        def __init__(self, input_size, hidden_size, kernel_size):
            super(ConvGRUCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.kernel_size = kernel_size
            padding = int((kernel_size - 1) / 2)
            self.ConvGates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=padding)
            self.Conv_ct = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, input, hidden):
            if hidden is None:
                size_h = [input.size(0), self.hidden_size] + list(input.size()[2:])
                hidden = input.new_zeros(size_h)

            gates = self.ConvGates(torch.cat((input, hidden), 1))
            reset_gate, update_gate = gates.chunk(2, 1)
            reset_gate = torch.sigmoid(reset_gate)
            update_gate = torch.sigmoid(update_gate)
            gated_hidden = reset_gate * hidden
            candidate = torch.tanh(self.Conv_ct(torch.cat((input, gated_hidden), 1)))
            return (1 - update_gate) * hidden + update_gate * candidate


class PPM(nn.Module):
    """
    多尺度金字塔池化模块：保持你的原始设计不变。
    输入一个特征图，输出多个 dilation 分支特征，作为 GNN 节点。
    """
    def __init__(self, chnn_in, rd_sc, dila):
        super(PPM, self).__init__()
        chnn = chnn_in // rd_sc
        convs = [nn.Sequential(
            nn.Conv2d(chnn_in, chnn, 3, padding=ii, dilation=ii, bias=False),
            nn.BatchNorm2d(chnn),
            nn.ReLU(inplace=True))
            for ii in dila]
        self.convs = nn.ModuleList(convs)

    def forward(self, inputs):
        feats = []
        for conv in self.convs:
            feat = conv(inputs)
            feats.append(feat)
        return feats


class GraphReasoning(nn.Module):
    """
    Bayesian Feature-Preserving Graph Reasoning

    与原始版本保持相同输入输出：
        inputs = (feat_rgb, feat_dep, nd_rgb, nd_dep)
        return feat_rgb_list, feat_dep_list

    核心改动：
    1) 仍然使用 RGB/IR 的多尺度特征作为节点；
    2) 仍然使用节点差值生成边权重；
    3) 将边权重解释为边可信后验概率 z_ij；
    4) 增加节点原始特征保留后验概率 r_i；
    5) 通过 r_i 在原始节点特征 h_i^0 和图更新特征之间自适应平衡，缓解特征模糊/过平滑。
    """

    def __init__(self, chnn_in, rd_sc, dila, n_iter):
        super().__init__()
        self.n_iter = n_iter

        # 1) 多尺度节点生成
        self.ppm_rgb = PPM(chnn_in, rd_sc, dila)
        self.ppm_dep = PPM(chnn_in, rd_sc, dila)

        self.num_scales = len(dila)
        self.n_node = self.num_scales * 2

        # 2) Leader Node 增强相关：保持你的原始设计
        chnn = chnn_in * 2 // rd_sc
        C_ca = [nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(chnn, chnn // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn // 4, chnn_in // rd_sc, 1, bias=False))
            for _ in range(2)]
        self.C_ca = nn.ModuleList(C_ca)

        # 3) 构建稀疏邻接表
        neighbors = self._build_sparse_neighbors(self.num_scales)

        # 4) Bayesian 图推理模型
        self.joint_graph = GraphModel(
            N=self.n_node,
            chnn_in=chnn_in // rd_sc,
            num_scales=self.num_scales,
            neighbors=neighbors,
            # 下面几个参数可以根据实验再调
            edge_prior=0.50,       # 边可信先验：0.5 表示不强制边开/关
            retain_prior=0.60,     # 节点保留先验：降低保守性，让 GNN 更新更容易参与融合
            sample_edges=False,    # 默认用后验均值，训练更稳定；可改 True 做随机采样
            sample_retention=False,
            temperature=0.67,
        )

        self.bayesian_loss = None

    @staticmethod
    def _build_sparse_neighbors(num_scales: int):
        """
        构建稀疏邻接（对称化）：
        - 同模态相邻尺度链式连接；
        - RGB P1/P2/P3 与 IR P1/P2/P3 六节点全连接。
        """
        N = num_scales * 2
        neighbors = [[] for _ in range(N)]

        def add_undirected_edge(edges_set, u, v):
            if u == v:
                return
            edges_set.add((u, v))
            edges_set.add((v, u))

        edges = set()

        # RGB 相邻尺度连接
        for s in range(num_scales - 1):
            add_undirected_edge(edges, s, s + 1)

        # IR 相邻尺度连接
        base = num_scales
        for s in range(num_scales - 1):
            add_undirected_edge(edges, base + s, base + s + 1)

        # 前三个尺度的 RGB/IR 全连接
        k = min(3, num_scales)
        rgb_nodes = [i for i in range(k)]
        ir_nodes = [num_scales + i for i in range(k)]

        for a in range(len(rgb_nodes)):
            for b in range(a + 1, len(rgb_nodes)):
                add_undirected_edge(edges, rgb_nodes[a], rgb_nodes[b])

        for a in range(len(ir_nodes)):
            for b in range(a + 1, len(ir_nodes)):
                add_undirected_edge(edges, ir_nodes[a], ir_nodes[b])

        for u in rgb_nodes:
            for v in ir_nodes:
                add_undirected_edge(edges, u, v)

        for u, v in edges:
            if v not in neighbors[u]:
                neighbors[u].append(v)

        return neighbors

    def _enh(self, Func, src, dst):
        """Leader Node 增强函数：保持你的原始设计。"""
        out = torch.sigmoid(Func(src)) * dst + dst
        return out

    def _inn(self, Func, feat):
        """
        内部推理函数。

        重要改动：
        - feat0 保存图推理开始前的原始节点特征；
        - 每一轮图更新都把 feat0 传给 GraphModel；
        - 这样节点保留门 r_i 保留的是原始 PPM 节点，而不是上一轮已经被混合后的节点。
        """
        feat = [fm.unsqueeze(1) for fm in feat]
        feat = torch.cat(feat, 1)  # [B, N, C, H, W]
        feat0 = feat

        loss_total = feat.new_tensor(0.0)
        for _ in range(self.n_iter):
            feat = Func(feat, init_inputs=feat0)
            loss_total = loss_total + Func.get_bayesian_loss()

        self.bayesian_loss = loss_total / max(self.n_iter, 1)

        feat = torch.split(feat, 1, 1)
        feat = [fm.squeeze(1) for fm in feat]
        return feat

    def get_bayesian_loss(self):
        """
        在训练代码里可选加入：
            loss = det_loss + lambda_bayes * model.xxx.get_bayesian_loss()

        如果你的外层模型不方便拿到这个模块，也可以先不加这个 loss，
        模块仍然可以作为概率门控残差 GNN 正常训练。
        """
        if self.bayesian_loss is None:
            return torch.tensor(0.0)
        return self.bayesian_loss

    def forward(self, inputs, node=False):
        feat_rgb, feat_dep, nd_rgb, nd_dep = inputs

        # 1) 生成多尺度节点
        feat_rgb = self.ppm_rgb(feat_rgb)
        feat_dep = self.ppm_dep(feat_dep)

        # 2) Leader Node 增强（可选）
        if node:
            feat_rgb = [self._enh(self.C_ca[0], nd_rgb, fm) for fm in feat_rgb]
            feat_dep = [self._enh(self.C_ca[1], nd_dep, fm) for fm in feat_dep]

        # 3) 联合图推理
        feats_joint = feat_rgb + feat_dep
        feats_out = self._inn(self.joint_graph, feats_joint)

        # 4) 拆分输出
        feat_rgb = feats_out[:self.num_scales]
        feat_dep = feats_out[self.num_scales:]

        return feat_rgb, feat_dep


class GraphModel(nn.Module):
    """
    Bayesian Feature-Preserving GraphModel

    输入:  [B, N, C, H, W]
    输出:  [B, N, C, H, W]

    相比原始 GraphModel 的核心改动：
    1) edge_mlp(h_i - h_j) 不再只是普通边权重，而是边可信后验概率 q(z_ij|h_i,h_j)；
    2) 增加 retain_mlp，预测节点原始特征保留概率 q(r_i|h_i^0,h_i^l,m_i)；
    3) 节点更新变成：
       h_i^{l+1} = r_i * h_i^0 + (1-r_i) * candidate_i
    4) 记录 KL loss 和 preserve loss，用于训练时作为正则项。
    """

    def __init__(
        self,
        N: int,
        chnn_in: int = 256,
        num_scales: int = 3,
        neighbors=None,
        use_reliability_gate: bool = False,  # 保留字段，兼容旧调用
        edge_prior: float = 0.50,
        retain_prior: float = 0.70,
        sample_edges: bool = False,
        sample_retention: bool = False,
        temperature: float = 0.67,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_node = N
        self.chnn = chnn_in
        self.num_scales = num_scales
        self.use_reliability_gate = use_reliability_gate

        if neighbors is None:
            neighbors = []
            for i in range(N):
                neighbors.append([j for j in range(N) if j != i])
        self.neighbors = neighbors

        hidden = max(chnn_in // 4, 8)

        # 边后验概率 q(z_ij)：仍然使用你的“特征差值作为边”的思想
        # z_ij 越大，说明 j 节点给 i 节点传递消息越可信。
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(chnn_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )

        # 节点保留后验概率 q(r_i)：决定保留多少原始节点特征 h_i^0。
        self.retain_mlp = nn.Sequential(
            nn.Conv2d(chnn_in * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )

        # ConvGRU 用于生成候选更新状态
        self.ConvGRU = ConvGRUCell(chnn_in, chnn_in, kernel_size=1)

        # residual 缩放参数：sigmoid(-2)≈0.119，比 -3 更鼓励图更新参与融合。
        self.gamma = nn.Parameter(torch.tensor(-2.0))

        # 先验概率，用 buffer 保证会随模型迁移到 GPU
        edge_prior = float(min(max(edge_prior, eps), 1.0 - eps))
        retain_prior = float(min(max(retain_prior, eps), 1.0 - eps))
        retain_prior_logit = torch.logit(torch.tensor(retain_prior)).item()
        nn.init.constant_(self.retain_mlp[-1].bias, retain_prior_logit)
        self.register_buffer("edge_prior", torch.tensor(edge_prior))
        self.register_buffer("retain_prior", torch.tensor(retain_prior))

        self.sample_edges = sample_edges
        self.sample_retention = sample_retention
        self.temperature = temperature
        self.eps = eps

        # 记录最近一次 forward 的正则项
        self._last_edge_kl = None
        self._last_retain_kl = None
        self._last_preserve_loss = None
        self._last_bayesian_loss = None

    def _bernoulli_kl(self, q, p):
        """KL[ Bernoulli(q) || Bernoulli(p) ]，返回均值。"""
        q = q.clamp(self.eps, 1.0 - self.eps)
        p = p.clamp(self.eps, 1.0 - self.eps)
        kl = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
        return kl.mean()

    def _relaxed_bernoulli(self, logits, temperature):
        """
        连续松弛 Bernoulli 采样。
        默认 sample_edges=False 不会使用；如果打开采样，用它近似二值边/保留门。
        """
        u = torch.rand_like(logits).clamp(self.eps, 1.0 - self.eps)
        logistic_noise = torch.log(u) - torch.log(1.0 - u)
        return torch.sigmoid((logits + logistic_noise) / temperature)

    def get_bayesian_loss(self):
        if self._last_bayesian_loss is None:
            return torch.tensor(0.0)
        return self._last_bayesian_loss

    def get_loss_items(self):
        """
        可用于日志打印：
            edge_kl, retain_kl, preserve = graph.get_loss_items()
        """
        return {
            "edge_kl": self._last_edge_kl,
            "retain_kl": self._last_retain_kl,
            "preserve": self._last_preserve_loss,
            "bayesian": self._last_bayesian_loss,
        }

    def forward(self, inputs, init_inputs=None):
        # inputs: [B, N, C, H, W]
        b, n, c, h, w = inputs.shape
        assert n == self.n_node, f"Expect N={self.n_node}, but got {n}"
        assert c == self.chnn, f"Expect C={self.chnn}, but got {c}"

        if init_inputs is None:
            init_inputs = inputs
        assert init_inputs.shape == inputs.shape, "init_inputs must have the same shape as inputs"

        feat_s = [inputs[:, ii, :] for ii in range(self.n_node)]
        feat_0 = [init_inputs[:, ii, :] for ii in range(self.n_node)]

        pred_s = []
        edge_kl_list = []
        retain_kl_list = []
        preserve_list = []

        for i in range(self.n_node):
            h_t = feat_s[i]
            h_0 = feat_0[i]
            nbrs = self.neighbors[i]

            if len(nbrs) == 0:
                pred_s.append(h_t)
                continue

            msg_list = []
            edge_prob_list = []
            edge_logit_list = []

            for j in nbrs:
                h_j = feat_s[j]

                # 你的原始思想：特征差值作为边。
                # 这里把差值映射为边可信后验概率 q(z_ij)。
                edge_diff = h_t - h_j
                edge_logit = self.edge_mlp(edge_diff)       # [B,1,H,W]
                edge_prob = torch.sigmoid(edge_logit)       # q(z_ij)

                if self.training and self.sample_edges:
                    z_ij = self._relaxed_bernoulli(edge_logit, self.temperature)
                else:
                    z_ij = edge_prob

                edge_logit_list.append(edge_logit)
                edge_prob_list.append(edge_prob)
                msg_list.append(h_j * z_ij)

            # 归一化注意力：区分“邻居重要性 alpha”和“边可信度 z”
            # edge_prob 控制边是否可信；softmax 控制多个邻居之间谁更重要。
            edge_logits = torch.stack(edge_logit_list, dim=1)  # [B,K,1,H,W]
            alpha = torch.softmax(edge_logits, dim=1)

            msgs = torch.stack(msg_list, dim=1)                # [B,K,C,H,W]
            m_t = (msgs * alpha).sum(dim=1)                    # [B,C,H,W]

            edge_probs = torch.stack(edge_prob_list, dim=1)    # [B,K,1,H,W]
            edge_kl_list.append(self._bernoulli_kl(edge_probs, self.edge_prior))

            # ConvGRU 得到候选更新状态
            h_update = self.ConvGRU(m_t, h_t)
            scale = torch.sigmoid(self.gamma)
            h_candidate = h_t + (h_update - h_t) * scale

            # 节点保留概率 r_i：越大，越保留原始节点 h_0
            retain_feat = torch.cat([
                h_0,
                h_t,
                h_candidate,
                torch.abs(h_candidate - h_0)
            ], dim=1)
            retain_logit = self.retain_mlp(retain_feat)
            retain_prob = torch.sigmoid(retain_logit)          # q(r_i)

            if self.training and self.sample_retention:
                r_i = self._relaxed_bernoulli(retain_logit, self.temperature)
            else:
                r_i = retain_prob

            retain_kl_list.append(self._bernoulli_kl(retain_prob, self.retain_prior))

            # Bayesian feature-preserving update
            h_out = r_i * h_0 + (1.0 - r_i) * h_candidate

            # 保留约束：当 r_i 大时，输出应更接近原始节点特征
            preserve_loss = (r_i * (h_out - h_0).pow(2)).mean()
            preserve_list.append(preserve_loss)

            pred_s.append(h_out)

        pred = torch.stack(pred_s, dim=1).contiguous()

        # 保存正则项。这里先只记录，不强制改变你的外层 loss。
        if len(edge_kl_list) > 0:
            self._last_edge_kl = torch.stack(edge_kl_list).mean()
        else:
            self._last_edge_kl = pred.new_tensor(0.0)

        if len(retain_kl_list) > 0:
            self._last_retain_kl = torch.stack(retain_kl_list).mean()
        else:
            self._last_retain_kl = pred.new_tensor(0.0)

        if len(preserve_list) > 0:
            self._last_preserve_loss = torch.stack(preserve_list).mean()
        else:
            self._last_preserve_loss = pred.new_tensor(0.0)

        # 默认组合权重比较小，防止正则项压过检测损失。
        # 你也可以在外层训练代码中自己重新加权。
        self._last_bayesian_loss = (
            1.0e-3 * self._last_edge_kl +
            1.0e-3 * self._last_retain_kl +
            1.0e-2 * self._last_preserve_loss
        )

        return pred
