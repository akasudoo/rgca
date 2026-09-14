import math
import unittest

import torch
import torch.nn.functional as F

from models.common import (
    ExpMaskFrequencyCrossAttention,
    FrequencyCrossAttention_S,
    HAFFormer,
    Windowed2DGOATCrossAttention,
)


def grad_norm(param):
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().float().abs().sum())


class FrequencyAttentionTests(unittest.TestCase):
    def test_exp_mask_channel_confidence_is_informative(self):
        module = ExpMaskFrequencyCrossAttention(4, 1, False)
        uniform = torch.full((1, 1, 4, 4), 0.25)
        sharp = torch.full((1, 1, 4, 4), 1e-3)
        sharp[..., 0] = 1.0 - 3e-3

        uniform_score = module.compute_channel_confidence(uniform)
        sharp_score = module.compute_channel_confidence(sharp)
        self.assertLess(float(uniform_score.max()), 1e-5)
        self.assertGreater(float(sharp_score.mean()), 0.95)
        self.assertFalse(torch.allclose(uniform_score, sharp_score))

        del module.head_dim
        legacy_checkpoint_score = module.compute_channel_confidence(sharp)
        self.assertTrue(torch.allclose(legacy_checkpoint_score, sharp_score))

        module.eval()
        q_feat = torch.randn(1, 4, 5, 7)
        k_feat = torch.randn(1, 4, 5, 7)
        uniform_route = module.build_route_attention(
            q_feat, k_feat, uniform, 5, 7)
        sharp_route = module.build_route_attention(
            q_feat, k_feat, sharp, 5, 7)
        self.assertFalse(torch.allclose(uniform_route, sharp_route))

    def test_channel_score_entropy_is_informative(self):
        module = FrequencyCrossAttention_S(4, 1, False, channel_score_mode='entropy')
        uniform = torch.full((1, 1, 4, 4), 0.25)
        sharp = torch.full((1, 1, 4, 4), 1e-3)
        sharp[..., 0] = 1.0 - 3e-3

        uniform_score = module.compute_channel_score(uniform)
        sharp_score = module.compute_channel_score(sharp)
        self.assertLess(float(uniform_score.max()), 1e-5)
        self.assertGreater(float(sharp_score.mean()), 0.95)

        module.channel_score_mode = 'legacy_mean'
        legacy_a = module.compute_channel_score(uniform)
        legacy_b = module.compute_channel_score(sharp)
        legacy_value = 1.0 / (1.0 + math.exp(-0.25))
        self.assertTrue(torch.allclose(legacy_a, torch.full_like(legacy_a, legacy_value)))
        self.assertTrue(torch.allclose(legacy_a, legacy_b))

    def test_eot_channel_prior_neutral_initialization(self):
        torch.manual_seed(0)
        module = FrequencyCrossAttention_S(16, 4, False, prior_mode='eot_channel_v2')
        logits = torch.randn(2, 4, 4, 4)
        q_feat = torch.randn(2, 16, 7, 9)
        out = module.apply_attention_prior(logits.clone(), q_feat=q_feat)
        self.assertLess(float((out - logits).abs().max()), 1e-6)

        grid = module.spatial_routing_prior.build_fourier_grid(
            7, 9, q_feat.device, module.spatial_routing_prior.prior_head[0].weight.dtype)
        spatial_prior = module.spatial_routing_prior.prior_head(grid)
        self.assertGreater(float(spatial_prior.abs().max()), 0.0)
        self.assertLess(float(spatial_prior.abs().max()), 1e-2)

    def test_first_backward_has_gradients(self):
        torch.manual_seed(1)
        module = HAFFormer(32).train()
        self.assertIsInstance(module.mhca_rgb, ExpMaskFrequencyCrossAttention)
        self.assertAlmostEqual(float(module.mhca_rgb.beta.detach()), 0.01, places=6)
        rgb = torch.randn(2, 32, 8, 12, requires_grad=True)
        ir = torch.randn(2, 32, 8, 12, requires_grad=True)
        target = torch.randn(2, 32, 8, 12)
        out = module([rgb, ir])
        loss = F.mse_loss(out, target)
        loss.backward()

        mhca = module.mhca_rgb
        checks = {
            'attention': grad_norm(mhca.qk.weight),
            'router': grad_norm(mhca.route_proj[-2].weight),
            'filter_basis': grad_norm(mhca.complex_weights),
            'beta': grad_norm(mhca.beta),
            'reweight_mlp': grad_norm(mhca.reweight_mlp[-1].weight),
            'mask_input': grad_norm(module.foreground_mask.mask_head[0].weight),
            'mask_output': grad_norm(module.foreground_mask.mask_head[-2].weight),
        }
        for name, value in checks.items():
            self.assertTrue(math.isfinite(value), name)
            self.assertGreater(value, 0.0, name)

    def test_channel_only_routing_bypasses_spatial_prior(self):
        torch.manual_seed(7)
        module = FrequencyCrossAttention_S(16, 4, False, routing_mode='entropy_channel_only').train()
        spatial_calls = []
        handle = module.spatial_routing_prior.register_forward_hook(
            lambda _module, _inputs, _output: spatial_calls.append(1))
        output = module([torch.randn(2, 16, 7, 9), torch.randn(2, 16, 7, 9)])
        output.square().mean().backward()
        handle.remove()

        self.assertEqual(spatial_calls, [])
        self.assertGreater(grad_norm(module.channel_prior.key_bias), 0.0)
        self.assertIsNone(module.spatial_routing_prior.prior_head[-1].weight.grad)

    def test_fusion_mask_and_bidirectional_attention_receive_gradients(self):
        torch.manual_seed(9)
        module = HAFFormer(32).train()
        rgb = torch.randn(2, 32, 8, 12)
        ir = torch.randn(2, 32, 8, 12)
        mask_outputs = []
        handle = module.foreground_mask.register_forward_hook(
            lambda _module, _inputs, output: mask_outputs.append(output))

        output = module([rgb, ir])
        output.square().mean().backward()
        handle.remove()

        mask = mask_outputs[0]
        self.assertTrue(torch.isfinite(mask).all())
        self.assertGreater(float(mask.std()), 1e-4)
        self.assertGreater(float(mask.min()), 0.0)
        self.assertLess(float(mask.max()), 1.0)

        checks = {
            'rgb_attention': module.mhca_rgb.qk.weight,
            'ir_attention': module.mhca_ir.qk.weight,
            'attention_projection': module.mhca_rgb.project_out.weight,
            'foreground_mask_input': module.foreground_mask.mask_head[0].weight,
            'foreground_mask_output': module.foreground_mask.mask_head[-2].weight,
        }
        for name, parameter in checks.items():
            value = grad_norm(parameter)
            self.assertTrue(math.isfinite(value), name)
            self.assertGreater(value, 0.0, name)
        self.assertFalse(hasattr(module, 'conv'))
        self.assertFalse(hasattr(module, 'dwconv'))
        self.assertFalse(hasattr(module, 'concat'))

    def test_foreground_mask_is_the_direct_modality_gate(self):
        torch.manual_seed(11)
        module = HAFFormer(32).eval()
        rgb = torch.randn(2, 32, 8, 12)
        ir = torch.randn(2, 32, 8, 12)
        with torch.no_grad():
            rgb_enhanced = module.mhca_rgb([rgb, ir]) + rgb
            ir_enhanced = module.mhca_ir([ir, rgb]) + ir
            mask = module.foreground_mask(rgb_enhanced, ir_enhanced)
            expected = rgb_enhanced + mask * (ir_enhanced - rgb_enhanced)
            actual = module([rgb, ir])

        self.assertLess(float((actual - expected).abs().max()), 1e-6)
        self.assertLess(abs(float(mask.mean()) - 0.5), 0.1)

    def test_goat2d_factorization_matches_explicit_formula(self):
        torch.manual_seed(2)
        module = Windowed2DGOATCrossAttention(64, num_heads=4, window_size=4, rank_x=1, rank_y=1,
                                              use_null_sink=False)
        module.alpha_y.data.normal_(0, 0.1)
        module.beta_y.data.normal_(0, 0.1)
        module.alpha_x.data.normal_(0, 0.1)
        module.beta_x.data.normal_(0, 0.1)

        y, x = module._coords(torch.device('cpu'), torch.float32)
        prior_factored = module.explicit_prior()
        dy = y[:, None] - y[None, :]
        dx = x[:, None] - x[None, :]
        wy = module.freq_y[0]
        wx = module.freq_x[0]
        manual = (
            module.alpha_y[:, 0, None, None] * torch.cos(wy * dy)
            + module.beta_y[:, 0, None, None] * torch.sin(wy * dy)
            + module.alpha_x[:, 0, None, None] * torch.cos(wx * dx)
            + module.beta_x[:, 0, None, None] * torch.sin(wx * dx)
        )
        self.assertLess(float((prior_factored - manual).abs().max()), 1e-5)

    def test_goat2d_sdpa_scaling(self):
        torch.manual_seed(3)
        module = Windowed2DGOATCrossAttention(64, num_heads=4, window_size=4, rank_x=1, rank_y=1,
                                              use_null_sink=False)
        module.alpha_y.data.normal_(0, 0.1)
        module.beta_y.data.normal_(0, 0.1)
        module.alpha_x.data.normal_(0, 0.1)
        module.beta_x.data.normal_(0, 0.1)

        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        q_total, k_total = module.compose_qk(q, k)
        logits_sdpa = torch.matmul(q_total * math.sqrt(module.head_dim), k_total.transpose(-2, -1))
        logits_sdpa = logits_sdpa / math.sqrt(module.head_dim)
        content = torch.matmul(q[..., :module.d_content], k[..., :module.d_content].transpose(-2, -1))
        content = content / math.sqrt(module.d_content)
        prior = module.explicit_prior().unsqueeze(0)
        self.assertLess(float((logits_sdpa - (content + prior)).abs().max()), 1e-5)

    def test_row_shift_invariance_and_translation_equivariance(self):
        module = Windowed2DGOATCrossAttention(64, num_heads=4, window_size=4, rank_x=1, rank_y=1,
                                              use_null_sink=False)
        module.alpha_y.data.normal_(0, 0.1)
        module.beta_y.data.normal_(0, 0.1)
        module.alpha_x.data.normal_(0, 0.1)
        module.beta_x.data.normal_(0, 0.1)
        prior = module.explicit_prior()
        row_shift = torch.randn(4, prior.shape[-2], 1)
        self.assertLess(float((prior.softmax(dim=-1) - (prior + row_shift).softmax(dim=-1)).abs().max()), 1e-6)

        y, x = module._coords(torch.device('cpu'), torch.float32)
        shifted = module.explicit_prior(coords=(y + 0.37, x - 0.19))
        self.assertLess(float((prior - shifted).abs().max()), 1e-5)

    def test_multisize_forward_backward(self):
        for h, w in [(8, 12), (13, 17), (20, 20), (64, 80)]:
            module = HAFFormer(32).train()
            rgb = torch.randn(1, 32, h, w, requires_grad=True)
            ir = torch.randn(1, 32, h, w, requires_grad=True)
            out = module([rgb, ir])
            self.assertEqual(tuple(out.shape), (1, 32, h, w))
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            self.assertGreater(grad_norm(module.mhca_rgb.beta), 0.0)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_amp_smoke(self):
        module = HAFFormer(32).cuda().train()
        with torch.no_grad():
            module.mhca_rgb.beta.fill_(0.01)
            module.mhca_ir.beta.fill_(0.01)
        rgb = torch.randn(1, 32, 8, 12, device='cuda')
        ir = torch.randn(1, 32, 8, 12, device='cuda')
        with torch.cuda.amp.autocast():
            out = module([rgb, ir])
            loss = out.square().mean()
        self.assertTrue(torch.isfinite(out).all())
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        self.assertGreater(grad_norm(module.mhca_rgb.beta), 0.0)
        checks = {
            'frequency_router': module.mhca_rgb.route_proj[-2].weight,
            'frequency_filter': module.mhca_rgb.complex_weights,
            'mask_input': module.foreground_mask.mask_head[0].weight,
            'mask_output': module.foreground_mask.mask_head[-2].weight,
        }
        for name, parameter in checks.items():
            value = grad_norm(parameter)
            self.assertTrue(math.isfinite(value), name)
            self.assertGreater(value, 0.0, name)


if __name__ == '__main__':
    unittest.main()
