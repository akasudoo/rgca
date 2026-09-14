import math
import unittest

import torch

from models.common import (
    HAFFormerRGCAMaskGFB,
    ReliabilityGuidedBidirectionalCrossAttention,
    SharedLatentDiagonalChannelPrior,
)
from models.yolo_test import Model
from train import (
    collect_cross_modal_attention_diagnostics,
    collect_fusion_gate_diagnostics,
    enable_cross_modal_attention_tracking,
    enable_fusion_gate_tracking,
    migrate_rgca_residual_scale_state_dict,
)


class RGCAAttentionTests(unittest.TestCase):
    def test_bidirectional_attention_shape_diagnostics_and_gradients(self):
        torch.manual_seed(11)
        # Exercise the archived fixed-prior branch explicitly; the production
        # default is deliberately no-prior.
        module = ReliabilityGuidedBidirectionalCrossAttention(
            32, prior_mode='fixed', residual_mode='bounded_scalar').train()
        enable_cross_modal_attention_tracking(module)
        rgb = torch.randn(2, 32, 8, 10)
        ir = torch.randn(2, 32, 8, 10)
        delta_rgb, delta_ir = module(rgb, ir)
        self.assertEqual(delta_rgb.shape, rgb.shape)
        self.assertEqual(delta_ir.shape, ir.shape)

        (delta_rgb.square().mean() + delta_ir.square().mean()).backward()
        diagnostics = collect_cross_modal_attention_diagnostics(module)
        self.assertEqual(len(diagnostics), 1)
        row = diagnostics[0]
        self.assertGreater(row[3], 0.0)  # RGB attention std
        self.assertGreater(row[4], 0.0)  # IR attention std
        self.assertGreater(row[6], 0.0)  # RGB reliability std
        self.assertGreater(row[8], 0.0)  # IR reliability std
        self.assertGreater(float(module.qkv.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(module.reliability_gate[-2].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(module.raw_residual_scale.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(module.channel_prior.raw_margin.grad.abs().sum()), 0.0)

    def test_zero_content_logits_fall_back_to_diagonal_prior(self):
        prior = SharedLatentDiagonalChannelPrior(
            num_heads=2, head_dim=4, margin_init=1.0)
        logits = torch.zeros(1, 2, 4, 4)
        probability = prior(logits).softmax(dim=-1)

        expected_diagonal = math.exp(1.0) / (math.exp(1.0) + 3.0)
        diagonal = probability.diagonal(dim1=-2, dim2=-1)
        off_diagonal = probability[..., 0, 1]
        self.assertTrue(torch.allclose(
            diagonal, torch.full_like(diagonal, expected_diagonal),
            atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.all(diagonal > off_diagonal.unsqueeze(-1)))
        self.assertTrue(torch.allclose(
            probability.sum(dim=-1), torch.ones_like(probability.sum(dim=-1))))

        (-diagonal.mean()).backward()
        self.assertGreater(float(prior.raw_margin.grad.abs().sum()), 0.0)

    def test_residual_scale_is_positive_bounded_and_migratable(self):
        module = ReliabilityGuidedBidirectionalCrossAttention(
            32, residual_mode='bounded_scalar')
        with torch.no_grad():
            module.raw_residual_scale.fill_(-100.0)
        scale = module.get_residual_scale()
        self.assertGreaterEqual(
            float(scale.min()), module.residual_min - 1e-7)
        self.assertLessEqual(
            float(scale.max()), module.residual_max + 1e-7)

        old_state = {
            'residual_scale': torch.full((2, 32, 1, 1), 0.04),
        }
        self.assertEqual(
            migrate_rgca_residual_scale_state_dict(old_state, module), 1)
        module.load_state_dict({
            'raw_residual_scale': old_state['raw_residual_scale'],
        }, strict=False)
        self.assertTrue(torch.allclose(
            module.get_residual_scale(),
            torch.full((2, 1, 1, 1), 0.04), atol=1e-6, rtol=1e-6))

    def test_primary_update_depends_on_complementary_modality(self):
        torch.manual_seed(12)
        module = ReliabilityGuidedBidirectionalCrossAttention(32).eval()
        rgb = torch.randn(1, 32, 8, 10)
        ir = torch.randn(1, 32, 8, 10)
        with torch.no_grad():
            delta_a, _ = module(rgb, ir)
            delta_b, _ = module(rgb, ir + 0.5 * torch.randn_like(ir))
        self.assertGreater(float((delta_a - delta_b).abs().mean()), 1e-6)

    def test_rgca_fusion_keeps_mask_gfb_and_has_no_frequency_state(self):
        torch.manual_seed(13)
        module = HAFFormerRGCAMaskGFB(32).train()
        enable_cross_modal_attention_tracking(module)
        enable_fusion_gate_tracking(module)
        output = module([
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ])
        output.mean().backward()
        self.assertEqual(output.shape, (2, 32, 8, 10))
        self.assertEqual(len(collect_cross_modal_attention_diagnostics(module)), 1)
        self.assertEqual(len(collect_fusion_gate_diagnostics(module)), 1)
        state_names = tuple(module.state_dict())
        self.assertFalse(any('complex_weight' in name for name in state_names))
        self.assertFalse(any(name.endswith('.beta') for name in state_names))
        self.assertFalse(any('lambda_sym' in name for name in state_names))
        self.assertFalse(any('sink' in name for name in state_names))

    def test_default_model_contains_exactly_four_rgca_blocks(self):
        model = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD.yaml', ch=3, nc=6)
        blocks = [
            layer for layer in model.model
            if isinstance(layer, HAFFormerRGCAMaskGFB)
        ]
        self.assertEqual(len(blocks), 4)
        attentions = [block.cross_modal_attention for block in blocks]
        self.assertEqual([module.prior_mode for module in attentions],
                         ['none'] * 4)
        self.assertTrue(all(module.channel_prior is None
                            for module in attentions))
        self.assertEqual(
            [tuple(module.residual_scale.shape) for module in attentions],
            [(2, 64, 1, 1), (2, 128, 1, 1),
             (2, 256, 1, 1), (2, 512, 1, 1)])
        self.assertTrue(all(not hasattr(module, 'raw_residual_scale')
                            for module in attentions))
        self.assertFalse(any(
            '.channel_prior.' in name for name, _ in model.named_parameters()))
        self.assertEqual(sum(parameter.numel()
                             for parameter in model.parameters()), 13743436)


if __name__ == '__main__':
    unittest.main()
