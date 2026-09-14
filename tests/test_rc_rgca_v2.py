import unittest

import torch

from models.common import (
    HAFFormerRCRGCAV2MaskGFB,
    ReliabilityConditionedBidirectionalCrossAttentionV2,
)
from train import (
    apply_rc_rgca_v2_training_corruption,
    compute_rc_rgca_v2_reliability_loss,
)


class RCRGCAV2Test(unittest.TestCase):
    def test_null_attention_increases_when_source_reliability_falls(self):
        module = ReliabilityConditionedBidirectionalCrossAttentionV2(32)
        query = torch.randn(2, module.num_heads, module.head_dim, 25)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        spatial = torch.ones(2, 1, 5, 5)
        _, high = module._conditioned_attention(
            query, key, value, spatial, torch.full((2, 1), 0.95), 0)
        _, low = module._conditioned_attention(
            query, key, value, spatial, torch.full((2, 1), 0.05), 0)
        self.assertGreater(
            float(low[..., -1].mean()), float(high[..., -1].mean()))

    def test_reliability_loss_reaches_controller_and_null_key(self):
        torch.manual_seed(7)
        block = HAFFormerRCRGCAV2MaskGFB(32)
        block.train()
        rgb = torch.rand(4, 32, 16, 16)
        ir = torch.rand_like(rgb)
        output = block([rgb, ir])
        target = torch.tensor([
            [1.0, 1.0, 1.0],
            [0.1, 1.0, 1.0],
            [1.0, 0.1, 1.0],
            [1.0, 1.0, 0.1],
        ])
        loss = output.square().mean() + compute_rc_rgca_v2_reliability_loss(
            block, target, {
                'reliability_local_gain': 0.5,
                'reliability_ranking_gain': 0.25,
            })
        loss.backward()
        controller = block.cross_modal_attention
        self.assertGreater(
            float(controller.reliability_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(controller.raw_null_logit.grad.abs().sum()), 0.0)

    def test_corruption_returns_bounded_targets_and_preserves_shape(self):
        rgb = torch.rand(8, 3, 64, 64)
        ir = torch.rand_like(rgb)
        out_rgb, out_ir, target = apply_rc_rgca_v2_training_corruption(
            rgb, ir, {
                'reliability_corruption_probability': 1.0,
                'reliability_pair_corruption_probability': 1.0,
                'reliability_modality_dropout_probability': 0.2,
                'reliability_max_shift_fraction': 0.04,
            })
        self.assertEqual(out_rgb.shape, rgb.shape)
        self.assertEqual(out_ir.shape, ir.shape)
        self.assertEqual(tuple(target.shape), (8, 3))
        self.assertTrue(bool(((target >= 0.0) & (target <= 1.0)).all()))
        self.assertTrue(bool((target[:, 2] < 1.0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FP16')
    def test_half_precision_evaluation(self):
        block = HAFFormerRCRGCAV2MaskGFB(32).cuda().half().eval()
        rgb = torch.rand(2, 32, 16, 16, device='cuda').half()
        ir = torch.rand_like(rgb)
        with torch.no_grad():
            output = block([rgb, ir])
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertIsNone(
            block.cross_modal_attention.last_reliability_predictions)


if __name__ == '__main__':
    unittest.main()
