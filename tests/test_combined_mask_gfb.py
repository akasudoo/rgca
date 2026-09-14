import unittest

import torch

from models.common import (
    HAFFormerFrequencyMaskGFB,
    HAFFormerMaskGFBOnly,
    HAFFormerRGCAMaskGFB,
)
from models.yolo_test import Model
from train import (
    collect_frequency_route_diagnostics,
    collect_fusion_gate_diagnostics,
    enable_frequency_route_tracking,
    enable_fusion_gate_tracking,
)


class CombinedMaskGFBTests(unittest.TestCase):
    def test_mask_gfb_only_neutral_mask_preserves_gfb(self):
        torch.manual_seed(3)
        module = HAFFormerMaskGFBOnly(32).eval()
        final_mask_conv = module.foreground_mask.mask_head[-2]
        with torch.no_grad():
            final_mask_conv.weight.zero_()
            final_mask_conv.bias.zero_()
        inputs = [torch.randn(2, 32, 8, 10), torch.randn(2, 32, 8, 10)]
        with torch.no_grad():
            expected = module.fusion(inputs[0], inputs[1])
            actual = module(inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-5))

    def test_neutral_mask_preserves_original_gfb(self):
        torch.manual_seed(4)
        module = HAFFormerFrequencyMaskGFB(32).eval()
        final_mask_conv = module.foreground_mask.mask_head[-2]
        with torch.no_grad():
            final_mask_conv.weight.zero_()
            final_mask_conv.bias.zero_()
        inputs = [torch.randn(2, 32, 8, 10), torch.randn(2, 32, 8, 10)]
        with torch.no_grad():
            rgb = module.mhca_rgb(inputs) + inputs[0]
            ir = module.mhca_ir([inputs[1], inputs[0]]) + inputs[1]
            expected = module.fusion(rgb, ir)
            actual = module(inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-5))

    def test_combined_block_has_nonconstant_routes_masks_and_gradients(self):
        torch.manual_seed(5)
        module = HAFFormerFrequencyMaskGFB(32).train()
        enable_frequency_route_tracking(module)
        enable_fusion_gate_tracking(module)
        output = module([
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ])
        output.square().mean().backward()

        routes = collect_frequency_route_diagnostics(module)
        fusion = collect_fusion_gate_diagnostics(module)
        self.assertEqual(len(routes), 2)
        self.assertEqual(len(fusion), 1)
        self.assertGreater(min(row[2] for row in routes), 0.0)
        self.assertGreater(fusion[0][2], 0.0)
        self.assertGreater(fusion[0][6], 0.0)
        self.assertGreater(
            float(module.foreground_mask.mask_head[-2].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.fusion.dwconv.weight.grad.abs().sum()), 0.0)

    def test_default_m3fd_yaml_uses_four_frequency_free_rgca_blocks(self):
        model = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD.yaml', ch=3, nc=6)
        blocks = [
            layer for layer in model.model
            if isinstance(layer, HAFFormerRGCAMaskGFB)
        ]
        self.assertEqual(len(blocks), 4)
        self.assertFalse(any(
            isinstance(layer, HAFFormerFrequencyMaskGFB)
            for layer in model.model))

    def test_mask_gfb_only_yaml_has_four_attention_free_blocks(self):
        model = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD_MaskGFBOnly.yaml',
            ch=3, nc=6)
        blocks = [
            layer for layer in model.model
            if isinstance(layer, HAFFormerMaskGFBOnly)
        ]
        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(not hasattr(block, 'cross_modal_attention')
                            for block in blocks))
        self.assertTrue(all(not hasattr(block, 'mhca_rgb') for block in blocks))


if __name__ == '__main__':
    unittest.main()
