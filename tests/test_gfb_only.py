import unittest

import torch

from models.common import GatedModalFusion, HAFFormerGFBOnly
from models.yolo_test import Model


class GFBOnlyTests(unittest.TestCase):
    def test_block_is_exact_original_gfb(self):
        torch.manual_seed(17)
        module = HAFFormerGFBOnly(32).eval()
        inputs = [
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ]
        with torch.no_grad():
            expected = module.fusion(inputs[0], inputs[1])
            actual = module(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertIsInstance(module.fusion, GatedModalFusion)
        self.assertFalse(hasattr(module, 'foreground_mask'))
        self.assertFalse(hasattr(module, 'mhca_rgb'))
        self.assertFalse(hasattr(module, 'reliability_gate'))

    def test_yaml_has_four_strict_gfb_only_blocks(self):
        model = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD_GFBOnly.yaml',
            ch=3, nc=6)
        blocks = [
            layer for layer in model.model
            if isinstance(layer, HAFFormerGFBOnly)
        ]
        self.assertEqual(model.yaml['architecture_variant'],
                         'c0_full_dual_gfb_only')
        self.assertEqual(len(blocks), 4)
        self.assertEqual(len(model.model), 46)


if __name__ == '__main__':
    unittest.main()
