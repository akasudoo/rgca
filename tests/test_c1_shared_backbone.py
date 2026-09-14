import unittest

import torch

from models.common import C3AKCMambaLite, HAFFormerRGCAMaskGFB
from models.yolo_test import Model
from train import build_dual_stream_backbone_state_dict
from utils.torch_utils import torch_load


CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C1_SharedC4C5.yaml'
WEIGHTS = 'yolov5s.pt'


class C1SharedBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(CFG, ch=3, nc=6).float()
        checkpoint = torch_load(WEIGHTS, map_location='cpu')
        cls.source_state = checkpoint['model'].float().state_dict()

    def test_structure_and_parameter_count(self):
        self.assertEqual(
            self.model.yaml.get('architecture_variant'),
            'c1_dual_to_c3_shared_c4_c5')
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 8_045_752)
        self.assertEqual(sum(
            isinstance(module, HAFFormerRGCAMaskGFB)
            for module in self.model.modules()), 2)
        self.assertFalse(any(
            isinstance(module, C3AKCMambaLite)
            for module in self.model.modules()))
        self.assertEqual(self.model.model[5].f, -4)
        self.assertEqual(self.model.model[-1].f, [28, 31, 34, 37])

    def test_four_detection_scales_have_expected_shapes(self):
        self.model.train()
        with torch.no_grad():
            outputs = self.model(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 3, 128, 128))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 3, 32, 32, 11), (1, 3, 16, 16, 11),
             (1, 3, 8, 8, 11), (1, 3, 4, 4, 11)])

    def test_classic_yolov5s_fully_initializes_mapped_backbone(self):
        transfer, report = build_dual_stream_backbone_state_dict(
            self.source_state, self.model.state_dict(),
            self.model.yaml['pretrained_backbone_map'])
        self.assertEqual(report['format'], 'focus_spp')
        self.assertEqual(report['layout'], 'explicit_map')
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertGreater(report['rgb_items'], 0)
        self.assertEqual(report['rgb_items'], report['ir_items'])
        self.assertGreater(report['shared_items'], 0)

        result = self.model.load_state_dict(transfer, strict=False)
        mapped_prefixes = tuple(
            f'model.{layer}.'
            for layer in self.model.yaml['pretrained_backbone_map'])
        self.assertFalse(any(
            key.startswith(mapped_prefixes) for key in result.missing_keys))

        for rgb_layer, ir_layer in zip(range(5), range(5, 10)):
            rgb_prefix = f'model.{rgb_layer}.'
            for rgb_key, rgb_value in transfer.items():
                if rgb_key.startswith(rgb_prefix):
                    suffix = rgb_key[len(rgb_prefix):]
                    self.assertTrue(torch.equal(
                        rgb_value, transfer[f'model.{ir_layer}.{suffix}']))


if __name__ == '__main__':
    unittest.main()
