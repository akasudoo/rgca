import unittest

import torch

from models.common import C3, D3, D3C3
from models.yolo_test import Model
from train import (
    NEW_D3_INTERNAL_PREFIXES,
    _configured_model_definition,
    build_c1_to_deep_p4p5_state_dict,
    build_dual_stream_backbone_state_dict,
    parse_opt,
)
from utils.torch_utils import torch_load


C1_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C1_SharedC4C5.yaml'
C4_DEEP_CFG = (
    'models/transformer/yolov5s_LCAFNet_M3FD_C4_DeepP4P5D3C3.yaml')
C1_BEST = (
    'runs/train/'
    'exp_rgca_mask_gfb_c1_dual_to_c3_shared_c4c5_joint200_yolov5s_m3fd/'
    'weights/best.pt')


class C4DeepP4P5D3C3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        cls.c1 = Model(C1_CFG, ch=3, nc=6).float()
        torch.manual_seed(23)
        cls.c4 = Model(C4_DEEP_CFG, ch=3, nc=6).float()

    def test_budget_and_exact_replacement_scope(self):
        self.assertEqual(
            self.c4.yaml['architecture_variant'],
            'c4_c1_deep_p4p5_d3c3_pan')
        c1_params = sum(p.numel() for p in self.c1.parameters())
        c4_params = sum(p.numel() for p in self.c4.parameters())
        self.assertEqual(c1_params, 8_045_752)
        self.assertEqual(c4_params, 7_418_424)
        self.assertEqual(c1_params - c4_params, 627_328)
        self.assertEqual(
            [i for i, module in enumerate(self.c4.model)
             if isinstance(module, D3C3)], [34, 37])
        self.assertEqual(sum(isinstance(m, D3) for m in self.c4.modules()), 2)
        self.assertTrue(all(isinstance(self.c4.model[i], C3)
                            for i in [20, 24, 28, 31]))
        self.assertEqual(self.c4.model[-1].f, [28, 31, 34, 37])

    def test_c1_is_unchanged_through_p3_output(self):
        c1_state, c4_state = self.c1.state_dict(), self.c4.state_dict()
        unchanged_keys = [
            key for key in c1_state
            if key.startswith(tuple(f'model.{i}.' for i in range(34)))]
        self.assertTrue(unchanged_keys)
        self.assertTrue(all(torch.equal(c1_state[key], c4_state[key])
                            for key in unchanged_keys))
        self.assertTrue(all(isinstance(self.c4.model[i], type(self.c1.model[i]))
                            for i in range(34)))

    def test_four_detection_scales_have_expected_shapes(self):
        self.c4.train()
        with torch.no_grad():
            outputs = self.c4(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 3, 128, 128))
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 3, 32, 32, 11), (1, 3, 16, 16, 11),
             (1, 3, 8, 8, 11), (1, 3, 4, 4, 11)])

    def test_central_kernel_override_changes_only_two_blocks(self):
        opt = parse_opt([
            '--cfg', C4_DEEP_CFG, '--d3-kernel-size', '5'])
        model = Model(_configured_model_definition(opt), ch=3, nc=6)
        kernels = [m.large_dw.conv.kernel_size for m in model.modules()
                   if isinstance(m, D3)]
        self.assertEqual(kernels, [(5, 5)] * 2)
        self.assertEqual(
            [i for i, module in enumerate(model.model)
             if isinstance(module, D3C3)], [34, 37])

    def test_classic_yolov5s_initializes_complete_c1_backbone(self):
        checkpoint = torch_load('yolov5s.pt', map_location='cpu')
        transfer, report = build_dual_stream_backbone_state_dict(
            checkpoint['model'].float().state_dict(), self.c4.state_dict(),
            self.c4.yaml['pretrained_backbone_map'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertFalse(any(key.startswith(('model.34.', 'model.37.'))
                             for key in transfer))

    def test_c1_best_transfer_excludes_only_new_d3_mixers(self):
        checkpoint = torch_load(C1_BEST, map_location='cpu')
        source = checkpoint['model'].float().state_dict()
        target = self.c4.state_dict()
        transfer, report = build_c1_to_deep_p4p5_state_dict(source, target)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertEqual(report['target_items'], 558)
        self.assertEqual(report['excluded_items'], 60)
        self.assertFalse(any(key.startswith(NEW_D3_INTERNAL_PREFIXES)
                             for key in transfer))
        unchanged = [key for key in target
                     if not key.startswith(NEW_D3_INTERNAL_PREFIXES)]
        self.assertEqual(set(transfer), set(unchanged))
        self.assertTrue(all(torch.equal(transfer[key], source[key])
                            for key in unchanged))
        for layer in (34, 37):
            self.assertTrue(any(
                key.startswith(f'model.{layer}.cv1.') for key in transfer))
            self.assertTrue(any(
                key.startswith(f'model.{layer}.cv2.') for key in transfer))
            self.assertTrue(any(
                key.startswith(f'model.{layer}.cv3.') for key in transfer))


if __name__ == '__main__':
    unittest.main()
