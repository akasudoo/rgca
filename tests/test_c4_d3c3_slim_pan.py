import unittest

import torch

from models.common import (
    C3,
    D3,
    D3C3,
    ModalityPrivateLowRankInject,
    ModalityPrivateLowRankStem,
)
from models.yolo_test import Model
from train import (
    _configured_model_definition,
    build_dual_stream_backbone_state_dict,
    parse_opt,
)
from utils.torch_utils import torch_load


C1_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C1_SharedC4C5.yaml'
C4_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C4_D3C3SlimPAN.yaml'


class C4D3C3SlimPanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        cls.c1 = Model(C1_CFG, ch=3, nc=6).float()
        torch.manual_seed(11)
        cls.c4 = Model(C4_CFG, ch=3, nc=6).float()

    def test_structure_budget_and_controlled_scope(self):
        self.assertEqual(self.c4.yaml['architecture_variant'],
                         'c4_c1_d3c3_slim_pan')
        c1_params = sum(p.numel() for p in self.c1.parameters())
        c4_params = sum(p.numel() for p in self.c4.parameters())
        self.assertEqual(c1_params, 8_045_752)
        self.assertEqual(c4_params, 7_118_776)
        self.assertEqual(c1_params - c4_params, 926_976)
        self.assertEqual(sum(isinstance(m, D3C3) for m in self.c4.modules()), 6)
        self.assertEqual(sum(isinstance(m, D3) for m in self.c4.modules()), 6)
        self.assertFalse(any(isinstance(m, (ModalityPrivateLowRankStem,
                                            ModalityPrivateLowRankInject))
                             for m in self.c4.modules()))
        self.assertEqual(self.c4.model[-1].f, [28, 31, 34, 37])
        self.assertEqual([i for i, m in enumerate(self.c4.model)
                          if isinstance(m, D3C3)],
                         [20, 24, 28, 31, 34, 37])

    def test_c1_backbone_is_bit_exact_before_pan(self):
        c1_state, c4_state = self.c1.state_dict(), self.c4.state_dict()
        backbone_keys = [key for key in c1_state
                         if key.startswith(tuple(f'model.{i}.' for i in range(17)))]
        self.assertTrue(backbone_keys)
        self.assertTrue(all(torch.equal(c1_state[key], c4_state[key])
                            for key in backbone_keys))
        self.assertTrue(all(isinstance(self.c4.model[i], type(self.c1.model[i]))
                            for i in range(17)))

    def test_d3_operator_order_groups_and_gradients(self):
        block = D3(32, kernel_size=7).train()
        x = torch.randn(2, 32, 12, 12, requires_grad=True)
        output = block(x)
        output.square().mean().backward()
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertEqual(block.dsconv1[1].conv.groups, 32)
        self.assertEqual(block.large_dw.conv.groups, 32)
        self.assertEqual(block.large_dw.conv.kernel_size, (7, 7))
        self.assertEqual(block.dsconv2[1].conv.groups, 32)
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(x.grad.abs().sum(), 0)

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

    def test_central_kernel_override(self):
        opt = parse_opt(['--cfg', C4_CFG, '--d3-kernel-size', '5'])
        model = Model(_configured_model_definition(opt), ch=3, nc=6)
        kernels = [m.large_dw.conv.kernel_size for m in model.modules()
                   if isinstance(m, D3)]
        self.assertEqual(kernels, [(5, 5)] * 6)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7_100_344)

    def test_classic_yolov5s_initializes_complete_c1_backbone(self):
        checkpoint = torch_load('yolov5s.pt', map_location='cpu')
        transfer, report = build_dual_stream_backbone_state_dict(
            checkpoint['model'].float().state_dict(), self.c4.state_dict(),
            self.c4.yaml['pretrained_backbone_map'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertFalse(any(key.startswith(tuple(
            f'model.{i}.' for i in [20, 24, 28, 31, 34, 37]))
                             for key in transfer))


if __name__ == '__main__':
    unittest.main()
