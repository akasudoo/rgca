import unittest

import torch

from models.common import HAFFormerRGCAMaskGFB
from models.yolo_test import Model
from train import (
    build_dual_stream_backbone_state_dict,
    build_full_dual_rgca_to_c1_5_state_dict,
)
from utils.torch_utils import torch_load


CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C1_5_SharedC5.yaml'
WEIGHTS = 'yolov5s.pt'


class C15SharedC5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(CFG, ch=3, nc=6).float()
        checkpoint = torch_load(WEIGHTS, map_location='cpu')
        cls.source_state = checkpoint['model'].float().state_dict()

    def test_structure_parameter_budget_and_detection_scales(self):
        self.assertEqual(
            self.model.yaml['architecture_variant'],
            'c1_5_dual_to_c4_shared_c5')
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            9_322_562)
        self.assertEqual(sum(
            isinstance(module, HAFFormerRGCAMaskGFB)
            for module in self.model.modules()), 3)
        self.assertEqual(self.model.model[7].f, -4)
        self.assertEqual(self.model.model[14].f, [2, 9])
        self.assertEqual(self.model.model[15].f, [4, 11])
        self.assertEqual(self.model.model[16].f, [6, 13])
        self.assertEqual(self.model.model[-1].f, [31, 34, 37, 40])

        self.model.train()
        with torch.no_grad():
            outputs = self.model(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 3, 128, 128))
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 3, 32, 32, 11), (1, 3, 16, 16, 11),
             (1, 3, 8, 8, 11), (1, 3, 4, 4, 11)])

    def test_canonical_checkpoint_initializes_both_c4_branches_and_shared_c5(self):
        transfer, report = build_dual_stream_backbone_state_dict(
            self.source_state, self.model.state_dict(),
            self.model.yaml['pretrained_backbone_map'])
        self.assertEqual(report['format'], 'focus_spp')
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], 348)
        self.assertEqual(report['rgb_items'], 150)
        self.assertEqual(report['ir_items'], 150)
        self.assertEqual(report['shared_items'], 48)

        # The two independent C4 branches begin identically, while C5 is
        # represented only once after P4 fusion.
        for rgb_layer, ir_layer in zip(range(7), range(7, 14)):
            rgb_prefix = f'model.{rgb_layer}.'
            for rgb_key, rgb_value in transfer.items():
                if rgb_key.startswith(rgb_prefix):
                    suffix = rgb_key[len(rgb_prefix):]
                    self.assertTrue(torch.equal(
                        rgb_value, transfer[f'model.{ir_layer}.{suffix}']))
        self.assertTrue(any(key.startswith('model.17.') for key in transfer))
        self.assertTrue(any(key.startswith('model.19.') for key in transfer))

    def test_gradients_reach_modal_c4_p4_fusion_shared_c5_and_detect(self):
        model = Model(CFG, ch=3, nc=6).float().train()
        rgb = torch.randn(1, 3, 128, 128)
        ir = torch.randn(1, 3, 128, 128)
        loss = sum(output.float().square().mean()
                   for output in model(rgb, ir))
        loss.backward()
        parameters = dict(model.named_parameters())
        names = (
            'model.6.cv1.conv.weight',
            'model.13.cv1.conv.weight',
            'model.16.cross_modal_attention.qkv.weight',
            'model.17.conv.weight',
            'model.41.m.0.weight',
        )
        for name in names:
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)


class C0ToC15MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = Model(CFG, ch=3, nc=6).float()
        checkpoint = torch_load(
            'runs/train/exp_rgca_mask_gfb_uniform_lr/weights/best.pt',
            map_location='cpu')
        cls.source = checkpoint['model'].float().state_dict()

    def test_all_target_states_are_initialized_without_random_tensors(self):
        target_state = self.target.state_dict()
        transfer, report = build_full_dual_rgca_to_c1_5_state_dict(
            self.source, target_state)
        self.assertEqual(report['target_items'], 676)
        self.assertEqual(report['expected_target_items'], 676)
        self.assertEqual(report['rgb_items'], 150)
        self.assertEqual(report['ir_items'], 150)
        self.assertEqual(report['fusion_items'], 102)
        self.assertEqual(report['shared_c5_items'], 48)
        self.assertEqual(report['head_items'], 226)
        self.assertEqual(report['missing'], [])
        self.assertEqual(set(transfer), set(target_state))

    def test_exact_copies_and_c5_average_follow_the_declared_mapping(self):
        transfer, _ = build_full_dual_rgca_to_c1_5_state_dict(
            self.source, self.target.state_dict())
        self.assertTrue(torch.equal(
            transfer['model.6.cv1.conv.weight'],
            self.source['model.6.cv1.conv.weight']))
        self.assertTrue(torch.equal(
            transfer['model.13.cv1.conv.weight'],
            self.source['model.16.cv1.conv.weight']))
        self.assertTrue(torch.equal(
            transfer['model.16.cross_modal_attention.qkv.weight'],
            self.source['model.22.cross_modal_attention.qkv.weight']))
        self.assertTrue(torch.equal(
            transfer['model.17.conv.weight'],
            (self.source['model.7.conv.weight']
             + self.source['model.17.conv.weight']) * 0.5))
        self.assertTrue(torch.equal(
            transfer['model.41.m.0.weight'],
            self.source['model.45.m.0.weight']))


if __name__ == '__main__':
    unittest.main()
