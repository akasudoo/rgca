import unittest

import torch

from models.common import (
    HAFFormerRGCAMaskGFB,
    HAFFormerRGCANoReliabilityMaskGFB,
)
from models.yolo_test import Model
from train import build_dual_stream_backbone_state_dict
from utils.torch_utils import torch_load


CFG = (
    'models/transformer/'
    'yolov5s_LCAFNet_M3FD_RGCANoReliability_MaskGFB.yaml'
)


class RGCANoReliabilityTests(unittest.TestCase):
    def test_identity_gate_has_no_reliability_parameters(self):
        torch.manual_seed(31)
        module = HAFFormerRGCANoReliabilityMaskGFB(32).train()
        attention = module.cross_modal_attention
        self.assertFalse(attention.use_reliability_gate)
        self.assertIsNone(attention.reliability_gate)
        self.assertFalse(any(
            'reliability_gate' in name for name in module.state_dict()))

        attention.export_reliability_maps = True
        rgb = torch.randn(2, 32, 8, 10)
        ir = torch.randn(2, 32, 8, 10)
        output = module([rgb, ir])
        output.mean().backward()
        maps = attention.last_reliability_maps
        self.assertTrue(torch.equal(
            maps['r_ir_to_rgb'], torch.ones_like(maps['r_ir_to_rgb'])))
        self.assertTrue(torch.equal(
            maps['r_rgb_to_ir'], torch.ones_like(maps['r_rgb_to_ir'])))
        self.assertGreater(float(attention.qkv.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(attention.local_value.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(
            module.foreground_mask.mask_head[0].weight.grad.abs().sum()), 0.0)

    def test_yaml_changes_only_the_rgca_wrapper_type(self):
        torch.manual_seed(123)
        model = Model(CFG, ch=3, nc=6)
        blocks = [
            layer for layer in model.model
            if type(layer) is HAFFormerRGCANoReliabilityMaskGFB
        ]
        self.assertEqual(
            model.yaml['architecture_variant'],
            'c0_full_dual_rgca_no_reliability_mask_gfb')
        self.assertEqual(len(blocks), 4)
        self.assertEqual(len(model.model), 46)
        self.assertTrue(all(
            block.cross_modal_attention.prior_mode == 'none'
            for block in blocks))
        self.assertFalse(any(
            '.reliability_gate.' in name
            for name, _ in model.named_parameters()))

        torch.manual_seed(123)
        full = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD.yaml', ch=3, nc=6)
        full_blocks = [
            layer for layer in full.model
            if type(layer) is HAFFormerRGCAMaskGFB
        ]
        removed = sum(
            parameter.numel()
            for block in full_blocks
            for parameter in block.cross_modal_attention.reliability_gate.parameters()
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in full.parameters())
            - sum(parameter.numel() for parameter in model.parameters()),
            removed)
        full_state = full.state_dict()
        ablation_state = model.state_dict()
        shared_keys = [
            key for key in full_state if '.reliability_gate.' not in key
        ]
        self.assertEqual(set(shared_keys), set(ablation_state))
        for key in shared_keys:
            self.assertTrue(
                torch.equal(full_state[key], ablation_state[key]), key)

    def test_canonical_yolov5s_initializes_both_backbones(self):
        model = Model(CFG, ch=3, nc=6).float()
        checkpoint = torch_load('yolov5s.pt', map_location='cpu')
        transferred, report = build_dual_stream_backbone_state_dict(
            checkpoint['model'].float().state_dict(), model.state_dict(),
            model.yaml['pretrained_backbone_map'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['rgb_items'], report['ir_items'])
        self.assertGreater(report['rgb_items'], 0)
        self.assertEqual(len(transferred), report['target_items'])


if __name__ == '__main__':
    unittest.main()
