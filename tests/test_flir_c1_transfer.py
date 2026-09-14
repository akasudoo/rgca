import unittest

import torch

from models.yolo_test import Model
from train import build_full_dual_lcafnet_to_c1_state_dict
from utils.torch_utils import torch_load


SOURCE = 'LCAFNet_LLVIP.pt'
TARGET = 'models/transformer/yolov5s_LCAFNet_LLVIP_C1_SharedC4C5.yaml'


class LLVIPC1TransferTests(unittest.TestCase):
    def test_strict_structural_migration(self):
        checkpoint = torch_load(SOURCE, map_location='cpu')
        source = checkpoint['model'].float().state_dict()
        model = Model(TARGET, ch=3, nc=1).float()
        initial = model.state_dict()

        transferred, report = build_full_dual_lcafnet_to_c1_state_dict(
            source, initial)

        self.assertEqual(report['target_items'], 518)
        self.assertEqual(report['rgb_items'], 90)
        self.assertEqual(report['ir_items'], 90)
        self.assertEqual(report['shared_items'], 108)
        self.assertEqual(report['head_items'], 226)
        self.assertEqual(report['gfb_items'], 4)
        self.assertEqual(len(report['fresh_fusion_items']), 64)
        self.assertFalse(report['unexpected_missing'])

        self.assertTrue(torch.equal(
            transferred['model.0.conv.weight'], source['model.0.conv.weight']))
        self.assertTrue(torch.equal(
            transferred['model.5.conv.weight'], source['model.10.conv.weight']))
        self.assertTrue(torch.equal(
            transferred['model.12.conv.weight'],
            (source['model.5.conv.weight']
             + source['model.15.conv.weight']) * 0.5))
        self.assertTrue(torch.equal(
            transferred['model.38.m.0.weight'], source['model.45.m.0.weight']))
        self.assertTrue(torch.equal(
            transferred['model.10.fusion.conv.0.weight'],
            source['model.20.conv.0.weight']))

        fresh_key = 'model.10.cross_modal_attention.qkv.weight'
        self.assertNotIn(fresh_key, transferred)
        fresh_value = initial[fresh_key].clone()
        model.load_state_dict(transferred, strict=False)
        self.assertTrue(torch.equal(model.state_dict()[fresh_key], fresh_value))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8_030_332)


if __name__ == '__main__':
    unittest.main()
