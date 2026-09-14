import unittest

import torch

from models.yolo_test import Model
from train import (
    build_ablation_model_config,
    build_dual_stream_backbone_state_dict,
    focus_weight_to_conv6,
)
from utils.torch_utils import torch_load


CFG = 'models/transformer/yolov5s_LCAFNet_M3FD.yaml'
BECP_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_BECP.yaml'
WEIGHTS = 'yolov5s.pt'


class StrictAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = torch_load(WEIGHTS, map_location='cpu')
        cls.source_model = checkpoint['model'].float().eval()
        cls.source_state = cls.source_model.state_dict()

    def build_model(self, variant):
        cfg = build_ablation_model_config(
            CFG, variant, deterministic_init=True, seed=7)
        return Model(cfg, ch=3, nc=6).float().eval()

    def test_old_focus_kernel_conversion_is_numerically_equivalent(self):
        source_stem = self.source_model.model[0]
        target = self.build_model('frequency_mask')
        transfer, _ = build_dual_stream_backbone_state_dict(
            self.source_state, target.state_dict())
        target.load_state_dict(transfer, strict=False)
        image = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected = source_stem(image)
            actual = target.model[0](image)
        self.assertTrue(torch.allclose(expected, actual, atol=6e-5, rtol=1e-4))

    def test_focus_converter_places_all_four_pixel_offsets(self):
        weight = torch.arange(4 * 3 * 3, dtype=torch.float32).view(1, 4, 3, 3)
        converted = focus_weight_to_conv6(weight)
        self.assertTrue(torch.equal(converted[0, 0, 0::2, 0::2], weight[0, 0]))
        self.assertTrue(torch.equal(converted[0, 0, 1::2, 0::2], weight[0, 1]))
        self.assertTrue(torch.equal(converted[0, 0, 0::2, 1::2], weight[0, 2]))
        self.assertTrue(torch.equal(converted[0, 0, 1::2, 1::2], weight[0, 3]))

    def test_old_weight_fully_initializes_both_modal_backbones(self):
        target = self.build_model('frequency_mask')
        transfer, report = build_dual_stream_backbone_state_dict(
            self.source_state, target.state_dict())
        self.assertEqual(report['format'], 'focus_spp')
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['rgb_items'], 198)
        self.assertEqual(report['ir_items'], 198)
        for rgb_key, value in transfer.items():
            parts = rgb_key.split('.', 2)
            if parts[0] == 'model' and parts[1].isdigit() and int(parts[1]) < 10:
                ir_key = f'model.{int(parts[1]) + 10}.{parts[2]}'
                self.assertTrue(torch.equal(value, transfer[ir_key]))

    def test_old_weight_initializes_becp_model_backbones_only(self):
        target = Model(BECP_CFG, ch=3, nc=6).float().eval()
        transfer, report = build_dual_stream_backbone_state_dict(
            self.source_state, target.state_dict())
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['rgb_items'], 198)
        self.assertEqual(report['ir_items'], 198)
        result = target.load_state_dict(transfer, strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(any(
            '.channel_prior.' in key for key in result.missing_keys))
        self.assertFalse(any(
            key.startswith(tuple(f'model.{index}.' for index in range(20)))
            for key in result.missing_keys))

    def test_all_variants_keep_identical_seeded_detection_head(self):
        variants = [
            'mean', 'original_gfb', 'frequency_gfb',
            'original_mask', 'frequency_mask']
        states = [self.build_model(variant).state_dict() for variant in variants]
        reference = {
            key: value for key, value in states[0].items()
            if key.startswith(tuple(f'model.{index}.' for index in range(24, 46)))
        }
        self.assertTrue(reference)
        for state in states[1:]:
            self.assertEqual(reference.keys(), {
                key for key in state
                if key.startswith(tuple(f'model.{index}.' for index in range(24, 46)))
            })
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, state[key]), key)

    def test_shared_gfb_and_mask_initializations_are_paired(self):
        original_gfb = self.build_model('original_gfb').state_dict()
        frequency_gfb = self.build_model('frequency_gfb').state_dict()
        original_mask = self.build_model('original_mask').state_dict()
        frequency_mask = self.build_model('frequency_mask').state_dict()

        gfb_keys = [key for key in original_gfb if '.fusion.' in key]
        mask_keys = [key for key in original_mask if '.foreground_mask.' in key]
        self.assertTrue(gfb_keys)
        self.assertTrue(mask_keys)
        for key in gfb_keys:
            self.assertTrue(torch.equal(original_gfb[key], frequency_gfb[key]), key)
        for key in mask_keys:
            self.assertTrue(torch.equal(original_mask[key], frequency_mask[key]), key)


if __name__ == '__main__':
    unittest.main()
