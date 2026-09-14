import unittest

import torch
import yaml

from models.common import (
    ModalityPrivateLowRankDownsample,
    ModalityPrivateLowRankInject,
    ModalityPrivateLowRankStem,
)
from models.yolo_test import Model
from train import (
    build_dual_stream_backbone_state_dict,
)
from utils.torch_utils import torch_load


CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C2_PrivateLowRank.yaml'
WEIGHTS = 'yolov5s.pt'


class C2PrivateLowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(CFG, ch=3, nc=6).float()

    def test_structure_parameter_budget_and_archived_config(self):
        self.assertEqual(
            self.model.yaml.get('architecture_variant'),
            'c2_c1_private_low_rank_c4_c5')
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 8_075_262)
        stems = [
            module for module in self.model.modules()
            if isinstance(module, ModalityPrivateLowRankStem)]
        injectors = [
            module for module in self.model.modules()
            if isinstance(module, ModalityPrivateLowRankInject)]
        downsamplers = [
            module for module in self.model.modules()
            if isinstance(module, ModalityPrivateLowRankDownsample)]
        self.assertEqual(len(stems), 1)
        self.assertEqual(len(injectors), 2)
        self.assertEqual(len(downsamplers), 1)
        self.assertEqual(stems[0].rank, 16)
        self.assertEqual([module.rank for module in injectors], [16, 16])
        self.assertEqual(self.model.model[-1].f, [32, 35, 38, 41])
        self.assertEqual(stems[0].reduce.conv.groups, 2)
        for module in injectors:
            self.assertEqual(module.expand.groups, 2)
            self.assertEqual(module.gate_score.groups, 2)
            self.assertAlmostEqual(float(module.private_scale), 0.1, places=6)

    def test_centralized_rank_and_scale_override_yaml(self):
        with open(CFG) as stream:
            definition = yaml.safe_load(stream)
        for layer in definition['backbone'] + definition['head']:
            if layer[2] == 'ModalityPrivateLowRankStem':
                layer[3][0] = 8
            elif layer[2] == 'ModalityPrivateLowRankInject':
                layer[3][0] = 0.05
        definition['private_bypass_rank'] = 8
        definition['private_bypass_initial_scale'] = 0.05
        self.assertEqual(definition['private_bypass_rank'], 8)
        self.assertAlmostEqual(definition['private_bypass_initial_scale'], 0.05)
        model = Model(definition, ch=3, nc=6)
        stems = [
            module for module in model.modules()
            if isinstance(module, ModalityPrivateLowRankStem)]
        injectors = [
            module for module in model.modules()
            if isinstance(module, ModalityPrivateLowRankInject)]
        self.assertEqual(stems[0].rank, 8)
        self.assertEqual([module.rank for module in injectors], [8, 8])
        self.assertTrue(all(
            abs(float(module.private_scale) - 0.05) < 1e-6
            for module in injectors))

    def test_four_detection_scales_have_expected_shapes(self):
        self.model.train()
        with torch.no_grad():
            outputs = self.model(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 3, 128, 128))
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 3, 32, 32, 11), (1, 3, 16, 16, 11),
             (1, 3, 8, 8, 11), (1, 3, 4, 4, 11)])

    def test_private_paths_receive_gradients_and_emit_diagnostics(self):
        stem = ModalityPrivateLowRankStem(32, rank=8).train()
        inject_c4 = ModalityPrivateLowRankInject(64, 16).train()
        downsample = ModalityPrivateLowRankDownsample(16).train()
        inject_c5 = ModalityPrivateLowRankInject(128, 16).train()
        inject_c4.track_private_bypass_stats = True
        inject_c5.track_private_bypass_stats = True
        private_c4 = stem([
            torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16)])
        output_c4 = inject_c4([torch.randn(2, 64, 8, 8), private_c4])
        private_c5 = downsample(private_c4)
        output_c5 = inject_c5([torch.randn(2, 128, 4, 4), private_c5])
        (output_c4.square().mean() + output_c5.square().mean()).backward()
        self.assertEqual(tuple(output_c4.shape), (2, 64, 8, 8))
        self.assertEqual(tuple(output_c5.shape), (2, 128, 4, 4))
        self.assertEqual(len(inject_c4.last_private_bypass_stats), 10)
        self.assertEqual(len(inject_c5.last_private_bypass_stats), 10)
        reduce_gradient = stem.reduce.conv.weight.grad.reshape(2, -1)
        self.assertTrue(torch.all(reduce_gradient.abs().sum(1) > 0))
        self.assertGreater(inject_c4.gate_score.weight.grad.abs().sum(), 0)
        self.assertGreater(inject_c5.gate_score.weight.grad.abs().sum(), 0)
        self.assertGreater(inject_c4.private_scale_logit.grad.abs(), 0)
        self.assertGreater(inject_c5.private_scale_logit.grad.abs(), 0)

    def test_grouped_stem_has_no_cross_modal_connection(self):
        stem = ModalityPrivateLowRankStem(8, rank=4).eval()
        with torch.no_grad():
            packed = stem([
                torch.randn(1, 8, 16, 16), torch.zeros(1, 8, 16, 16)])
        rgb_state, ir_state = packed.chunk(2, dim=1)
        self.assertGreater(rgb_state.abs().sum(), 0)
        self.assertEqual(float(ir_state.abs().sum()), 0.0)

    def test_classic_yolov5s_initializes_every_unchanged_backbone_tensor(self):
        checkpoint = torch_load(WEIGHTS, map_location='cpu')
        transfer, report = build_dual_stream_backbone_state_dict(
            checkpoint['model'].float().state_dict(), self.model.state_dict(),
            self.model.yaml['pretrained_backbone_map'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertGreater(report['shared_items'], 0)
        self.assertFalse(any(
            key.startswith(('model.14.', 'model.15.', 'model.19.', 'model.20.'))
            for key in transfer))


if __name__ == '__main__':
    unittest.main()
