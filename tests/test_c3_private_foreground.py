import unittest

import torch

from models.common import ModalityPrivateLowRankInject, ModalityPrivateLowRankStem
from models.yolo_test import Model
from train import TRAINING_CONFIG
from utils.private_foreground import (
    build_private_foreground_heatmap,
    compute_private_foreground_supervision,
)


C2_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C2_PrivateLowRank.yaml'
C3_CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_C3_GTPrivateForeground.yaml'


class C3PrivateForegroundTests(unittest.TestCase):
    def test_structure_budget_and_training_only_heads(self):
        model = Model(C3_CFG, ch=3, nc=6).float()
        self.assertEqual(model.yaml['architecture_variant'],
                         'c3_c2_gt_private_foreground')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8_075_330)
        injectors = [m for m in model.modules()
                     if isinstance(m, ModalityPrivateLowRankInject)]
        self.assertEqual(len(injectors), 2)
        self.assertTrue(all(m.foreground_head is not None for m in injectors))
        self.assertTrue(all(m.foreground_head.groups == 2 for m in injectors))
        self.assertEqual(sum(m.foreground_head.weight.numel()
                             + m.foreground_head.bias.numel()
                             for m in injectors), 68)
        self.assertTrue(TRAINING_CONFIG['model_cfg'].endswith(
            'yolov5s_LCAFNet_LLVIP_C0_FullDual.yaml'))

        inputs = torch.randn(1, 3, 64, 64)
        model.train()
        with torch.no_grad():
            model(inputs, inputs)
        self.assertEqual([tuple(m.last_private_foreground_logits.shape)
                          for m in injectors], [(1, 2, 4, 4), (1, 2, 2, 2)])
        model.eval()
        with torch.no_grad():
            model(inputs, inputs)
        self.assertTrue(all(m.last_private_foreground_logits is None
                            for m in injectors))

    def test_heatmap_and_auxiliary_gradients(self):
        targets = torch.tensor([
            [0, 1, 0.5, 0.5, 0.25, 0.25],
            [1, 2, 0.2, 0.7, 0.10, 0.15]], dtype=torch.float32)
        heatmap = build_private_foreground_heatmap(targets, 2, 8, 8)
        self.assertEqual(tuple(heatmap.shape), (2, 8, 8))
        self.assertEqual(float(heatmap[0].max()), 1.0)
        self.assertGreater(float(heatmap.sum()), 2.0)

        model = Model(C3_CFG, ch=3, nc=6).float().train()
        inputs = torch.randn(2, 3, 64, 64)
        model(inputs, inputs)
        loss, rows = compute_private_foreground_supervision(
            model, targets, 2, (0.03, 0.02), 1.0, collect_stats=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(rows), 2)
        loss.backward()
        injectors = [m for m in model.modules()
                     if isinstance(m, ModalityPrivateLowRankInject)]
        self.assertTrue(all(m.foreground_head.weight.grad.abs().sum() > 0
                            for m in injectors))
        stem = next(m for m in model.modules()
                    if isinstance(m, ModalityPrivateLowRankStem))
        self.assertGreater(stem.reduce.conv.weight.grad.abs().sum(), 0)
        self.assertTrue(all(m.last_private_foreground_logits is None
                            for m in injectors))

    def test_c2_to_c3_transfer_preserves_inference_path_exactly(self):
        torch.manual_seed(17)
        c2 = Model(C2_CFG, ch=3, nc=6).float().eval()
        torch.manual_seed(17)
        c3 = Model(C3_CFG, ch=3, nc=6).float().eval()
        common = {key: value for key, value in c3.state_dict().items()
                  if '.foreground_head.' not in key}
        self.assertTrue(all(torch.equal(value, c2.state_dict()[key])
                            for key, value in common.items()))
        c3.load_state_dict(c2.state_dict(), strict=False)
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            c2_output = c2(inputs, inputs)[0]
            c3_output = c3(inputs, inputs)[0]
        self.assertTrue(torch.equal(c2_output, c3_output))


if __name__ == '__main__':
    unittest.main()
