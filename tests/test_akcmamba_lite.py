import contextlib
import io
import unittest

import torch

from models.common import (
    AdaptiveKernelConv2d,
    C3AKCMambaLite,
    HAFFormerRGCAMaskGFB,
    SelectiveAxisStateSpace2D,
)
from models.yolo_test import Model
from train import (
    build_dual_stream_backbone_state_dict,
    collect_akcmamba_lite_diagnostics,
    enable_akcmamba_lite_tracking,
)
from utils.torch_utils import torch_load


CFG = 'models/transformer/yolov5s_LCAFNet_M3FD_AKCMambaLite.yaml'


class AKCMambaLiteTests(unittest.TestCase):
    def test_adaptive_kernel_zero_start_shape_and_gradients(self):
        torch.manual_seed(21)
        module = AdaptiveKernelConv2d(16, 24, num_points=5).train()
        self.assertEqual(float(module.offset.weight.abs().sum()), 0.0)
        self.assertEqual(float(module.offset.bias.abs().sum()), 0.0)
        x = torch.randn(2, 16, 12, 14, requires_grad=True)
        output = module(x)
        self.assertEqual(output.shape, (2, 24, 12, 14))
        output.square().mean().backward()
        self.assertGreater(float(module.offset.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(module.offset.weight.grad).all())

    def test_selective_state_space_propagates_nonlocal_context(self):
        torch.manual_seed(22)
        module = SelectiveAxisStateSpace2D(16, state_ratio=0.5).eval()
        base = torch.zeros(1, 16, 9, 11)
        changed = base.clone()
        changed[:, :, 4, 1] = 1.0
        with torch.no_grad():
            output_base = module(base)
            output_changed = module(changed)
        # A perturbation at column 1 influences a distant location in the same
        # row through the bidirectional horizontal state recurrence.
        distant_change = (
            output_changed[:, :, 4, 9] - output_base[:, :, 4, 9]
        ).abs().mean()
        self.assertGreater(float(distant_change), 1e-8)

    def test_full_model_has_three_akc_neck_blocks_and_no_prior(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = Model(CFG, ch=3, nc=6).eval()
        neck = [module for module in model.modules()
                if isinstance(module, C3AKCMambaLite)]
        fusion = [module for module in model.modules()
                  if isinstance(module, HAFFormerRGCAMaskGFB)]
        self.assertEqual(len(neck), 3)
        self.assertEqual(len(fusion), 4)
        self.assertTrue(all(
            module.cross_modal_attention.channel_prior is None
            for module in fusion))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 13626973)
        enable_akcmamba_lite_tracking(model)
        with torch.no_grad():
            prediction = model(
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 3, 64, 64))
        self.assertEqual(prediction[0].shape, (1, 1020, 11))
        diagnostics = collect_akcmamba_lite_diagnostics(model)
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(all(len(row) == 11 for row in diagnostics))

    def test_classic_yolov5s_still_initializes_both_backbones(self):
        with contextlib.redirect_stdout(io.StringIO()):
            target = Model(CFG, ch=3, nc=6).float().eval()
        checkpoint = torch_load('yolov5s.pt', map_location='cpu')
        source = (checkpoint.get('ema') or checkpoint['model']).float()
        transfer, report = build_dual_stream_backbone_state_dict(
            source.state_dict(), target.state_dict())
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['rgb_items'], 198)
        self.assertEqual(report['ir_items'], 198)
        result = target.load_state_dict(transfer, strict=False)
        backbone_prefix = tuple(f'model.{index}.' for index in range(20))
        self.assertFalse(any(
            key.startswith(backbone_prefix) for key in result.missing_keys))


if __name__ == '__main__':
    unittest.main()
