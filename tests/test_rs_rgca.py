import unittest

import torch

from models.common import ReliabilityGuidedBidirectionalCrossAttention
from train import compute_rs_rgca_reliability_loss


class RSRGCATests(unittest.TestCase):
    def test_auxiliary_hook_preserves_forward_and_detaches_inputs(self):
        torch.manual_seed(17)
        module = ReliabilityGuidedBidirectionalCrossAttention(
            32, reduction=2, num_heads=4).train()
        rgb = torch.randn(2, 32, 8, 8, requires_grad=True)
        ir = torch.randn(2, 32, 8, 8, requires_grad=True)

        baseline = module(rgb, ir)
        module.enable_rs_reliability_supervision = True
        supervised = module(rgb, ir)
        self.assertTrue(torch.equal(baseline[0], supervised[0]))
        self.assertTrue(torch.equal(baseline[1], supervised[1]))

        prediction = module.last_rs_reliability_predictions
        auxiliary = (
            prediction['r_ir_to_rgb'].mean()
            + prediction['r_rgb_to_ir'].mean())
        auxiliary.backward()
        input_gradient = sum(
            0.0 if value.grad is None else float(value.grad.abs().sum())
            for value in (rgb, ir))
        gate_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.reliability_gate.parameters()
            if parameter.grad is not None)
        self.assertEqual(input_gradient, 0.0)
        self.assertGreater(gate_gradient, 0.0)

    def test_directional_loss_uses_source_modality_quality(self):
        class PredictionHolder(torch.nn.Module):
            pass

        target = torch.tensor([[1.0, 0.2, 1.0]])
        hyp = {'rs_reliability_ranking_gain': 0.25}

        aligned = PredictionHolder()
        aligned.last_rs_reliability_predictions = {
            'r_ir_to_rgb': torch.tensor([[0.2]], requires_grad=True),
            'r_rgb_to_ir': torch.tensor([[1.0]], requires_grad=True),
        }
        aligned_loss, _, _ = compute_rs_rgca_reliability_loss(
            aligned, target, hyp)

        swapped = PredictionHolder()
        swapped.last_rs_reliability_predictions = {
            'r_ir_to_rgb': torch.tensor([[1.0]], requires_grad=True),
            'r_rgb_to_ir': torch.tensor([[0.2]], requires_grad=True),
        }
        swapped_loss, _, _ = compute_rs_rgca_reliability_loss(
            swapped, target, hyp)

        self.assertEqual(float(aligned_loss), 0.0)
        self.assertGreater(float(swapped_loss), float(aligned_loss))


if __name__ == '__main__':
    unittest.main()
