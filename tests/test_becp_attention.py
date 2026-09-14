import unittest

import torch

from models.common import (
    BayesianEvidentialCorrespondencePrior,
    HAFFormerBECPMaskGFB,
)
from models.yolo_test import Model
from train import (
    apply_becp_epoch_schedule,
    becp_evidence_loss_scale,
    becp_evidential_regularization,
    collect_becp_diagnostics,
    enable_cross_modal_attention_tracking,
    set_becp_prior_warmup,
)


class BECPCorrespondenceTests(unittest.TestCase):
    def test_zero_warmup_scale_is_exact_no_prior(self):
        torch.manual_seed(21)
        prior = BayesianEvidentialCorrespondencePrior(2, 4).eval()
        rgb = torch.randn(3, 2, 4, 4)
        ir = torch.randn(3, 2, 4, 4)
        out_rgb, out_ir = prior(rgb, ir)
        self.assertTrue(torch.equal(out_rgb, rgb.float()))
        self.assertTrue(torch.equal(out_ir, ir.float()))

    def test_evidence_loss_updates_tiny_head(self):
        torch.manual_seed(22)
        block = HAFFormerBECPMaskGFB(32).train()
        scale, count = set_becp_prior_warmup(block, epoch=10, warmup_epochs=10)
        self.assertEqual(scale, 1.0)
        self.assertEqual(count, 1)
        enable_cross_modal_attention_tracking(block)
        output = block([
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ])
        regularization, components = becp_evidential_regularization(
            block, torch.device('cpu'))
        (output.square().mean() + 0.02 * regularization).backward()
        prior = block.cross_modal_attention.channel_prior
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in prior.parameters()
            if parameter.grad is not None)
        self.assertGreater(gradient, 1e-5)
        self.assertGreater(components[0], 0.0)
        self.assertGreater(components[2], 0.0)
        rows = collect_becp_diagnostics(block)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][2], 0.0)  # gate std

    def test_fulltrain_feature_burnin_keeps_evidence_controller_fixed(self):
        torch.manual_seed(24)
        block = HAFFormerBECPMaskGFB(32).train()
        set_becp_prior_warmup(
            block, epoch=9, warmup_epochs=20, delay_epochs=20)
        output = block([
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ])
        regularization, _ = becp_evidential_regularization(
            block, torch.device('cpu'))
        loss_scale = becp_evidence_loss_scale(9, 10)
        (output.square().mean() + loss_scale * regularization).backward()
        prior = block.cross_modal_attention.channel_prior
        evidence_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in prior.parameters()
            if parameter.grad is not None)
        fusion_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in block.cross_modal_attention.qkv.parameters()
            if parameter.grad is not None)
        self.assertEqual(evidence_gradient, 0.0)
        self.assertGreater(fusion_gradient, 0.0)

    def test_group_a_keeps_controller_but_blocks_its_gradient_at_epoch_199(self):
        torch.manual_seed(25)
        block = HAFFormerBECPMaskGFB(32).train()
        prior_scale, count, loss_scale = apply_becp_epoch_schedule(
            block, epoch=199, loss_delay_epochs=10,
            prior_warmup_epochs=20, prior_delay_epochs=20,
            off_control=True)
        self.assertEqual((prior_scale, count, loss_scale), (0.0, 1, 0.0))
        output = block([
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 8, 10),
        ])
        regularization, _ = becp_evidential_regularization(
            block, torch.device('cpu'))
        (output.square().mean() + loss_scale * regularization).backward()
        prior = block.cross_modal_attention.channel_prior
        controller_parameters = sum(
            parameter.numel() for parameter in prior.parameters())
        controller_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in prior.parameters()
            if parameter.grad is not None)
        self.assertEqual(controller_parameters, 76)
        self.assertEqual(controller_gradient, 0.0)

    def test_log_odds_objective_separates_matched_and_mismatched_pairs(self):
        torch.manual_seed(23)
        prior = BayesianEvidentialCorrespondencePrior(2, 4).train()
        identity = torch.eye(4).view(1, 1, 4, 4)
        positive_rgb = 2.0 * identity.repeat(8, 2, 1, 1)
        positive_ir = positive_rgb.clone()
        negative_rgb = torch.roll(positive_rgb, shifts=1, dims=-2)
        negative_ir = negative_rgb.clone()
        optimizer = torch.optim.Adam(prior.parameters(), lr=0.03)

        initial_gap = None
        initial_rank = None
        final_gap = None
        final_rank = None
        for _ in range(60):
            prior(
                positive_rgb, positive_ir,
                negative_rgb, negative_ir)
            evidential, kl, ranking = prior.get_becp_loss_components()
            if initial_gap is None:
                initial_gap = prior._last_log_odds[2]
                initial_rank = float(ranking.detach())
            loss = evidential + 0.1 * kl + 0.5 * ranking
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_gap = prior._last_log_odds[2]
            final_rank = float(ranking.detach())

        self.assertGreater(final_gap, initial_gap + 0.2)
        self.assertLess(final_rank, initial_rank)

    def test_model_has_four_lightweight_becp_blocks(self):
        model = Model(
            'models/transformer/yolov5s_LCAFNet_M3FD_BECP.yaml', ch=3, nc=6)
        blocks = [
            module for module in model.modules()
            if isinstance(module, HAFFormerBECPMaskGFB)
        ]
        self.assertEqual(len(blocks), 4)
        evidence_parameters = sum(
            parameter.numel()
            for block in blocks
            for submodule in (
                block.cross_modal_attention.channel_prior.descriptor_norm,
                block.cross_modal_attention.channel_prior.evidence_head,
            )
            for parameter in submodule.parameters())
        self.assertEqual(evidence_parameters, 304)


if __name__ == '__main__':
    unittest.main()
