import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from models.common import HAFFormer
from models.yolo_test import Model
from train import (
    DEFAULT_ATTENTION_CONTROL_LR_RATIO,
    DEFAULT_BECP_CFG,
    DEFAULT_BECP_DIFFERENTIAL_LR,
    DEFAULT_BECP_EPOCHS,
    DEFAULT_BECP_EXPERIMENT_NAME,
    DEFAULT_BECP_HYP,
    DEFAULT_BECP_LOSS_DELAY_EPOCHS,
    DEFAULT_BECP_OFF_CONTROL,
    DEFAULT_BECP_PRETRAINED_FUSION_LR_RATIO,
    DEFAULT_BECP_PRETRAINED_LR_RATIO,
    DEFAULT_BECP_PRETRAINED_SCOPE,
    DEFAULT_BECP_PRIOR_ADAPT_EPOCHS,
    DEFAULT_BECP_PRIOR_DELAY_EPOCHS,
    DEFAULT_BECP_PRIOR_JOINT_LR_FACTOR,
    DEFAULT_BECP_PRIOR_LR_RATIO,
    DEFAULT_BECP_PRIOR_WARMUP_EPOCHS,
    DEFAULT_BECP_WEIGHTS,
    DEFAULT_TRAIN_CFG,
    DEFAULT_TRAIN_AFSS,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_CLOSE_MOSAIC,
    DEFAULT_TRAIN_CUDNN_BENCHMARK,
    DEFAULT_TRAIN_DIFFERENTIAL_LR,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_EXPERIMENT_NAME,
    DEFAULT_TRAIN_HYP,
    DEFAULT_TRAIN_IMAGE_SIZE,
    DEFAULT_TRAIN_OPTIMIZER,
    DEFAULT_TRAIN_PRETRAINED_SCOPE,
    DEFAULT_TRAIN_PRIOR_ADAPT_EPOCHS,
    DEFAULT_TRAIN_SEED,
    DEFAULT_TRAIN_WORKERS,
    DEFAULT_TRAIN_WEIGHTS,
    HYPERPARAMETERS,
    apply_becp_epoch_schedule,
    becp_evidence_loss_scale,
    build_dual_stream_backbone_state_dict,
    collect_frequency_route_diagnostics,
    compute_warmup_iterations,
    enable_frequency_route_tracking,
    is_attention_control_parameter,
    is_frequency_prior_parameter,
    set_becp_prior_warmup,
)
from train_route_fix import validate_route_diagnostics


class TrainingStrategyTests(unittest.TestCase):
    def test_default_entrypoint_is_llvip_c0_yolov5s_joint100(self):
        self.assertEqual(DEFAULT_TRAIN_EPOCHS, 100)
        self.assertEqual(DEFAULT_TRAIN_BATCH_SIZE, 8)
        self.assertEqual(DEFAULT_TRAIN_IMAGE_SIZE, (1024, 1024))
        self.assertEqual(DEFAULT_TRAIN_OPTIMIZER, 'SGD')
        self.assertEqual(DEFAULT_TRAIN_SEED, 1)
        self.assertEqual(DEFAULT_TRAIN_WORKERS, 8)
        self.assertFalse(DEFAULT_TRAIN_CUDNN_BENCHMARK)
        self.assertEqual(DEFAULT_TRAIN_CLOSE_MOSAIC, 10)
        self.assertFalse(DEFAULT_TRAIN_AFSS)
        self.assertEqual(DEFAULT_TRAIN_PRIOR_ADAPT_EPOCHS, 0)
        self.assertEqual(DEFAULT_TRAIN_PRETRAINED_SCOPE, 'backbone')
        self.assertFalse(DEFAULT_TRAIN_DIFFERENTIAL_LR)
        self.assertEqual(Path(DEFAULT_TRAIN_WEIGHTS).name, 'yolov5s.pt')
        self.assertTrue(Path(DEFAULT_TRAIN_WEIGHTS).is_file())
        self.assertEqual(
            Path(DEFAULT_TRAIN_CFG).name,
            'yolov5s_LCAFNet_LLVIP_C0_FullDual.yaml')
        self.assertTrue(Path(DEFAULT_TRAIN_CFG).is_file())
        self.assertEqual(
            Path(DEFAULT_TRAIN_HYP).name,
            'hyp.scratch.yaml')
        self.assertTrue(Path(DEFAULT_TRAIN_HYP).is_file())
        self.assertIn(
            'c0_joint100_yolov5s_llvip_1024_exp_mask_protocol',
            DEFAULT_TRAIN_EXPERIMENT_NAME)
        self.assertEqual(HYPERPARAMETERS['lr0'], 0.01)
        self.assertEqual(HYPERPARAMETERS['warmup_epochs'], 3.0)
        self.assertEqual(HYPERPARAMETERS['mosaic'], 1.0)

    def test_c0_llvip_hyperparameters_match_training_file(self):
        with open(DEFAULT_TRAIN_HYP) as stream:
            active_hyp = yaml.safe_load(stream)
        self.assertEqual(active_hyp, HYPERPARAMETERS)

    def test_c0_llvip_structure_and_canonical_backbone_transfer(self):
        model = Model(DEFAULT_TRAIN_CFG, ch=3, nc=1).float()
        self.assertEqual(model.yaml['architecture_variant'],
                         'c0_full_dual_rgca')
        checkpoint = torch.load(
            DEFAULT_TRAIN_WEIGHTS, map_location='cpu', weights_only=False)
        transfer, report = build_dual_stream_backbone_state_dict(
            checkpoint['model'].float().state_dict(), model.state_dict(),
            model.yaml['pretrained_backbone_map'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['target_items'], report['expected_target_items'])
        self.assertEqual(report['rgb_items'], report['ir_items'])
        self.assertGreater(report['rgb_items'], 0)
        self.assertEqual(report['shared_items'], 0)
        self.assertEqual(len(transfer), report['target_items'])

    def test_historical_group_d_constants_remain_reproducible(self):
        self.assertEqual(DEFAULT_BECP_EPOCHS, 50)
        self.assertFalse(DEFAULT_BECP_OFF_CONTROL)
        self.assertEqual(DEFAULT_BECP_PRETRAINED_SCOPE, 'all')
        self.assertTrue(DEFAULT_BECP_DIFFERENTIAL_LR)
        self.assertIn('group_D_becp_finetune50', DEFAULT_BECP_EXPERIMENT_NAME)
        self.assertTrue(Path(DEFAULT_BECP_WEIGHTS).is_file())
        self.assertEqual(Path(DEFAULT_BECP_WEIGHTS).name, 'best.pt')
        self.assertEqual(
            Path(DEFAULT_BECP_WEIGHTS).parent.parent.name,
            'exp_rgca_mask_gfb_uniform_lr')
        self.assertTrue(Path(DEFAULT_BECP_CFG).is_file())
        self.assertTrue(Path(DEFAULT_BECP_HYP).is_file())

    def test_finetune_differential_learning_rate_ratios(self):
        self.assertEqual(DEFAULT_ATTENTION_CONTROL_LR_RATIO, 1.0)
        self.assertEqual(DEFAULT_BECP_PRETRAINED_LR_RATIO, 0.1)
        self.assertEqual(DEFAULT_BECP_PRETRAINED_FUSION_LR_RATIO, 0.2)
        self.assertEqual(DEFAULT_BECP_PRIOR_LR_RATIO, 5.0)
        self.assertEqual(DEFAULT_BECP_PRIOR_JOINT_LR_FACTOR, 0.2)
        with open(DEFAULT_BECP_HYP) as stream:
            hyp = yaml.safe_load(stream)
        self.assertEqual(hyp['lr0'], 0.002)
        self.assertEqual(hyp['warmup_epochs'], 1.0)

    def test_group_d_finetune_calibrates_then_ramps_prior(self):
        module = torch.nn.Module()
        module.prior = torch.nn.Module()
        module.prior.set_prior_warmup_scale = lambda value: None
        self.assertEqual(DEFAULT_BECP_PRIOR_ADAPT_EPOCHS, 5)
        self.assertEqual(DEFAULT_BECP_LOSS_DELAY_EPOCHS, 0)
        self.assertEqual(
            apply_becp_epoch_schedule(
                module, epoch=0,
                loss_delay_epochs=DEFAULT_BECP_LOSS_DELAY_EPOCHS,
                prior_warmup_epochs=DEFAULT_BECP_PRIOR_WARMUP_EPOCHS,
                prior_delay_epochs=DEFAULT_BECP_PRIOR_DELAY_EPOCHS,
                off_control=DEFAULT_BECP_OFF_CONTROL),
            (0.0, 1, 1.0))
        self.assertEqual(
            apply_becp_epoch_schedule(
                module, epoch=4,
                loss_delay_epochs=DEFAULT_BECP_LOSS_DELAY_EPOCHS,
                prior_warmup_epochs=DEFAULT_BECP_PRIOR_WARMUP_EPOCHS,
                prior_delay_epochs=DEFAULT_BECP_PRIOR_DELAY_EPOCHS,
                off_control=DEFAULT_BECP_OFF_CONTROL)[::2],
            (0.0, 1.0))
        self.assertAlmostEqual(
            apply_becp_epoch_schedule(
                module, epoch=5,
                loss_delay_epochs=DEFAULT_BECP_LOSS_DELAY_EPOCHS,
                prior_warmup_epochs=DEFAULT_BECP_PRIOR_WARMUP_EPOCHS,
                prior_delay_epochs=DEFAULT_BECP_PRIOR_DELAY_EPOCHS,
                off_control=DEFAULT_BECP_OFF_CONTROL)[0],
            0.1)
        self.assertEqual(
            apply_becp_epoch_schedule(
                module, epoch=14,
                loss_delay_epochs=DEFAULT_BECP_LOSS_DELAY_EPOCHS,
                prior_warmup_epochs=DEFAULT_BECP_PRIOR_WARMUP_EPOCHS,
                prior_delay_epochs=DEFAULT_BECP_PRIOR_DELAY_EPOCHS,
                off_control=DEFAULT_BECP_OFF_CONTROL)[0],
            1.0)

    def test_fulltrain_becp_loss_and_prior_have_separate_stages(self):
        self.assertEqual(becp_evidence_loss_scale(9, 10), 0.0)
        self.assertEqual(becp_evidence_loss_scale(10, 10), 1.0)
        module = torch.nn.Module()
        module.prior = torch.nn.Module()
        module.prior.set_prior_warmup_scale = lambda value: None
        self.assertEqual(
            set_becp_prior_warmup(
                module, epoch=19, warmup_epochs=20, delay_epochs=20)[0],
            0.0)
        self.assertAlmostEqual(
            set_becp_prior_warmup(
                module, epoch=20, warmup_epochs=20, delay_epochs=20)[0],
            0.05)
        self.assertEqual(
            set_becp_prior_warmup(
                module, epoch=39, warmup_epochs=20, delay_epochs=20)[0],
            1.0)

    def test_group_a_forces_prior_and_evidence_loss_off_for_every_epoch(self):
        module = torch.nn.Module()
        module.prior = torch.nn.Module()
        scales = []
        module.prior.set_prior_warmup_scale = scales.append
        for epoch in (0, 9, 10, 19, 20, 39, 199, 10000):
            prior_scale, count, loss_scale = apply_becp_epoch_schedule(
                module, epoch=epoch, loss_delay_epochs=10,
                prior_warmup_epochs=20, prior_delay_epochs=20,
                off_control=True)
            self.assertEqual((prior_scale, count, loss_scale), (0.0, 1, 0.0))
        self.assertEqual(scales, [0.0] * 8)

    def test_becp_prior_is_delayed_then_ramped(self):
        module = torch.nn.Module()
        module.prior = torch.nn.Module()
        scales = []

        def set_scale(value):
            scales.append(value)

        module.prior.set_prior_warmup_scale = set_scale
        scale, count = set_becp_prior_warmup(
            module, epoch=4, warmup_epochs=10, delay_epochs=5)
        self.assertEqual((scale, count), (0.0, 1))
        scale, _ = set_becp_prior_warmup(
            module, epoch=5, warmup_epochs=10, delay_epochs=5)
        self.assertAlmostEqual(scale, 0.1)
        scale, _ = set_becp_prior_warmup(
            module, epoch=14, warmup_epochs=10, delay_epochs=5)
        self.assertEqual(scale, 1.0)

    def test_short_run_warmup_is_capped(self):
        warmup = compute_warmup_iterations(
            warmup_epochs=3.0,
            batches_per_epoch=210,
            min_iters=1000,
            remaining_epochs=5,
            max_fraction=0.2,
        )
        self.assertEqual(warmup, 210)

    def test_full_run_respects_epoch_warmup(self):
        warmup = compute_warmup_iterations(
            warmup_epochs=3.0,
            batches_per_epoch=210,
            min_iters=0,
            remaining_epochs=200,
            max_fraction=0.2,
        )
        self.assertEqual(warmup, 630)

    def test_resume_has_no_new_warmup(self):
        warmup = compute_warmup_iterations(
            warmup_epochs=3.0,
            batches_per_epoch=210,
            min_iters=0,
            remaining_epochs=20,
            max_fraction=0.2,
            resumed=True,
        )
        self.assertEqual(warmup, 0)

    def test_active_temperature_uses_no_decay_prior_group(self):
        self.assertTrue(
            is_frequency_prior_parameter('model.20.mhca_rgb.temperature'))

    def test_rgca_prior_and_bounded_controls_use_no_decay_group(self):
        names = (
            'model.20.cross_modal_attention.channel_prior.raw_margin',
            'model.20.cross_modal_attention.raw_residual_scale',
            'model.20.cross_modal_attention.cross_mix_logit',
            'model.20.cross_modal_attention.raw_temperature',
        )
        self.assertTrue(all(is_attention_control_parameter(name) for name in names))

    def test_selective_state_context_scale_uses_no_decay_group(self):
        self.assertTrue(is_attention_control_parameter(
            'model.38.state_space.context_scale'))

    def test_route_diagnostics_collect_real_forward_stats(self):
        torch.manual_seed(3)
        module = HAFFormer(32).eval()
        enable_frequency_route_tracking(module)
        with torch.no_grad():
            module([
                torch.randn(2, 32, 8, 12),
                torch.randn(2, 32, 8, 12),
            ])
        rows = collect_frequency_route_diagnostics(module)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[2] > 0.0 for row in rows))

    def test_two_stage_launcher_rejects_collapsed_route(self):
        with TemporaryDirectory() as temp_dir:
            diagnostics = Path(temp_dir) / 'route_diagnostics.csv'
            diagnostics.write_text(
                'epoch,module,confidence_mean,confidence_std,confidence_min,'
                'confidence_max,beta,temperature\n'
                '0,route,0.1,0,0.1,0.1,0.01,1.0\n')
            with self.assertRaises(RuntimeError):
                validate_route_diagnostics(
                    Path(temp_dir), expected_epochs=1, expected_branches=1)


if __name__ == '__main__':
    unittest.main()
