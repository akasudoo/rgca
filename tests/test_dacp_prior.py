import torch

from models.common import DetectionAlignedForegroundBackgroundPrior
from train import (
    dacp_balanced_heatmap_loss,
    dacp_health_passes,
    dacp_objectness_regularization,
    dacp_parameter_allowed,
    set_dacp_prior_warmup,
)
from utils.general import keep_force_fp32_modules


def _attention_inputs(batch=2, heads=4, channels=8, height=8, width=6):
    tokens = height * width
    tensors = [
        torch.nn.functional.normalize(
            torch.randn(batch, heads, channels, tokens), dim=-1)
        for _ in range(4)
    ]
    logits = torch.randn(batch, heads, channels, channels)
    features = (
        torch.randn(batch, 16, height, width),
        torch.randn(batch, 16, height, width),
    )
    return logits, features, tensors


def test_zero_initialized_dacp_is_exact_logit_identity():
    prior = DetectionAlignedForegroundBackgroundPrior(4, 8)
    logits, (rgb, ir), (q_rgb, k_rgb, q_ir, k_ir) = _attention_inputs()
    mask = prior.predict_objectness(rgb, ir)
    out_rgb, out_ir = prior(
        logits, logits.clone(), q_rgb, k_rgb, q_ir, k_ir, mask)
    assert torch.equal(out_rgb, logits.float())
    assert torch.equal(out_ir, logits.float())


def test_nonzero_full_ramp_start_remains_bounded():
    prior = DetectionAlignedForegroundBackgroundPrior(
        4, 8, rho_max=0.1, rho_init=0.01)
    assert torch.equal(prior.get_effective_rho(), torch.zeros(2, 4))
    prior.set_dacp_warmup_scale(1.0)
    assert torch.allclose(
        prior.get_effective_rho(), torch.full((2, 4), 0.01), atol=1e-7)


def test_dacp_residual_is_bounded_by_ten_percent_content_rms():
    prior = DetectionAlignedForegroundBackgroundPrior(4, 8, rho_max=0.1)
    prior.set_dacp_warmup_scale(1.0)
    prior.raw_rho.data.fill_(4.0)
    prior.track_dacp_stats = True
    logits, (rgb, ir), (q_rgb, k_rgb, q_ir, k_ir) = _attention_inputs()
    mask = prior.predict_objectness(rgb, ir)
    prior(logits, logits.clone(), q_rgb, k_rgb, q_ir, k_ir, mask)
    assert prior.last_dacp_stats[8] <= 0.10001


def test_box_supervision_reaches_the_mask_head():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prior = DetectionAlignedForegroundBackgroundPrior(4, 8)

    model = Wrapper()
    rgb = torch.randn(2, 16, 10, 10)
    ir = torch.randn(2, 16, 10, 10)
    model.prior.predict_objectness(rgb, ir)
    targets = torch.tensor([
        [0.0, 0.0, 0.5, 0.5, 0.25, 0.3],
        [1.0, 1.0, 0.2, 0.3, 0.15, 0.2],
    ])
    loss, rows = dacp_objectness_regularization(
        model, targets, torch.device('cpu'))
    loss.backward()
    assert len(rows) == 1
    assert torch.isfinite(loss)
    assert model.prior.mask_head[0].weight.grad.abs().sum() > 0


def test_dacp_schedule_and_stage_gradient_policy():
    prior = DetectionAlignedForegroundBackgroundPrior(4, 8)
    assert [set_dacp_prior_warmup(prior, epoch, 10, 5)[0]
            for epoch in (0, 4, 5, 14, 15)] == [0.0, 0.0, 0.1, 1.0, 1.0]
    mask_name = 'model.20.cross_modal_attention.channel_prior.mask_head.0.weight'
    rho_name = 'model.20.cross_modal_attention.channel_prior.raw_rho'
    head_name = 'model.44.m.0.weight'
    assert dacp_parameter_allowed(mask_name, 0, 5, 15)
    assert not dacp_parameter_allowed(rho_name, 0, 5, 15)
    assert dacp_parameter_allowed(rho_name, 5, 5, 15)
    assert not dacp_parameter_allowed(head_name, 5, 5, 15)
    assert dacp_parameter_allowed(head_name, 15, 5, 15)


def test_checkpoint_half_conversion_preserves_dacp_fp32():
    prior = DetectionAlignedForegroundBackgroundPrior(4, 8).half()
    keep_force_fp32_modules(prior)
    assert prior.raw_rho.dtype == torch.float32
    assert prior.mask_head[0].weight.dtype == torch.float32
    rgb = torch.randn(1, 16, 6, 6, dtype=torch.float16)
    ir = torch.randn(1, 16, 6, 6, dtype=torch.float16)
    assert prior.predict_objectness(rgb, ir).dtype == torch.float32


def test_balanced_heatmap_pushes_foreground_up_and_background_down():
    logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 2, 2] = 1.0
    focal, dice, _ = dacp_balanced_heatmap_loss(logits, target)
    (focal + 0.5 * dice).backward()
    assert logits.grad[..., 2, 2].item() < 0.0
    background_grad = logits.grad[target == 0]
    assert background_grad.mean().item() > 0.0


def test_health_gate_requires_all_three_conditions():
    healthy = {
        'mean_foreground_gap': 0.006,
        'positive_gap_fraction': 0.8,
        'max_prior_content_ratio': 0.01,
    }
    assert dacp_health_passes(healthy, 0.005, 0.75, 0.001)
    for key in healthy:
        failed = dict(healthy)
        failed[key] = 0.0
        assert not dacp_health_passes(failed, 0.005, 0.75, 0.001)
