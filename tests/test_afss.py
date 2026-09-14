import numpy as np

from utils.afss import AFSSScheduler, DynamicSubsetSampler


def test_afss_uses_full_dataset_until_scores_exist():
    scheduler = AFSSScheduler(10, warmup_epochs=3)
    selected, stats = scheduler.select(epoch=4)
    assert selected == list(range(10))
    assert stats['selected_total'] == 10


def test_afss_paper_sampling_ratios_and_hard_coverage():
    scheduler = AFSSScheduler(100, warmup_epochs=3, seed=7)
    precision = np.r_[np.full(50, 0.9), np.full(30, 0.7), np.full(20, 0.4)]
    recall = precision.copy()
    scheduler.update_scores(precision, recall)

    selected, stats = scheduler.select(epoch=3)

    assert stats == {
        'easy': 50,
        'moderate': 30,
        'hard': 20,
        'selected_easy': 1,
        'selected_moderate': 12,
        'selected_hard': 20,
        'selected_total': 33,
    }
    assert set(range(80, 100)).issubset(selected)


def test_afss_moderate_short_term_coverage():
    scheduler = AFSSScheduler(10, warmup_epochs=0, moderate_ratio=0.4, seed=1)
    scheduler.update_scores(np.full(10, 0.7), np.full(10, 0.7))
    scheduler.last_used[:] = 2
    scheduler.last_used[:4] = 0

    selected, _ = scheduler.select(epoch=3)

    assert set(range(4)).issubset(selected)
    assert len(selected) == 4


def test_afss_state_round_trip():
    scheduler = AFSSScheduler(5)
    scheduler.update_scores(np.arange(5) / 5, np.arange(5) / 6)
    scheduler.mark_used([1, 3], completed_epoch=7)
    restored = AFSSScheduler(5)
    restored.load_state_dict(scheduler.state_dict())

    np.testing.assert_allclose(restored.precision, scheduler.precision)
    np.testing.assert_allclose(restored.recall, scheduler.recall)
    np.testing.assert_array_equal(restored.last_used, scheduler.last_used)
    assert restored.has_scores


def test_dynamic_subset_sampler_is_deterministic_and_distributed():
    data = list(range(10))
    rank0 = DynamicSubsetSampler(data, seed=5, num_replicas=2, rank=0)
    rank1 = DynamicSubsetSampler(data, seed=5, num_replicas=2, rank=1)
    subset = [1, 3, 4, 8]
    rank0.set_indices(subset)
    rank1.set_indices(subset)
    rank0.set_epoch(2)
    rank1.set_epoch(2)

    indices0, indices1 = list(rank0), list(rank1)
    assert set(indices0 + indices1) == set(subset)
    assert set(indices0).isdisjoint(indices1)
