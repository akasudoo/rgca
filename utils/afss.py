"""Anti-Forgetting Sampling Strategy (AFSS) for paired RGB-IR training."""

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from utils.general import box_iou, non_max_suppression, xywh2xyxy


class DynamicSubsetSampler(Sampler):
    """A mutable, epoch-shuffled subset sampler.

    The sampler yields original dataset indices. In distributed training each
    rank receives a padded, non-overlapping strided slice, matching the basic
    behavior of ``DistributedSampler``.
    """

    def __init__(self, data_source, seed=0, num_replicas=1, rank=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.indices = list(range(len(data_source)))

    def set_indices(self, indices):
        indices = [int(i) for i in indices]
        if not indices:
            raise ValueError('AFSS selected an empty training subset')
        if min(indices) < 0 or max(indices) >= len(self.data_source):
            raise IndexError('AFSS subset contains an out-of-range dataset index')
        self.indices = indices

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rank_indices(self):
        indices = self.indices.copy()
        random.Random(self.seed + self.epoch).shuffle(indices)
        if self.num_replicas > 1:
            total = int(math.ceil(len(indices) / self.num_replicas)) * self.num_replicas
            indices += indices[:total - len(indices)]
            indices = indices[self.rank:total:self.num_replicas]
        return indices

    def __iter__(self):
        return iter(self._rank_indices())

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.num_replicas))


class AFSSScheduler:
    """Paper-faithful easy/moderate/hard sampling with anti-forgetting."""

    def __init__(self, num_images, warmup_epochs=3, update_interval=5,
                 easy_threshold=0.85, hard_threshold=0.55,
                 easy_ratio=0.02, moderate_ratio=0.40,
                 easy_review_interval=10, moderate_coverage_interval=3,
                 seed=0):
        self.num_images = int(num_images)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.easy_threshold = float(easy_threshold)
        self.hard_threshold = float(hard_threshold)
        self.easy_ratio = float(easy_ratio)
        self.moderate_ratio = float(moderate_ratio)
        self.easy_review_interval = int(easy_review_interval)
        self.moderate_coverage_interval = int(moderate_coverage_interval)
        self.seed = int(seed)
        self.precision = np.zeros(self.num_images, dtype=np.float32)
        self.recall = np.zeros(self.num_images, dtype=np.float32)
        self.last_used = np.zeros(self.num_images, dtype=np.int32)
        self.has_scores = False

    @property
    def sufficiency(self):
        return np.minimum(self.precision, self.recall)

    def should_update_scores(self, completed_epoch):
        """Return whether scores should be refreshed after a 1-based epoch."""
        return (completed_epoch >= self.warmup_epochs and
                (completed_epoch - self.warmup_epochs) % self.update_interval == 0)

    def update_scores(self, precision, recall):
        precision = np.asarray(precision, dtype=np.float32)
        recall = np.asarray(recall, dtype=np.float32)
        if precision.shape != (self.num_images,) or recall.shape != (self.num_images,):
            raise ValueError('AFSS precision/recall arrays do not match the training dataset')
        self.precision = precision
        self.recall = recall
        self.has_scores = True

    def _sample(self, candidates, count, rng):
        candidates = [int(i) for i in candidates]
        if count <= 0 or not candidates:
            return []
        if count >= len(candidates):
            return candidates
        return rng.sample(candidates, count)

    def select(self, epoch):
        """Select original dataset indices for a zero-based training epoch."""
        if epoch < self.warmup_epochs or not self.has_scores:
            selected = list(range(self.num_images))
            return selected, {
                'easy': 0, 'moderate': 0, 'hard': self.num_images,
                'selected_easy': 0, 'selected_moderate': 0,
                'selected_hard': self.num_images, 'selected_total': self.num_images
            }

        completed_epoch = int(epoch)  # epochs completed before this epoch starts
        sufficiency = self.sufficiency
        easy = np.flatnonzero(sufficiency > self.easy_threshold).tolist()
        moderate = np.flatnonzero(
            (sufficiency >= self.hard_threshold) &
            (sufficiency <= self.easy_threshold)).tolist()
        hard = np.flatnonzero(sufficiency < self.hard_threshold).tolist()
        rng = random.Random(self.seed + epoch)

        # Continuous review: 2% easy images, at most half forced from images
        # unseen for at least 10 epochs.
        easy_count = min(len(easy), int(math.ceil(self.easy_ratio * len(easy))))
        easy_forced_pool = [
            i for i in easy
            if completed_epoch - int(self.last_used[i]) >= self.easy_review_interval
        ]
        easy_forced_count = min(len(easy_forced_pool), easy_count // 2)
        selected_easy_forced = self._sample(easy_forced_pool, easy_forced_count, rng)
        easy_remaining = [i for i in easy if i not in set(selected_easy_forced)]
        selected_easy = selected_easy_forced + self._sample(
            easy_remaining, easy_count - len(selected_easy_forced), rng)

        # Short-term coverage: 40% moderate images, prioritizing every image
        # that has missed the previous two epochs.
        moderate_count = min(
            len(moderate), int(math.ceil(self.moderate_ratio * len(moderate))))
        moderate_forced_pool = [
            i for i in moderate
            if completed_epoch - int(self.last_used[i]) >= self.moderate_coverage_interval
        ]
        selected_moderate_forced = self._sample(
            moderate_forced_pool, min(len(moderate_forced_pool), moderate_count), rng)
        moderate_remaining = [
            i for i in moderate if i not in set(selected_moderate_forced)
        ]
        selected_moderate = selected_moderate_forced + self._sample(
            moderate_remaining, moderate_count - len(selected_moderate_forced), rng)

        selected = selected_easy + selected_moderate + hard
        rng.shuffle(selected)
        return selected, {
            'easy': len(easy), 'moderate': len(moderate), 'hard': len(hard),
            'selected_easy': len(selected_easy),
            'selected_moderate': len(selected_moderate),
            'selected_hard': len(hard),
            'selected_total': len(selected)
        }

    def mark_used(self, indices, completed_epoch):
        self.last_used[np.asarray(indices, dtype=np.int64)] = int(completed_epoch)

    def state_dict(self):
        return {
            'num_images': self.num_images,
            'precision': self.precision.tolist(),
            'recall': self.recall.tolist(),
            'last_used': self.last_used.tolist(),
            'has_scores': self.has_scores,
        }

    def load_state_dict(self, state):
        if int(state.get('num_images', -1)) != self.num_images:
            raise ValueError('AFSS checkpoint state belongs to a different-sized dataset')
        self.precision = np.asarray(state['precision'], dtype=np.float32)
        self.recall = np.asarray(state['recall'], dtype=np.float32)
        self.last_used = np.asarray(state['last_used'], dtype=np.int32)
        self.has_scores = bool(state['has_scores'])


def _match_image_predictions(pred, labels, iou_threshold):
    """Return class-aware TP count at a single IoU threshold."""
    if not len(pred) or not len(labels):
        return 0
    target_classes = labels[:, 0]
    target_boxes = xywh2xyxy(labels[:, 1:5])
    matched_targets = set()
    true_positives = 0
    for prediction_index in pred[:, 4].argsort(descending=True):
        prediction = pred[prediction_index]
        candidate_indices = torch.nonzero(
            target_classes == prediction[5], as_tuple=False).view(-1)
        candidate_indices = [
            int(i) for i in candidate_indices if int(i) not in matched_targets
        ]
        if not candidate_indices:
            continue
        candidates = torch.as_tensor(candidate_indices, device=pred.device)
        ious = box_iou(prediction[:4].view(1, 4), target_boxes[candidates]).view(-1)
        best_iou, best_position = ious.max(0)
        if best_iou >= iou_threshold:
            matched_targets.add(candidate_indices[int(best_position)])
            true_positives += 1
    return true_positives


@torch.no_grad()
def evaluate_image_precision_recall(model, dataloader, dataset_size, path_to_index,
                                    device, opt, conf_threshold=0.25,
                                    iou_threshold=0.5):
    """Evaluate unaugmented training images and return per-image P/R arrays."""
    was_training = model.training
    model.eval()
    precision = np.zeros(dataset_size, dtype=np.float32)
    recall = np.zeros(dataset_size, dtype=np.float32)

    for imgs, targets, paths, _ in tqdm(dataloader, desc='AFSS state update'):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        height, width = imgs.shape[2:]
        targets[:, 2:] *= torch.tensor(
            [width, height, width, height], device=device)
        imgs_rgb, imgs_ir = imgs[:, :3], imgs[:, 3:]
        mode = getattr(opt, 'single_modality', 'none')
        if mode == 'vis':
            model_rgb, model_ir = imgs_rgb, imgs_rgb
        elif mode == 'ir':
            model_rgb, model_ir = imgs_ir, imgs_ir
        elif mode == 'concat':
            model_rgb, model_ir = torch.cat([imgs_rgb, imgs_ir], 1), imgs_ir
        else:
            model_rgb, model_ir = imgs_rgb, imgs_ir

        output = model(model_rgb, model_ir)
        predictions = output[0] if isinstance(output, (tuple, list)) else output
        predictions = non_max_suppression(
            predictions, conf_threshold, iou_threshold,
            multi_label=True, agnostic=getattr(opt, 'single_cls', False))

        for batch_index, pred in enumerate(predictions):
            labels = targets[targets[:, 0] == batch_index, 1:]
            if getattr(opt, 'single_cls', False) and len(pred):
                pred[:, 5] = 0
            true_positives = _match_image_predictions(pred, labels, iou_threshold)
            num_predictions, num_targets = len(pred), len(labels)
            image_precision = (
                true_positives / num_predictions if num_predictions else
                (1.0 if num_targets == 0 else 0.0))
            image_recall = (
                true_positives / num_targets if num_targets else 1.0)
            key = str(Path(paths[batch_index]))
            if key not in path_to_index:
                raise KeyError(f'AFSS evaluation image is absent from training set: {key}')
            image_index = path_to_index[key]
            precision[image_index] = image_precision
            recall[image_index] = image_recall

    if was_training:
        model.train()
    return precision, recall
