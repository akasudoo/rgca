"""Validation for paired RGB/IR oriented detection using exact rotated IoU."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets_obb import create_dataloader_obb_rgb_ir
from utils.general import check_img_size, colorstr, logger
from utils.metrics import ap_per_class
from utils.obb_utils import non_max_suppression_obb, process_batch_obb
from utils.torch_utils import select_device


def _dota_continuous_ap50(stats, nc):
    """DOTA-devkit VOC-style continuous AP at rotated IoU 0.5."""
    correct, confidence, predicted_class, target_class = stats
    order = np.argsort(-confidence)
    correct = correct[order, 0]
    predicted_class = predicted_class[order]
    per_class = np.zeros(nc, dtype=np.float64)
    for class_id in range(nc):
        prediction_mask = predicted_class == class_id
        target_count = int((target_class == class_id).sum())
        if not target_count or not prediction_mask.any():
            continue
        false_positive = (1.0 - correct[prediction_mask]).cumsum()
        true_positive = correct[prediction_mask].cumsum()
        recall = true_positive / target_count
        precision = true_positive / np.maximum(
            true_positive + false_positive, np.finfo(np.float64).eps)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        changes = np.where(recall[1:] != recall[:-1])[0]
        per_class[class_id] = np.sum(
            (recall[changes + 1] - recall[changes]) *
            precision[changes + 1])
    return per_class


@torch.no_grad()
def test(data, weights=None, batch_size=16, imgsz=640, conf_thres=0.001,
         iou_thres=0.5, model=None, single_cls=False, dataloader=None,
         save_dir=Path(''), save_txt=False, save_conf=False, verbose=False,
         plots=False, wandb_logger=None, compute_loss=None, is_coco=False,
         opt=None, labels_list=None, save_json=False, task='val',
         max_det=300, max_nms=3000, multi_label=False, **kwargs):
    del (save_txt, save_conf, plots, wandb_logger, is_coco, labels_list,
         save_json, kwargs)
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(getattr(opt, 'device', '0'))
        weight_path = Path(weights)
        if not weight_path.is_file():
            raise FileNotFoundError(
                f'OBB checkpoint does not exist: {weight_path}. '
                'Run train.py successfully first; best.pt is created after '
                'the first completed validation epoch.')
        model = attempt_load(weights, map_location=device)
    model.eval()
    half = device.type != 'cpu' and next(model.parameters()).dtype == torch.float16
    if isinstance(data, (str, Path)):
        with open(data) as stream:
            data = yaml.safe_load(stream)
    names = model.module.names if hasattr(model, 'module') else model.names
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    nc = 1 if single_cls else int(data['nc'])
    if dataloader is None:
        split = task if task in ('train', 'val', 'test') else 'val'
        rgb, ir = data[f'{split}_rgb'], data[f'{split}_ir']
        stride = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, stride)
        dataloader = create_dataloader_obb_rgb_ir(
            rgb, ir, imgsz, batch_size, stride, opt, rect=True, pad=0.5,
            prefix=colorstr(f'{split}: '))[0]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    losses = torch.zeros(4, device=device)
    seen, inference_time, nms_time = 0, 0.0, 0.0
    progress = tqdm(dataloader, desc='OBB validation')
    for images, targets, paths, shapes in progress:
        del paths, shapes
        images = images.to(device, non_blocking=True)
        images = (images.half() if half else images.float()) / 255.0
        targets = targets.to(device)
        started = time.time()
        prediction, _, train_output = model(images[:, :3], images[:, 3:])
        inference_time += time.time() - started
        if compute_loss is not None:
            losses += compute_loss(
                [layer.float() for layer in train_output], targets)[1]
        started = time.time()
        outputs = non_max_suppression_obb(
            prediction.float(), conf_thres, iou_thres,
            multi_label=multi_label, agnostic=single_cls,
            max_det=max_det, max_nms=max_nms)
        nms_time += time.time() - started
        for image_index, detections in enumerate(outputs):
            labels = targets[targets[:, 0] == image_index, 1:7]
            target_classes = labels[:, 0].tolist()
            seen += 1
            if single_cls and len(detections):
                detections[:, 6] = 0
            correct = process_batch_obb(detections, labels, iouv)
            stats.append((correct.cpu(), detections[:, 5].cpu(),
                          detections[:, 6].cpu(), target_classes))
    true_positive = false_positive = false_negative = f1_value = 0
    precision = recall = map50 = mean_ap = 0.0
    # Returned to train.py for per-class validation/test AP50 logging.
    # This is deliberately DOTA OBB AP50, not the per-class AP50:95 array
    # returned by the horizontal YOLO evaluator.
    per_class_ap50 = np.zeros(nc, dtype=np.float64)
    ap_class = np.array([], dtype=np.int32)
    if stats:
        stats = [np.concatenate(values, 0) for values in zip(*stats)]
        if len(stats[0]) and stats[0].any():
            (true_positive, false_positive, false_negative, precision_values,
             recall_values, ap, f1_values, ap_class) = ap_per_class(*stats)
            # The paper comparison follows DOTA Task1 AP50.  Use the official
            # devkit's continuous VOC integral for the reported/best mAP50;
            # retain YOLO's 0.5:0.95 metric as an additional diagnostic.
            dota_ap50 = _dota_continuous_ap50(stats, nc)
            ap50, class_ap = dota_ap50[ap_class], ap.mean(1)
            precision, recall = precision_values.mean(), recall_values.mean()
            map50, mean_ap = ap50.mean(), class_ap.mean()
            f1_value = f1_values[0] if len(f1_values) else 0
            per_class_ap50[:] = dota_ap50
            logger.info('%20s%12s%12s%12s%12s%12s' %
                        ('Class', 'Images', 'Labels', 'P', 'R', 'DOTA-OBB AP50'))
            counts = np.bincount(stats[3].astype(np.int64), minlength=nc)
            logger.info('%20s%12d%12d%12.4g%12.4g%12.4g' %
                        ('all', seen, counts.sum(), precision, recall, map50))
            if verbose:
                for index, class_id in enumerate(ap_class):
                    logger.info('%20s%12d%12d%12.4g%12.4g%12.4g' %
                                (names[class_id], seen, counts[class_id],
                                 precision_values[index], recall_values[index],
                                 ap50[index]))
    loss_values = (losses / max(len(dataloader), 1)).cpu().tolist()
    tp0 = float(true_positive[0]) if not np.isscalar(true_positive) else float(true_positive)
    fp0 = float(false_positive[0]) if not np.isscalar(false_positive) else float(false_positive)
    fn0 = float(false_negative[0]) if not np.isscalar(false_negative) else float(false_negative)
    timing = (inference_time / max(seen, 1) * 1000,
              nms_time / max(seen, 1) * 1000,
              (inference_time + nms_time) / max(seen, 1) * 1000,
              imgsz, imgsz, batch_size)
    model.float()
    results = (tp0, fp0, fn0, float(f1_value), float(precision),
               float(recall), float(map50), float(mean_ap), *loss_values)
    return results, per_class_ap50, [0.0] * 10, timing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='data/multispectral/DroneVehicle_OBB.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--max-nms', type=int, default=3000)
    parser.add_argument('--multi-label', action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--task', choices=('train', 'val', 'test'), default='val')
    args = parser.parse_args()
    test(args.data, weights=args.weights, batch_size=args.batch_size,
         imgsz=args.img_size, conf_thres=args.conf_thres,
         iou_thres=args.iou_thres, opt=args, task=args.task, verbose=True,
         max_det=args.max_det, max_nms=args.max_nms,
         multi_label=args.multi_label)
