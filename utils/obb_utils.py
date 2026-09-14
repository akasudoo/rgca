"""Geometry, CSL encoding, rotated NMS, and exact rotated-IoU matching."""

import math

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS


ANGLE_BINS = 180


def regular_theta(theta):
    """Map radians to the long-edge interval [-pi/2, pi/2)."""
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def gaussian_circular_label(angle_degrees, bins=ANGLE_BINS, radius=2.0):
    """Circular smooth label centered on an angle in [0, 180)."""
    indices = np.arange(bins, dtype=np.float32)
    center = float(angle_degrees) % bins
    distance = np.abs(indices - center)
    distance = np.minimum(distance, bins - distance)
    return np.exp(-(distance ** 2) / (2.0 * float(radius) ** 2))


def poly2rbox(polygons, bins=ANGLE_BINS, radius=2.0, with_csl=False):
    """Convert four-point polygons to (cx, cy, long, short, theta radians)."""
    polygons = np.asarray(polygons, dtype=np.float32).reshape(-1, 8)
    boxes, labels = [], []
    for polygon in polygons:
        (cx, cy), (width, height), angle = cv2.minAreaRect(
            polygon.reshape(4, 2))
        theta = -float(angle) * math.pi / 180.0
        if width < height:
            width, height = height, width
            theta += math.pi / 2
        theta = regular_theta(theta)
        boxes.append((cx, cy, width, height, theta))
        if with_csl:
            labels.append(gaussian_circular_label(
                theta * 180.0 / math.pi + 90.0, bins, radius))
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if with_csl:
        return boxes, np.asarray(labels, dtype=np.float32).reshape(-1, bins)
    return boxes


def rbox2poly(boxes):
    """Convert (cx, cy, long, short, theta radians) boxes to polygons."""
    is_tensor = isinstance(boxes, torch.Tensor)
    if is_tensor:
        center, width, height, theta = (
            boxes[:, :2], boxes[:, 2:3], boxes[:, 3:4], boxes[:, 4:5])
        cosine, sine = torch.cos(theta), torch.sin(theta)
        vector1 = torch.cat((width / 2 * cosine, -width / 2 * sine), -1)
        vector2 = torch.cat((-height / 2 * sine, -height / 2 * cosine), -1)
        return torch.cat((center + vector1 + vector2,
                          center + vector1 - vector2,
                          center - vector1 - vector2,
                          center - vector1 + vector2), -1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    center, width, height, theta = np.split(boxes, (2, 3, 4), axis=-1)
    cosine, sine = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate((width / 2 * cosine, -width / 2 * sine), -1)
    vector2 = np.concatenate((-height / 2 * sine, -height / 2 * cosine), -1)
    return np.concatenate((center + vector1 + vector2,
                           center + vector1 - vector2,
                           center - vector1 - vector2,
                           center - vector1 + vector2), -1)


def _opencv_rect(box):
    return ((float(box[0]), float(box[1])),
            (max(float(box[2]), 1e-4), max(float(box[3]), 1e-4)),
            float(box[4]) * 180.0 / math.pi)


def rotated_iou_matrix(boxes1, boxes2):
    """Exact polygon IoU via OpenCV; intended for validation, not gradients."""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 5)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 5)
    output = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    if not len(boxes1) or not len(boxes2):
        return output
    polygons1 = rbox2poly(boxes1).reshape(-1, 4, 2)
    polygons2 = rbox2poly(boxes2).reshape(-1, 4, 2)
    min1, max1 = polygons1.min(1), polygons1.max(1)
    min2, max2 = polygons2.min(1), polygons2.max(1)
    areas1 = boxes1[:, 2] * boxes1[:, 3]
    areas2 = boxes2[:, 2] * boxes2[:, 3]
    for i, box1 in enumerate(boxes1):
        candidates = np.where(
            (max1[i, 0] > min2[:, 0]) & (max2[:, 0] > min1[i, 0]) &
            (max1[i, 1] > min2[:, 1]) & (max2[:, 1] > min1[i, 1]))[0]
        for j in candidates:
            _, points = cv2.rotatedRectangleIntersection(
                _opencv_rect(box1), _opencv_rect(boxes2[j]))
            if points is None:
                continue
            intersection = abs(cv2.contourArea(cv2.convexHull(points)))
            union = float(areas1[i] + areas2[j] - intersection)
            output[i, j] = intersection / union if union > 0 else 0.0
    return output


def non_max_suppression_obb(prediction, conf_thres=0.001, iou_thres=0.5,
                            multi_label=False, agnostic=False, max_det=300,
                            max_nms=3000):
    """Decode CSL angles and run vectorized class-aware rotated NMS on GPU."""
    bins = ANGLE_BINS
    nc = prediction.shape[-1] - 5 - bins
    class_end = 5 + nc
    outputs = []
    for raw in prediction:
        raw = raw[raw[:, 4] > conf_thres]
        if not len(raw):
            outputs.append(torch.zeros((0, 7), device=prediction.device))
            continue
        raw[:, 5:class_end] *= raw[:, 4:5]
        theta = (raw[:, class_end:].argmax(1).float() - 90.0)
        theta = theta * math.pi / 180.0
        if multi_label and nc > 1:
            row, cls = (raw[:, 5:class_end] > conf_thres).nonzero(
                as_tuple=False).T
            detections = torch.cat((raw[row, :4], theta[row, None],
                                    raw[row, cls + 5, None],
                                    cls[:, None].float()), 1)
        else:
            confidence, cls = raw[:, 5:class_end].max(1, keepdim=True)
            detections = torch.cat((raw[:, :4], theta[:, None], confidence,
                                    cls.float()), 1)
            detections = detections[confidence.view(-1) > conf_thres]
        if not len(detections):
            outputs.append(torch.zeros((0, 7), device=prediction.device))
            continue
        order = detections[:, 5].argsort(descending=True)[:max_nms]
        detections = detections[order]

        # Shift only the box centers so different classes never suppress each
        # other. batch_probiou and Fast-NMS stay on the prediction device,
        # avoiding the former 10k-box GPU->CPU OpenCV bottleneck.
        boxes = detections[:, :5].clone()
        if not agnostic:
            class_stride = boxes[:, :4].abs().max().clamp(min=1.0) * 2.0 + 1.0
            boxes[:, :2] += detections[:, 6:7] * class_stride
        kept = TorchNMS.fast_nms(
            boxes, detections[:, 5], float(iou_thres),
            iou_func=batch_probiou)
        detections = detections[kept[:max_det]]
        outputs.append(detections)
    return outputs


def process_batch_obb(detections, labels, iouv):
    """Return correct-detection flags using one-to-one exact rotated IoU."""
    correct = np.zeros((len(detections), len(iouv)), dtype=bool)
    if not len(labels) or not len(detections):
        return torch.as_tensor(correct, device=detections.device)
    det = detections.detach().float().cpu().numpy()
    lab = labels.detach().float().cpu().numpy()
    same_class = lab[:, 0:1] == det[None, :, 6]
    ious = np.zeros((len(lab), len(det)), dtype=np.float32)
    # Exact polygon IoU is retained for the reported DOTA metrics, but only
    # boxes from the same class are compared after GPU NMS.
    for class_id in np.intersect1d(lab[:, 0], det[:, 6]):
        label_indices = np.flatnonzero(lab[:, 0] == class_id)
        detection_indices = np.flatnonzero(det[:, 6] == class_id)
        ious[np.ix_(label_indices, detection_indices)] = rotated_iou_matrix(
            lab[label_indices, 1:6], det[detection_indices, :5])
    for threshold_index, threshold in enumerate(iouv.cpu().numpy()):
        label_index, detection_index = np.nonzero(
            (ious >= threshold) & same_class)
        if not len(label_index):
            continue
        matches = np.stack((label_index, detection_index,
                            ious[label_index, detection_index]), 1)
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return torch.as_tensor(correct, device=detections.device)
