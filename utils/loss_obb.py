"""YOLOv5 CSL oriented-box loss."""

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.loss import FocalLoss, smooth_BCE
from utils.torch_utils import is_parallel


class ComputeLossOBB:
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        hyp = model.hyp
        self.sort_obj_iou = False
        cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        obj_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        theta_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp.get('theta_pw', 1.0)], device=device))
        self.cp, self.cn = smooth_BCE(hyp.get('label_smoothing', 0.0))
        gamma = hyp['fl_gamma']
        if gamma > 0:
            cls_loss, obj_loss = FocalLoss(cls_loss, gamma), FocalLoss(obj_loss, gamma)
            theta_loss = FocalLoss(theta_loss, gamma)
        detector = (model.module.model[-1] if is_parallel(model)
                    else model.model[-1])
        self.stride = detector.stride
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            detector.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(self.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.BCEtheta = cls_loss, obj_loss, theta_loss
        self.gr, self.hyp, self.autobalance = model.gr, hyp, autobalance
        for attribute in ('na', 'nc', 'nl', 'anchors', 'angle_bins'):
            setattr(self, attribute, getattr(detector, attribute))

    def __call__(self, predictions, targets):
        device = targets.device
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, theta_targets = self.build_targets(
            predictions, targets)
        for layer_index, prediction in enumerate(predictions):
            batch, anchor, grid_y, grid_x = indices[layer_index]
            object_target = torch.zeros_like(prediction[..., 0], device=device)
            count = batch.shape[0]
            if count:
                selected = prediction[batch, anchor, grid_y, grid_x]
                xy = selected[:, :2].sigmoid() * 2.0 - 0.5
                wh = (selected[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[layer_index]
                predicted_box = torch.cat((xy, wh), 1)
                iou = bbox_iou(
                    predicted_box.T, tbox[layer_index], x1y1x2y2=False,
                    CIoU=True)
                lbox += (1.0 - iou).mean()
                score = iou.detach().clamp(0).type(object_target.dtype)
                object_target[batch, anchor, grid_y, grid_x] = (
                    (1.0 - self.gr) + self.gr * score)
                class_end = 5 + self.nc
                if self.nc > 1:
                    true_class = torch.full_like(
                        selected[:, 5:class_end], self.cn, device=device)
                    true_class[range(count), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(selected[:, 5:class_end], true_class)
                ltheta += self.BCEtheta(
                    selected[:, class_end:],
                    theta_targets[layer_index].type(selected.dtype))
            object_loss = self.BCEobj(prediction[..., 4], object_target)
            lobj += object_loss * self.balance[layer_index]
            if self.autobalance:
                self.balance[layer_index] = (
                    self.balance[layer_index] * 0.9999 +
                    0.0001 / object_loss.detach().item())
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp.get('theta', 0.5)
        batch_size = predictions[0].shape[0]
        total = lbox + lobj + lcls + ltheta
        return total * batch_size, torch.cat((lbox, lobj, lcls, ltheta)).detach()

    def build_targets(self, predictions, targets):
        anchor_count, target_count = self.na, targets.shape[0]
        target_classes, target_boxes, indices, selected_anchors = [], [], [], []
        theta_targets = []
        anchor_indices = (torch.arange(anchor_count, device=targets.device)
                          .float().view(anchor_count, 1).repeat(1, target_count))
        targets = torch.cat((targets.repeat(anchor_count, 1, 1),
                             anchor_indices[:, :, None]), 2)
        offset_scale = 0.5
        offsets_template = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=targets.device).float() * offset_scale
        for layer_index in range(self.nl):
            anchors = self.anchors[layer_index]
            feature_width = predictions[layer_index].shape[3]
            feature_height = predictions[layer_index].shape[2]
            transformed = targets.clone()
            transformed[:, :, 2:6] /= self.stride[layer_index]
            if target_count:
                ratio = transformed[:, :, 4:6] / anchors[:, None]
                match = torch.max(ratio, 1.0 / ratio).max(2)[0] < self.hyp['anchor_t']
                transformed = transformed[match]
                xy = transformed[:, 2:4]
                inverse_xy = torch.tensor(
                    [feature_width, feature_height], device=targets.device) - xy
                left, top = ((xy % 1 < offset_scale) & (xy > 1)).T
                right, bottom = ((inverse_xy % 1 < offset_scale) &
                                 (inverse_xy > 1)).T
                mask = torch.stack((torch.ones_like(left), left, top,
                                    right, bottom))
                transformed = transformed.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(xy)[None] +
                           offsets_template[:, None])[mask]
            else:
                transformed = targets[0]
                offsets = 0
            batch, target_class = transformed[:, :2].long().T
            xy = transformed[:, 2:4]
            wh = transformed[:, 4:6]
            grid = (xy - offsets).long()
            grid_x, grid_y = grid.T
            anchor = transformed[:, -1].long()
            indices.append((batch, anchor,
                            grid_y.clamp_(0, feature_height - 1),
                            grid_x.clamp_(0, feature_width - 1)))
            target_boxes.append(torch.cat((xy - grid, wh), 1))
            selected_anchors.append(anchors[anchor])
            target_classes.append(target_class)
            theta_targets.append(transformed[:, 7:-1])
        return (target_classes, target_boxes, indices, selected_anchors,
                theta_targets)
