import os
import numpy as np
import torch


def distance_cal(gt_bboxes, bboxes):
    x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    gt_x_center = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_y_center = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    distance = 1 / torch.sqrt(((x_center[:, None] - gt_x_center[None, :]) ** 2 + (y_center[:, None] - gt_y_center[None, :]) ** 2)) 
    aspect = 1 / torch.sqrt((w[:, None] - gt_w[None, :]) ** 2 + (h[:, None] - gt_h[None, :]) ** 2)
    return distance.T, aspect.T
