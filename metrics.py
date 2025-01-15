import torch
import numpy as np
from medpy import metric as me

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()


def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric


def test_SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'hd95':
            pred_ = _threshold(pred, threshold=0.5)
            label_ = _threshold(label, threshold=0.5)
            pred_ = pred_.cpu().detach().numpy()[0]
            label_ = label_.cpu().detach().numpy()[0]
            if pred_.sum() > 0 and label_.sum() > 0:
                hd95 = me.binary.hd95(pred_, label_)
            else:
                hd95 = 0
            metric_list.append(hd95)
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric