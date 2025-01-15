import logging
import os
import torch
import numpy as np
import torch.nn as nn

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels

def select_random_points(pr, gt, point_num=9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x, y] == 0 and one_gt[x, y] == 1:
                label = 1
            elif one_pred[x, y] == 1 and one_gt[x, y] == 0:
                label = 0
            elif one_pred[x, y] == 1 and one_gt[x, y] == 1:
                label = 1
            elif one_pred[x, y] == 0 and one_gt[x, y] == 0:
                label = 0
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)

def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logit = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num=point_num)
    batched_input["mask_inputs"] = low_res_masks_logit
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input

def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    return batched_input

#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss

class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):

    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss