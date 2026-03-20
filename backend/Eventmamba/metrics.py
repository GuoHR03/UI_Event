import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # print(dist)
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen

def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target[:, :2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:, 2]  # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()

def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    # print(target.size(),prediction.size())
    target = target.reshape(-1, 2)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers

def px_euclidean_ab(target, prediction, width_scale, height_scale):
    """
    分别计算 a (长轴) and b (短轴) 的总像素绝对误差
    :param target: (N, 2) tensor, 包含 [a, b] 的真实值
    :param prediction: (N, 2) tensor, 包含 [a, b] 的预测值
    :param width_scale: a 的缩放因子 (通常是图像宽度 W)
    :param height_scale: b 的缩放因子 (通常是图像高度 H)
    :return: a的总误差, b的总误差, 样本总数
    """
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    diff = target - prediction
    diff[:, 0] *= width_scale 
    diff[:, 1] *= height_scale

    abs_diff = torch.abs(diff)
    total_a_error = torch.sum(abs_diff[:, 0])
    total_b_error = torch.sum(abs_diff[:, 1])
    
    sample_numbers = target.shape[0]
    return total_a_error, total_b_error, sample_numbers

def px_euclidean_angle(target, prediction):
    """
    计算角度误差，自动处理周期性 (椭圆周期为 pi)
    :param target: (N, 1) or (N,) 弧度制 [-pi/2, pi/2]
    :param prediction: (N, 1) or (N,) 弧度制 [-pi/2, pi/2]
    :return: 总角度误差(度), 样本数
    """
    target = target.reshape(-1)
    prediction = prediction.reshape(-1)
    diff = target - prediction
    diff = (diff + torch.pi/2) % torch.pi - torch.pi/2
    abs_diff = torch.abs(diff)
    abs_diff_deg = abs_diff * 180 / torch.pi
    total_angle_error = torch.sum(abs_diff_deg)
    sample_numbers = target.shape[0]
    
    return total_angle_error, sample_numbers

def compute_ellipse_iou(pred_params, target_params, H, W):
    """
    计算椭圆的 IoU (Intersection over Union)。
    params: [B, 5] -> (x, y, a, b, angle)
    """
    B = pred_params.shape[0]
    y = torch.arange(H, device=pred_params.device).float()
    x = torch.arange(W, device=pred_params.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid_y = yy.expand(B, -1, -1)
    grid_x = xx.expand(B, -1, -1)
    def generate_mask(params):
        cx = params[:, 0].view(B, 1, 1) * W
        cy = params[:, 1].view(B, 1, 1) * H
        a  = params[:, 2].view(B, 1, 1) * W
        b  = params[:, 3].view(B, 1, 1) * H
        theta = params[:, 4].view(B, 1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dx = grid_x - cx
        dy = grid_y - cy
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2
        mask = (ellipse_eq <= 1.0).float()
        return mask
    pred_mask = generate_mask(pred_params)
    target_mask = generate_mask(target_params)

    intersection = (pred_mask * target_mask).sum(dim=(1, 2))
    union = pred_mask.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.sum()

class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_vector: torch.Tensor, target_vector: torch.Tensor) -> torch.Tensor:

        cos_sim = F.cosine_similarity(pred_vector, target_vector, dim=1, eps=1e-8) 
        
        loss_per_sample = 1.0 - cos_sim
        
        if self.reduction == 'mean':
            loss = torch.mean(loss_per_sample)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_per_sample)
        else:
            loss = loss_per_sample
            
        return loss