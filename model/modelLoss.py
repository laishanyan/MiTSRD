import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 类别权重

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def full_loss(x, labels, z_A, z_E):
    weight = torch.tensor([0.4, 0.6]).to('cuda')
    loss_cls = F.cross_entropy(x, labels, weight=weight)
    z_A_norm = F.normalize(z_A, dim=1)
    z_E_norm = F.normalize(z_E, dim=1)
    loss_orth = (z_A_norm * z_E_norm).sum(dim=1).mean()  # 平均相似度
    lambda_orth = 0.15
    loss = loss_cls + lambda_orth * loss_orth

    return loss