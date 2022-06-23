import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=1, gamma=2, neg_weights=0.5, pos_weights=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.neg_weights = neg_weights
        self.pos_weights = pos_weights

    def forward(self, pred, target):
        loss_pos = - self.alpha * torch.pow(1 - pred[target==1], self.gamma) * (pred[target==1]).log()
        loss_neg = - (1 - self.alpha) * torch.pow(pred[target==0], self.gamma) * (1 - pred[target==0]).log()

        assert len(loss_pos) != 0 or len(loss_neg) != 0, 'Invalid loss.'
        # operate mean operation on an empty list will lead to NaN
        if len(loss_pos) == 0:
            loss_mean = self.neg_weights * loss_neg.mean()
        elif len(loss_neg) == 0:
            loss_mean = self.pos_weights * loss_pos.mean()
        else:
            loss_mean = self.pos_weights * loss_pos.mean() + self.neg_weights * loss_neg.mean()
        
        return loss_mean