# https://arxiv.org/pdf/1708.02002.pdf
# https://amaarora.github.io/2020/06/29/FocalLoss.html#how-to-implement-this-in-code

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2., reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha #torch.tensor([alpha, 1-alpha]).cuda()
        self.reduction = reduction

    def forward(self, pred, target):
        target_one_hot = F.one_hot(target, num_classes=2)

        CE_loss = F.cross_entropy(pred, target_one_hot.float(), reduction='none')
        # at = self.alpha.gather(0, target.data.view(-1))
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha*(1-pt)**self.gamma * CE_loss

        # pred = pred[:,0]
        # weight = torch.pow(1.0-pred, self.gamma)
        # loss = -1 * target * torch.log(pred)
        # focal_loss = self.alpha * weight * F.log_softmax(pred, dim=1) #at*((1 - pt) ** self.gamma * ce_loss).mean()

        # weight = torch.pow(1.0-pred, self.gamma)
        # focal_loss = self.alpha * weight * F.log_softmax(pred, dim=1) #at*((1 - pt) ** self.gamma * ce_loss).mean()
        
        if self.reduction == 'none':
            focal_loss = focal_loss
        elif self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)
        else:
            raise NotImplementedError(f"invalid reduction mode: {self.reduction}")

        return focal_loss


# TODO: need to separate the positive and negative labels before feeding here
# Would call mse twice for each combo before calling the forward pass
# Then sum up all msfe loss
# http://203.170.84.89/~idawis33/DataScienceLab/publication/IJCNN15.wang.final.pdf
class MSFELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(MSFELoss, self).__init__()
        self.reduction = reduction
        self.FPE = 0
        self.FNE = 0

    # TODO beforehand need to change target to either 0/1
    def mse(self, pred, target, f_positive=True):
        if f_positive:
            self.FNE = F.mse_loss(pred, target)
        else:
            self.FPE = F.mse_loss(pred, target)

    def forward(self):
        msfe_loss = self.FNE**2 + self.FPE**2
        return msfe_loss