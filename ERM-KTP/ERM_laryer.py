import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LearnableMaskLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask = torch.nn.Parameter(torch.full((feature_dim,num_classes),0.5))

    def get_channel_mask(self):
        c_mask = self.mask
        return c_mask

    def get_density(self):
        return torch.norm(self.mask, p=1)/torch.numel(self.mask)

    def get_CSI(self):
        csi = 0
        mask = self.mask.transpose(0,1)
        for idx in range(mask.size(0)):
            x = mask[idx].view(1,-1)
            for idy in range(mask.size(0)):
                if idx != idy:
                    y = mask[idy].view(1,-1)
                    csi += torch.cosine_similarity(x, y, dim=-1)
        return csi

    def _icnn_mask(self, x, labels):
        if self.training:
            index_mask = torch.zeros(x.shape, device=x.device)
            for idx, la in enumerate(labels):
                index_mask[idx, :, :, :] = self.mask[:, la].view(-1, self.mask.shape[0], 1, 1)
            return index_mask * x
        else:
            return x

    def loss_function(self):
        mask = self.mask.view(self.mask.size(0), -1)
        inner_product = torch.triu(torch.mm(mask.transpose(0,1), mask), diagonal=1).sum()

        l1_reg = torch.norm(self.mask, p=1)
        l1_reg = torch.relu(l1_reg - torch.numel(self.mask) * 0.1)
        return l1_reg + 0.1 * inner_product

    def channel_express(self, x, target):
        mask = (torch.where(self.mask[:,target]>0, torch.zeros_like(self.mask[:,target]), torch.ones_like(self.mask[:,target])))
        if isinstance (target, list):
            mask = (torch.sum(mask, dim=1)==len(target))
        
        mask = mask.view(-1,1,1).repeat(x.size(0),1,1,1)
        
        return x * mask

    def clip_lmask(self):

        lmask = self.mask
        lmask = lmask / torch.max(lmask, dim=1)[0].view(-1, 1)
        lmask = torch.clamp(lmask, min=0, max=1)
        self.mask.data = lmask

    def forward(self, x, labels, last_layer_mask=None):
        if (last_layer_mask is not None):
            self.last_layer_mask = last_layer_mask


        x = self._icnn_mask(x, labels)

        return x, self.loss_function()