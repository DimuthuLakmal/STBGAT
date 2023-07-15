import torch


class Masked_MAE_Loss(torch.nn.Module):
    def __init__(self):
        super(Masked_MAE_Loss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(v_ - v)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
