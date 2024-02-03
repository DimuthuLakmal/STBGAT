import torch
from torch import nn

from models.sgat.gat import GAT


class SGATEmbedding(nn.Module):
    def __init__(self, sgat_configs):
        super(SGATEmbedding, self).__init__()
        self.gats = GAT(sgat_configs)

    def forward(self, x):
        out = self.gats(x)
        return out