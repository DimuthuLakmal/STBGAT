import torch
from torch import nn

from models.sgat.gat import GAT
from models.sgat.gat_layer_skip import GATV2


class SGATEmbedding(nn.Module):
    def __init__(self, sgat_configs):
        super(SGATEmbedding, self).__init__()

        if sgat_configs['skip_conn']:
            self.gats = nn.ModuleList([
                GATV2(sgat_configs) for _ in range(sgat_configs['seq_len'])
            ])
        else:
            self.gats = GAT(sgat_configs)

        # self.gat = GATV2(n_layers=n_layers,
        #                  first_in_f_size=first_in_f_size,
        #                  out_f_sizes=out_f_sizes,
        #                  n_heads=n_heads,
        #                  alpha=alpha,
        #                  dropout=dropout,
        #                  edge_dim=edge_dim)

    def forward(self, x):
        out = self.gats(x)
        return out