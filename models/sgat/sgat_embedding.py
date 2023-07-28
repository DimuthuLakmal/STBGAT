import torch
from torch import nn

from models.sgat.gat import GAT
from models.sgat.gat_layer_skip import GATV2


class SGATEmbedding(nn.Module):
    def __init__(self, n_layers, first_in_f_size, out_f_sizes, n_heads, alpha, dropout, edge_dim, seq_len):
        super(SGATEmbedding, self).__init__()
        self.gats = nn.ModuleList([
            GATV2(n_layers=n_layers,
                  first_in_f_size=first_in_f_size,
                  out_f_sizes=out_f_sizes,
                  n_heads=n_heads,
                  alpha=alpha,
                  dropout=dropout,
                  edge_dim=edge_dim) for _ in range(seq_len)
        ])

        # self.gat = GATV2(n_layers=n_layers,
        #                  first_in_f_size=first_in_f_size,
        #                  out_f_sizes=out_f_sizes,
        #                  n_heads=n_heads,
        #                  alpha=alpha,
        #                  dropout=dropout,
        #                  edge_dim=edge_dim)

    def forward(self, x):
        seq_size = len(x[0])

        hidden_f = ()
        for t in range(seq_size):
            batch_data = list(zip(*x))[t]
            hidden_f = (*hidden_f, self.gats[t](batch_data))

        return torch.stack(hidden_f)
