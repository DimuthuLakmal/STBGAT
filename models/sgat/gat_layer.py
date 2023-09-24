from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv

from models.sgat.gat_conv.gat_conv_v8.gat_conv_v8 import GATConvV8


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout, alpha=0.2, concat=True, edge_dim=-1, seq_len=36):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.
        self.n_heads = n_heads
        self.seq_len = seq_len

        if edge_dim == -1:
            self.conv = GATConvV8([in_features, in_features], out_features, heads=n_heads, dropout=dropout,
                                  concat=concat, seq_len=seq_len)
        else:
            self.conv = GATConvV8([in_features, in_features], out_features, heads=n_heads, dropout=dropout,
                                  concat=concat, edge_dim=edge_dim, seq_len=seq_len)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        return x