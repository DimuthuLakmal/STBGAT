import torch
from torch import nn
import torch.nn.functional as F

from models.sgat.gat_layer import GATLayer


class GAT(nn.Module):
    def __init__(self, n_layers, first_in_f_size, out_f_sizes, edge_dim=1, n_heads=8, alpha=0.2, dropout=0.2):
        super(GAT, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            in_f_size = out_f_sizes[l - 1] * n_heads[l - 1] if l else first_in_f_size
            concat = True if l < (n_layers - 1) else False
            gat_layer = GATLayer(in_f_size, out_f_sizes[l], n_heads=n_heads[l], dropout=dropout, alpha=alpha,
                                 concat=concat, edge_dim=edge_dim)
            self.layer_stack.append(gat_layer)

    def forward(self, batch_data):
        # ToDo: implement using pytorch geometric batch data
        out = ()
        for data in batch_data:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            for l, gat_layer in enumerate(self.layer_stack):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = gat_layer(x, edge_attr=edge_attr, edge_index=edge_index)
                if l < (self.n_layers - 1):
                    x = F.elu(x)

            out = (*out, x)

        return torch.stack(out)
