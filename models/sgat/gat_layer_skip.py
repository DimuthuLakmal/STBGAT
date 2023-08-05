import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

from models.sgat.gat_conv.gat_conv_v4.gat_conv_v4 import GATConvV4
from models.sgat.gat_conv.gat_conv_v6.gat_conv_v6 import GATConvV6
from models.sgat.gat_conv.gat_conv_v7.gat_conv_v7 import GATConvV7


class GATV2(nn.Module):
    def __init__(self, configs):
        super(GATV2, self).__init__()

        n_layers = configs['n_layers']
        dropout = configs['dropout']
        first_in_f_size = configs['first_in_f_size']
        edge_dim = configs['edge_dim']

        self.n_layers = n_layers
        self.dropout = dropout

        # Skip connection correction temp
        self.hid = 16
        self.hid2 = 16
        self.hid3 = 64
        self.in_head = 8
        self.in_head2 = 4
        self.out_head = 1
        self.dropout_skip = 0.8  # 0.7

        self.conv1 = GATConvV4(first_in_f_size, self.hid, heads=self.in_head, dropout=dropout, edge_dim=edge_dim)

        # 2nd layer
        self.conv2 = GATConvV4(self.hid * self.in_head, self.hid2, heads=self.out_head, dropout=dropout, concat=False,
                               edge_dim=edge_dim)
        self.skip_conv = GATConvV6(self.hid * self.in_head, self.hid2, heads=self.out_head, dropout=self.dropout_skip,
                                   concat=False, edge_dim=edge_dim)

        # 3rd layer
        self.conv3 = GATConvV4(self.hid2 * self.in_head2, self.hid3, heads=self.out_head, dropout=dropout, concat=False,
                               edge_dim=edge_dim)

        self.skip_conv2 = GATConvV6(self.hid2 * self.in_head2, self.hid3, heads=self.out_head, dropout=self.dropout_skip,
                                   concat=False, edge_dim=edge_dim)

        self.lin_skip_conv = Linear(self.hid2, self.hid2)
        # self.lin_skip_conv = Linear(self.hid2 * self.in_head2, self.hid2 * self.in_head2)
        self.lin_skip_conv2 = Linear(self.hid3, self.hid3)
        # self.lin_skip_conv3 = Linear(2 * self.hid2, self.hid2)

        self.norm = nn.LayerNorm(self.hid2)
        self.norm2 = nn.LayerNorm(self.hid3)

        # self.layer_stack = nn.ModuleList()
        # for l in range(n_layers):
        #     in_f_size = out_f_sizes[l - 1] * n_heads[l - 1] if l else first_in_f_size
        #     concat = True if l < (n_layers - 1) else False
        #     gat_layer = GATLayerV4(in_f_size, out_f_sizes[l], n_heads=n_heads[l], dropout=dropout, alpha=alpha,
        #                            concat=concat, layer_index=l, edge_dim=edge_dim, dropout_skip=dropout_skip)
        #     self.layer_stack.append(gat_layer)

    def forward(self, batch_data):
        # ToDo: implement using pytorch geometric batch data
        out = ()
        for data in batch_data:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            # x, edge_index = data.x, data.edge_index

            x = F.dropout(x, p=self.dropout, training=self.training)
            x, skip_out_prev = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
            x_ = x
            x = F.elu(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x1, skip_out_prev2 = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)

            x2 = self.skip_conv(x, x_skip=skip_out_prev, edge_index=edge_index, edge_attr=edge_attr)
            x2 = F.dropout(x2, p=self.dropout_skip, training=self.training)  # 0.7
            x2 = self.lin_skip_conv(x2)
            x = x1 + x2
            x = self.norm(x)
            x_ = x
            # x = F.elu(x)

            # # 3rd layer
            # x = F.dropout(x, p=self.dropout, training=self.training)
            # attn_edge = self.conv_edge3(x, edge_index=edge_index, edge_attr=edge_attr)
            # x3, skip_out = self.conv3(x, edge_index=edge_index, edge_attr=edge_attr, attn_edge=attn_edge)
            #
            # x4 = self.skip_conv2(x, x_skip=skip_out_prev2, edge_index=edge_index, edge_attr=edge_attr)
            # x4 = F.dropout(self.lin_skip_conv2(x4), p=0.5, training=self.training)
            # x = x3 + x4
            # x = F.elu(x)

            out = (*out, x)

        return torch.stack(out)

    # def forward(self, batch_data):
    #     out = ()
    #     for data in batch_data:
    #         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    #
    #         skip_out = None
    #         for l, gat_layer in enumerate(self.layer_stack):
    #             x, skip_out = gat_layer(x, edge_attr=edge_attr, edge_index=edge_index, skip_out_prev=skip_out)
    #             if l < (self.n_layers - 1):
    #                 x = F.elu(x)
    #
    #         out = (*out, x)
    #
    #     return torch.stack(out)
