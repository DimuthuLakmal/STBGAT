from torch import nn
from torch.nn import MultiheadAttention


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        return self.dropout(self.cross_attn(query, key, value)[0])
