from torch import nn

from models.transformer.multi_head_attention import MultiHeadAttention


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads, mask=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        return self.dropout(self.cross_attn(query, key, value))
