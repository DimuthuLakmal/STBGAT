from torch import nn
from torch.nn import MultiheadAttention

from models.transformer.position_wise_feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout=0.2):
        super(EncoderBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = PositionWiseFeedForward(embed_dim, expansion_factor * embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # self attention
        attention_out = self.attention(query, key, value)[0]  # 32x10x512

        # add and normalization
        attention_residual_out = self.dropout1(attention_out) + value  # 32x10x512
        norm1_out = self.norm1(attention_residual_out)  # 32x10x512

        # positionwise ffn
        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512

        # add and normalization
        feed_fwd_residual_out = self.dropout2(feed_fwd_out) + norm1_out  # 32x10x512
        norm2_out = self.norm2(feed_fwd_residual_out)  # 32x10x512

        return norm2_out
