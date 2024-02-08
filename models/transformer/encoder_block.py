from torch import nn

from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.position_wise_feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, configs):
        super(EncoderBlock, self).__init__()

        emb_dim = configs['emb_dim']
        n_heads = configs['n_heads']
        expansion_factor = configs['expansion_factor']
        src_dropout = configs['src_dropout']
        ff_dropout = configs['ff_dropout']

        self.attention = MultiHeadAttention(emb_dim, n_heads, mask=False)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = PositionWiseFeedForward(emb_dim, expansion_factor * emb_dim)

        self.dropout1 = nn.Dropout(src_dropout)
        self.dropout2 = nn.Dropout(ff_dropout)

    def forward(self, query, key, value):
        # self attention
        attention_out = self.attention(query, key, value)  # 32x10x512

        # add and normalization
        attention_residual_out = self.dropout1(attention_out) + value  # 32x10x512
        norm1_out = self.norm1(attention_residual_out)  # 32x10x512

        # positionwise ffn
        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512

        # add and normalization
        feed_fwd_residual_out = self.dropout2(feed_fwd_out) + norm1_out  # 32x10x512
        norm2_out = self.norm2(feed_fwd_residual_out)  # 32x10x512

        return norm2_out
