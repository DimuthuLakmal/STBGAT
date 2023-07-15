from torch import nn
from torch.nn import MultiheadAttention

from models.transformer.cross_attention import CrossAttentionLayer
from models.transformer.position_wise_feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout=0.2, n_cross_attn_layers=5):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, n_heads, dropout=dropout) for _ in range(n_cross_attn_layers)
        ])
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = PositionWiseFeedForward(embed_dim, expansion_factor * embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, enc_x, tgt_mask):
        # self attention
        attention = self.self_attention(x, x, x, attn_mask=tgt_mask)  # 32x10x512
        x = self.norm1(x + self.dropout1(attention[0]))

        # cross attention
        cross_attn = None
        for idx, layer in enumerate(self.cross_attn_layers):
            if idx == 0:
                cross_attn = layer(x, enc_x[idx], enc_x[idx])
            else:
                cross_attn += layer(x, enc_x[idx], enc_x[idx])
        x = self.norm2(x + cross_attn)

        # positionwise ffn
        ff_output = self.feed_forward(x)
        out = self.norm3(x + self.dropout2(ff_output))

        return out
