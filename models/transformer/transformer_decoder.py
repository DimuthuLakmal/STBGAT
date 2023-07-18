import torch
from torch import nn

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.decoder_block import DecoderBlock
from models.transformer.token_embedding import TokenEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, out_dim, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, dropout=0.2,
                 enc_features=5, sgat_settings=None, merge_embed=False, max_lookup_len=0,
                 cross_attn_features=True, per_enc_feature_len=12, offset=1):

        super(TransformerDecoder, self).__init__()

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=embed_dim)
        self.graph_embedding = SGATEmbedding(n_layers=sgat_settings['n_layers'],
                                             first_in_f_size=sgat_settings['first_in_f_size'],
                                             out_f_sizes=sgat_settings['out_f_sizes'],
                                             n_heads=sgat_settings['n_heads'],
                                             alpha=sgat_settings['alpha'],
                                             dropout=sgat_settings['dropout'],
                                             edge_dim=sgat_settings['edge_dim'])
        self.position_embedding = PositionalEmbedding(max_lookup_len, embed_dim)
        # by merging embeddings we increase the num embeddings
        self.merge_embed = merge_embed
        if merge_embed:
            embed_dim = embed_dim * 2

        self.conv_q_layer = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

        self.conv_q_layers = nn.ModuleList([
                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
                for _ in range(num_layers)
            ])
        # decoder input masking for convolution operation
        self.offset = offset
        self.seq_len = seq_len
        self.emb_dim = embed_dim

        self.enc_features = enc_features
        self.per_enc_feature_len = per_enc_feature_len
        self.cross_attn_features = cross_attn_features
        self.conv_k_layers = nn.ModuleList([
            nn.ModuleList(
                [nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1) for i in
                 range(cross_attn_features)])
            for j in range(num_layers)
        ])

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads, dropout=dropout,
                             n_cross_attn_layers=cross_attn_features)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, out_dim)

    def calculate_masked_src(self, x, conv_q, tgt_mask, device='cuda'):
        batch_size = x.shape[0]

        x = x.repeat(1, self.seq_len, 1).view(batch_size, self.seq_len, self.seq_len, self.emb_dim)

        x = tgt_mask.transpose(2, 3) * x
        x = x.permute(0, 3, 1, 2)
        x = conv_q(x)

        out = torch.zeros((batch_size, self.emb_dim, self.seq_len)).to(device)
        for i in range(self.seq_len):
            out[:, :, i] = x[:, :, i, i]

        return out.permute(0, 2, 1)

    def create_conv_mask(self, x, device='cuda'):
        tri = torch.tril(torch.ones((self.seq_len, self.seq_len)))
        tri[:, :self.offset] = 1
        tgt_mask_conv = tri.repeat(1, self.emb_dim).view(-1, self.emb_dim, self.seq_len)
        tgt_mask_conv = tgt_mask_conv.expand(x.shape[0], self.seq_len, self.emb_dim, self.seq_len).to(device)
        return tgt_mask_conv

    def forward(self, x, graph_x, enc_x, tgt_mask, lookup_idx=None, local_trends=True, device='cuda'):
        embed_x = None
        embed_graph_x = None
        if x is not None:
            embed_x = self.embedding(x)
        if graph_x is not None:
            embed_graph_x = self.graph_embedding(graph_x).transpose(2, 1)

        embed_out = None
        if embed_graph_x is not None and embed_x is not None and self.merge_embed:
            embed_out = torch.concat((embed_x, embed_graph_x), dim=-1)
        elif embed_graph_x is None and embed_x is not None:
            embed_out = embed_x
        elif embed_graph_x is not None and embed_x is None:
            embed_out = embed_graph_x

        embed_out = embed_out.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
        embed_shp = embed_out.shape
        embed_out = embed_out.reshape(embed_shp[0], embed_shp[1] * embed_shp[2], embed_shp[3])  # (36, 4 * 170, 16)
        embed_out = embed_out.permute(1, 0, 2)

        x = self.position_embedding(embed_out, lookup_idx)  # 32x10x512

        tgt_mask_conv = self.create_conv_mask(x, device)

        for idx, layer in enumerate(self.layers):
            if local_trends:
                # x = self.conv_q_layer(x.transpose(2, 1)).transpose(2, 1)
                x = self.calculate_masked_src(x, self.conv_q_layers[idx], tgt_mask_conv, device)

            enc_xs = []
            for idx_k, f_layer in enumerate(self.conv_k_layers[idx]):
                if self.enc_features > 1:
                    enc_xs.append(f_layer(enc_x[idx_k].transpose(2, 1)).transpose(2, 1))
                else:
                    start = idx_k * self.per_enc_feature_len
                    enc_xs.append(f_layer(enc_x[0][:, start: start + self.per_enc_feature_len].transpose(2, 1)).transpose(2, 1))

            x = layer(x, enc_xs, tgt_mask)

        out = self.fc_out(x)

        out = out.permute(1, 0, 2)
        out = out.reshape(embed_shp[0], embed_shp[1], embed_shp[2], out.shape[-1])
        out = out.permute(1, 0, 2, 3)

        return out
