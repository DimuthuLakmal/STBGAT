import torch
from torch import nn

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.decoder_block import DecoderBlock
from models.transformer.token_embedding import TokenEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, configs):

        super(TransformerDecoder, self).__init__()

        self.emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
        self.merge_emb = configs['merge_emb']
        max_lookup_len = configs['max_lookup_len']
        self.lookup_idx = configs['lookup_idx']

        n_layers = configs['n_layers']
        out_dim = configs['out_dim']

        self.offset = configs['seq_offset']
        self.seq_len = configs['seq_len']
        self.enc_features = configs['enc_features']
        self.per_enc_feature_len = configs['per_enc_feature_len']
        self.cross_attn_features = configs['decoder_block']['cross_attn_features']

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=self.emb_dim)
        configs['sgat']['seq_len'] = configs['seq_len']
        self.graph_embedding = SGATEmbedding(configs['sgat'])
        self.graph_embedding_semantic = SGATEmbedding(configs['sgat'])
        # by merging embeddings we increase the num embeddings
        if self.merge_emb:
            self.emb_dim = self.emb_dim * 3
        self.position_embedding = PositionalEmbedding(max_lookup_len, self.emb_dim)

        # convolution related
        self.local_trends = configs['local_trends']

        self.conv_q_layer = nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1)
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        padding_size = 1
        if self.offset == 1:
            padding_size = 2
        self.conv_q_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=(1, 3), stride=1,
                      padding=(0, padding_size), bias=False)
            for _ in range(n_layers)
        ])

        self.conv_k_layers = nn.ModuleList([
            nn.ModuleList(
                [nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1) for i in
                 range(self.cross_attn_features)])
            for j in range(n_layers)
        ])

        configs['decoder_block']['emb_dim'] = self.emb_dim
        self.layers = nn.ModuleList(
            [
                DecoderBlock(configs['decoder_block'])
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(self.emb_dim, out_dim)

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

    def forward(self, x, graph_x, graph_x_semantic, enc_x, tgt_mask, device):
        embed_x = None
        embed_graph_x = None
        embed_graph_x_semantic = None
        if x is not None:
            embed_x = self.embedding(x)
        if graph_x is not None:
            embed_graph_x = self.graph_embedding(graph_x).transpose(0, 1)
        if graph_x_semantic is not None:
            embed_graph_x_semantic = self.graph_embedding_semantic(graph_x_semantic).transpose(0, 1)

        embed_out = None
        if embed_graph_x is not None and embed_graph_x_semantic is not None and embed_x is not None:
            if self.merge_emb:
                embed_out = torch.concat((embed_x, embed_graph_x), dim=-1)
            else:
                embed_out = embed_x + embed_graph_x + embed_graph_x_semantic
                embed_out = self.emb_norm(embed_out)
        elif embed_graph_x is None and embed_x is not None:
            embed_out = embed_x
        elif embed_graph_x is not None and embed_x is None:
            embed_out = embed_graph_x

        embed_out = embed_out.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
        embed_shp = embed_out.shape
        embed_out = embed_out.reshape(embed_shp[0], embed_shp[1] * embed_shp[2], embed_shp[3])  # (36, 4 * 170, 16)
        embed_out = embed_out.permute(1, 0, 2)

        x = self.position_embedding(embed_out, self.lookup_idx)  # 32x10x512

        tgt_mask_conv = self.create_conv_mask(x, device)

        for idx, layer in enumerate(self.layers):
            if self.local_trends:
                # x = self.conv_q_layer(x.transpose(2, 1)).transpose(2, 1)
                x = self.calculate_masked_src(x, self.conv_q_layers[idx], tgt_mask_conv, device)

            enc_xs = []
            for idx_k, f_layer in enumerate(self.conv_k_layers[idx]):
                if self.enc_features > 1:
                    enc_xs.append(f_layer(enc_x[idx_k].transpose(2, 1)).transpose(2, 1))
                else:
                    start = idx_k * self.per_enc_feature_len
                    enc_xs.append(
                        f_layer(enc_x[0][:, start: start + self.per_enc_feature_len].transpose(2, 1)).transpose(2, 1))

            x = layer(x, enc_xs, tgt_mask)

        out = self.fc_out(x)

        out = out.permute(1, 0, 2)
        out = out.reshape(embed_shp[0], embed_shp[1], embed_shp[2], out.shape[-1])
        out = out.permute(1, 0, 2, 3)

        return out
