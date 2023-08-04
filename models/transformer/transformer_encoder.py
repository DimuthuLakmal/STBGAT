import torch
from torch import nn

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.encoder_block import EncoderBlock
from models.transformer.token_embedding import TokenEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, configs: dict):
        super(TransformerEncoder, self).__init__()

        emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
        self.merge_emb = configs['merge_emb']
        emb_expansion_factor = configs['emb_expansion_factor']
        max_lookup_len = configs['max_lookup_len']
        self.lookup_idx = configs['lookup_idx']

        n_layers = configs['n_layers']

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=emb_dim)
        configs['sgat']['seq_len'] = configs['seq_len']
        self.graph_embedding = SGATEmbedding(configs['sgat'])
        self.graph_embedding_semantic = SGATEmbedding(configs['sgat'])

        # convolution related
        self.local_trends = configs['local_trends']

        # by merging embeddings we increase the num embeddings
        if self.merge_emb:
            emb_dim = emb_dim * emb_expansion_factor
        self.positional_encoder = PositionalEmbedding(max_lookup_len, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)

        # to do local trend analysis
        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(n_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(n_layers)])

        configs['encoder_block']['emb_dim'] = emb_dim
        self.layers = nn.ModuleList(
            [EncoderBlock(configs['encoder_block']) for i in range(n_layers)])

    def forward(self, x, graph_x, graph_x_semantic):
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

        out = self.positional_encoder(embed_out, self.lookup_idx)
        for (layer, conv_q, conv_k) in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
            if self.local_trends:
                out_transposed = out.transpose(2, 1)
                q = conv_q(out_transposed).transpose(2, 1)
                k = conv_k(out_transposed).transpose(2, 1)
                v = out
            else:
                q, k, v = out, out, out

            out = layer(q, k, v)

        return out  # 32x10x512
