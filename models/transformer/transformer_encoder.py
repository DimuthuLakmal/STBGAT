import torch
from torch import nn

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.encoder_block import EncoderBlock
from models.transformer.token_embedding import TokenEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, sgat_settings=None,
                 merge_embed=False, dropout=0.2, max_lookup_len=0):
        super(TransformerEncoder, self).__init__()

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=embed_dim)
        self.graph_embedding = SGATEmbedding(n_layers=sgat_settings['n_layers'],
                                             first_in_f_size=sgat_settings['first_in_f_size'],
                                             out_f_sizes=sgat_settings['out_f_sizes'],
                                             n_heads=sgat_settings['n_heads'],
                                             alpha=sgat_settings['alpha'],
                                             dropout=sgat_settings['dropout'],
                                             edge_dim=sgat_settings['edge_dim'],
                                             seq_len=seq_len)

        self.graph_embedding_semantic = SGATEmbedding(n_layers=sgat_settings['n_layers'],
                                                      first_in_f_size=sgat_settings['first_in_f_size'],
                                                      out_f_sizes=sgat_settings['out_f_sizes'],
                                                      n_heads=sgat_settings['n_heads'],
                                                      alpha=sgat_settings['alpha'],
                                                      dropout=sgat_settings['dropout'],
                                                      edge_dim=sgat_settings['edge_dim'],
                                                      seq_len=seq_len)

        # by merging embeddings we increase the num embeddings
        self.merge_embed = merge_embed
        # if merge_embed:
        #     embed_dim = embed_dim * 3

        self.positional_encoder = PositionalEmbedding(max_lookup_len, embed_dim)

        # to do local trend analysis
        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(num_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(num_layers)])

        self.layers = nn.ModuleList(
            [EncoderBlock(embed_dim, expansion_factor, n_heads, dropout) for i in range(num_layers)])

        self.temp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_x, graph_x_semantic, lookup_idx=None, local_trends=True):
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
        if embed_graph_x is not None and embed_graph_x_semantic is not None and embed_x is not None and self.merge_embed:
            # embed_out = torch.concat((embed_x, embed_graph_x), dim=-1)
            embed_out = embed_x + embed_graph_x + embed_graph_x_semantic
            embed_out = self.temp_norm(embed_out)
        elif embed_graph_x is None and embed_x is not None:
            embed_out = embed_x
        elif embed_graph_x is not None and embed_x is None:
            embed_out = embed_graph_x

        embed_out = embed_out.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
        embed_shp = embed_out.shape
        embed_out = embed_out.reshape(embed_shp[0], embed_shp[1] * embed_shp[2], embed_shp[3])  # (36, 4 * 170, 16)
        embed_out = embed_out.permute(1, 0, 2)

        out = self.positional_encoder(embed_out, lookup_idx)
        for (layer, conv_q, conv_k) in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
            if local_trends:
                out_transposed = out.transpose(2, 1)
                q = conv_q(out_transposed).transpose(2, 1)
                k = conv_k(out_transposed).transpose(2, 1)
                v = out
            else:
                q, k, v = out, out, out

            out = layer(q, k, v)

        return out  # 32x10x512
