import torch
from torch import nn, Tensor

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.encoder_block import EncoderBlock
from models.transformer.token_embedding import TokenEmbedding

from torch_geometric.transforms import ToDevice
import torch_geometric.data as data


class TransformerEncoder(nn.Module):
    def __init__(self, configs: dict):
        super(TransformerEncoder, self).__init__()

        self.emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
        dropout_e = configs['dropout_e']
        max_lookup_len = configs['max_lookup_len']
        self.lookup_idx = configs['lookup_idx']

        # embedding and positional encoder
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=self.emb_dim)
        self.positional_encoder = PositionalEmbedding(max_lookup_len, self.emb_dim)

        # convolution related (to do local trend analysis)
        self.local_trends = configs['local_trends']
        n_layers = configs['n_layers']
        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(n_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1)
             for _ in range(n_layers)])

        # encoder attention blocks
        configs['encoder_block']['emb_dim'] = self.emb_dim
        self.layers = nn.ModuleList(
            [EncoderBlock(configs['encoder_block']) for i in range(n_layers)])

        # graph related
        self.device = configs['device']
        self.edge_index = configs['edge_index']
        self.edge_attr = configs['edge_attr']
        self.edge_details = configs['edge_details']
        self.graph_input = configs['graph_input']
        self.graph_semantic_input = configs['graph_semantic_input']
        graph_out_size = configs['sgat']['out_f_sizes'][-1]
        configs['sgat']['seq_len'] = configs['seq_len']

        self.graph_embedding = SGATEmbedding(configs['sgat'])
        self.graph_embedding_semantic = SGATEmbedding(configs['sgat'])

        # encoder output layers
        self.out_norm = nn.LayerNorm(graph_out_size)
        self.out_e_lin = nn.Linear(self.emb_dim, graph_out_size)
        self.dropout_e = nn.Dropout(dropout_e)

    def _create_graph(self, x, edge_index, edge_attr):
        graph = data.Data(x=(Tensor(x[0]), Tensor(x[1])),
                          edge_index=torch.LongTensor(edge_index),
                          y=None,
                          edge_attr=Tensor(edge_attr))
        return graph

    def _derive_graphs(self, x_batch):
        to = ToDevice(self.device)

        x_batch_graphs = []
        x_batch_graphs_semantic = []
        for idx, x_all_t in enumerate(x_batch):
            semantic_edge_index, semantic_edge_attr = self.edge_details
            x_src = x_all_t.permute(1, 0, 2)  # N, T, F
            x_src = x_src.reshape(x_src.shape[0], -1)  # N, T*F

            if self.graph_input:
                graph = self._create_graph((x_src, x_src), self.edge_index, self.edge_attr)
                x_batch_graphs.append(to(graph))

            if self.graph_semantic_input:
                graph_semantic = self._create_graph((x_src, x_src), semantic_edge_index, semantic_edge_attr)
                x_batch_graphs_semantic.append(to(graph_semantic))

        return x_batch_graphs, x_batch_graphs_semantic

    def _organize_matrix(self, mat):
        mat = mat.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
        mat_shp = mat.shape
        mat = mat.reshape(mat_shp[0], mat_shp[1] * mat_shp[2], mat_shp[3])  # (36, 4 * 170, 16)
        mat = mat.permute(1, 0, 2)  # (4 * 170, 36, 16)
        return mat

    def forward(self, x, enc_idx):
        embed_out = self.embedding(x)
        embed_out = self._organize_matrix(embed_out)

        out_e = self.positional_encoder(embed_out, self.lookup_idx)
        for (layer, conv_q, conv_k) in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
            if self.local_trends:
                out_transposed = out_e.transpose(2, 1)
                q = conv_q(out_transposed).transpose(2, 1)
                k = conv_k(out_transposed).transpose(2, 1)
                v = out_e
            else:
                q, k, v = out_e, out_e, out_e

            out_e = layer(q, k, v)

        if enc_idx == 0:
            graph_x = out_e

            graph_x = graph_x.reshape(x.shape[0], x.shape[2], x.shape[1], graph_x.shape[-1])
            graph_x = graph_x.permute(0, 2, 1, 3)
            out_g_dis, out_g_semantic = self._derive_graphs(graph_x)

            if self.graph_input:
                out_g_dis = self.graph_embedding(out_g_dis)  # (4, 307, 64)
            if self.graph_semantic_input:
                out_g_semantic = self.graph_embedding_semantic(out_g_semantic) # (4, 307, 64)

            if self.graph_input and self.graph_semantic_input:
                out_g = self.out_norm(out_g_dis + out_g_semantic)
            elif self.graph_input and not self.graph_semantic_input:
                out_g = out_g_dis
            elif not self.graph_input and self.graph_semantic_input:
                out_g = out_g_semantic
            elif not self.graph_input and not self.graph_semantic_input:
                out_e = self.dropout_e(self.out_e_lin(out_e))
                return out_e

            out = self.dropout_e(self.out_e_lin(out_e)) + self._organize_matrix(out_g)
            return out  # 32x10x512

        else:
            out_e = self.dropout_e(self.out_e_lin(out_e))
            return out_e