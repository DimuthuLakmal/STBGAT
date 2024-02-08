import torch
from torch import nn, Tensor

from models.sgat.gat import GAT
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.encoder_block import EncoderBlock
from models.transformer.token_embedding import TokenEmbedding

from torch_geometric.transforms import ToDevice
import torch_geometric.data as data

from utils.transformer_utils import organize_matrix


class TransformerEncoder(nn.Module):
    def __init__(self, configs: dict):
        super(TransformerEncoder, self).__init__()

        self.emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
        dropout_e_rep = configs['dropout_e_rep']
        dropout_e_normal = configs['dropout_e_normal']
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
        self.sem_edge_details = configs['sem_edge_details']
        self.graph_input = configs['graph_input']
        self.graph_semantic_input = configs['graph_semantic_input']
        graph_out_size = configs['sgat']['out_f_sizes'][-1]
        configs['sgat']['seq_len'] = configs['seq_len']
        configs['sgat']['dropout_g'] = configs['sgat']['dropout_g_dis']
        if self.graph_input:
            self.graph_embedding_dis = GAT(configs['sgat'])
        configs['sgat']['dropout_g'] = configs['sgat']['dropout_g_sem']
        if self.graph_semantic_input:
            self.graph_embedding_semantic = GAT(configs['sgat'])

        # encoder output layers
        self.out_norm = nn.LayerNorm(graph_out_size)
        self.out_e_lin = nn.Linear(self.emb_dim, graph_out_size)
        self.dropout_e_rep = nn.Dropout(dropout_e_rep)
        self.dropout_e_normal = nn.Dropout(dropout_e_normal)

    def _create_graph(self, x, edge_index, edge_attr):
        graph = data.Data(x=(Tensor(x[0]), Tensor(x[1])),
                          edge_index=torch.LongTensor(edge_index),
                          y=None,
                          edge_attr=Tensor(edge_attr))
        return graph

    def _derive_graphs(self, x_batch):
        """
        Create graph structure for distance based on semantic graphs (nodes and edges with attributes)
        Parameters
        ----------
        x_batch: Tensor, nodes data

        Returns
        -------
        (x_batch_graphs, x_batch_graphs_semantic): tuple(torch_geometric.data.Data, torch_geometric.data.Data)
        """
        to = ToDevice(self.device)

        x_batch_graphs = []
        x_batch_graphs_semantic = []
        for idx, x_all_t in enumerate(x_batch):
            x_src = x_all_t.permute(1, 0, 2)  # N, T, F
            x_src = x_src.reshape(x_src.shape[0], -1)  # N, T*F

            if self.graph_input:
                graph = self._create_graph((x_src, x_src), self.edge_index, self.edge_attr)
                x_batch_graphs.append(to(graph))

            if self.graph_semantic_input:
                semantic_edge_index, semantic_edge_attr = self.sem_edge_details
                graph_semantic = self._create_graph((x_src, x_src), semantic_edge_index, semantic_edge_attr)
                x_batch_graphs_semantic.append(to(graph_semantic))

        return x_batch_graphs, x_batch_graphs_semantic

    def forward(self, x, enc_idx):
        embed_out = self.embedding(x)
        embed_out_shp = embed_out.shape
        embed_out = organize_matrix(embed_out)

        out_e = self.positional_encoder(embed_out, self.lookup_idx)
        for (layer, conv_q, conv_k) in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
            if self.local_trends:
                out_e = out_e.view(-1, embed_out_shp[1], embed_out_shp[3])
                out_transposed = out_e.transpose(2, 1)
                q = conv_q(out_transposed).transpose(2, 1)
                k = conv_k(out_transposed).transpose(2, 1)
                v = out_e
            else:
                q, k, v = out_e, out_e, out_e

            q = q.reshape(embed_out_shp[0], embed_out_shp[2], embed_out_shp[1], embed_out_shp[3])
            v = v.reshape(embed_out_shp[0], embed_out_shp[2], embed_out_shp[1], embed_out_shp[3])
            k = k.reshape(embed_out_shp[0], embed_out_shp[2], embed_out_shp[1], embed_out_shp[3])
            out_e = layer(q, k, v)  # output of temporal encoder layer

        if enc_idx == 0:
            graph_x = out_e

            graph_x = graph_x.reshape(x.shape[0], x.shape[2], x.shape[1], graph_x.shape[-1])
            graph_x = graph_x.permute(0, 2, 1, 3)
            out_g_dis, out_g_semantic = self._derive_graphs(graph_x)

            if self.graph_input:
                out_g_dis = self.graph_embedding_dis(out_g_dis)  # (4, 307, 64)
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

            # add and norm of the output temporal encoder layer and graphs output
            out = self.dropout_e_normal(self.out_e_lin(out_e)) + out_g.transpose(1, 2)
            return out  # 32x10x512

        # We avoided applying graph structure to the representative vector in some datasets.
        else:
            out_e = self.dropout_e_rep(self.out_e_lin(out_e))
            return out_e