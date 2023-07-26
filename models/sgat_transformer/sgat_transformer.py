import torch
from torch import nn

from models.transformer.transformer_decoder import TransformerDecoder
from models.transformer.transformer_encoder import TransformerEncoder
from utils.data_utils import create_lookup_index


class SGATTransformer(nn.Module):
    def __init__(
            self,
            device: str,
            sgat_first_in_f_size: int,
            sgat_n_layers: int,
            sgat_out_f_sizes: list,
            sgat_n_heads: list,
            sgat_alpha: float = 0.2,
            sgat_dropout: float = 0.2,
            sgat_edge_dim: int = 1,
            transformer_merge_emb=False,
            transformer_enc_seq_len: int = 12,
            transformer_dec_seq_len: int = 12,
            transformer_dec_seq_offset: int = 1,
            transformer_input_dim: int = 1,
            transfomer_emb_dim: int = 16,  # input to transformers will be embedded to this dim
            transformer_n_layers: int = 2,
            transformer_expansion_factor: int = 4,
            transformer_n_heads: int = 4,
            transformer_enc_features: int = 5,  # number of encoders
            transformer_out_dim: int = 1,
            transformer_dropout: float = 0.2,
            transformer_lookup_index: bool = True,
            transformer_cross_attn_features: int = 1,
            transformer_per_enc_feature_len: int = 12,
            transformer_dec_out_start_idx: int = 2,
            transformer_dec_out_end_idx: int = -2
    ):
        super(SGATTransformer, self).__init__()

        self.device = device
        self.enc_features = transformer_enc_features
        self.dec_seq_len = transformer_dec_seq_len
        self.emb_dim = transfomer_emb_dim
        self.merge_emb = transformer_merge_emb
        self.dec_seq_offset = transformer_dec_seq_offset
        self.dec_out_start_idx = transformer_dec_out_start_idx
        self.dec_out_end_idx = transformer_dec_out_end_idx

        sgat_settings = {
            'n_layers': sgat_n_layers,
            'first_in_f_size': sgat_first_in_f_size,
            'out_f_sizes': sgat_out_f_sizes,
            'n_heads': sgat_n_heads,
            'alpha': sgat_alpha,
            'dropout': sgat_dropout,
            'edge_dim': sgat_edge_dim
        }

        self.lookup_idx_enc = None
        self.lookup_idx_dec = None
        max_lookup_len_enc = None
        max_lookup_len_dec = None
        if transformer_lookup_index:
            self.lookup_idx_enc, max_lookup_len_enc = create_lookup_index(merge=True if transformer_enc_features == 1 else False)

            start_idx_lk_dec = max_lookup_len_enc - transformer_dec_seq_offset
            self.lookup_idx_dec = [i for i in range(start_idx_lk_dec, start_idx_lk_dec + transformer_dec_seq_len)]
            max_lookup_len_dec = start_idx_lk_dec + transformer_dec_seq_len

        self.encoders = nn.ModuleList([
            TransformerEncoder(seq_len=transformer_enc_seq_len,
                               input_dim=transformer_input_dim,
                               embed_dim=transfomer_emb_dim,
                               num_layers=transformer_n_layers,
                               expansion_factor=transformer_expansion_factor,
                               n_heads=transformer_n_heads,
                               sgat_settings=sgat_settings,
                               merge_embed=transformer_merge_emb,
                               dropout=transformer_dropout,
                               max_lookup_len=max_lookup_len_enc if max_lookup_len_enc else transformer_enc_seq_len)
            for _ in range(transformer_enc_features)
        ])

        self.decoder = TransformerDecoder(input_dim=transformer_input_dim,
                                          out_dim=transformer_out_dim,
                                          embed_dim=transfomer_emb_dim,
                                          seq_len=transformer_dec_seq_len,
                                          offset=transformer_dec_seq_offset,
                                          num_layers=transformer_n_layers,
                                          expansion_factor=transformer_expansion_factor,
                                          n_heads=transformer_n_heads,
                                          dropout=transformer_dropout,
                                          enc_features=transformer_enc_features,
                                          sgat_settings=sgat_settings,
                                          merge_embed=transformer_merge_emb,
                                          cross_attn_features=transformer_cross_attn_features,
                                          per_enc_feature_len=transformer_per_enc_feature_len,
                                          max_lookup_len=max_lookup_len_dec if max_lookup_len_dec else transformer_dec_seq_len)

    def create_mask(self, batch_size, device):
        trg_mask = torch.triu(torch.ones((self.dec_seq_len, self.dec_seq_len)))\
            .fill_diagonal_(0).bool().expand(batch_size * 8, self.dec_seq_len, self.dec_seq_len)
        return trg_mask.to(device)

    def forward(self, x, graph_x, y=None, graph_y=None, train=True):
        # TODO: We can't guarentee that always x presents. So have to replace the way of finding shape
        enc_outs = torch.zeros((self.enc_features, x.shape[0] * x.shape[2], x.shape[1], self.emb_dim)).to(self.device)

        for idx, encoder in enumerate(self.encoders):
            x_i = x[:, :, :, idx: idx + 1] if x is not None else None
            graph_x_i = graph_x[idx] if graph_x is not None else None
            lookup_idx_i = self.lookup_idx[idx] if self.enc_features > 1 else self.lookup_idx_enc

            enc_out = encoder(x_i, graph_x_i, lookup_idx_i, True)
            enc_outs[idx] = enc_out

        tgt_mask = self.create_mask(enc_outs.shape[1], self.device)

        if train:
            dec_out = self.decoder(y, graph_y, enc_outs, tgt_mask=tgt_mask, local_trends=True,
                                   lookup_idx=self.lookup_idx_dec, device=self.device)
            return dec_out[:, self.dec_out_start_idx: self.dec_out_end_idx]
        else:
            final_out = torch.zeros_like(y)
            dec_out_len = self.dec_seq_len - self.dec_seq_offset
            for i in range(dec_out_len):
                dec_out = self.decoder(y, graph_y, enc_outs, tgt_mask=tgt_mask, local_trends=True,
                                       lookup_idx=self.lookup_idx_dec, device=self.device)

                y[:, i + self.dec_seq_offset] = dec_out[:, i + self.dec_out_start_idx]
                for batch in range(y.shape[0]):
                    graph_y[batch][i + self.dec_seq_offset].x = dec_out[batch, i + self.dec_out_start_idx]

                final_out[:, i + self.dec_seq_offset] = dec_out[:, i + self.dec_out_start_idx]

            return final_out[:, self.dec_seq_offset:]
