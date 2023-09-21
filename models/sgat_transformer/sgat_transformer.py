import torch
from torch import nn

from models.transformer.transformer_decoder import TransformerDecoder
from models.transformer.transformer_encoder import TransformerEncoder


class SGATTransformer(nn.Module):
    def __init__(self, configs: dict):
        super(SGATTransformer, self).__init__()

        tf_configs = configs['transformer']

        self.device = configs['device']

        self.emb_dim = tf_configs['encoder']['emb_dim']
        self.merge_emb = tf_configs['encoder']['merge_emb']
        self.enc_features = tf_configs['encoder']['features']
        self.enc_seq_len = tf_configs['encoder']['seq_len']
        self.enc_emb_expansion_factor = tf_configs['encoder']['emb_expansion_factor']

        self.dec_seq_len = tf_configs['decoder']['seq_len']
        self.dec_seq_offset = tf_configs['decoder']['seq_offset']
        self.dec_out_start_idx = tf_configs['decoder']['out_start_idx']
        self.dec_out_end_idx = tf_configs['decoder']['out_end_idx']

        encoder_configs = tf_configs['encoder']
        encoder_configs['device'] = self.device
        self.encoders = nn.ModuleList([
            TransformerEncoder(encoder_configs) for _ in range(self.enc_features)
        ])

        decoder_configs = tf_configs['decoder']
        decoder_configs['enc_features'] = self.enc_features
        decoder_configs['device'] = self.device
        self.decoder = TransformerDecoder(decoder_configs)

    def _create_mask(self, batch_size, device):
        trg_mask = torch.triu(torch.ones((self.dec_seq_len, self.dec_seq_len)))\
            .fill_diagonal_(0).bool().expand(batch_size * 8, self.dec_seq_len, self.dec_seq_len)
        return trg_mask.to(device)

    def _create_enc_out(self, x):
        emb_dim = self.emb_dim if not self.merge_emb else self.emb_dim * self.enc_emb_expansion_factor
        enc_outs = torch.zeros((self.enc_features, x[0].shape[0] * x[0].shape[2], x[0].shape[1], emb_dim * 4))\
            .to(self.device)
        return enc_outs

    def forward(self, x, time_idx, y=None, train=True):
        enc_outs = self._create_enc_out(x)
        tgt_mask = self._create_mask(enc_outs.shape[1], self.device)

        for idx, encoder in enumerate(self.encoders):
            x_i = x[idx]

            enc_out = encoder(x_i, time_idx, idx)
            enc_outs[idx] = enc_out

        if train:
            dec_out = self.decoder(y, enc_outs, tgt_mask=tgt_mask, device=self.device)
            return dec_out[:, self.dec_out_start_idx: self.dec_out_end_idx]
        else:
            final_out = torch.zeros_like(y)
            dec_out_len = self.dec_seq_len - self.dec_seq_offset
            for i in range(dec_out_len):

                dec_out = self.decoder(y, enc_outs, tgt_mask=tgt_mask, device=self.device)
                y[:, i + self.dec_seq_offset, :, 0:1] = dec_out[:, i + self.dec_out_start_idx]
                final_out[:, i + self.dec_seq_offset] = dec_out[:, i + self.dec_out_start_idx]

            return final_out[:, self.dec_seq_offset:]
