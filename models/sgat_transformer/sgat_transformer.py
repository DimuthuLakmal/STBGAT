import torch
from torch import nn

from models.transformer.transformer_decoder import TransformerDecoder
from models.transformer.transformer_encoder import TransformerEncoder


class SGATTransformer(nn.Module):
    def __init__(self, configs: dict):
        super(SGATTransformer, self).__init__()

        tf_configs = configs['transformer']

        self.device = configs['device']

        self.enc_features = tf_configs['encoder']['features']
        self.enc_seq_len = tf_configs['encoder']['seq_len']

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

        self.enc_out_size = encoder_configs['sgat']['out_f_sizes'][-1]
        self.decoder_heads = decoder_configs['decoder_block']['n_heads']

    def _create_mask(self, batch_size: int, device: str):
        """
        Create a mask for temporal decoder, so that, it can't look ahead decoder input seq when calculating attention
        Parameters
        ----------
        batch_size, int
        device, str, cpu or cuda

        Returns
        -------
        tgt_mask: Tensor, mask tensor
        """
        trg_mask = torch.triu(torch.ones((self.dec_seq_len, self.dec_seq_len)))\
            .fill_diagonal_(0).bool().expand(batch_size * self.decoder_heads, self.dec_seq_len, self.dec_seq_len)
        return trg_mask.to(device)

    def _create_enc_out(self, x_shp: int):
        """
        Create a zero valued tensor to hold encoder outputs
        Parameters
        ----------
        x_shp: int, encoder's 0th sequence shape

        Returns
        -------
        enc_outs: Tensor
        """
        enc_outs = torch.zeros((self.enc_features, x_shp[0] * x_shp[2], x_shp[1], self.enc_out_size)).to(self.device)
        return enc_outs

    def forward(self, x, y=None, train=True):
        enc_outs = self._create_enc_out(x[0].shape)
        tgt_mask = self._create_mask(enc_outs.shape[1], self.device)

        # get encoder outputs
        for idx, encoder in enumerate(self.encoders):
            x_i = x[idx]

            enc_out = encoder(x_i, idx)
            enc_outs[idx] = enc_out

        # if model in test or finetune stage, decoder will be fed with input in autoregressive manner.
        # Otherwise, it will be fed in parallel at once.
        if train:
            dec_out = self.decoder(y, enc_outs, tgt_mask=tgt_mask, device=self.device)
            return dec_out[:, self.dec_out_start_idx: self.dec_out_end_idx]
        else:
            dec_out_len = self.dec_seq_len - self.dec_seq_offset
            for i in range(dec_out_len):
                y_input = torch.tensor(y)
                dec_out = self.decoder(y_input, enc_outs, tgt_mask=tgt_mask, device=self.device)
                y[:, i + self.dec_seq_offset, :, 0:1] = dec_out[:, i + self.dec_out_start_idx]

            return y[:, self.dec_seq_offset:]
