"""
# https://arxiv.org/abs/1803.09017
Based on https://github.com/sovaai/sova-tts-engine/blob/master/modules/gst.py
"""
import torch
import torch.nn as nn

from layers import LinearNorm, Conv2Block

class ReferenceEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        reference_encoder_filters = [32, 32, 64, 64, 128, 128]

        channels = zip([1] + reference_encoder_filters[:-1], reference_encoder_filters)

        self.convs = torch.nn.ModuleList(
            [Conv2Block(
                in_channels, out_channels, kernel_size=3, stride=2,
                padding=1, dilation=1, bias=True, w_init_gain='relu', 
                activation=nn.ReLU, bn=True)
            for in_channels, out_channels in channels]
        )

        self.conv_params = {
            "kernel_size": 3,
            "stride": 2,
            "pad": 1,
            "n_convs": len(reference_encoder_filters)
        }

        self.n_mels = hparams.n_mel_channels

        out_channels = self.calculate_size(dim_size=self.n_mels, **self.conv_params)

        self.gru = torch.nn.GRU(
            input_size=reference_encoder_filters[-1] * out_channels,
            hidden_size=hparams.encoder_embedding_dim // 2, # 128 in the paper
            batch_first=True
        )


    def forward(self, inputs, input_lengths=None):
       
        N = inputs.size(0)
        out = inputs.contiguous().view(N, 1, -1, self.n_mels) # [N, 1, Ty, n_mels]
        for conv in self.convs:
            out = conv(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        if input_lengths is not None:
            _input_lengths = self.calculate_size(input_lengths, **self.conv_params).cpu()
            out = nn.utils.rnn.pack_padded_sequence(
                out, _input_lengths, batch_first=True, enforce_sorted=False
            )
        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, N, E//2]
        return out.squeeze(0)


    @staticmethod
    def calculate_size(dim_size, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            dim_size = (dim_size - kernel_size + 2 * pad) // stride + 1
        return dim_size


class STL(torch.nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, hparams):
        super().__init__()

        self.stl_num_heads = 8
        self.stl_token_num = 10

        self.embed = torch.nn.Parameter(torch.FloatTensor(
            self.stl_token_num, hparams.encoder_embedding_dim // self.stl_num_heads
        ))
        d_q = hparams.encoder_embedding_dim // 2
        d_k = hparams.encoder_embedding_dim // self.stl_num_heads

        self.attention = MultiHeadAttention(
            query_dim=d_q,
            key_dim=d_k,
            num_units=hparams.encoder_embedding_dim,
            num_heads=self.stl_num_heads
        )

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)


    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]

        weights, style_emb = self.attention(query, keys)
        weights = weights.squeeze(2).transpose(1, 0)  # [N, num_heads, token_num]

        return weights, style_emb


class MultiHeadAttention(torch.nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = LinearNorm(query_dim, num_units, bias=False)
        self.W_key = LinearNorm(key_dim, num_units, bias=False)
        self.W_value = LinearNorm(key_dim, num_units, bias=False)


    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = torch.nn.functional.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return scores, out


class GST(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = ReferenceEncoder(hparams)
        self.stl = STL(hparams)

    def forward(self, inputs, input_lengths=None):
        weights, style_emb = self.stl(self.encoder(inputs, input_lengths=input_lengths))

        outputs = {
            "style_emb":style_emb,
            "gst_weights":weights
        }

        return outputs


    def inference(self, encoder_outputs, reference_mel=None, token_idx=None):
        device = str(next(self.stl.parameters()).device)
        dtype = torch.half if 'cuda' in device else torch.float

        style_embedding = None
        if reference_mel is not None:
            out = self.forward(reference_mel)
            style_embedding = out['style_emb'].expand_as(encoder_outputs)
        elif token_idx is not None:
            encoder_embedding_dim = encoder_outputs.size(-1)
            query = torch.zeros(1, 1, encoder_embedding_dim // 2).to(device=device, dtype=dtype)
            token = torch.tanh(self.stl.embed[token_idx]).view(1, 1, -1)
            _, style_embedding = self.stl.attention(query, token)

        return style_embedding

class TPSEGST(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.gru = torch.nn.GRU(
            input_size=hparams.encoder_embedding_dim,
            hidden_size=64,
            batch_first=True
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(64,hparams.encoder_embedding_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.encoder_embedding_dim//2,hparams.encoder_embedding_dim),
        )

    def forward(self, x):
        # Detaching from main graph to not send gradient to the GST layer
        x = x.contiguous().detach()
        self.gru.flatten_parameters()
        _, y = self.gru(x)
        y = y.squeeze(0).unsqueeze(1)
        y = torch.tanh(self.linear(y))

        return y