from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths, dropout_frame
import contextlib
import numpy as np
import random
from text.symbols import ctc_symbols
from gst import GST, TPSEGST


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights



@contextlib.contextmanager
def temp_seed(seed=None):
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.relu(linear(x))
            if self.training:
                x = F.dropout(x, p=0.5, training=True)
            else:
                w = x.numel()
                b = np.expand_dims(np.random.binomial(1, p=0.5, size=w),axis=0)
                b = torch.tensor(b, dtype=torch.float16).to(x.device).view(x.shape)
                x = x * b * (1/0.5)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, x):
        try:
            with torch.no_grad():
                for conv in self.convolutions:
                    x = F.dropout(F.relu(conv(x)), 0.5, self.training)

                x = x.transpose(1, 2)

                # self.lstm.flatten_parameters()
                outputs, _ = self.lstm(x)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            outputs = None

        # removes unused memory but may increase time a bit
        torch.cuda.empty_cache()

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        if hparams.use_gst:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.token_embedding_size
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing

        # Maximazing Mutual Inforamtion
        # https://arxiv.org/abs/1909.01145
        # https://github.com/bfs18/tacotron2
        self.use_mmi = hparams.use_mmi

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        lp_out_dim = hparams.decoder_rnn_dim if self.use_mmi else hparams.n_mel_channels * hparams.n_frames_per_step

        self.mel_layer = None
        if not self.use_mmi:
            self.linear_projection = LinearNorm(
                hparams.decoder_rnn_dim + self.encoder_embedding_dim,
                lp_out_dim
            )
        else:
            self.linear_projection = nn.Sequential(
                LinearNorm(
                    hparams.decoder_rnn_dim + self.encoder_embedding_dim,
                    lp_out_dim,
                    w_init_gain='relu'
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.mel_layer = nn.Sequential(
                LinearNorm(
                    hparams.decoder_rnn_dim,
                    hparams.decoder_rnn_dim,
                    w_init_gain='relu'
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                LinearNorm(
                    in_dim=hparams.decoder_rnn_dim,
                    out_dim=hparams.n_mel_channels * hparams.n_frames_per_step
                )
            )

        gate_in_dim = hparams.decoder_rnn_dim if self.use_mmi else \
            hparams.decoder_rnn_dim + self.encoder_embedding_dim

        self.gate_layer = LinearNorm(
            gate_in_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, decoder_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        if decoder_outputs:
            decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
            decoder_outputs = decoder_outputs.transpose(1, 2)
        else:
            decoder_outputs = None

        return mel_outputs, gate_outputs, alignments, decoder_outputs

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        if self.use_mmi:
            mel_output = self.mel_layer(decoder_output)
            decoder_hidden_attention_context = decoder_output
        else:
            mel_output = decoder_output
            decoder_output = None

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return mel_output, gate_prediction, self.attention_weights, decoder_output

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if self.p_teacher_forcing >= random.random() or len(mel_outputs) == 0:
                decoder_input = decoder_inputs[len(mel_outputs)]
            else:
                decoder_input = self.prenet(mel_outputs[-1])

            mel_output, gate_output, attention_weights, decoder_output = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
            if decoder_output is not None:
                decoder_outputs += [decoder_output.squeeze(1)]

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments, decoder_outputs

    def inference(self, memory, seed=None, suppress_gate=False):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        with temp_seed(seed):
            while True:
                decoder_input = self.prenet(decoder_input)
                mel_output, gate_output, alignment, _ = self.decode(decoder_input)

                mel_outputs += [mel_output.squeeze(1)]
                gate_outputs += [gate_output]
                alignments += [alignment]

                # if decoder_output is not None:
                #     decoder_outputs += [decoder_output.squeeze(1)]

                if not suppress_gate and torch.sigmoid(gate_output.data) > self.gate_threshold:
                    break
                elif len(mel_outputs) == self.max_decoder_steps:
                    print("Warning! Reached max decoder steps")
                    break

                decoder_input = mel_output

        mel_outputs, gate_outputs, alignments, _ = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments


class MIEsitmator(nn.Module):
    def __init__(self, vocab_size, decoder_dim, hidden_size, dropout=0.5):
        super(MIEsitmator, self).__init__()
        self.proj = nn.Sequential(
            LinearNorm(decoder_dim, hidden_size, bias=True, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.ctc_proj = LinearNorm(hidden_size, vocab_size + 1, bias=True)
        self.ctc = nn.CTCLoss(blank=vocab_size, reduction='none')

    def forward(self, decoder_outputs, target_phones, decoder_lengths, target_lengths):
        out = self.proj(decoder_outputs)
        log_probs = self.ctc_proj(out).log_softmax(dim=2)
        log_probs = log_probs.transpose(1, 0)
        ctc_loss = self.ctc(log_probs, target_phones, decoder_lengths, target_lengths)
        # average by number of frames since taco_loss is averaged.
        ctc_loss = (ctc_loss / decoder_lengths.float()).mean()
        return ctc_loss

class Embeder(nn.Module):
    def __init__(self, hparams):
        super(Embeder, self).__init__()
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.end_symbols_ids = hparams.end_symbols_ids
        
    def forward(self, x):
        emb = self.embedding(x)
        if self.end_symbols_ids:
            s = torch.tensor(self.end_symbols_ids, requires_grad=False).to(x.device)
            end_vectors = self.embedding(s)
            for b in range(x.size(0)):
                seq = x[b].cpu().detach().numpy().tolist()
                vec = None
                for i in range(x.size(1),0,-1):
                    if seq[i-1] in self.end_symbols_ids:
                        _id = self.end_symbols_ids.index(seq[i-1])
                        vec = end_vectors[_id,:]*1.5
                        continue
                    if vec is not None:
                        emb[b,i-1] = emb[b,i-1] + vec

        return emb


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        # self.embedding = Embeder(hparams)
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.drop_frame_rate = hparams.drop_frame_rate
        self.use_mmi = hparams.use_mmi
        self.use_gst = hparams.use_gst

        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        if self.use_mmi:
            vocab_size = len(ctc_symbols)
            decoder_dim = hparams.decoder_rnn_dim
            self.mi = MIEsitmator(vocab_size, decoder_dim, decoder_dim, dropout=0.5)
        else:
            self.mi = None

        self.gst = None
        if self.use_gst:
            self.gst = GST(hparams)
            self.tpse_gst = TPSEGST(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text, ctc_text_lengths, guide_mask  = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        ctc_text = to_gpu(ctc_text).long()
        ctc_text_lengths = to_gpu(ctc_text_lengths).long()
        guide_mask = to_gpu(guide_mask).float()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             ctc_text, ctc_text_lengths),
            (mel_padded, gate_padded,guide_mask ))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mel_mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            if outputs[0] is not None:
                float_mask = (~mask).float().unsqueeze(1)
                outputs[0] = outputs[0] * float_mask
            outputs[1].data.masked_fill_(mel_mask, 0.0)
            outputs[2].data.masked_fill_(mel_mask, 0.0)
            outputs[3].data.masked_fill_(mel_mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs, minimize=False):
        text_inputs, text_lengths, mels, max_len, output_lengths, *_ = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        if self.drop_frame_rate > 0. and self.training:
            # mels shape (B, n_mel_channels, T_out),
            mels = dropout_frame(mels, self.global_mean, output_lengths, self.drop_frame_rate)

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        emb_text = self.encoder(embedded_inputs, text_lengths)
        encoder_outputs = emb_text

        tpse_gst_outputs = None
        gst_output = None
        if self.gst is not None:
            gst_outputs = self.gst(mels, output_lengths)
            emb_gst = gst_outputs.repeat(1, emb_text.size(1), 1)
            tpse_gst_outputs = self.tpse_gst(encoder_outputs)
            encoder_outputs = torch.cat((emb_text, emb_gst), dim=2)

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if not minimize:
            return self.parse_output(
                [decoder_outputs, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, tpse_gst_outputs, gst_outputs],
                output_lengths)
        else:
            return self.parse_output(
                [decoder_outputs, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, tpse_gst_outputs, gst_outputs],
                output_lengths)[2]

    def inference(self, inputs, seed=None, reference_mel=None, token_idx=None, scale=1.0):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        emb_text = self.encoder.inference(embedded_inputs)
        encoder_outputs = emb_text

        if self.gst is not None:
            if reference_mel is not None:
                emb_gst = self.gst(reference_mel)*scale
            elif token_idx is not None:
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size, dtype=torch.float16).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[token_idx].unsqueeze(0).expand(1, -1, -1)
                emb_gst = self.gst.stl.attention(query, key)*scale
            else:
                emb_gst = self.tpse_gst(emb_text)*scale

            emb_gst = emb_gst.repeat(1, emb_text.size(1), 1)
         
            encoder_outputs = torch.cat(
                    (emb_text, emb_gst), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, seed=seed)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, emb_gst])

        return outputs
