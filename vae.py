
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)


        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)            
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class VAE(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, hp.vae_dim],[N, hp.vae_dim],[N, hp.vae_dim]
    '''

    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.mean = nn.Linear(hp.ref_enc_gru_size, hp.vae_dim)
        self.log_var = nn.Linear(hp.ref_enc_gru_size, hp.vae_dim)
        self.emb = nn.Linear(hp.vae_dim, hp.vae_embedding)
        self.hp = hp

    def forward(self, inputs, input_lengths=None):
        reference = self.encoder(inputs, input_lengths)

        mean = self.mean(reference)
        log_var = self.log_var(reference)
        std = torch.exp(log_var)
        z = torch.randn(mean.shape[0], self.hp.vae_dim).to(reference.device)
        output = mean + z * std
        emb = self.emb(output)
        emb = torch.unsqueeze(emb, 1)
        return emb, mean, log_var

def vae_weight(hp, iteration):
    if iteration < hp.vae_warming_up and iteration % 100 < 1:
        w1 = torch.tensor(hp.vae_init_weights + iteration / 100 * hp.vae_weight_multiplier, dtype=torch.float32)
    else:
        w1 = torch.tensor(0.0, dtype=torch.float32)
    
    if iteration > hp.vae_warming_up and iteration % 400 < 1:
        w2 = torch.tensor(
            hp.vae_init_weights \
            + ((iteration-hp.vae_warming_up) / 400 * hp.vae_weight_multiplier \
            + hp.vae_warming_up / 100 * hp.vae_weight_multiplier), \
            dtype=torch.float32)
    else:
        w2 = torch.tensor(0.0, dtype=torch.float32)
    
    return torch.maximum(w1, w2)

if __name__ == '__main__':
    from hparams import create_hparams

    hp = create_hparams()
    x = torch.randn(4,500,80)
    y = VAE(hp)(x)
    print(y[0].shape)
    print(y)