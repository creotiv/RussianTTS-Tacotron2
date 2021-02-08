import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    mask = get_mask_from_lengths(lengths).float()
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.float() * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    dropped_mels = (mels * (1.0 - drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
