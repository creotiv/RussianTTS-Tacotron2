import numpy as np
from scipy.io.wavfile import read
import torch
import cv2
import math



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

def guide_attention_slow(text_lengths, mel_lengths, max_txt=None, max_mel=None):
    b = len(text_lengths)
    if max_txt is None:
        max_txt= np.max(text_lengths)
    if max_mel is None:
        max_mel = np.max(mel_lengths)
    guide = np.ones((b, max_txt, max_mel), dtype=np.float32)
    mask = np.zeros((b, max_txt, max_mel), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(max_txt):
            for t in range(max_mel):
                if n < N and t < T:
                    W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (0.2 ** 2)))
                    M[n][t] = 1.0
                elif t >= T and n < N:
                    W[n][t] = 1.0 - np.exp(-((float(n - N - 1) / N)** 2 / (2.0 * (0.2 ** 2))))
    if len(guide) == 1:
        cv2.imwrite('messigray2.png',(guide[0]*255).astype(np.uint8))
        return guide[0], mask[0]
    return guide, mask

def rotate_image(image, angle, center=(0,25)):
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def guide_attention_fast(txt_len, mel_len, max_txt, max_mel):
    img_h,img_w = 50,300

    _max_mel = max_mel + img_w

    c = math.sqrt(txt_len**2+mel_len**2)
    alpha =  - np.arcsin(txt_len/c)*(180/math.pi)

    img = cv2.imread('grad.png')[:,:,0]
    # img = cv2.resize(img,(300,15))
    # img_h,img_w = 15,300
    scale = int(max_mel//img_w + 20)
    img = np.concatenate([img]*scale,1)
    base_img = img.copy()
    h,w = max_txt+img_h, _max_mel
    mask = np.zeros((h,w), dtype=np.float32)
    mask[:img_h,:img.shape[1]] = img[:img_h,:_max_mel]
    mask = rotate_image(mask, alpha)

    y = txt_len
    mask[:,mel_len:_max_mel] = 0

    if mel_len < max_mel:
        hw = _max_mel-mel_len
        hs = txt_len
        he = txt_len+img_h
        mask[hs:he,mel_len:_max_mel] = base_img[:,0:hw]

    mask = mask[img_h//2:-img_h//2,:max_mel]
    mask[txt_len:,:] = 0
    return (255 - mask)/255

