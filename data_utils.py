import random
import os
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text, guide_attention_fast
from text import text_to_sequence, sequence_to_ctc_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.ds_path = hparams.dataset_path
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.clean_non_existent()
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        print('Dataset files:',len(self.audiopaths_and_text))

    def clean_non_existent(self):
        out = []
        for el in self.audiopaths_and_text:
            if os.path.exists(os.path.join(self.ds_path,'wavs',"%s.wav" % el[0])):
                out.append(el)
            else:
                print(el[0])
        self.audiopaths_and_text = out

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[2].strip() or audiopath_and_text[1].strip()
        text, ctc_text = self.get_text(text)
        mel = self.get_mel(os.path.join(self.ds_path,'wavs',audiopath+'.wav'))
        guide_mask = torch.FloatTensor(guide_attention_fast(len(text),mel.shape[-1],200,1000))
        return (text, ctc_text, mel, guide_mask)

    def get_mel(self, filename):
        if not os.path.exists(filename+'.npy'):
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            np.save(filename+'.npy',melspec)
        else:
            melspec = torch.from_numpy(np.load(filename+'.npy'))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        sequence = text_to_sequence(text, self.text_cleaners)
        text_norm = torch.IntTensor(sequence)
        ctc_text_norm = torch.IntTensor(sequence_to_ctc_sequence(sequence))
        return text_norm, ctc_text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        max_ctc_txt_len = max([len(x[1]) for x in batch])
        ctc_text_paded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_paded .zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][1]
            ctc_text_paded[i, :ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        guide_padded = torch.FloatTensor(len(batch), 200, 1000)
        guide_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            guide = batch[ids_sorted_decreasing[i]][3]
            guide_padded[i, :, :] = guide

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text_paded, ctc_text_lengths, guide_padded