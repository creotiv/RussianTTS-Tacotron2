import matplotlib
import matplotlib.pylab as plt
import IPython.display as ipd

import sys
sys.path.append('hifigan/')
import numpy as np
import torch
import librosa
import librosa.display
import math
import json
import os
import soundfile as sf
import librosa

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, symbol_to_id
from text.cleaners import transliteration_ua_cleaners, english_cleaners, transliteration_cleaners,transliteration_cleaners_with_stress
from text.rudict import RuDict
from PIL import Image 
import time

from torch.nn import functional as F

from sklearn.metrics.pairwise import cosine_similarity as cs


from hifigan.meldataset import MAX_WAV_VALUE
from hifigan.models import Generator
from hifigan.env import AttrDict

from audio_processing import get_mel
import streamlit as st

def plot_data(st, data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    plt.savefig('out.png')
    image = Image.open('out.png')
    st.image(image, use_column_width=True)

def load_tts_model(path, hparams):
    model = load_model(hparams)
    model.load_state_dict(torch.load(path)['state_dict'])
    _ = model.cuda().eval().half()
    return model

@st.cache()
def load_vocoder_model():
    def load_checkpoint(filepath, device):
        assert os.path.isfile(filepath)
        checkpoint_dict = torch.load(filepath, map_location=device)
        return checkpoint_dict

    device = torch.device('cuda')
    with open('hifigan/config.json') as fp:
        json_config = json.load(fp)
        h = AttrDict(json_config)
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint("hifigan/g_02500000", device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def inference(mel, generator, loundess=20):
    mel = mel.type(torch.float32)
    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = F.normalize(audio.detach(), dim=0).cpu().numpy()#.astype('int16')
        return audio * loundess 

def main():
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.end_symbols_ids = [symbol_to_id[s] for s in '?!.']
    hparams.use_gst = False

    st.title("Text To Speech Demo")

    st.sidebar.title("Settings")
    seed = st.sidebar.text_input('Seed', value='')

    seed = int(seed) if seed.strip() else None

    checkpoint_path = st.sidebar.selectbox("Weights",[
        #"weights/dga/checkpoint_186000_no_emph",
        #"weights/dga/checkpoint_350000",
        #"weights/gst/tpgst_old_100500", 
	"weights/tpgst/checkpoint_200k_10ktp"
    ])
    custom_path = st.sidebar.text_input('Custom weights', value="")
    if 'gst' in checkpoint_path:
        hparams.use_gst = True
    checkpoint_path = custom_path or checkpoint_path
        
    model = load_tts_model(checkpoint_path, hparams)
    vocoder = load_vocoder_model()

    #cleaner = "transliteration_cleaners"
    #if st.sidebar.checkbox('Use automatic emphasizer'):
    cleaner = "transliteration_cleaners_with_stress"

    _text = st.selectbox("Predefined text",[
        "Н+очь, +улица, фон+арь, апт+ека. Бессм+ысленный, и т+усклый св+ет. Жив+и ещ+е х+оть ч+етверть в+ека - Вс+ё б+удет т+ак. Исх+ода н+ет.",
        "мн+е хот+елось б+ы сказ+ать к+ак я призн+ателен вс+ем прис+утсвующим сд+есь.",
        "Тв+орог или твор+ог, к+озлы или козл+ы, з+амок или зам+ок.", 
        "Вс+е смеш+алось в д+оме Обл+онских. Жен+а узн+ала, что муж был в св+язи с б+ывшею в их д+оме франц+уженкою-гуверн+анткой, и объяв+ила м+ужу, что не м+ожет ж+ить с ним в одн+ом д+оме. Полож+ение это продолж+алось уже третий д+ень и муч+ительно ч+увствовалось и сам+ими супр+угами, и вс+еми чл+енами семь+и, и домоч+адцами.",
        "Я открыл зам+ок и вошел в з+амок, сь+ев жарк+ое я п+онял как+ое сейч+ас ж+аркое лето в укра+ине.",
"Молод+ой парн+ишка Т+анг С+ан одн+ажды оступ+ился и сл+едуя сво+им жел+аниям и пр+ихотям вор+ует секр+етные уч+ения в своей школе боев+ых искусств.",
        "Тетрагидропиранилциклопентилтетрагидропиридопиридиновые вещества",
        "Я открыл замок и вошел в замок, сьев жаркое я понял какое сейчас жаркое лето в украине.",
    ])
    text = st.text_area('What to say?', value=_text, height=250)
    run = st.button('Generate')
    text = text or _text

    if run:
        start = time.perf_counter()    

        sequence = np.array(text_to_sequence(text, [cleaner]))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        _, mel_outputs, mel_outputs_postnet, _, alignments, _ = model.inference(sequence,seed=seed)
        audio = inference(mel_outputs_postnet, vocoder)
        
        elapsed = time.perf_counter() - start
        st.text('Generated in %s sec.' % elapsed)

        plot_data(st,(mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))

        sf.write('out.wav', audio, hparams.sampling_rate)
        st.audio('out.wav')

if __name__ == "__main__":
    main()

