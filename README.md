# Tacotron 2 (with HiFi-GAN)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [RUSLAN dataset](https://ruslan-corpus.github.io/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

## Generated samples
https://soundcloud.com/andrey-nikishaev/sets/russian-tts-nvidia-tacotron2

## New
* Added Diagonal guided attention (DGA) from another model https://arxiv.org/abs/1710.08969
* Added Maximizing Mutual Information for Tacotron (MMI) https://arxiv.org/abs/1909.01145
    - Can't make it work as showed in paper
    - DGA still gives better results, and much cleaner
* Added Russian text preparation with simple stress dictionary (za'mok i zamo'k)
* Using HiFi GAN

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [RUSLAN dataset](https://ruslan-corpus.github.io/)
2. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
3. CD into this repo: `cd tacotron2`
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Ruslan Model] or [LJ Speech model]
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Ruslan Model] or [LJ Speech model]
2. Download published [HiFi-GAN Model] (Universal model recommended for non-English languages)
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[HiFi-Gan](https://github.com/jik876/hifi-gan) HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

## Acknowledgements
This implementation uses code from the following repos: [Nvidia/Tacotron2](https://github.com/NVIDIA/tacotron2)


[Ruslan Model]: https://drive.google.com/file/d/1CCC0_v3cL5qrLFBsSBuNE3_QgZfDH7wl/view?usp=sharing
[HiFi-GAN Model]: https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y
[LJ Speech model]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
