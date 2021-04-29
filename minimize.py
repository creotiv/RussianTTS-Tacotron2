import os
import time
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from train import validate, prepare_dataloaders, Tacotron2Loss

def model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = ('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    return size

def load_base_model():
    hparams = create_hparams()
    checkpoint_path = "weights/checkpoint_200k_44ktp_natasha"
    model = Tacotron2(hparams).cpu()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
    _ = model.eval().cpu()
    return hparams,model

def prepare_model(model, hparams):

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    criterion = Tacotron2Loss(hparams, iteration=0)


    validate(model, criterion, valset, 0, hparams.batch_size, 1, collate_fn)

def minimize():
    hparams, model = load_base_model()

    base_model_size = model_size(model)
    print(base_model_size)

    s = time.perf_counter()
    prepare_model(model, hparams)
    print(time.perf_counter() - s)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear, nn.LSTMCell, nn.GRUCell}, dtype=torch.qint8
    )

    s = time.perf_counter()
    prepare_model(quantized_model, hparams)
    print(time.perf_counter() - s)
    # # Fuse Conv, bn and relu
    # stat_quant.fuse_model()
    # # we will not see Batch Norm instead will see replaced it with Identity
    # print(stat_quant)

    # quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # print(quantized_model.qconfig)

    # # this aware quantization engine thet we are going to inference few inputs 
    # # so Observer can get enough data for building optimiation maps
    # torch.quantization.prepare(quantized_model, inplace=True)

    # prepare_model(quantized_model, hparams)

    # # now we converting our model to int8
    # torch.quantization.convert(quantized_model, inplace=True)

    base_model_size = model_size(quantized_model)
    print(base_model_size)

minimize()