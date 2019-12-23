#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:39:47 2019

@author: lukeum
"""

import sys
import numpy as np
import torch
import os
import argparse

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
#from stft import STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def plot_alignment_to_numpy(alignment, path, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(path)


def infer(checkpoint_path, griffin_iters, text, out_filename):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()#.half()

    sequence = np.array(text_to_sequence(text, ['chinese_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    #audio = audio.astype('int16')
    audio_path = os.path.join('samples', "{}_synthesis.wav".format(out_filename))
    write(audio_path, hparams.sampling_rate, audio)
    print(audio_path)
    plot_alignment_to_numpy(alignments.squeeze().cpu().detach().numpy().T,os.path.join('samples', "{}_attention.png".format(out_filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str,help='text to infer',default='好在#1众弟兄#2看到了#1把她#1追了#1回来#4。')
    parser.add_argument('-s', '--steps', type=int,help='griffin lim iters', default=60)
    parser.add_argument('-c', '--checkpoint', type=str,help='checkpoint path',default='./outdir/checkpoint_32000')
    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='sample')
    args = parser.parse_args()
    infer(args.checkpoint, args.steps, args.text, args.out_filename)
    
    text = '你问我支持不支持，我说支持。'
    sequence = np.array(text_to_sequence(text, ['chinese_cleaners']))[None, :]
