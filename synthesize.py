#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:51:41 2019

@author: lukeum
"""

import matplotlib.pylab as plt

import os
import sys
import numpy as np
import torch
import transformers
import soundfile
import argparse
from praatio import tgio
from librosa import resample
from tqdm import tqdm

sys.path.append('./waveglow/')
sys.path.append('./text')
from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence,sequence_to_pinyin
from denoiser import Denoiser
from text.cleaners import text_normalize
from text.txt2pinyin import txt2pinyin




def plot_data(data, path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.savefig(path)
    plt.close()


def load_bert(path):
    '''
    Load the Chinese Bert model in the specified folder
    '''
    config_path = os.path.join(path,'chinese_wwm_ext_pytorch/bert_config.json')
    model_path = os.path.join(path,'chinese_wwm_ext_pytorch/pytorch_model.bin')
    vocab_path = os.path.join(path, 'chinese_wwm_ext_pytorch/vocab.txt')
    
    
    config = transformers.BertConfig.from_pretrained(config_path)
    config.output_hidden_states=True
    
    model = transformers.BertModel.from_pretrained(model_path,config=config)
    model.eval()
    
    tokenizer = transformers.BertTokenizer(vocab_path)
    
    return model, tokenizer





def extract_embeddings(model,tokenizer,text,upsampling=True):
    '''
    Extract embeddings from the pre-trained bert model.
    Apply upsampling to ensure that embedding length are the same as the phoneme length
    '''
    
    clean_text = text_normalize(text)
    pinyin_seq = txt2pinyin(clean_text)
    phon_seq = [i for syl in pinyin_seq for i in syl]
    
    inputs = torch.tensor(tokenizer.encode(clean_text)).unsqueeze(0)
    outputs = model(inputs)    
    h = outputs[2][-1].cpu().detach().numpy()
#    del outputs
    
    
    assert h.shape[1] == len(pinyin_seq)

    features = [np.tile(h[:,i,:],[1,len(syl),1]) for i,syl in enumerate(pinyin_seq)]
    features = np.concatenate(features,axis=1)
    
    assert features.shape[1] == len(phon_seq)
    assert features.shape[2] == 768
    assert features.shape[0] == 1
    
    return torch.tensor(features).cuda().half()



def Create_textgrid(phones,out_path,raw=False):
    ''' 
    Create a textgrid based on the alignment
        sample: an pd DataFrame of an alignment file
        out_path: the output path
    '''
    tg = tgio.Textgrid()
    syl_tier = tgio.IntervalTier('phones',[],0,sample.iloc[-1,1]+sample.iloc[-1,2])
    entries = []
    
    if raw:
        for i in range(len(sample)):
            
            ph = (sample.iloc[i,3],sample.iloc[i,3]+sample.iloc[i,4],sample.iloc[i,-1])
            entries.append(ph)
    else:
        for i in range(len(sample)):
            ph = (sample.iloc[i,1],sample.iloc[i,1]+sample.iloc[i,2],sample.iloc[i,-1])
            entries.append(ph)
        
    syl_tier = syl_tier.new(entryList=entries)
    tg.addTier(syl_tier)
    out_path = os.path.join(out_path,sample.iloc[0,0]+'.TextGrid')
    tg.save(out_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',default=None,required=True,type=str)
    parser.add_argument('--use_bert',action='store_true')
    parser.add_argument('--bert_folder',default="./",type=str)
    parser.add_argument('--tacotron_path', type=str)
    parser.add_argument('--waveglow_path',type=str)
    parser.add_argument('--resample',default=None, type=str)
    parser.add_argument('--out_dir',required=True,type=str)
    parser.add_argument('--text_grid',action='store_true',default=None)
    parser.add_argument('--alignment', action='store_true',default=None)
    
    
    
    args = parser.parse_args()
    
    hparams = create_hparams()
    hparams.bert = args.use_bert
    hparams.sampling_rate = 22050


    
    # load Tacotron 2 
    checkpoint_path = args.tacotron_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda().eval().half()
    
    # load Waveglow
    waveglow_path = args.waveglow_path
    waveglow = torch.load(waveglow_path)
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    denoiser = Denoiser(waveglow)
    
    # load Chinese BERT
    if hparams.bert:
        bert, tokenizer= load_bert(args.bert_folder)
    
    # Extract phonemic features
    with open(args.text,'r') as f:
        texts = []
        for line in f.readlines():
            name, sen = line.strip().split(' ')
            if sen[-1] not in  ['。','？','！']:
                texts.append((name, sen+'。'))
            else:
                texts.append((name, sen))
            
    for i, (name, text) in tqdm(enumerate(texts)):

        phone_seq = np.array(text_to_sequence(text, ['chinese_cleaners']))[None, :]
        phones = torch.autograd.Variable(
            torch.from_numpy(phone_seq)).cuda().long()
        if hparams.bert == False:
            sequence = phones
        # Extract BERT embeddings
        else:
            features = extract_embeddings(bert,tokenizer,text)
            sequence = (phones, features)
        
        
        
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        
        if args.alignment:
            plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T), os.path.join(args.out_dir,'fig_%s.png'%(i)))
        

        
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        
        audio = np.concatenate((np.zeros(2048),audio[0].data.cpu().numpy(),np.zeros(2048)))
        
#        name = 'sample_'+str(i)
#        audio = audio[0].data.cpu().numpy()
        soundfile.write(os.path.join(args.out_dir,'%s.wav'%(name)),audio,hparams.sampling_rate)
        
        
        # Downsampled to 16000 for forced alignment with Kaldi
        if args.resample:
            new_y = resample(audio,hparams.sampling_rate,16000)
            soundfile.write(os.path.join(args.out_dir,'%s_resample.wav'%(name)),new_y,16000)
    
    
        # Generate textgrid annotations based on attention alignment
        if args.text_grid:
            alignment = alignments.float().data.cpu().numpy()[0].T
            frames = np.argmax(alignment,axis=0)
            frames = [phone_seq[0,i] for i in frames]
            pinyin_seq = sequence_to_pinyin(frames)
            
            duration = 256/hparams.sampling_rate*(len(frames)+16)
            
            times = []
           
            for i, p in enumerate(pinyin_seq):
                if i != len(pinyin_seq)-1:
                    if p != pinyin_seq[i+1]:
                           times.append((p,i))
                else:
                    times.append((p,i))
        
        
            
            tg = tgio.Textgrid()
            syl_tier = tgio.IntervalTier('phones',[],0,duration)
            entries = []
            
            entries.append((0,2048/hparams.sampling_rate,'sil'))
            for i, p in enumerate(times):
                
                if i == 0:
                    ph = (2048/hparams.sampling_rate,(p[1]+8)*256/hparams.sampling_rate,p[0])
                else:
                    ph = ((p_last[1]+8)*256/hparams.sampling_rate,(p[1]+8)*256/hparams.sampling_rate,p[0])
                p_last = p
                entries.append(ph)
        
                
            syl_tier = syl_tier.new(entryList=entries)
            tg.addTier(syl_tier)
            out_path = os.path.join(args.out_dir, 'textgrid','%s.TextGrid'%(name))
            tg.save(out_path)
