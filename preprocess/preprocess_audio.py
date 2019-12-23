#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:08:41 2019

@author: lukeum
"""
import os
import re
import numpy as np
import argparse
from tqdm import tqdm
from scipy.io import wavfile
import librosa
import soundfile

def get_sil(path_to_sfs_file):
    '''
    Get the indices of the start and end frame for non-silent intervals
    '''
    
    with open(path_to_sfs_file,'r') as f:
        content = f.readlines()
    start = float(content[0].split()[0])/10e6-0.05
    end = float(content[-2].split()[0])/10e6
    
    return start, end



def load_wav(path,sampling_rate):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    if sr != sampling_rate:
        x = librosa.resample(x, sr, sampling_rate)
    x = np.clip(x, -1.0, 1.0)
    return x


def save_wav(wav, path,sampling_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sampling_rate, wav.astype(np.int16))

'''
sample = "./BZNSYP/sfs/000081.sfs"
s, e = get_sil(sample)
import matplotlib.pyplot as plt
y = load_wav("./BZNSYP/Wave/000081.wav")
plt.plot(y)
plt.plot(y[round(s*sr):round(e*sr)])
wavfile.write("demo.wav",22050,y)
'''


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data","--path_to_data",dest='path', help="Please specify the path to data",
                        default="./BZNSYP")
    parser.add_argument("-sr","--sampling_rate",dest='sr', help="Please specify the path to data",
                        default=22050)



    args = parser.parse_args()
    path = args.path 

    path_to_sil = os.path.join(args.path,"sfs") 
    path_to_audio = os.path.join(args.path,"Wave")
    outdir = os.path.join(args.path,"wav_trimmed") 
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir) 
        
    filelists = os.listdir(path_to_audio)
    pattern = re.compile('(.*?)\.wav')
#corrected the transcription for the following sound files    
#    filelists = ['003806.wav','004227.wav']
    for f in tqdm(filelists):
        try:
            name = re.match(pattern,f).group(1)
            start_time, end_time = get_sil(os.path.join(path_to_sil,name+".sfs"))
        
            y = load_wav(os.path.join(path_to_audio,f),args.sr)
            save_wav(y[round(start_time*args.sr):round(end_time*args.sr)],os.path.join(outdir,f),args.sr)
#        soundfile.write(os.path.join(outdir,f),y[round(start_time*args.sr):round(end_time*args.sr)],args.sr)                
        except:
            print(f)    
    
  

        
        
    
    
    
    
    
    
    
    
    
