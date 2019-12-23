#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:58:41 2019

@author: lukeum
"""

import re
import os
import argparse
import pandas as pd





def partition(texts,path,out,shuffle=True):
    '''
    Partion files into three distinct sets
    '''
    
    if shuffle==True:
        texts = texts.sample(frac=1,random_state=233).reset_index(drop=True)
    
    portion = {}
    portion['train'] = texts[:9800]
    portion['dev'] = texts[9800:9900]
    portion['test'] = texts[9900:]
    

    for key, value in portion.items():
        with open(os.path.join(out,"mandarin_text_%s.txt"%(key)),'w') as f:
            for j in value[0]:
                file, text = j.split()
                file = os.path.join(path,'wav_trimmed',file+'.wav')
                f.write(file+'|'+text+'\n')
                

                
                



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-path","--path_to_data",dest='path', help="Please specify the path to data",
                        default="./BZNSYP")
    parser.add_argument("-out","--path_to_output",dest='out', help="Please specify the output path",
                        default='./filelists')

    args = parser.parse_args()
    
    
    texts = pd.read_csv(os.path.join(args.path,"ProsodyLabeling/000001-010000.txt"),header=None)
    texts = texts.iloc[::2,:]

    partition(texts,args.path,args.out)
    
