#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:46:25 2019

@author: lukeum
"""


import os
import torch
import pandas as pd
import numpy as np
import transformers
import argparse
from text.cleaners import text_normalize
from txt2pinyin import txt2pinyin
from tqdm import tqdm


def load_model(path):
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
    h = outputs[0].cpu().detach().numpy()
#    del outputs
    h = h[:,1:-1,:]
    
    
    assert h.shape[1] == len(pinyin_seq)

    features = [np.tile(h[:,i,:],[1,len(syl),1]) for i,syl in enumerate(pinyin_seq)]
    features = np.concatenate(features,axis=1)
    
    assert features.shape[1] == len(phon_seq)
    assert features.shape[2] == 768
    assert features.shape[0] == 1
    
    return features

        

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--path_to_model",dest='path', help="Please specify the path to data",
                    default="./BZNSYP")

    args = parser.parse_args()    
    
    model, tokenizer= load_model(args.path)
    
    text_file = os.path.join(args.path,"ProsodyLabeling/000001-010000.txt") 
    all_texts = pd.read_csv(text_file,header=None).iloc[::2,:]
    
    for sample in tqdm(all_texts[0]):
        
        file, text = sample.split()
        features = extract_embeddings(model,tokenizer,text)
        filename = os.path.join(args.path,'bert',file)
        np.save(filename,features)

