#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:47:38 2019

@author: lukeum
"""
import re
import sys
sys.path.append("./preprocess")
from txt2pinyin import txt2pinyin
import argparse
import pandas as pd
from collections import Counter
from tqdm import tqdm

_special_char = [(re.compile('%s' % x[0]), x[1]) for x in [
  ('Ｂ', '哔'),
  ('Ｐ', '披'),
  ('[—”…）（“；]',''),
  ('[(#1)(#2)(#3)(#4)]','')
]]

def text_normalize(text):
  for regex, replacement in _special_char:
    text = re.sub(regex, replacement, text)
  return text



def generate_dict(texts):
    
    dictionary = Counter()
    
    for i in tqdm(range(len(texts))):
        
        clean_text = text_normalize(texts.iloc[i,1])
                
        py = txt2pinyin(clean_text)
        
        sequence = [i for syl in py for i in syl]

        dictionary.update(sequence)
        
    return dictionary



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-path","--path_to_text",dest='path', help="Please specify the path to text",
                        default="../BZNSYP")
    parser.add_argument("-out","--path_to_dict",dest='out', help="Please specify the path to text",
                        default="../text")
    args = parser.parse_args()
    
    
    texts = pd.read_csv(os.path.join(args.path,"ProsodyLabeling/000001-010000.txt"),sep='\t',header=None)
    texts = texts.iloc[::2,:]
    
    dictionary = generate_dict(texts)

    with open(os.path.join(args.out,'phone_dict.txt'),'w') as d:
        
        for k in sorted(dictionary.keys()):
            d.write(k+'\n')
            
            
