#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:11:00 2019

@author: lukeum
"""

import re
import os
import sys

import argparse
from praatio import tgio
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from tqdm import tqdm
#from nnmnkwii.util import example_label_file, example_question_file




def get_entries(path_to_textgrid,textgrid):
    '''
    Get all entries from a textgrid file.
    Each entry consists of three elements
        - start time
        - end time
        - phone label
    '''
    entries = textgrid.tierDict[textgrid.tierNameList[0]].entryList
    return entries
    
 
    
def generate_individual_sfs(path_to_sfs,tg_file,entries):
    '''
    Convert a textgrid to a .sfs file for generating frame based textual feature
    '''
    name_pattern = re.compile(r'(\d+)\.interval')
    file_name = re.match(name_pattern,tg_file).group(1)+'.sfs'
    
    
    with open(os.path.join(path_to_sfs,file_name),'w') as f:
        for e in entries:
            if e[2] =="sil":
                phs_type = 's'
            elif e[2] == 'sp1':
                phs_type = 'd'
            elif re.match(r'\w+\d$',e[2]):
                phs_type = 'b'
            else:
                phs_type = 'a'
            
            f.write(str(round(e[1]*10e6))+' '+phs_type+'\n')



def get_all_sfs(path):
    '''
    Batch processing
    Generate .sfs files from textgrids
    
    ./PhoneLabeling/007017.interval has the wrong tier name
    '''
    path_to_textgrid = os.path.join(path,"PhoneLabeling")
    tg_files = os.listdir(path_to_textgrid)
    
    path_to_sfs = os.path.join(path,"sfs")
    if not os.path.isdir(path_to_sfs):
        os.mkdir(path_to_sfs)
    files = os.listdir(path_to_sfs)
        
    print("Generating sfs alignment files!")
    for tg_file in tqdm(tg_files):
        
        tg = tgio.openTextgrid(os.path.join(path_to_textgrid,tg_file))
        
        entries = get_entries(path_to_textgrid,tg)
        
        generate_individual_sfs(path_to_sfs,tg_file,entries)

    print("\n"+"Done!"+"\n")
    


        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--path_to_data",dest='path', help="Please specify the path to data",
                    default="../BZNSYP")
	
    parser.add_argument("-f","--features",dest='feat', help="Please specify if features are to be generated",
                default=None)

    args = parser.parse_args()
    
    path = args.path
    
    get_all_sfs(path)
    
    if args.feat:
        proLab = get_prosodic_labels(path)
        generate_alignment(path,proLab)
