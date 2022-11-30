#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import argparse

class_ids = {'English':0, 'Hindi':1, 'Tamil':2, 'Telugu':3}

def create_meta(files_list,store_loc,mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    
    if mode=='test':
        meta_store = store_loc+'/out_of_dom.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    else:
        print('Error in creating meta files')
    
def extract_files(folder_path):
    all_lang_folders = sorted(glob.glob(folder_path+'/*/'))
    test_lists = []
    
    for lang_folderpath in all_lang_folders:
        
        language = lang_folderpath.split('/')[-2]
        all_files = sorted(glob.glob(lang_folderpath+'/*.wav'))
        train_nums = len(all_files)
        
        for i in range(len(all_files)):
            to_write = all_files[i]+' '+str(class_ids[language])
            test_lists.append(to_write)
            
    return test_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--processed_data", default="/media/out_of_domain_LID", type=str,help='Dataset path')
    parser.add_argument("--meta_store_path", default="../meta/", type=str,help='Save directory after processing')
    config = parser.parse_args()
    
    test_list= extract_files(config.processed_data)
    create_meta(test_list,config.meta_store_path,mode='test')

