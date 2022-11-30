#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils import utils
from tqdm import tqdm
import librosa

def extract_features(audio_filepath, noise):
    features = utils.feature_extraction(audio_filepath, noise)
    return features 

def FE_pipeline(feature_list,store_loc,mode):
    
    input_data=[]
    input_labels=[]
    noise_path = "/media/musan/noise/free-sound/mergedNoise.wav"
    noise, noise_sr = librosa.load(noise_path, sr=16000)
    
    folder = os.path.join(store_loc, mode)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for row in tqdm(feature_list):
        filepath = row.split(' ')[0]
        lang_id = row.split(' ')[1]
        extract_feats = extract_features(filepath, noise)
        input_data.append(extract_feats)
        input_labels.append(int(lang_id))
        
    np.save(os.path.join(folder, 'xvec_features.npy'),input_data)
    np.save(os.path.join(folder, 'xvec_labels.npy'),input_labels)
    

if __name__ == '__main__':
    store_loc = './projectDataFeats'
    
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
        
    read_train = [line.rstrip('\n') for line in open('meta/training.txt')]
    FE_pipeline(read_train,store_loc,mode='train')
    
    read_test = [line.rstrip('\n') for line in open('meta/testing.txt')]
    FE_pipeline(read_test,store_loc,mode='test')
    
    read_val = [line.rstrip('\n') for line in open('meta/validation.txt')]
    FE_pipeline(read_val,store_loc,mode='validation')
    
    
    
    
    
