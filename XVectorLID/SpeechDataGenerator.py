#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from utils import utils
        
class SpeechDataGenerator():
    def __init__(self, feat_file, label_file, maxlen=100):
        self.npy_data=np.load(feat_file, allow_pickle=True)
        self.labels=np.load(label_file, allow_pickle=True)
        self.spec_len=maxlen

    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        class_id = self.labels[idx]
        features = self.npy_data[idx].T
        randtime = np.random.randint(0, features.shape[1]-self.spec_len)
        mfbanks = features[:, randtime:randtime+self.spec_len]
        sample = {'features': torch.from_numpy(np.ascontiguousarray(mfbanks)), 'labels': torch.from_numpy(np.ascontiguousarray(int(class_id)))}
        return sample
        
class TestSpeechDataGenerator():
    def __init__(self, feat_file, label_file):
        self.npy_data=np.load(feat_file, allow_pickle=True)
        self.labels=np.load(label_file, allow_pickle=True)

    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        class_id = self.labels[idx]
        features = self.npy_data[idx].T
        sample = {'features': torch.from_numpy(np.ascontiguousarray(features)), 'labels': torch.from_numpy(np.ascontiguousarray(int(class_id)))}
        return sample
