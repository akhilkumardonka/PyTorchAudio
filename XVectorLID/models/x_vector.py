#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from torchsummary import summary

#####################################################
#Network Architecture
#####################################################

class Xvector(nn.Module):
    ######################################################
    ##Takes input as wave file and produces speaker logits
    ######################################################
    def __init__(self, outClasses):
        super(Xvector, self).__init__()

        self.outClasses = outClasses
        self.n_mels     = 64
        self.log_input  = True      
 
        self.instancenorm   = nn.InstanceNorm1d(self.n_mels)
        self.torchfb        = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=self.n_mels)
                )

        p_dropout = 0.1 

        self.tdnn1 = nn.Conv1d(in_channels=self.n_mels, out_channels=64, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(64, momentum=0.1, affine=True)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(64, momentum=0.1, affine=True)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(64, momentum=0.1, affine=True)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(64, momentum=0.1, affine=True)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(128, momentum=0.1, affine=True)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(256,256)
        self.bn_fc1 = nn.BatchNorm1d(256, momentum=0.1, affine=True)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(256,256)
        self.bn_fc2 = nn.BatchNorm1d(256, momentum=0.1, affine=True)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)
 
        self.fc3 = nn.Linear(256,self.outClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        x_input = x
        
        #if self.training:
        #  x_input = self.instancenorm(x_input)
        #x  = x.reshape(-1,x.size()[-1])
        #if self.training:
        #    with torch.no_grad():
        #        with torch.cuda.amp.autocast(enabled=False):
        #            x = self.torchfb(x)+1e-6
        #            if self.log_input: x = x.log()
        #            x_input = self.instancenorm(x)
        #x_input = x
        
        x = self.dropout_tdnn1(self.bn_tdnn1(self.tdnn1(x_input)))
        x = self.dropout_tdnn2(self.bn_tdnn2(self.tdnn2(x)))
        x = self.dropout_tdnn3(self.bn_tdnn3(self.tdnn3(x)))
        x = self.dropout_tdnn4(self.bn_tdnn4(self.tdnn4(x)))
        x = self.dropout_tdnn5(self.bn_tdnn5(self.tdnn5(x)))

        eps = 0.0000001
        if self.training:
            shape = x.size()
            noise = torch.FloatTensor(shape)
            noise = noise.to("cuda")
            torch.randn(shape, out=noise)
            x += noise*eps

        mean = x.mean(dim=2)
        variance =  x.std(dim=2)
        stats = torch.cat((mean,variance),1)

        xvec = self.fc1(stats)
        x = self.dropout_fc1(self.bn_fc1(xvec))
        x = self.dropout_fc2(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        
        # else:
        #     x = self.fc1(stats) #Typically considering this activation potential as speaker embedding generalizes well.
        #     #x = self.dropout_fc1(self.bn_fc1(self.fc1(stats)))
        
        #     #x = self.fc2(x) #These activation potentials are slightly overfitted to training speakers and may not generalize well.
        
        return x, xvec
        
if __name__ == '__main__':
	model = Xvector(4)
	summary(model, (64, 100))
