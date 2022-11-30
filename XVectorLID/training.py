import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader   
from torch import optim
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os
import librosa
import random
import sklearn.metrics
from models.x_vector import Xvector
from SpeechDataGenerator import SpeechDataGenerator
from utils.utils import collate_fn

############### plotting ###############

def plotInfo(train_vals, valid_vals, title, filename):
    epochs = np.arange(0, len(train_vals), dtype=int)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_vals, color='r', label='Training')
    plt.plot(epochs, valid_vals, color='b', label='Validation')

    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

#######################################
#Train the network
#######################################

training_filepath = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/train/xvec_features.npy"
testing_filepath = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/test/xvec_features.npy"
validation_filepath = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/validation/xvec_features.npy"

training_labels = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/train/xvec_labels.npy"
testing_labels = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/test/xvec_labels.npy"
validation_labels = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/validation/xvec_labels.npy"

outClasses = 4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model=Xvector(outClasses).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss()

batch_size = 8
num_epochs = 50

### Data related
dataset_train = SpeechDataGenerator(training_filepath, training_labels, maxlen=300)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

dataset_val = SpeechDataGenerator(validation_filepath, validation_labels, maxlen=300)
dataloader_val = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

dataset_test = SpeechDataGenerator(testing_filepath, testing_labels, maxlen=300)
dataloader_test = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

check_points_path='/home/akhil/SpeechSystemsProject/XVector_LID/checkpoints'
if not os.path.exists(check_points_path):
    os.system('mkdir -p %s'%check_points_path)

def train(dataloader_train, epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    
    for i_batch, sample_batched in enumerate(dataloader_train):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device),labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits,x_vec = model(features)
        loss = criterion(pred_logits,labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
    return mean_acc, mean_loss
    
    
def validation(dataloader_val,epoch):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            loss = criterion(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total validation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
        
        model_save_path = os.path.join(check_points_path, 'best_check_point_'+str(epoch)+'_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
    	
        return mean_acc, mean_loss
        
if __name__ == '__main__':

    train_means_acc = []
    val_means_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        tr_mean, tr_loss = train(dataloader_train,epoch)
        vl_mean, vl_loss = validation(dataloader_val,epoch)
        train_means_acc.append(tr_mean)
        train_loss.append(tr_loss)
        val_means_acc.append(vl_mean)
        val_loss.append(vl_loss)
        
    plotInfo(train_loss, val_loss, "Loss", "tdnn_loss.png")
    plotInfo(train_means_acc, val_means_acc, "Accuracy", "tdnn_acc.png")
