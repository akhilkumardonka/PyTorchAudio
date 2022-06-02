import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import pickle

from acousticscenedataset import AcousticScene
from models import ASC10Model
from utilities import LearningRateHelpers, PlotHelpers

def load_data(batch_size, train_asc, validate_asc):
    return (DataLoader(train_asc, batch_size, shuffle=True), DataLoader(validate_asc, batch_size, shuffle=False))

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, device, change_lr=None):
    
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        batch_losses=[]

        # changing learning rates
        if change_lr:
            optimizer = change_lr(optimizer, epoch)

        # forward and backward propagations with train data
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()

        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

        # getting validation losses
        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []

        # forward and backward propagations with validation data
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            batch_losses.append(loss.item())

        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)

        # predictions & accuracy
        predictions = trace_yhat.argmax(axis=1)
        accuracy = np.mean(predictions==trace_y)
    
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
    
if __name__ == "__main__":
    
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE * 10
    BASE_DIR = "../datasets/TAU-urban-acoustic-scenes-2020-mobile-development"
    
    if torch.cuda.is_available():
        device = "cuda:3"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        win_length = 640,
        hop_length = 320,
        n_mels = 40
    )
    
    print("Creating Training and Validation PyTorch Datasets & Batch Iterators : ===================>\n")
    
    tfname = 'train_pickle.pkl'
    vfname = 'valid_pickle.pkl'
    pickleDirectory = "./dataset_pickle/"
    
    if(os.path.isfile(pickleDirectory + tfname)):
        file = open(pickleDirectory + tfname, 'rb')     
        train_asc = pickle.load(file)
    else:
        TRAIN_ANNOTATIONS_FILE = BASE_DIR + "/evaluation_setup/fold1_train.csv"
        train_asc = AcousticScene(TRAIN_ANNOTATIONS_FILE, BASE_DIR, 'filename', 'scene_label', mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
        file = open(pickleDirectory + tfname, 'ab')
        pickle.dump(train_asc, file)                     
        file.close()
            
    print("Total Train Files : ", len(train_asc))
    
    if(os.path.isfile(pickleDirectory + vfname)):
        file = open(pickleDirectory + vfname, 'rb')     
        validate_asc = pickle.load(file)
    else:
        VALIDATION_ANNOTATIONS_FILE = BASE_DIR + "/evaluation_setup/fold1_evaluate.csv"
        validate_asc = AcousticScene(VALIDATION_ANNOTATIONS_FILE, BASE_DIR, 'filename', 'scene_label', mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
        file = open(pickleDirectory + vfname, 'ab')
        pickle.dump(validate_asc, file)                     
        file.close()
        
    print("Total Test Files : ", len(validate_asc))
    
    train_iter, valid_iter = load_data(BATCH_SIZE, train_asc, validate_asc)
    print("Finished\n\n")
    
    print("Printing Model Summary : ===================>\n")
    model = ASC10Model(input_shape=(1,40,501), batch_size=64, num_cats=10)
    summary(model.cuda(), input_size=(1, 1, 40, 501))
    model.to(device)
    
    # initialise loss funtion & optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    print("Initiating Model Training : ===================>\n")
    train_losses = []
    valid_losses = []
    train(model, loss_fn, train_iter, valid_iter, EPOCHS, optimizer, train_losses, valid_losses, device)

    # save model and loss plots
    PlotHelpers.lossPlot(train_losses, valid_losses)
    
    with open('asc10resnet.pth','wb') as f:
        torch.save(model, f)
        
    print("Trained neural network saved at asc10resnet.pth")
