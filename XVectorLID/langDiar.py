import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader   
from models.x_vector import Xvector
from utils.utils import collate_fn
from SpeechDataGenerator import TestSpeechDataGenerator
import matplotlib.pyplot as plt
from sklearn import metrics
import librosa

def load_wav(audio_filepath, min_dur_sec=3):
    audio_data,sr  = librosa.load(audio_filepath,sr=16000)
    len_file = len(audio_data)    
    if len_file <int(min_dur_sec*sr):
        dummy=np.zeros((1,int(min_dur_sec*sr)-len_file))
        extened_wav = np.concatenate((audio_data,dummy[0]))
    else:
        extened_wav = audio_data
    return extened_wav

def feature_extraction(filepath, min_dur_sec=3):
    audio_data = load_wav(filepath,min_dur_sec=min_dur_sec)
    mel_filter_banks = librosa.feature.melspectrogram(y = audio_data, n_mels=64, win_length=400, hop_length=160)
    mag_T = np.log10(mel_filter_banks+1e-5).T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    return (mag_T - mu) / (std + 1e-5)

best_model_path = "/home/akhil/SpeechSystemsProject/XVector_LID/checkpoints/best_check_point_43_0.06625567802500584"

outClasses = 4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 1
model=Xvector(outClasses)
model.load_state_dict(torch.load(best_model_path)["model"])
model.eval()
model = model.to(device)

                
if __name__ == '__main__':
	wavfile = "./test_audio_wavs/hindi_english.wav"
	feature = feature_extraction(wavfile)
	win_length = 100 # 1 second segment
	hop = 20 # 0.2 second segment hop
	feat = torch.from_numpy(np.ascontiguousarray(feature.T))
	predictions = []
	for j in range(0, feat.shape[-1]-win_length, hop):
		inp = feat[:,j:j+win_length]
		inp = inp.to(device)
		pred_logits,x_vec = model(inp[None,:,:])
		predictions.append(np.argmax(pred_logits.detach().cpu().numpy(),axis=1))
	print(predictions)
