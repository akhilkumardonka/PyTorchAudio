import numpy as np
import librosa
import math

def load_wav(audio_filepath, min_dur_sec=3):
    audio_data,sr  = librosa.load(audio_filepath,sr=16000)
    
    len_file = len(audio_data)

    if len_file <int(min_dur_sec*sr):
        dummy=np.zeros((1,int(min_dur_sec*sr)-len_file))
        extened_wav = np.concatenate((audio_data,dummy[0]))
    else:
        extened_wav = audio_data
    
    return extened_wav

def feature_extraction(filepath, noise, min_dur_sec=3):
    audio_data = load_wav(filepath, min_dur_sec=min_dur_sec)
    
    # adding noise
    randtime = np.random.randint(0, len(noise)-len(audio_data))
    rand_noise = noise[randtime:randtime+len(audio_data)]
    speech_rms = np.sqrt(np.mean(audio_data**2))
    noise_rms = np.sqrt(np.mean(rand_noise**2))
    snr_db = 20
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / (speech_rms + 1e-5)
    audio_data = (scale * audio_data + rand_noise) / 2
    mel_filter_banks = librosa.feature.melspectrogram(y = audio_data, n_mels=64, win_length=400, hop_length=160)
    
    # # vad
    # energies = librosa.feature.rms(y=audio_data, frame_length = 400, hop_length = 160)
    # indexes = energies[0] > 0.02
    # if len(indexes) == mel_filter_banks.shape[1]:
    #   mel_filter_banks = mel_filter_banks[:,indexes]
    	
    mag_T = np.log10(mel_filter_banks+1e-5).T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    return (mag_T - mu) / (std + 1e-5)
    
def collate_fn(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['features'])
        targets.append((sample['labels']))
    return specs, targets
