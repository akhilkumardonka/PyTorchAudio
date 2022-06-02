import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from tqdm import tqdm

class AcousticScene(Dataset):
    
    def __init__(self, annotations_file, base_dir, in_col, out_col, transformation, target_sample_rate, num_samples, device):
        
        self.df = pd.read_csv(annotations_file, sep="\t")
        self.data = []
        self.labels = []
        self.c2i={}
        self.i2c={}
        self.categories = sorted(self.df[out_col].unique())
        
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        for i, category in enumerate(self.categories):
            self.c2i[category]=i
            self.i2c[i]=category
        for ind in tqdm(range(len(self.df))):
            row = self.df.iloc[ind]
            file_path = os.path.join(base_dir, row[in_col])
                
            signal, sr = torchaudio.load(file_path)
            signal = signal.to(self.device)
        
            # resampling to bring uniformity
            signal = self._resample_if_necessary(signal, sr)

            # data cleaning to convert all wav files into monoaudio
            signal = self._mix_down_if_necessary(signal)

            # Preprocessing Audio of Different Length
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)

            #applying transformation
            signal = self.transformation(signal)
            
            self.data.append(signal)
            self.labels.append(self.c2i[row[out_col]])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):        
        return self.data[index], self.labels[index]
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        # no. of channels : signal.shape[0]
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

if __name__ == "__main__":
    BASE_DIR = "/home/akhil/models/siplab_model/datasets/TAU-urban-acoustic-scenes-2020-mobile-development"
    ANNOTATIONS_FILE = BASE_DIR + "/meta.csv"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE * 10

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        win_length = 640,
        hop_length = 320,
        n_mels = 40
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    asc = AcousticScene(ANNOTATIONS_FILE, BASE_DIR, 'filename', 'scene_label', mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print("Total Train Files : ", len(asc))
