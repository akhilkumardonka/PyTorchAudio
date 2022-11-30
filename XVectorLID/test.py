import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader   
from models.x_vector import Xvector
from utils.utils import collate_fn
from SpeechDataGenerator import TestSpeechDataGenerator
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import librosa

# testing_filepath = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/out_of_dom/xvec_features.npy"
# testing_labels = "/home/akhil/SpeechSystemsProject/XVector_LID/projectDataFeats/out_of_dom/xvec_labels.npy"
# best_model_path = "/home/akhil/SpeechSystemsProject/XVector_LID/checkpoints/best_check_point_43_0.06625567802500584"

# outClasses = 4
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# batch_size = 1
# model=Xvector(outClasses)
# model.load_state_dict(torch.load(best_model_path)["model"])
# model.eval()
# model = model.to(device)
# dataset_test = TestSpeechDataGenerator(testing_filepath, testing_labels)
# dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# def test(dataloader_val):
#     with torch.no_grad():
#         full_preds=[]
#         full_gts=[]
#         x_vecs = []
#         for i_batch, sample_batched in enumerate(dataloader_val):
#             features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
#             labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
#             features, labels = features.to(device),labels.to(device)
#             pred_logits,x_vec = model(features)
#             predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
#             x_vecs.append(x_vec.detach().cpu().numpy().tolist())
#             for pred in predictions:
#                 full_preds.append(pred)
#             for lab in labels.detach().cpu().numpy():
#                 full_gts.append(lab)
        
#         return np.array(full_preds), np.array(full_gts), np.array(x_vecs)

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
                
if __name__ == '__main__':
	# di = {0: "English", 1: "Hindi", 2: "Tamil", 3: "Telugu"}
	# pred, true, x_vecs = test(dataloader_test)
	# confusion = metrics.confusion_matrix(true, pred)
	# #print(metrics.classification_report(true, pred, labels=["English", "Hindi", "Tamil", "Telugu"]))
	# x_vecs = np.squeeze(x_vecs, axis=1)
	# correct = np.sum(pred == true)
	# accuracy = correct/len(true)
	# print(accuracy)
	# display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels=["English", "Hindi", "Tamil", "Telugu"])
	# display.plot()
	# plt.show()
	
	# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	# tsne_results = tsne.fit_transform(x_vecs)
	
	# df_subset = pd.DataFrame()
	
	# df_subset['Dimension 1'] = tsne_results[:,0]
	# df_subset['Dimension 2'] = tsne_results[:,1]
	# df_subset['True Labels'] = true
	# df_subset = df_subset.replace({'True Labels': di})

	# plt.figure(figsize=(16,10))
	# sns.scatterplot(
	#     x='Dimension 1', y='Dimension 2',
	#     hue='True Labels',
	#     palette=sns.color_palette("hls", 4),
	#     data=df_subset,
	#     legend="full",
	#     alpha=0.7
	# )
	# plt.show()

	wavfile = "/home/akhil/SpeechSystemsProject/XVector_LID/diarization_test_files/hindi_english/HinEng_Codeswitch.wav"
	feature = feature_extraction(wavfile)[:2000,:]
	
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(feature)

	df_subset = pd.DataFrame()
	
	df_subset['Dimension 1'] = tsne_results[:,0]
	df_subset['Dimension 2'] = tsne_results[:,1]
	df_subset['True Labels'] = np.ones(feature.shape[0])

	plt.figure(figsize=(16,10))
	sns.scatterplot(
	    x='Dimension 1', y='Dimension 2',
	    hue='True Labels',
	    palette=sns.color_palette("hls", 4),
	    data=df_subset,
	    legend="full",
	    alpha=0.7
	)
	plt.show()


