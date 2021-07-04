from torch.utils.data import Dataset
import torch
import torchaudio
from  torchaudio import transforms
import os
import shutil
import re
from emotion import Emotion
import torch.nn.functional as F

# filename = "data/dev/DC_su03.wav"
# waveform, sample_rate = torchaudio.load(filename)
# mel_specgram = transforms.MelSpectrogram(sample_rate, pad_mode="reflect", center=True, win_length=300)(waveform)
# print(mel_specgram.shape)
# print(waveform.shape)
#
# waveform2 = F.pad(waveform, (0,70000), "constant", value=0)
# mel_specgram = transforms.MelSpectrogram(sample_rate, pad_mode="reflect", center=True, win_length=300)(waveform2)
# print(mel_specgram.shape)
# print(waveform2.shape)
# torchaudio.save("temp.wav", waveform2, sample_rate)


filename = "data/dev/JK_f08.wav"
waveform, sample_rate = torchaudio.load(filename)
info = torchaudio.info(filename)
print(info.num_frames)
print(info.sample_rate)
mel_specgram = transforms.MelSpectrogram(sample_rate, pad_mode="reflect", center=True, win_length=300)(waveform)
print(mel_specgram.shape)
print(waveform.shape)

min = 1000000
max = 0
total = 0
for root, dirs, files in os.walk("data/raw"):
    for filename in files:
        waveform, sample_rate = torchaudio.load(f"data/raw/{filename}")
        length = waveform.shape[1]
        if length > max:
            max = length
        if length < min:
            min = length
        total = total + length

print(min)
print(max)
print(total/len(os.listdir("data/raw")))