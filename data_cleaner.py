from os import walk

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchaudio
from  torchaudio import transforms
import os
import shutil
import re
from emotion import Emotion
import torch.nn.functional as F
import random
from torchaudio import functional as AF
import simpleaudio as sa
import time
import librosa
import soundfile

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()
  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show()


min_len = 100000000
max_len = 0
min_name = ""
max_name = ""
total = 0
counter = 0
# for i in os.listdir("data/raw"):
#   waveform, sample_rate = torchaudio.load(f"data/raw/{i}")
#   w = librosa.effects.trim(waveform[0], top_db=20)
#   length = len(waveform[0])
#   if length > max_len:
#     max_len = length
#     max_name = i
#   if length < min_len:
#     min_len = length
#     min_name = i
#   total += length
#
# print(min_len)
# print(min_name)
# print(max_len)
# print(max_name)
# print(total/(len(os.listdir("data/raw"))))
# w = w[0].reshape(1,len(w[0]))
# plot_waveform(w, sample_rate)
# plot_specgram(w, sample_rate)
# play_obj = sa.play_buffer(w.numpy(),1,4, sample_rate)
# # exit(0)

# low = 0;
# for i in os.listdir("data/clean"):
#   print(f"Processing {i}")
#   waveform, sample_rate = torchaudio.load(f"data/clean/{i}")
#   length = len(waveform[0])
#   #waveform = librosa.effects.trim(waveform[0], top_db=top_db)
#   #soundfile.write(f"data/cleaner/{i}", waveform[0], sample_rate)
#   #length = len(waveform[0])
#   if length < 60000:
#     low = low +1
#   if length > max_len:
#     max_len = length
#     max_name = i
#   if length < min_len:
#     min_len = length
#     min_name = i
#   total += length

for i in os.listdir("data/clean"):
  print(f"Processing {i}")
  waveform, sample_rate = torchaudio.load(f"data/clean/{i}")
  length = len(waveform[0])

  if length >= 60000:
    waveform = librosa.util.fix_length(waveform[0], 60000)
    soundfile.write(f"data/cleaner/{i}", waveform, sample_rate)
    length = len(waveform)

    if length > max_len:
      max_len = length
      max_name = i
    if length < min_len:
      min_len = length
      min_name = i
    total += length
    counter += 1

print(min_len)
print(min_name)
print(max_len)
print(max_name)
print(counter)
print(len(os.listdir("data/clean")))
print(total/counter)

exit(0)