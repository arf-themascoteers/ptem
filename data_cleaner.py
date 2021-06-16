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





for i in os.listdir("data/clean"):
  print(f"Processing {i}")
  waveform, sample_rate = torchaudio.load(f"data/clean/{i}")
  length = len(waveform[0])

  if length >= 60000:
    waveform = librosa.util.fix_length(waveform[0], 60000)
    soundfile.write(f"data/cleaner/{i}", waveform, sample_rate)
    length = len(waveform)


