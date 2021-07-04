import torchaudio
import os
import librosa
import soundfile


for i in os.listdir("data/raw"):
  path = os.path.join("data/raw", i)
  if not os.path.isdir(path):
    continue
  for j in os.listdir(path):
    file = os.path.join(path, j)
    print(f"Processing {file}")
    waveform, sample_rate = torchaudio.load(file)
    length = len(waveform[0])

    if length >= 60000:
      waveform = librosa.util.fix_length(waveform[0], 60000)

    soundfile.write(f"data/cleaner/{i}_{j}", waveform, sample_rate)
    length = len(waveform)


