from dataset_savee import DatasetSavee
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import librosa
import librosa.display

ds = DatasetSavee("dev")
dl = DataLoader(ds, batch_size=1, shuffle=False)

for waveform, mel_specgram, sample_rate, emotion, emotion_index, speaker, speaker_index in dl:
  librosa.display.specshow(mel_specgram[0][0].numpy(), y_axis='mel', x_axis='time');
  plt.title('Mel Spectrogram');
  plt.colorbar(format='%+2.0f dB');
  plt.show()
  plt.plot(waveform[0][0])
  plt.show()
