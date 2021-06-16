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

class DatasetSavee(Dataset):
    def __init__(self, mode):
        self.ROOT = "data/cleaner"

        self.DEV = "data/dev"
        self.TRAIN = "data/train"
        self.TEST = "data/test"

        self.mode = mode.strip().lower()
        self.dir = None

        if self.mode == "dev":
            self.dir = self.DEV
        elif self.mode == "train":
            self.dir = self.TRAIN
        elif self.mode == "test":
            self.dir = self.TEST
        else:
            self.dir = "data/cleaner"
            self.mode = "all"

        if not os.path.exists("data/dev"):
            self.prepare()

        self.files = os.listdir(self.dir)
        self.emotions = []
        self.speakers = []

        for f in self.files:
            m = re.search('(.+?)_(.+?)[0-9][0-9].wav', f)
            self.speakers.append(m.group(1))
            self.emotions.append(m.group(2))

        self.unique_emotions = list(set(self.emotions))
        self.unique_emotions.sort()
        self.unique_speakers = list(set(self.speakers))
        self.unique_speakers.sort()

    def prepare(self):
        self.delete_dirs()
        self.create_dirs()
        prepared_files = self.get_emotion_file_dictionary()

        for key, value in prepared_files.items():
            n_dev, n_train, n_test = self.get_mode_counts(len(value))
            self.process_file_list(value, n_dev, n_train)

    def get_mode_counts(self, size):
        n_test = int(size // 10 * 1.5)
        n_train = int(size // 10 * 8)
        n_dev = size - (n_test + n_train)
        return n_dev, n_train, n_test

    def process_file_list(self, files, n_dev, n_train):
        random.shuffle(files)
        self.make_data_for(files[0:n_dev], self.DEV)
        self.make_data_for(files[n_dev: n_dev + n_train], self.TRAIN)
        self.make_data_for(files[n_dev + n_train:], self.TEST)

    def make_data_for(self, list, mode):
        for file in list:
            shutil.copyfile(f"data/cleaner/{file}", f"{mode}/{file}")

    def __get_mode_counts__(self, size):
        n_test = int(size // 10 * 1.5)
        n_train = int(size // 10 * 8)
        n_dev = size - (n_test + n_train)
        return n_dev, n_train, n_test

    def get_emotion_file_dictionary(self):
        prepared_files = {}
        for root, dirs, files in os.walk(self.ROOT):
            for filename in files:
                m = re.search('(.+?)_(.+?)[0-9][0-9].wav', filename)
                if m:
                    emotion = m.group(2)
                    if emotion not in prepared_files:
                        prepared_files[emotion] = []
                    prepared_files[emotion].append(filename)
        return prepared_files

    def delete_dir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def delete_dirs(self):
        self.delete_dir(self.DEV)
        self.delete_dir(self.TRAIN)
        self.delete_dir(self.TEST)

    def create_dirs(self):
        os.mkdir(self.DEV)
        os.mkdir(self.TRAIN)
        os.mkdir(self.TEST)

    def stat(self):
        print(f"Total dev files: {len(os.listdir(self.DEV))}")
        print(f"Total train files: {len(os.listdir(self.TRAIN))}")
        print(f"Total test files: {len(os.listdir(self.TEST))}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path = os.path.join(self.dir, self.files[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        mel_specgram = transforms.MelSpectrogram(sample_rate, normalized=True)(waveform)
        emotion = self.emotions[idx]
        emotion_index = self.unique_emotions.index(emotion)
        speaker = self.speakers[idx]
        speaker_index = self.unique_speakers.index(speaker)
        return mel_specgram, sample_rate, emotion, emotion_index, speaker, speaker_index

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = DatasetSavee("all")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    for waveform, mel_specgram, sample_rate, emotion, emotion_index, speaker, speaker_index in dataloader:
        print(mel_specgram.shape, emotion, speaker)
        print(mel_specgram[0,0,0:2,0:2])
        break
