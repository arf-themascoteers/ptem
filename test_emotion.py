from dataset_savee import DatasetSavee
from emotion_net import EmotionNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

def test():
    BATCH_SIZE = 1

    working_set = DatasetSavee("test")
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = EmotionNet()
    model.load_state_dict(torch.load("models/emotion.h5"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, sample_rate, emotion, emotion_index, speaker, speaker_index in dataloader:
            y_pred = model(mel)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(emotion_index.data.view_as(pred)).sum()
            total += 1

    print(f"{correct} correct among {len(working_set)}")

test()
exit(0)