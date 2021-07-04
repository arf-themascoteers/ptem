from dataset_savee import DatasetSavee
from emotion_net import EmotionNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

QUICK = False

def train():
    NUM_EPOCHS = 100
    BATCH_SIZE = 400

    mode = "train"
    if QUICK:
        mode = "dev"

    working_set = DatasetSavee(mode)
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = EmotionNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch  in range(0, NUM_EPOCHS):
        for wav, mel, sample_rate, emotion, emotion_index, speaker, speaker_index in dataloader:
            optimizer.zero_grad()
            y_pred = model(mel)
            loss = F.nll_loss(y_pred, emotion_index)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    torch.save(model.state_dict(), 'models/emotion.h5')
    return model

train()
exit(0)


