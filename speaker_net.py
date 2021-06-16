import torch.nn as nn
import torch.nn.functional as F


class SpeakerNet(nn.Module):
    def __init__(self):
        super(SpeakerNet, self).__init__()
        self.n_labels = 35
        self.net = nn.Sequential(
            nn.Conv2d(1,16, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(16, 32, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(66816,1000),
            nn.ReLU(),
            nn.Linear(1000, 4)

        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)