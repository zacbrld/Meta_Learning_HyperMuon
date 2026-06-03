import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """3-layer MLP for CIFAR-10. All weight matrices are 2D → Muon-native."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
