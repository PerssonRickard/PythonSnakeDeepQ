import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class valueFunctionNetwork(nn.Module):
    def __init__(self):
        super(valueFunctionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(16, 8, 8)
        self.conv2 = nn.Conv2d(32, 4, 4)
        self.fc1 = nn.Linear(32*74*74, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32*74*74)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


valueFunction = valueFunctionNetwork()

print(torch.cuda.is_available())



