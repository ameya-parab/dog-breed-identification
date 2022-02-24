import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.drop1 = nn.Dropout(0.20)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.drop3 = nn.Dropout(0.30)

        self.fc1 = nn.Linear(in_features=5184, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=120)

    def forward(self, data):

        data = self.conv1(data)
        data = F.leaky_relu(data)
        data = F.max_pool2d(data, 2)
        data = self.drop1(data)

        data = self.conv2(data)
        data = F.leaky_relu(data)
        data = F.max_pool2d(data, 2)
        data = self.drop2(data)

        data = self.conv3(data)
        data = F.leaky_relu(data)
        data = F.max_pool2d(data, 2)
        data = self.drop3(data)

        data = torch.flatten(data, start_dim=1)

        data = self.fc1(data)
        data = F.leaky_relu(data)

        data = self.fc2(data)
        data = F.leaky_relu(data)

        data = self.fc3(data)

        return data
