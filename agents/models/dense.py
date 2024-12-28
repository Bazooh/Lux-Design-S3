import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(23, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 16 * 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 16, 5)

        return x


class IndependantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        xs = torch.empty((x.shape[0], 16, 5), dtype=torch.float32)

        pos_sum = x[:, 6:, :, :].sum(dim=1)
        for i in range(16):
            xi = x[:, :6, :, :]
            pos = x[:, 6 + i, :, :]

            xi = torch.cat((xi, (pos_sum - pos).unsqueeze(1), pos.unsqueeze(1)), dim=1)

            xi = F.relu(self.conv1(xi))
            xi = F.relu(self.conv2(xi))
            xi = F.max_pool2d(xi, 2)
            xi = F.relu(self.conv3(xi))
            xi = F.max_pool2d(xi, 2)
            xi = xi.view(-1, 128 * 4 * 4)
            xi = F.relu(self.fc1(xi))
            xi = self.fc2(xi)
            xs[:, i, :] = xi

        return xs
