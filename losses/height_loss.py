import torch.nn as nn


class HeightLoss(nn.Module):
    def __init__(self, min_height=-300.0, max_height=300.0, eps=1e-6):
        super().__init__()

        self.min_height = min_height
        self.max_height = max_height
        self.eps = eps

    def forward(self, y, z):
        height_range = self.max_height - self.min_height

        loss = ((y - z).abs()).mean()
        return loss
