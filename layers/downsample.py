import torch
from torch.nn import functional as F
from torch import nn

from layers.double_convolution import DoubleConvolution


class DownSample(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConvolution(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)