from configuration import Configuration
from layers.double_convolution import DoubleConvolution
from layers.downsample import DownSample
from layers.upsample import UpSample
from models.base_model import BaseModel

import torch
from torch.nn import functional as F, ReLU
from torch import nn


class UHeightNet(BaseModel):

    def __init__(self,
                 configuration: Configuration,
                 input_channels: int = 3,
                 num_layers: int = 5,
                 features_start: int = 64, ):
        super().__init__(configuration)

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        self.num_layers = num_layers

        layers = [DoubleConvolution(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(DownSample(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(UpSample(feats, feats // 2))
            feats //= 2

        layers.append(nn.Conv2d(feats, 1, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.upsample = nn.UpsamplingBilinear2d(configuration.output_size)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        z = self.layers[-1](xi[-1])
        return self.upsample(z)
