import torch
from kornia.contrib import VisionTransformer
from torch.nn import ReLU
from torchvision.models import resnet18, resnet152
from torchvision.transforms import transforms

from configuration import Configuration
from layers.double_convolution import DoubleConvolution
from layers.upsample import UpSample
from models.base_model import BaseModel

import torch.nn as nn
from torch.nn import functional as F


class PyramidNetEncoder(nn.Module):

    def __init__(self, pretrained):
        super().__init__()

        encoder_model = resnet18(pretrained=pretrained)
        encoder_layers = list(encoder_model.children())

        self.encoder_layer0 = nn.Sequential(*encoder_layers[:3])
        self.encoder_layer1 = nn.Sequential(*encoder_layers[3:5])
        self.encoder_layer2 = nn.Sequential(*encoder_layers[5:8])

    def forward(self, x):
        encoded0 = self.encoder_layer0(x)
        encoded1 = self.encoder_layer1(encoded0)
        encoded2 = self.encoder_layer2(encoded1)

        return encoded0, encoded1, encoded2


class PyramidNet(BaseModel):
    def __init__(self, configuration: Configuration, encoders: int = 4):
        super().__init__(configuration)

        self.encoders = nn.ModuleList([PyramidNetEncoder(True) for x in range(0, encoders)])
        self.deconv0 = nn.LazyConvTranspose2d(32, kernel_size=3, stride=4)
        self.act1 = nn.ReLU(inplace=True)
        self.deconv1 = nn.LazyConvTranspose2d(16, kernel_size=3, stride=2)
        self.act2 = nn.ReLU(inplace=True)
        self.deconv2 = nn.LazyConvTranspose2d(1, kernel_size=3, stride=2)
        self.act3 = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingNearest2d(configuration.output_size)

    def forward(self, x):
        encoded0 = []
        encoded1 = []
        encoded2 = []

        for i, encoder in enumerate(self.encoders):
            e0, e1, e2 = encoder(x)
            encoded0.append(e0)
            encoded1.append(e1)
            encoded2.append(e2)

        encoded0 = torch.cat(encoded0, dim=1)
        encoded1 = torch.cat(encoded1, dim=1)
        encoded2 = torch.cat(encoded2, dim=1)

        decoded = self.deconv0(encoded2)
        decoded = self.act1(decoded)
        decoded = self.deconv1(decoded)
        decoded = self.act2(decoded)
        decoded = self.deconv2(decoded)
        decoded = self.act3(decoded)

        upsampled = self.upsample(decoded)

        return upsampled
