from torch.nn import ReLU
from torchvision.models import resnet18, resnet152
from torchvision.transforms import transforms

from configuration import Configuration
from layers.upsample import UpSample
from models.base_model import BaseModel

import torch.nn as nn


class PyramidNetResNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        encoder_model = resnet152(pretrained=True)
        encoder_layers = list(encoder_model.children())

        self.encoder_layer0 = nn.Sequential(*encoder_layers[:3])
        self.encoder_layer1 = nn.Sequential(*encoder_layers[3:5])
        self.encoder_layer2 = nn.Sequential(*encoder_layers[5:6])
        self.encoder_layer3 = nn.Sequential(*encoder_layers[6:7])
        self.encoder_layer4 = nn.Sequential(*encoder_layers[7:8])
        self.encoder_layer5 = nn.Sequential(*encoder_layers[8:9])

    def forward(self, x):
        encoded0 = self.encoder_layer0(x)
        encoded1 = self.encoder_layer1(encoded0)
        encoded2 = self.encoder_layer2(encoded1)
        encoded3 = self.encoder_layer3(encoded2)
        encoded4 = self.encoder_layer4(encoded3)
        encoded5 = self.encoder_layer5(encoded4)

        return encoded5


class PyramidNet(BaseModel):
    def __init__(self, configuration: Configuration):
        super().__init__(configuration)

        self.encoder = PyramidNetResNetEncoder()

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
