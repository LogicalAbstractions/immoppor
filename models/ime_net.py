from configuration import Configuration
from models.backbones.se_net import se_net154
from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImeNetUpProjection(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class ImeNetSeEncoder(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super().__init__()

        self.base = nn.Sequential(*list(original_model.children())[:-3])
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.down = ImeNetUpProjection(64, 128)

    def forward(self, x):
        x_block0 = self.base[0][0:6](x)
        x = self.base[0][6:](x_block0)

        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)
        return x_block0, x_block1, x_block2, x_block3, x_block4


class ImeNetDecoder(nn.Module):

    def __init__(self, num_features=2048):
        super().__init__()

        self.conv = nn.Conv2d(num_features, num_features //
                              2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = ImeNetUpProjection(1024, 512)  # out 512 channels

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = ImeNetUpProjection(512, 256)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = ImeNetUpProjection(256, 128)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = ImeNetUpProjection(128, 64)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))

        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_block3 = F.relu(self.bn1(self.conv1(x_block3)))  # 512
        cx_d1 = torch.cat((x_d1, x_block3), 1)  # 512
        cx_d1 = F.relu(self.bn1(self.conv1(cx_d1)))  # 512

        x_d2 = self.up2(cx_d1, [x_block2.size(2), x_block2.size(3)])
        x_block2 = F.relu(self.bn2(self.conv2(x_block2)))
        cx_d2 = torch.cat((x_d2, x_block2), 1)
        cx_d2 = F.relu(self.bn2(self.conv2(cx_d1)))

        x_d3 = self.up3(cx_d2, [x_block1.size(2), x_block1.size(3)])

        x_block1 = F.relu(self.bn3(self.conv3(x_block1)))
        cx_d3 = torch.cat((x_d3, x_block1), 1)
        cx_d3 = F.relu(self.bn3(self.conv3(cx_d3)))

        x_d4 = self.up4(cx_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        cx_d4 = torch.cat((x_d4, x_block0), 1)
        cx_d4 = F.relu(self.bn3(self.conv4(cx_d4)))

        return cx_d4  # 128 chanel


class ImeNetMultiFusion(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super().__init__()

        self.up0 = ImeNetUpProjection(
            num_input_features=64, num_output_features=16)

        self.up1 = ImeNetUpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.up2 = ImeNetUpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.up3 = ImeNetUpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = ImeNetUpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            80, 80, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(80)

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4, size):
        x_m0 = self.up0(x_block0, size)
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m0, x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class ImeNetConvolution(nn.Module):
    def __init__(self, block_channel):
        super().__init__()

        self.conv0 = nn.Conv2d(208, 144,
                               kernel_size=1, stride=1)

        self.bn0 = nn.BatchNorm2d(144)

        self.conv1 = nn.Conv2d(144, 144,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(144)

        self.conv2 = nn.Conv2d(144, 144, kernel_size=5, stride=1, padding=2, bias=True)

        self.bn2 = nn.BatchNorm2d(144)

        self.conv3 = nn.Conv2d(144, 72, kernel_size=3, padding=1, stride=1)

        self.bn3 = nn.BatchNorm2d(72)

        self.conv4 = nn.Conv2d(72, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(x3)

        return x4


class ImeNet(BaseModel):
    def __init__(self, configuration: Configuration):
        super().__init__(configuration)

        block_channel = [256, 512, 1024, 2048]

        self.encoder = ImeNetSeEncoder(se_net154(pretrained='imagenet'))
        self.decoder = ImeNetDecoder(num_features=2048)
        self.multifusion = ImeNetMultiFusion(block_channel)
        self.conv = ImeNetConvolution(block_channel)
        self.upsample = nn.Upsample(configuration.output_size)

    def forward(self, x):
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.encoder(x)
        x_decoded = self.decoder(x_block0, x_block1, x_block2, x_block3, x_block4)

        x_mff = self.multifusion(x_block0, x_block1, x_block2, x_block3, x_block4,
                                 [x_decoded.size(2), x_decoded.size(3)])

        out = self.conv(torch.cat((x_decoded, x_mff), 1))
        return self.upsample(out)
