import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImeSobel(nn.Module):

    def __init__(self):
        super().__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class ImeLoss(nn.Module):
    def __init__(self, output_size: tuple[int, int], batch_size: int):
        super().__init__()
        self.sobel = ImeSobel()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.ones = torch.ones(batch_size, 1, output_size[0], output_size[1]).float().cuda()

    def forward(self, output, depth):
        depth_grad = self.sobel(depth)
        output_grad = self.sobel(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, self.ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, self.ones), 1)
        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        return loss
