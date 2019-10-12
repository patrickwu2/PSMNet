import torch
import torch.nn as nn
import torch.nn.functional as F


class ExNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 4x4 -> 32x32
        self.Rnet = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 2, stride=2),  # 8x8
            nn.ConvTranspose2d(1, 1, 2, stride=2),  # 16x16
            nn.ConvTranspose2d(1, 1, 2, stride=2)   # 32x32
        )
        self.Tnet = nn.Linear(16, 32)

    def forward(self, inputs):
        R_mat = self.Rnet(inputs)
        T_mat = self.Tnet(inputs.reshape(-1, 1, 1, 16))

        return R_mat, T_mat


