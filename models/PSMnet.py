import math
import torch
import torch.nn as nn
from models.costnet import CostNet
from models.exnet import ExNet
from models.stackedhourglass import StackedHourglass


class PSMNet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.extrinsic_net = ExNet()
        self.cost_net = CostNet()
        self.stackedhourglass = StackedHourglass(max_disp)
        self.D = max_disp

        self.__init_params()

    def forward(self, frame1, frame2, frame3, extr1, extr3):
        original_size = [self.D, frame1.size(2), frame1.size(3)]
        frame1_cost = self.cost_net(frame1)  # [B, 32, 1/4H, 1/4W]
        frame2_cost = self.cost_net(frame2)  # [B, 32, 1/4H, 1/4W]
        frame3_cost = self.cost_net(frame3)  # [B, 32, 1/4H, 1/4W]

        R1, T1 = self.extrinsic_net(extr1)
        R3, T3 = self.extrinsic_net(extr3)
        # reshape and multiply
        B, C, H, W = frame1_cost.size()

        frame1_cost = frame1_cost.permute(0, 2, 3, 1)
        frame1_cost = torch.matmul(frame1_cost, R1) + T1
        frame1_cost = frame1_cost.permute(0, 3, 1, 2)

        frame3_cost = frame3_cost.permute(0, 2, 3, 1)
        frame3_cost = torch.matmul(frame3_cost, R3) + T3
        frame3_cost = frame3_cost.permute(0, 3, 1, 2)
        # cost = torch.cat([left_cost, right_cost], dim=1)  # [B, 64, 1/4H, 1/4W]
        # B, C, H, W = cost.size()

        # print('left_cost')
        # print(left_cost[0, 0, :3, :3])


        cost_volume = torch.zeros(
                        B, C * 3, self.D // 4, H, W
                      ).type_as(frame1_cost)  # [B, 64, D, 1/4H, 1/4W]

        # for i in range(self.D // 4):
        #     cost_volume[:, :, i, :, i:] = cost[:, :, :, i:]

        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = frame1_cost[:, :, :, i:]
                cost_volume[:, C:2*C, i, :, i:] = frame2_cost[:, :, :, :-i]
                cost_volume[:, 2*C:, i, :, i:] = frame3_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = frame1_cost
                cost_volume[:, C:2*C, i, :, :] = frame2_cost
                cost_volume[:, 2*C:, i, :, :] = frame3_cost
        disp1, disp2, disp3 = self.stackedhourglass(
                                cost_volume,
                                out_size=original_size
                              )

        return disp1, disp2, disp3

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
