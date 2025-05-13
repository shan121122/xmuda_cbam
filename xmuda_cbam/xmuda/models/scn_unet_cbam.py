import torch
import torch.nn as nn

import sparseconvnet as scn
from xmuda.models.cbam import CBAM

DIMENSION = 3

class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels,
                 m=16,
                 block_reps=1,
                 residual_blocks=False,
                 full_scale=4096,
                 num_planes=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m))

        self.cbam = CBAM(m)  # 新增 CBAM

        self.output_layer = scn.OutputLayer(DIMENSION)

    def forward(self, x):
        features = self.sparseModel(x)
        # 直接对特征点应用CBAM (features.features形状为[N, C])
        features.features = self.cbam(features.features)
        output = self.output_layer(features)
        return output
    
    def get_features(self, x):
        x[0] = x[0].to(torch.int32)
        features = self.sparseModel(x)
        return self.cbam(features.unsqueeze(0)).squeeze(0)
