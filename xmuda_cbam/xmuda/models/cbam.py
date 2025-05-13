import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 确保hidden_channels至少为1
        hidden_channels = max(1, in_channels // reduction_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 2:  # [N, C]
            assert x.shape[1] == self.in_channels, f"Input channel {x.shape[1]} doesn't match initialized {self.in_channels}"
            avg_pool = torch.mean(x, dim=0)  # [C]
            max_pool, _ = torch.max(x, dim=0)  # [C]
            out = self.mlp(avg_pool) + self.mlp(max_pool)
            scale = self.sigmoid(out)  # [C]
            return x * scale.unsqueeze(0)  # [N, C] * [1, C]
        elif len(x.shape) == 4:  # [B, C, H, W]
            assert x.shape[1] == self.in_channels, f"Input channel {x.shape[1]} doesn't match initialized {self.in_channels}"
            avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # [B, C]
            max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)  # [B, C]
            out = self.mlp(avg_pool) + self.mlp(max_pool)  # [B, C]
            scale = self.sigmoid(out).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
            return x * scale  # 广播乘法
        else:
            raise NotImplementedError("Unsupported tensor shape for ChannelAttention")

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 只对2D特征应用空间注意力
        if len(x.shape) == 4:  # [B, C, H, W]
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_out = torch.cat([avg_out, max_out], dim=1)
            out = self.sigmoid(self.conv(x_out))
            return x * out
        return x  # 3D点云不应用空间注意力
    
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 对于3D特征，x的形状应该是[N, C] (N是点数，C是通道数)
        if len(x.shape) == 2:  # 处理3D点云特征
            # 通道注意力
            x = self.channel_attention(x)
            # 3D点云没有空间维度，跳过空间注意力
            return x
        else:  # 处理2D图像特征
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            return x