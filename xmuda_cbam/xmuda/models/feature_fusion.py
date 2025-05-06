import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv_2d = nn.Sequential(
            nn.Conv2d(in_channels_2d, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3d = nn.Sequential(
            nn.Conv1d(in_channels_3d, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合后再做一次整合
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat2d, feat3d, proj_indices):
        """
        feat2d: [B, C2, H, W]
        feat3d: [N, C3]
        proj_indices: [N, 2] — 每个点在图像上的位置 (x, y)
        """
        B, C2, H, W = feat2d.shape
        N, C3 = feat3d.shape

        # 处理2D特征
        feat2d_flat = self.conv_2d(feat2d).view(B, -1, H * W)  # [B, C_out, H*W]

        fused_feats = []  # 初始化输出

        if isinstance(proj_indices, list):
            proj_indices = torch.cat([torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in proj_indices], dim=0)
        proj_indices = proj_indices.to(feat2d.device).long()
        
        # 防止越界
        x_idx = torch.clamp(proj_indices[:, 0], 0, W - 1)
        y_idx = torch.clamp(proj_indices[:, 1], 0, H - 1)
        linear_idx = (y_idx + x_idx * H).long().clamp(0, H * W - 1)
        
        # 提取 2D 特征
        batch_indices = proj_indices[:, 2] if proj_indices.shape[1] > 2 else torch.zeros(len(proj_indices), device=feat2d.device).long()
        feat2d_mapped = feat2d_flat[batch_indices, :, linear_idx]  # [N, C_out]

        # 处理 3D 特征
        feat3d_mapped = self.conv_3d(feat3d.unsqueeze(0).transpose(1, 2)).squeeze(0).transpose(0, 1)  # [N, C_out]
        
        # 检查维度是否匹配
        assert feat2d_mapped.shape[0] == feat3d_mapped.shape[0], \
            f"Feature dimension mismatch: 2D features {feat2d_mapped.shape[0]} points, 3D features {feat3d_mapped.shape[0]} points"
    
        # 融合
        fused = torch.cat([feat2d_mapped, feat3d_mapped], dim=1)  # [N, 2*C_out]
        fused = self.fusion(fused)
    
        return fused