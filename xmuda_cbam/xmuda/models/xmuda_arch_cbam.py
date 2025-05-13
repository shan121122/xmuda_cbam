import torch
import torch.nn as nn

from xmuda.models.resnet34_unet_cbam import UNetResNet34
from xmuda.models.scn_unet_cbam import UNetSCN


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)


    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        feat_map = self.net_2d(img)  # shape: (B, C, H, W)
        if isinstance(feat_map, tuple):
            feat_map = feat_map[0]  # 取第一个元素（假设是张量）
        assert isinstance(feat_map, torch.Tensor), "feat_map must be Tensor" 

        # 2D-3D feature lifting
        img_feats = []
        for i in range(feat_map.shape[0]):
            # Get current image indices and validate them
            curr_indices = img_indices[i]
            # 确保索引是torch.long类型
            if not isinstance(curr_indices, torch.Tensor):
                curr_indices = torch.from_numpy(curr_indices).to(device=feat_map.device)
            curr_indices = curr_indices.long()
            # Clamp indices to valid range
            h, w = feat_map.shape[2], feat_map.shape[3]
            u = torch.clamp(curr_indices[:, 0], min=0, max=h-1)
            v = torch.clamp(curr_indices[:, 1], min=0, max=w-1)
            # Permute and index carefully
            permuted = feat_map.permute(0, 2, 3, 1)  # [B, H, W, C]
            img_feats.append(permuted[i][u, v])  # [N, C]

        img_feats = torch.cat(img_feats, 0)

        # linear
        seg_logit = self.linear(img_feats)

        preds = {
            'feat': feat_map,  # 完整的特征图 [B, C, H, W]
            'feats': img_feats,  # 采样后的特征 [N, C]
            'seg_logit': seg_logit,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats)

        if 'fused_feat' in data_batch:
            preds['fused_logit'] = self.fusion_head(data_batch['fused_feat'])

        return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch):
        
        feats = self.net_3d(data_batch['x'])
        
        # For SCN output, features are in feats.features
        if hasattr(feats, 'features'):
            features = feats.features
        else:
            features = feats

        seg_logit = self.linear(features)

        preds = {
            'feat': features,  # [N, C]
            'seg_logit': seg_logit,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats)

        return preds


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
