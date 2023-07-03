import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(out_channels)
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv_res(x)
        x = self.net(x)
        return self.relu(x + res)


class TransposedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class Detector(nn.Module):
    def __init__(self, hyper_paras: pl.utilities.parsing.AttributeDict) -> None:
        super().__init__()
        self.n_keypoints = hyper_paras.n_keypoints
        self.output_size = 32

        self.conv = nn.Sequential(
            ResBlock(3, 64),  # 64
            ResBlock(64, 128),  # 32
            ResBlock(128, 256),  # 16
            ResBlock(256, 512),  # 8
            TransposedBlock(512, 256),  # 16
            TransposedBlock(256, 128),  # 32
            nn.Conv2d(128, self.n_keypoints * 3, kernel_size=3, padding=1),
        )

        grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), torch.Size((1, 2, self.output_size, self.output_size)), align_corners=True).reshape(1, 1, self.output_size, self.output_size, 2)
        self.coord = nn.Parameter(grid, requires_grad=False)

        self.denominator = 20

    def forward(self, input_dict: dict) -> dict:
        img = F.interpolate(input_dict['img'], size=(128, 128), mode='bilinear', align_corners=False)
        feat_map = self.conv(img)
        prob_map, depth, vis = feat_map.chunk(3, dim=1)
        prob_map = prob_map.reshape(img.shape[0], self.n_keypoints, -1)
        prob_map = F.softmax(prob_map, dim=2)
        prob_map = prob_map.reshape(img.shape[0], self.n_keypoints, self.output_size, self.output_size)
        keypoints = (self.coord * prob_map.unsqueeze(-1)).sum(dim=(2, 3))
        depth = torch.tanh((depth * prob_map).sum(dim=(2, 3)) / self.denominator)
        keypoints = torch.cat((keypoints, depth.unsqueeze(-1)), dim=2)
        vis = (vis * prob_map).sum(dim=(2, 3))
        return {'keypoints': keypoints, 'vis': vis}


class Encoder(nn.Module):
    def __init__(self, hyper_paras: pl.utilities.parsing.AttributeDict) -> None:
        super().__init__()
        self.detector = Detector(hyper_paras)
        self.missing = 0.9
        self.block = 16

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict: dict, need_masked_img: bool=False) -> dict:
        mask_batch = self.detector(input_dict)
        if need_masked_img:
            damage_mask = torch.zeros(input_dict['img'].shape[0], 1, self.block, self.block, device=input_dict['img'].device).uniform_() > self.missing
            damage_mask = F.interpolate(damage_mask.to(input_dict['img']), size=input_dict['img'].shape[-1], mode='nearest')
            mask_batch['damaged_img'] = input_dict['img'] * damage_mask
        return mask_batch
