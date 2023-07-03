import torch
import torch.nn.functional as F
from torch import nn
from typing import Union
import pytorch_lightning as pl


def draw_lines(paired_keypoints: torch.Tensor, paired_vis: torch.Tensor, heatmap_size: int=16, thick: Union[float, torch.Tensor]=1e-2) -> torch.Tensor:
    """
    Draw lines on a grid.
    :param paired_keypoints: (batch_size, n_points, 2, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    bs, n_points, _, _ = paired_keypoints.shape
    start = paired_keypoints[:, :, 0, :]   # (batch_size, n_points, 2)
    end = paired_keypoints[:, :, 1, :]     # (batch_size, n_points, 2)
    paired_diff = end - start           # (batch_size, n_points, 2)
    grid = F.affine_grid(torch.eye(2, 3, device=paired_keypoints.device).unsqueeze(0), torch.Size((1, 2, heatmap_size, heatmap_size)), align_corners=True).reshape(1, 1, -1, 2)
    diff_to_start = grid - start.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)
    # (batch_size, n_points, heatmap_size**2)
    t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / (1e-8+paired_diff.square().sum(dim=-1, keepdim=True))

    diff_to_end = grid - end.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)

    before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
    after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
    between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

    squared_dist = (before_start + after_end + between_start_end).reshape(bs, n_points, heatmap_size, heatmap_size)
    heatmaps = torch.exp(- squared_dist / thick)
    
    vis_start = paired_vis[:, :, 0:1]
    vis_end = paired_vis[:, :, 1:2]
    before_vis_start = (t <= 0).float() * vis_start
    after_vis_end = (t >= 1).float() * vis_end
    between_vis_start_end = (0 < t).float() * (t < 1).float() * ((1 - t) * vis_start + t * vis_end)

    vis_maps = torch.sigmoid(before_vis_start + after_vis_end + between_vis_start_end)
    return heatmaps * vis_maps.reshape(bs, n_points, heatmap_size, heatmap_size)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hyper_paras: pl.LightningModule.hparams) -> None:
        super().__init__()
        self.n_keypoints = hyper_paras['n_keypoints']
        self.thick = 1e-3
        self.edge_idx = torch.tensor(hyper_paras['edge_idx']).transpose(1, 0)

        self.multiplier = nn.Parameter(torch.tensor(-4.0), requires_grad=True)

        self.down0 = nn.Sequential(
            nn.Conv2d(3 + 1, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down1 = DownBlock(64, 128)  # 64
        self.down2 = DownBlock(128, 256)  # 32
        self.down3 = DownBlock(256, 512)  # 16
        self.down4 = DownBlock(512, 512)  # 8

        self.up1 = UpBlock(512, 512)  # 16
        self.up2 = UpBlock(512 + 512, 256)  # 32
        self.up3 = UpBlock(256 + 256, 128)  # 64
        self.up4 = UpBlock(128 + 128, 64)  # 64

        self.conv = nn.Conv2d(64+64, 3, kernel_size=(3, 3), padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def rasterize(self, keypoints: torch.Tensor, vis: torch.Tensor, output_size: int=128) -> torch.Tensor:
        paired_keypoints = torch.stack([keypoints[:, self.edge_idx[0], :2], keypoints[:, self.edge_idx[1], :2]], dim=2)
        paired_vis = torch.stack([vis[:, self.edge_idx[0]], vis[:, self.edge_idx[1]]], dim=2)

        heatmap_sep = draw_lines(paired_keypoints, paired_vis, heatmap_size=output_size, thick=self.thick)
        heatmap = heatmap_sep.max(dim=1, keepdim=True)[0] * F.softplus(self.multiplier)
        return heatmap

    def forward(self, input_dict: dict) -> dict:
        heatmap = self.rasterize(input_dict['keypoints'], input_dict['vis'])

        x = torch.cat([input_dict['damaged_img'], heatmap], dim=1)

        down_128 = self.down0(x)
        down_64 = self.down1(down_128)
        down_32 = self.down2(down_64)
        down_16 = self.down3(down_32)
        down_8 = self.down4(down_16)
        up_8 = down_8
        up_16 = torch.cat([self.up1(up_8), down_16], dim=1)
        up_32 = torch.cat([self.up2(up_16), down_32], dim=1)
        up_64 = torch.cat([self.up3(up_32), down_64], dim=1)
        up_128 = torch.cat([self.up4(up_64), down_128], dim=1)
        img = self.conv(up_128)

        input_dict['heatmap'] = heatmap
        input_dict['img'] = img
        return input_dict


if __name__ == '__main__':
    model = Decoder({'z_dim': 256, 'n_keypoints': 10, 'n_embedding': 128, 'tau': 0.01})
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
