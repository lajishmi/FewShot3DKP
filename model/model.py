import importlib
import PIL
import pytorch_lightning as pl
import torch.utils.data
import wandb
from typing import Union
from torchvision import transforms
from utils_.loss import *
from utils_.visualization import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .encoder import Encoder
from .decoder import Decoder


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(self.hparams)
        self.decoder = Decoder(self.hparams)
        self.batch_size = self.hparams.batch_size
        self.test_func = importlib.import_module('datasets.' + self.hparams.dataset).test_epoch_end

        self.vgg_loss = VGGPerceptualLoss()
        self.geo3d_loss = Geo3DLoss()
        self.geo2d_loss = Geo2DLoss(self.hparams.sym_idx, self.hparams.n_keypoints)
        self.smooth_loss = SmoothLoss(self.hparams.edge_idx, self.hparams.n_keypoints)

        self.val_list = []
        self.test_list = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def forward(self, x: PIL.Image.Image) -> PIL.Image.Image:
        """
        :param x: a PIL image
        :return: an edge map of the same size as x with values in [0, 1] (normalized by max)
        """
        w, h = x.size
        x = self.transform(x).unsqueeze(0)
        x = x.to(self.device)
        kp = self.encoder({'img': x})['keypoints']
        edge_map = self.decoder.rasterize(kp, output_size=64)
        bs = edge_map.shape[0]
        edge_map = edge_map / (1e-8 + edge_map.reshape(bs, 1, -1).max(dim=2, keepdim=True)[0].reshape(bs, 1, 1, 1))
        edge_map = torch.cat([edge_map] * 3, dim=1)
        edge_map = F.interpolate(edge_map, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.clamp(edge_map + (x * 0.5 + 0.5)*0.5, min=0, max=1)
        x = transforms.ToPILImage()(x[0].detach().cpu())

        fig = plt.figure(figsize=(1, h/w), dpi=w)
        fig.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(x)
        kp = kp[0].detach().cpu() * 0.5 + 0.5
        kp[:, 1] *= w
        kp[:, 0] *= h
        plt.scatter(kp[:, 1], kp[:, 0], s=min(w/h, min(1, h/w)), marker='o')
        ncols, nrows = fig.canvas.get_width_height()
        fig.canvas.draw()
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
        plt.close(fig)
        return plot

    def training_step(self, batch, batch_idx):
        self.vgg_loss.eval()
        annotated_batch, unannotated_batch = batch
        n_shots = annotated_batch['img'].shape[0]
        batch = {'img': torch.cat([annotated_batch['img'], unannotated_batch['img']], dim=0)}
        out_batch = self.decoder(self.encoder(batch, need_masked_img=True))

        keypoint_loss = F.l1_loss(out_batch['keypoints'][:n_shots, :, :2], annotated_batch['keypoints'])
        self.log("keypoint_loss", keypoint_loss)

        l1_loss = F.l1_loss(out_batch['img'], batch['img'])
        self.log("l1_loss", l1_loss)

        perceptual_loss = self.vgg_loss(out_batch['img'], batch['img'])
        self.log("perceptual_loss", perceptual_loss)
        
        geo2d_loss = self.geo2d_loss(batch['img'], out_batch['keypoints'], self.encoder)
        self.log("geo2d_loss", geo2d_loss)

        geo3d_loss = self.geo3d_loss(out_batch['keypoints'])
        self.log("geo3d_loss", geo3d_loss)

        smooth_loss = self.smooth_loss(out_batch['keypoints'])
        self.log("smooth_loss", smooth_loss)
        
        return l1_loss + perceptual_loss + keypoint_loss + geo3d_loss * 0.1 + geo2d_loss + smooth_loss * 0.01

    def validation_step(self, batch, batch_idx):
        self.val_list += [batch]
        return batch

    def on_validation_epoch_end(self):
        self.log("val_loss", -self.global_step*1.0)
        imgs = denormalize(self.val_list[0]['img']).cpu()
        recon_batch = self.decoder(self.encoder(self.val_list[0], need_masked_img=True))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)

        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(self.val_list[0]['img']).cpu()), caption='original_image'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(imgs, scaled_kp)), caption='keypoints'),
                                                  ]})
        self.val_list = []
        
    def test_step(self, batch, batch_idx):
        kp = self.encoder(batch)['keypoints'][..., :2]
        out_batch = {'keypoints': batch['keypoints'].cpu(), 'det_keypoints': kp.cpu()}
        self.test_list += [out_batch]
        return out_batch

    def on_test_epoch_end(self):
        outputs = self.test_func(self.test_list)
        self.print("test_loss", outputs['val_loss'])
        self.log("test_loss", outputs['val_loss'])
        self.test_list = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=1e-3)
        return optimizer
