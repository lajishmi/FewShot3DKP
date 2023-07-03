import os
import torch
import torch.nn.functional as F
import torchvision
import kornia
import networkx as nx
import einops
import torch.nn as nn


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        os.environ['TORCH_HOME'] = os.path.abspath(os.getcwd())
        blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
        perceptual_loss = 0.0
        style_loss = 0.0

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            perceptual_loss += torch.nn.functional.l1_loss(x, y)

            # b, ch, h, w = x.shape
            # act_x = x.reshape(x.shape[0], x.shape[1], -1)
            # act_y = y.reshape(y.shape[0], y.shape[1], -1)
            # gram_x = act_x @ act_x.permute(0, 2, 1) / (ch * h * w)
            # gram_y = act_y @ act_y.permute(0, 2, 1) / (ch * h * w)
            # style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return perceptual_loss#, style_loss


class Geo2DLoss(torch.nn.Module):
    def __init__(self, sym_idx, n_keypoints):
        super().__init__()
        self.color_aug = kornia.augmentation.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=1.)
        self.geo_aug = kornia.augmentation.RandomAffine(degrees=60, translate=0.1, scale=(0.9, 1.1), p=1.)

        self.sym_idx = list(range(n_keypoints))
        for idx in sym_idx:
            self.sym_idx[idx[0]] = idx[1]
            self.sym_idx[idx[1]] = idx[0]

    def forward(self, img, keypoints, encoder):
        keypoints = keypoints[..., :2]
        with torch.no_grad():
            bs, _, w, h = img.shape
            trans_img = self.color_aug(img)
            trans_img = self.geo_aug(trans_img)
            tran_mat = kornia.geometry.conversions.normalize_homography(self.geo_aug.transform_matrix, (w, h), (w, h))

        trans_keypoints = kornia.geometry.keypoints.Keypoints(keypoints).transform_keypoints(tran_mat).to_tensor()

        with torch.no_grad():
            vis = (trans_keypoints[..., 0] > -1) * \
                (trans_keypoints[..., 1] > -1) * \
                (trans_keypoints[..., 0] < 1) * \
                (trans_keypoints[..., 1] < 1)

            permutated_idx = torch.randperm(bs)
            trans_img = trans_img[permutated_idx]
            vis = vis[permutated_idx]

            middle_idx = bs // 2
            trans_img_flip = torch.flip(trans_img[:middle_idx], dims=[3])
            vis_flip = vis[:middle_idx, self.sym_idx]
            trans_img = torch.cat([trans_img_flip, trans_img[middle_idx:]])
            vis = torch.cat([vis_flip, vis[middle_idx:]])

        trans_keypoints = trans_keypoints[permutated_idx]
        trans_keypoints_flip = torch.stack([-trans_keypoints[:middle_idx, :, 0], trans_keypoints[:middle_idx, :, 1]], dim=2)[:, self.sym_idx]
        trans_keypoints = torch.cat([trans_keypoints_flip, trans_keypoints[middle_idx:]])
        
        equi_loss = F.l1_loss(encoder({'img': trans_img})['keypoints'][..., :2] * vis.unsqueeze(-1),
                              trans_keypoints * vis.unsqueeze(-1))
        return equi_loss
    

def transformA2B(src, dst):
    with torch.no_grad():
        bs, num, dim = src.shape

        src_mean = src.mean(dim=1)
        dst_mean = dst.mean(dim=1)

        src_demean = src - src_mean.unsqueeze(1)
        dst_demean = dst - dst_mean.unsqueeze(1)

        A = dst_demean.transpose(-1, -2) @ src_demean / num

        d = torch.eye(dim, device=src.device).repeat(bs, 1, 1)
        neg_A = torch.det(A) < 0
        if torch.any(neg_A):
            d[neg_A][:, dim - 1, dim - 1] = -1

        U, S, V = torch.svd(A.float())

        scale = (S * torch.diagonal(d, dim1=1, dim2=2)).sum(dim=1) / src_demean.var(dim=1, unbiased=False).sum(dim=1)

        R = U @ d @ V.transpose(-1, -2)
        T = dst_mean - scale.reshape(bs, -1) * (R @ src_mean.unsqueeze(-1)).squeeze(-1)

    transformed_src = src @ (R.transpose(-1, -2) * scale.reshape(bs, 1, 1)) + T.unsqueeze(1)

    return transformed_src
    

class Geo3DLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints):
        randomized_keypoints = keypoints[torch.randperm(keypoints.size(0))]
        trans_keypoints = transformA2B(keypoints, randomized_keypoints)
        return (trans_keypoints - randomized_keypoints).abs().mean()
    

class SmoothLoss(torch.nn.Module):
    def __init__(self, edge_idx, n_keypoints):
        super().__init__()
        g = nx.Graph()
        g.add_edges_from(edge_idx)

        self.n_neighbors = 2
        self.edge_groups = [[] for _ in range(self.n_neighbors)]
        for idx in range(n_keypoints):
            neighbors_idx = nx.single_source_shortest_path_length(g, idx, cutoff=self.n_neighbors)
            for i, edge_group in enumerate(self.edge_groups):
                neighbors = [key for key, value in neighbors_idx.items() if value == i+1]
                if len(neighbors) == 2:
                    edge_group.append([[idx, neighbors[0]], [idx, neighbors[1]]])

        for i in range(self.n_neighbors):
            self.edge_groups[i] = torch.tensor(self.edge_groups[i])

    def forward(self, keypoints):
        losses = []
        for edge_group in self.edge_groups:
            vector1 = keypoints[:, edge_group[:, 0, 1], :] - keypoints[:, edge_group[:, 0, 0], :]
            vector2 = keypoints[:, edge_group[:, 1, 1], :] - keypoints[:, edge_group[:, 1, 0], :]
            losses.append(F.cosine_similarity(vector1, vector2, dim=-1).mean())
        return torch.stack(losses).mean()