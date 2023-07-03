import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms
import h5py


class AnnotatedTrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size, fewshot_idx, sym_idx):
        super().__init__()
        self.image_size = image_size
        self.data_root = data_root

        self.sym_idx = list(range(98))
        for idx in sym_idx:
            self.sym_idx[idx[0]] = idx[1]
            self.sym_idx[idx[1]] = idx[0]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with h5py.File(os.path.join(data_root, 'wflw.h5'), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][fewshot_idx])
            self.keypoints = torch.from_numpy(hf['train_landmark'][fewshot_idx])   # [-1, 1]
        #     self.imgs = torch.from_numpy(hf['train_img'][:200])
        #     self.keypoints = torch.from_numpy(hf['train_landmark'][:200])   # [-1, 1]

        # import yaml
        # with open('config/wflw.yaml', 'r') as stream:
        #     args = yaml.safe_load(stream)
        #     edge_idx = args['edge_idx']

        # for i in range(self.keypoints.shape[0]):
        #     import matplotlib.pyplot as plt
        #     plt.imshow(self.imgs[i].permute(1, 2, 0) * 0.5 + 0.5)

        #     for edge in edge_idx:
        #         plt.plot(self.keypoints[i, edge, 0].detach().cpu() * 64 + 64,
        #                  self.keypoints[i, edge, 1].detach().cpu() * 64 + 64, color='red')
        #     plt.savefig('./fewshots/{}.jpg'.format(i))
        #     plt.close()

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx] / 255)
        keypoints = self.keypoints[idx]

        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            keypoints = torch.stack([-keypoints[:, 0], keypoints[:, 1]], dim=1)
            keypoints = keypoints[self.sym_idx]
            
        sample = {'img': img, 'keypoints': keypoints}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class UnannotatedTrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.image_size = image_size
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with h5py.File(os.path.join(data_root, 'wflw.h5'), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255)}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.image_size = image_size
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with h5py.File(os.path.join(data_root, 'wflw.h5'), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test_img'][...])
            self.keypoints = torch.from_numpy(hf['test_landmark'][...])   # [-1, 1]

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255),
                  'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


def test_epoch_end(test_list):
    X = torch.cat([batch['det_keypoints'] for batch in test_list])
    y = torch.cat([batch['keypoints'] for batch in test_list])
    normalized_loss = (X - y).norm(dim=-1) / (y[:, 60, :] - y[:, 72, :]).norm(dim=-1).unsqueeze(-1)
    return {'val_loss': normalized_loss.mean()}
