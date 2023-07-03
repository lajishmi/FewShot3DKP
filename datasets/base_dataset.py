import importlib
import pytorch_lightning as pl
import torch
import torch.utils.data


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_root, image_size, batch_size, fewshot_idx, sym_idx, num_workers=6):
        super().__init__()
        self.n_shots = len(fewshot_idx)
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_root = data_root
        self.image_size = image_size
        self.num_workers = num_workers

        dataset = importlib.import_module('datasets.' + self.dataset)
        self.annotated_train_dataset = dataset.AnnotatedTrainSet(self.data_root, self.image_size, fewshot_idx, sym_idx)
        self.unannotated_train_dataset = dataset.UnannotatedTrainSet(self.data_root, self.image_size)
        self.test_dataset = dataset.TestSet(self.data_root, self.image_size)

    def train_dataloader(self):
        return [torch.utils.data.DataLoader(self.annotated_train_dataset, batch_size=min(self.batch_size, self.n_shots), num_workers=self.num_workers, shuffle=True, drop_last=True),
                torch.utils.data.DataLoader(self.unannotated_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)]

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
