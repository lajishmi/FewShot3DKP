import argparse
import os
from model.model import Model
import pytorch_lightning as pl

from datasets.base_dataset import DataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='wflw_10shots')
    args = parser.parse_args()

    model = Model.load_from_checkpoint(os.path.join('checkpoints', args.log, 'model.ckpt'))

    trainer = pl.Trainer(sync_batchnorm=True)
    datamodule = DataModule(model.hparams.dataset, model.hparams.data_root, 
                            model.hparams.image_size, model.hparams.batch_size,
                            model.hparams.fewshot_idx, model.hparams.sym_idx)
    trainer.test(model, datamodule=datamodule)
