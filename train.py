import argparse
import yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model.model import Model
from datasets.base_dataset import DataModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/wflw.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        args = yaml.safe_load(stream)

    args['log'] = '{0}_{1}shots'.format(args['dataset'], len(args['fewshot_idx']))
    wandb_logger = WandbLogger(name=args['log'], project="FewShot3DKP")
    datamodule = DataModule(args['dataset'], args['data_root'], args['image_size'], args['batch_size'], 
                            args['fewshot_idx'], args['sym_idx'])
    model = Model(**args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                       dirpath=os.path.join('checkpoints', args['log']),
                                                       filename='model')
    
    if args['debug'] == 0:
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)
        model.vgg_loss = torch.compile(model.vgg_loss)
        # model.geo3d_loss = torch.compile(model.geo3d_loss)
        # model.geo2d_loss = torch.compile(model.geo2d_loss)
        model.smooth_loss = torch.compile(model.smooth_loss)

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(fast_dev_run=args['debug'],
                         max_steps=20001, sync_batchnorm=True,
                         limit_val_batches=1,
                         check_val_every_n_epoch=1,
                         callbacks=checkpoint_callback, logger=wandb_logger,
                         )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
