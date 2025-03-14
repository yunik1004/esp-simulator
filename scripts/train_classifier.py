"""Train script"""

import argparse

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader

from esp_simulator.data.item_farm import ItemFarmClassificationDataset
from esp_simulator.models.item_farm import ItemFarmAntiCheater

from utils.environments import add_argument_environment


class TrainerModel(L.LightningModule):
    """Temporary model class for training and validation"""

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = ItemFarmAntiCheater(grid_size=self.hparams.grid_size)
        if self.hparams.ema:
            self.init_ema()

        self.criterion = nn.BCEWithLogitsLoss()

    def init_ema(self) -> None:
        """Initialize the ema model"""
        self.ema_model = AveragedModel(
            self.model, multi_avg_fn=get_ema_multi_avg_fn(self.hparams.ema_decay)
        )
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weight from the checkpoint

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        self.load_state_dict(checkpoint, strict=False)
        if self.hparams.ema:
            ema_model_exist = False
            for k in checkpoint.keys():
                if k.startswith("ema_model."):
                    ema_model_exist = True
                    break
            if not ema_model_exist:
                self.init_ema()

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"]
        y_hat = self.ema_model(x) if self.hparams.ema else self.model(x)
        loss = self.criterion(y_hat, y.float())
        self.log_dict({"train/loss": loss})
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.hparams.ema:
            self.ema_model.update_parameters(self.model)

    def on_train_epoch_end(self):
        if self.hparams.ema:
            update_bn(self.trainer.train_dataloader, self.ema_model)

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"]
        y_hat = self.ema_model(x) if self.hparams.ema else self.model(x)
        loss = self.criterion(y_hat, y.float())
        self.log_dict({"val/loss": loss}, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }


class TrainerDataModule(L.LightningDataModule):
    """Temporary data module class for training and validation"""

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

    def prepare_data(self):
        self.train_dataset = ItemFarmClassificationDataset(
            self.hparams.dataset_normal_train, self.hparams.dataset_esp_train
        )
        self.val_dataset = ItemFarmClassificationDataset(
            self.hparams.dataset_normal_val, self.hparams.dataset_esp_val
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    seed_everything(args.seed, workers=True)

    dm = TrainerDataModule(args)
    model = TrainerModel(args)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    logger = TensorBoardLogger(args.log_dir)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.val_epoch_freq, save_top_k=-1
    )

    # Trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=args.dist_strategy,
        callbacks=[checkpoint_callback],
        logger=False if args.validation else logger,
        deterministic=args.validation,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_epoch_freq,
    )

    if args.validation:
        trainer.validate(model, datamodule=dm)
        return

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=args.checkpoint if args.checkpoint and args.resume_train else None,
    )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Train script")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="training accelerator"
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="number of GPU nodes")
    parser.add_argument(
        "--dist_strategy", type=str, default="auto", help="model distribution strategy"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--resume_train", action="store_true", help="resume training state"
    )
    parser.add_argument("--log_dir", type=str, default=".", help="tensorboard log dir")
    parser.add_argument(
        "--validation", action="store_true", help="perform validation only"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--ema", action="store_true", help="perform EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="maximum number of epochs"
    )
    parser.add_argument(
        "--val_epoch_freq", type=int, default=1, help="validation epoch frequency"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for data loading"
    )
    parser.add_argument(
        "--dataset_normal_train", type=str, help="train dataset path of the normal user"
    )
    parser.add_argument(
        "--dataset_normal_val",
        type=str,
        help="validation dataset path of the normal user",
    )
    parser.add_argument(
        "--dataset_esp_train", type=str, help="train dataset path of the esp user"
    )
    parser.add_argument(
        "--dataset_esp_val", type=str, help="validation dataset path of the esp user"
    )
    add_argument_environment(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
