import argparse

import lightning as L
from lightning.pytorch import seed_everything
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
)
import numpy as np

from esp_simulator.data.item_farm import ItemFarmClassificationDataset
from esp_simulator.models.item_farm import (
    ItemFarmAntiCheater,
    ItemFarmRewardAntiCheater,
    ItemFarmTrajectoryLengthAntiCheater,
)

from utils.classifiers import add_argument_classifier
from utils.environments import add_argument_environment


class TesterModel(L.LightningModule):
    """Temporary model class for testing"""

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.cls_type == "heatmap":
            self.model = ItemFarmAntiCheater(
                grid_size=self.hparams.grid_size, features_dim=self.hparams.features_dim
            )
        elif self.hparams.cls_type == "length":
            self.model = ItemFarmTrajectoryLengthAntiCheater(
                grid_size=self.hparams.grid_size
            )
        elif self.hparams.cls_type == "reward":
            self.model = ItemFarmRewardAntiCheater(grid_size=self.hparams.grid_size)
        self.criterion = nn.BCEWithLogitsLoss()

        self.rewards = torch.from_numpy(
            np.load(self.hparams.dataset_esp_reward).astype(np.float32)
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weight from the checkpoint

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint, strict=False)

    def test_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["label"]
        id = batch["id"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.float())
        self.log_dict({"test/loss": loss}, on_epoch=True, sync_dist=True)
        return {"loss": loss, "y_hat": y_hat, "y": y, "id": id}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        y_hat = self.all_gather(outputs["y_hat"])
        y = self.all_gather(outputs["y"])
        id = self.all_gather(outputs["id"])
        if self.trainer.is_global_zero:
            self.pr_curve.update(y_hat, y)
            self.roc_curve.update(y_hat, y)

            if self.hparams.threshold is not None:
                self.confmat.update(y_hat, y)

                # Check false positives and false negatives
                y_hat_cls = y_hat.sigmoid() >= self.hparams.threshold
                y_cls = y.bool()

                fp = y_hat_cls & ~y_cls
                fn = ~y_hat_cls & y_cls

                self.fp.extend(id[fp].cpu().tolist())
                self.fn.extend(id[fn].cpu().tolist())

    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            self.pr_curve = BinaryPrecisionRecallCurve().to(self.device)
            self.roc_curve = BinaryROC().to(self.device)
            if self.hparams.threshold is not None:
                self.confmat = BinaryConfusionMatrix(
                    threshold=self.hparams.threshold
                ).to(self.device)
                self.fp = []
                self.fn = []

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            y = torch.cat(self.pr_curve.target, dim=0)
            y_hat = torch.cat(self.pr_curve.preds, dim=0).sigmoid()

            log_vals = y * torch.log(y_hat.sigmoid()) + (1 - y) * torch.log(
                1 - y_hat.sigmoid()
            )
            alt = y * torch.log(1 - y_hat.sigmoid())

            a = log_vals[y == 1].mean()
            b = log_vals[y == 0].mean()
            c = alt[y == 1].mean()
            r = self.rewards.mean()

            main_obj = a + b - 1 / self.hparams.adv_coef * r
            neg_cross_entropy = a + b
            cheater_obj_orig = r - self.hparams.adv_coef * a
            cheater_obj_prac = r + self.hparams.adv_coef * c
            print(
                f"Main obj: {main_obj}\nneg_cross_entropy: {neg_cross_entropy}\ncheater_obj_orig: {cheater_obj_orig}\ncheater_obj_practical: {cheater_obj_prac}"
            )


class TesterDataModule(L.LightningDataModule):
    """Temporary data module class for testing"""

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

    def prepare_data(self):
        self.dataset = ItemFarmClassificationDataset(
            self.hparams.dataset_normal, self.hparams.dataset_esp
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
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

    dm = TesterDataModule(args)
    model = TesterModel(args)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    # Trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=args.dist_strategy,
        logger=False,
        deterministic=True,
    )

    trainer.test(model, datamodule=dm)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Evaluation script of the classifier")
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of GPU nodes")
    parser.add_argument(
        "--dist_strategy", type=str, default="auto", help="model distribution strategy"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--output_pr_curve_image",
        type=str,
        default="./output/pr_curve.png",
        help="name of the precision-recall curve image",
    )
    parser.add_argument(
        "--output_roc_curve_image",
        type=str,
        default="./output/roc_curve.png",
        help="name of the ROC curve image",
    )
    parser.add_argument(
        "--output_confmat_image",
        type=str,
        default="./output/confmat.png",
        help="name of the confusion matrix image",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for data loading"
    )
    parser.add_argument(
        "--dataset_normal", type=str, help="dataset path of the normal user"
    )
    parser.add_argument("--dataset_esp", type=str, help="dataset path of the esp user")
    parser.add_argument(
        "--dataset_esp_reward", type=str, help="dataset path of the esp user"
    )
    parser.add_argument("--threshold", type=float, help="classifier threshold")
    parser.add_argument(
        "--adv_coef", type=float, default=1.0, help="adversarial training coefficient"
    )
    add_argument_environment(parser)
    add_argument_classifier(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
