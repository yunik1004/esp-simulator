import argparse

import lightning as L
from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
)

from esp_simulator.data.item_farm import ItemFarmClassificationDataset
from esp_simulator.models.blackjack import (
    BlackjackAntiCheater,
    BlackjackRewardAntiCheater,
    BlackjackTrajectoryLengthAntiCheater,
)

from utils.classifiers import add_argument_classifier
from utils.environments import add_argument_environment


class TesterModel(L.LightningModule):
    """Temporary model class for testing"""

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.cls_type == "trajectory":
            self.model = BlackjackAntiCheater(features_dim=self.hparams.features_dim)
        elif self.hparams.cls_type == "length":
            self.model = BlackjackTrajectoryLengthAntiCheater()
        elif self.hparams.cls_type == "reward":
            self.model = BlackjackRewardAntiCheater()
        self.criterion = nn.BCEWithLogitsLoss()

    def load_checkpoint(
        self, checkpoint_path: str, device: torch.device
    ) -> None:
        """Load model weight from the checkpoint

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
            # Precision-Recall
            precision, recall, thresholds_pr = self.pr_curve.compute()

            ap = -torch.trapz(precision, recall)
            self.log_dict({"test/AP": ap})

            precision_random = precision[0]

            fig_pr, ax_pr = self.pr_curve.plot(score=True)
            ax_pr.plot(
                [0, 1],
                [precision_random.cpu(), precision_random.cpu()],
                linestyle="dashed",
            )
            ax_pr.dataLim.y0 = 0
            ax_pr.dataLim.y1 = 1
            ax_pr.autoscale_view()

            if self.hparams.threshold is not None:
                threshold = self.hparams.threshold
                threshold_pr_idx = torch.where(thresholds_pr >= threshold)[0][0]
                precision_thr = precision[threshold_pr_idx]
                recall_thr = recall[threshold_pr_idx]

                self.log_dict({f"test/threshold={threshold}/precision": precision_thr})
                self.log_dict({f"test/threshold={threshold}/recall": recall_thr})
                ax_pr.plot(recall_thr.cpu(), precision_thr.cpu(), "x", color="k")

            output_pr_path = self.hparams.output_pr_curve_image
            fig_pr.savefig(output_pr_path)

            # ROC
            fpr, tpr, thresholds_roc = self.roc_curve.compute()
            auroc = torch.trapz(tpr, fpr)

            self.log_dict({"test/AUROC": auroc})

            fig_roc, ax_roc = self.roc_curve.plot(score=True)
            ax_roc.plot([0, 1], [0, 1], linestyle="dashed")

            if self.hparams.threshold is not None:
                threshold = self.hparams.threshold
                threshold_roc_idx = torch.where(thresholds_roc >= threshold)[0][-1]
                fpr_thr = fpr[threshold_roc_idx]
                tpr_thr = tpr[threshold_roc_idx]

                self.log_dict({f"test/threshold={threshold}/fpr": fpr_thr})
                ax_roc.plot(fpr_thr.cpu(), tpr_thr.cpu(), "x", color="k")

            output_roc_path = self.hparams.output_roc_curve_image
            fig_roc.savefig(output_roc_path)

            # Confusion matrix
            if self.hparams.threshold is not None:
                fig_conf, ax_conf = self.confmat.plot(
                    cmap="summer", labels=["Normal", "ESP"]
                )
                ax_conf.xaxis.set_label_position("top")
                ax_conf.xaxis.set_ticks_position("top")

                output_conf_path = self.hparams.output_confmat_image
                fig_conf.savefig(output_conf_path)

                # Print false positive, false negative
                print(f"False Positives: {self.fp}")
                print(f"False Negatives: {self.fn}")


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
    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    dm = TesterDataModule(args)
    model = TesterModel(args)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, device=fabric.device)

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
    parser.add_argument("--threshold", type=float, help="classifier threshold")
    add_argument_environment(parser)
    add_argument_classifier(parser)
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
