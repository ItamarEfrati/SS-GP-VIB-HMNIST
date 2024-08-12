import io

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from torch import nn
from typing import Optional
from torch.optim import Optimizer
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from torchvision.transforms import transforms
from sktime.dists_kernels.dtw import DtwDist

from utils.model_utils import get_confusion_matrix_image, get_precision_recall_curve_image, get_roc_curve_image


class ImagePlot(Callback):
    def __init__(self):
        image_shape = None

    def log_images(self, outputs, trainer, pl_module, split):
        tensors = torch.concat([outputs[0], outputs[1]]).reshape(-1, 28, 28, 1).permute(0, 3, 1, 2)
        grid = torchvision.utils.make_grid(tensors, nrow=10)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)

        pl_module.logger.experiment.log({f'{split} reconstruction images': wandb.Image(image), "epoch": pl_module.current_epoch})

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if batch_idx == 0:
            self.log_images((outputs['reconstruction'][:4], outputs['original'][:4]), trainer, pl_module, 'Train')

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if batch_idx == 0:
            self.log_images((outputs['reconstruction'][:4], outputs['original'][:4]), trainer, pl_module, 'Val')

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if batch_idx == 0:
            self.log_images((outputs['reconstruction'][:4], outputs['original'][:4]), trainer, pl_module, 'Test')


class MyEarlyStopping(Callback):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for m, v in trainer.lightning_module.val_metrics.items():
            if 'Accuracy' in m:
                d = v

        if d.compute().item() == 1:
            trainer.should_stop = True


class LogEvaluationMetrics(Callback):
    def __init__(self, num_classes):
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.num_classes = num_classes

    def log_metrics_images(self, outputs, trainer, pl_module, split):
        predictions = torch.cat([tmp['predictions'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        probabilities = torch.cat([tmp['probabilities'] for tmp in outputs])

        confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.num_classes).to(pl_module.device)
        confusion_matrix = confusion_matrix(predictions, targets)

        cm_im = get_confusion_matrix_image(confusion_matrix, self.num_classes)
        prc_im = get_precision_recall_curve_image(targets, probabilities)
        rocc_im = get_roc_curve_image(targets, probabilities)

        trainer.logger.experiment.add_image(f"{split} Confusion Matrix", cm_im, trainer.current_epoch)
        trainer.logger.experiment.add_image(f"{split} Precision Recall Curve", prc_im, trainer.current_epoch)
        trainer.logger.experiment.add_image(f"{split} ROC Curve", rocc_im, trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(self.train_outputs) > 0:
            self.log_metrics_images(self.train_outputs, trainer, pl_module, 'Train')
            self.train_outputs = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics_images(self.val_outputs, trainer, pl_module, 'Validation')
        self.val_outputs = []

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics_images(self.test_outputs, trainer, pl_module, 'Test')
        self.test_outputs = []

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.train_outputs.append(outputs)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.val_outputs.append(outputs)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.test_outputs.append(outputs)


class WarmupStart(Callback):

    def __init__(self, warmup_steps, lr):
        self.warmup_steps = warmup_steps
        self.lr = lr

    def on_before_optimizer_step(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int
    ) -> None:
        if self.warmup_steps == 0:
            return
        if trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_step == self.warmup_steps:
            trainer.progress_bar_callback.print("Finish warmup")


class XaviarInit(Callback):

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        for module in pl_module._modules.values():
            if 'net' in dir(module):
                for layer in list(module.net):
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
                        continue
                    if 'weight' in dir(layer):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        layer.bias.data.zero_()


class ParameterInitialization(Callback):

    def __init__(self, init_method):
        self.init_method = init_method

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.init_method is None:
            return
        for module in pl_module._modules.values():
            if 'net' in dir(module):
                for layer in list(module.net):
                    if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)
                        continue
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                        if self.init_method == 'kaiming':
                            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                        elif self.init_method == 'xavier':
                            nn.init.xavier_uniform_(layer.weight)
                        else:
                            raise Exception(f"Wrong init name for {self.init_method}")
                        layer.bias.data.zero_()


class LatentSpaceSaver(Callback):
    def __init__(self):
        self.train_outputs = []
        self.train_predictions = []
        self.train_target = []
        self.test_outputs = []
        self.test_predictions = []
        self.test_target = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        train_outputs = torch.cat(self.train_outputs)
        train_predictions = torch.cat(self.train_predictions)
        train_target = torch.cat(self.train_target)
        torch.save(train_outputs, 'train_z.pt')
        torch.save(train_predictions, 'train_pred.pt')
        torch.save(train_target, 'train_target.pt')
        self.train_outputs = []
        self.train_predictions = []
        self.train_target = []

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_outputs = torch.cat(self.test_outputs)
        self.test_predictions = torch.cat(self.test_predictions)
        self.test_target = torch.cat(self.test_target)
        torch.save(self.test_outputs, 'test_z.pt')
        torch.save(self.test_predictions, 'test_pred.pt')
        torch.save(self.test_target, 'test_target.pt')

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.train_outputs.append(outputs['latent'])
        self.train_predictions.append(outputs['preds'])
        self.train_target.append(outputs['target'])

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.test_outputs.append(outputs['latent'])
        self.test_predictions.append(outputs['preds'])
        self.test_target.append(outputs['target'])


class PlotLatentSpace(Callback):
    def __init__(self):
        self.val_outputs = []
        self.val_predictions = []
        self.val_target = []

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch % 5 == 0:
            # Concatenate validation outputs, predictions, and targets
            val_outputs = torch.cat(self.val_outputs).cpu().numpy()
            val_predictions = torch.cat(self.val_predictions).cpu().numpy()
            val_target = torch.cat(self.val_target).cpu().numpy()

            d = DtwDist(weighted=True, derivative=True)
            distmat = d.transform(val_outputs)

            # Apply t-SNE to the latent space
            tsne = TSNE(n_components=2, metric='precomputed', init='random')
            val_outputs_2d = tsne.fit_transform(distmat)

            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Plot the 2D latent space with true labels
            scatter_true = axes[0].scatter(val_outputs_2d[:, 0], val_outputs_2d[:, 1], c=val_target, cmap='viridis',
                                           alpha=0.5)
            axes[0].set_title("Validation Latent Space with True Labels")
            fig.colorbar(scatter_true, ax=axes[0])

            # Plot the 2D latent space with validation predictions
            scatter_pred = axes[1].scatter(val_outputs_2d[:, 0], val_outputs_2d[:, 1], c=val_predictions,
                                           cmap='viridis', alpha=0.5)
            axes[1].set_title("Validation Latent Space with Predictions")
            fig.colorbar(scatter_pred, ax=axes[1])

            # Convert the plot to a numpy array
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)

            pl_module.logger.experiment.log(
                {"Validation Latent Space": wandb.Image(image), "epoch": pl_module.current_epoch})

        # Reset the stored outputs
        self.val_outputs = []
        self.val_predictions = []
        self.val_target = []

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.val_outputs.append(outputs['latent'])
        self.val_predictions.append(outputs['preds'])
        self.val_target.append(outputs['target'])
