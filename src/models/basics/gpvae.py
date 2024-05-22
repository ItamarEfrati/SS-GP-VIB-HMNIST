import os

import torch
import torchmetrics
import pytorch_lightning as pl

from torch.distributions import MultivariateNormal

from src.models.abstracts.abstract_vib import AbstractVIB
from utils.model_utils import get_gp_prior


class GPVAE(pl.LightningModule):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 kernel,
                 sigma,
                 length_scale,
                 kernel_scales,
                 beta,
                 num_samples,
                 # image_preprocessor,
                 timeseries_encoder,
                 encoder,
                 decoder,
                 monitor_metric,
                 num_classes,
                 is_ensemble,
                 sample_during_evaluation,
                 ignore=None,
                 **kwargs):
        super().__init__(**kwargs)
        ignore = ['encoder', 'decoder'] if ignore is None else ignore + ['encoder', 'decoder']
        self.save_hyperparameters(ignore=ignore)

        # self.image_preprocessor = image_preprocessor
        self.timeseries_encoder = timeseries_encoder
        self.z_dim_time_length = self.timeseries_encoder.encoding_series_size
        self.encoder = encoder(z_dim_time_length=self.z_dim_time_length)
        self.decoder = decoder
        self.prior = None
        self.beta = beta
        self.num_samples = num_samples

        # metrics
        metrics = torchmetrics.MetricCollection({
            'MSE': torchmetrics.MeanSquaredError(num_classes=num_classes),
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def r_z(self):
        if self.prior is None:
            self.prior = get_gp_prior(kernel=self.hparams.kernel, kernel_scales=self.hparams.kernel_scales,
                                      time_length=self.z_dim_time_length, sigma=self.hparams.sigma,
                                      length_scale=self.hparams.length_scale, z_dim=self.encoder.encoding_size,
                                      device=self.device)
        return self.prior

    def encode(self, x):
        # transpose features and time dimensions to run cnn over time per feature
        # x = x.transpose(-2, -1)
        x = self.timeseries_encoder(x)
        pz_x = self.encoder(x)
        return pz_x

    def decode(self, z, is_ensemble=False):
        return self.decoder(z, is_ensemble)

    def compute_kl_divergence(self, q):
        kl = torch.distributions.kl.kl_divergence(q, self.r_z())
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl.sum(-1)

    def forward(self, x, is_sample, is_ensemble):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose batch shape with num samples shape (B, num_samples, z_dim)
        z = z.permute(1, 0, 3, 2)
        qx_z = self.decode(z, is_ensemble)
        return pz_x, qx_z

    def run_forward_step(self, batch, is_sample, stage):
        x = self.get_batch(batch)

        pz_x, qx_z = self.forward(x, is_ensemble=self.hparams.is_ensemble, is_sample=is_sample)
        log_likelihood = self.compute_log_likelihood(qx_z, x)
        kl = self.compute_kl_divergence(pz_x)

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        return {'log': {'loss': loss,
                        'kl_mean': kl.mean(),
                        'mean_negative_log_likelihood': (-log_likelihood).mean()},
                'latent': pz_x.mean,
                'reconstruction': qx_z.mean,
                'original': x}

    def step(self, batch, stage):
        is_sample = any([stage is 'train',
                         all([stage is not 'train', self.hparams.sample_during_evaluation])])
        forward_outputs = self.run_forward_step(batch, is_sample, stage)
        log_dict = forward_outputs.pop('log')
        forward_outputs['loss'] = log_dict['loss']
        log_dict = {f'{stage}_{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return forward_outputs

    def get_batch(self, batch):
        x, y = batch[0], batch[1]
        return x

    # region Loss computations

    def compute_log_likelihood(self, prob, target):
        log_likelihood = prob.log_prob(target)
        return log_likelihood.sum(dim=[1, 2])

    # endregion

    # region Pytorch lightning overwrites

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            lr_scheduler = {
                "scheduler": self.hparams.scheduler(optimizer=optimizer),
                "monitor": self.hparams.monitor_metric,
                "interval": 'epoch',
                "frequency": 1
            }
            return [optimizer], [lr_scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='train')
        self.train_metrics.update(step_output['reconstruction'], step_output['original'])
        self.log_dict(self.train_metrics, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='val')
        self.val_metrics.update(step_output['reconstruction'], step_output['original'])
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        return step_output

    def test_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='test')
        self.test_metrics.update(step_output['reconstruction'], step_output['original'])
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True)
        return step_output

    # endregion
