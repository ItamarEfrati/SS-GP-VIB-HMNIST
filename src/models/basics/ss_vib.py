import os

import torch
import torchmetrics
import pytorch_lightning as pl

from torch.distributions import MultivariateNormal

from models.basics.vib import VIB
from models.decoders import Decoder
from src.models.abstracts.abstract_vib import AbstractVIB


class SemiSupervisedVIB(VIB, pl.LightningModule):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 is_vae,
                 data_beta,
                 data_decoder: Decoder,
                 ignore=None,
                 **kwargs):
        ignore = ['encoder', 'decoder', 'data_decoder'] if ignore is None else ignore + ['encoder',
                                                                                         'decoder',
                                                                                         'data_decoder']
        kwargs['ignore'] = ignore
        super().__init__(**kwargs)
        self.is_vae = is_vae
        self.data_decoder = data_decoder

    def decode(self, z, is_train=True):
        if is_train:
            z_labeled, z_unlabeled = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
            px_z = self.data_decoder(z_unlabeled)
            qy_z = self.label_decoder(z_labeled)
            qy_z_full = self.label_decoder(z)
        else:
            qy_z = self.label_decoder(z)
            px_z = self.data_decoder(z)
            qy_z_full = qy_z

        return qy_z, px_z, qy_z_full

    def forward(self, x, is_sample=True, is_train=True):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose batch shape with num samples shape (B, num_samples, z_dim)
        z = z.transpose(0, 1)
        qy_z, px_z, qy_z_full = self.decode(z, is_train)
        return pz_x, qy_z, px_z, qy_z_full

    def run_forward_step(self, batch, is_sample, stage):
        is_train = stage is 'train'
        x, y, x_unlabeled = self.get_x_y(batch, is_train)
        if is_train:
            x_temp = torch.concat([x, x_unlabeled])
            x_reconstruction_origin = x_unlabeled
        else:
            x_temp = x
            x_reconstruction_origin = x
        pz_x, qy_z, px_z, qy_z_full = self.forward(x_temp, is_sample, is_train)

        kl = self.compute_kl_divergence(pz_x)
        label_log_likelihood = self.compute_log_likelihood(qy_z, y)

        log_values = {'mean_label_negative_log_likelihood': (-label_log_likelihood).mean()}
        data_log_likelihood = self.compute_log_likelihood(px_z, x_reconstruction_origin, is_multinomial=False)
        data_log_likelihood = data_log_likelihood.sum(dim=[1, 2])
        if self.is_vae:
            reconstruction_error = data_log_likelihood
            if is_train:
                kl = kl[x_reconstruction_origin.shape[0]:]
        else:
            if is_train:
                reconstruction_error = torch.concat([label_log_likelihood, data_log_likelihood])
            else:
                reconstruction_error = label_log_likelihood
        batch_size = data_log_likelihood.shape[0]
        kl[batch_size:] = kl[batch_size:] * self.hparams.data_beta
        kl[:batch_size] = kl[:batch_size] * self.hparams.beta
        log_values['mean_data_negative_log_likelihood'] = -data_log_likelihood.mean()

        entropy = qy_z_full.entropy().mean()
        log_values['qy_z_entropy'] = entropy
        elbo = reconstruction_error - kl
        elbo = elbo.mean() + entropy
        loss = -elbo
        probabilities, y_pred = self.get_y_pred(qy_z)
        log_values['loss'] = loss
        log_values['kl_mean'] = kl.mean()
        return {'log': log_values,
                'preds': y_pred,
                'probs': probabilities,
                'target': y,
                'reconstruction': px_z.mean,
                'original': x_reconstruction_origin,
                'latent': pz_x.mean}

    def get_x_y(self, batch, is_train=True):
        if is_train:
            labeled_data, unlabeled_data = batch[0], batch[1]
            x_unlabeled = unlabeled_data[0]
            x, y = labeled_data[0], labeled_data[1]
        else:
            x_unlabeled = None
            x, y = batch[0], batch[1]

        # the model is expecting (B, input_dim)
        x = torch.flatten(x, 1)
        x_unlabeled = torch.flatten(x_unlabeled, 1)
        return x, y, x_unlabeled

    def get_y_pred(self, qy_z):
        probabilities = qy_z.mean if not self.hparams.is_ensemble else qy_z.mean.mean(1)
        y_pred = torch.argmax(probabilities, dim=1)
        probabilities = probabilities[:, 1] if self.hparams.num_classes == 2 else probabilities
        return probabilities, y_pred

    # region Loss computations

    def compute_log_likelihood(self, prob, target, class_weight=None, is_multinomial=True):
        if class_weight is None:
            if is_multinomial:
                target = torch.nn.functional.one_hot(target.long(), num_classes=prob.event_shape[-1])
            log_likelihood = prob.log_prob(target)
        else:
            nll = torch.nn.NLLLoss(reduction='none', weight=class_weight)
            log_likelihood = -nll(prob.logits, target.long())
        return log_likelihood

    # endregion
