import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributions as dist

from models.variational.vib import VIB
from models.decoders import Decoder


class SemiSupervisedVIB(VIB, pl.LightningModule):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 is_vae,
                 is_ssl,
                 entropy_coef,
                 data_beta,
                 reconstruction_coef,
                 triplet_loss_coef,
                 data_decoder: Decoder,
                 triplet_threshold,
                 classification_coef,
                 ignore=None,
                 **kwargs):
        ignore = ['encoder', 'decoder', 'data_decoder'] if ignore is None else ignore + ['encoder',
                                                                                         'decoder',
                                                                                         'data_decoder']
        kwargs['ignore'] = ignore
        super().__init__(**kwargs)
        assert not (is_vae and is_ssl), "vae cannot be ssl"
        self.data_beta = data_beta if data_beta != -1 else self.beta
        self.data_decoder = data_decoder
        self.current_acc = 0

    def decode(self, z, is_train=True):
        if is_train and self.hparams.is_ssl:
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
        if is_train and self.hparams.is_ssl:
            x_temp = torch.concat([x, x_unlabeled])
            x_reconstruction_origin = x_unlabeled
        else:
            x_temp = x
            x_reconstruction_origin = x
        pz_x, qy_z, px_z, qy_z_full = self.forward(x_temp, is_sample, is_train)

        kl = self.compute_kl_divergence(pz_x)
        label_log_likelihood = self.hparams.classification_coef * self.compute_log_likelihood(qy_z, y)

        data_log_likelihood = self.compute_log_likelihood(px_z, x_reconstruction_origin, is_multinomial=False)
        data_log_likelihood = data_log_likelihood.mean(dim=[1, 2])
        batch_size = data_log_likelihood.shape[0]
        triplet_loss = 0
        if self.hparams.is_vae:
            reconstruction_error = data_log_likelihood
            kl = kl * self.data_beta
        elif self.hparams.is_ssl and is_train:
            reconstruction_error = torch.concat(
                [label_log_likelihood, self.hparams.reconstruction_coef * data_log_likelihood])
            kl[batch_size:] = kl[batch_size:] * self.data_beta
            kl[:batch_size] = kl[:batch_size] * self.hparams.beta

            # triplet loss
            if (self.hparams.triplet_loss_coef > 0) and (self.current_acc > 0.7):
                triplet_loss = self.get_triplet_loss(batch_size, qy_z_full, pz_x, y)
            else:
                triplet_loss = 0

        else:
            reconstruction_error = label_log_likelihood
            kl = kl * self.beta

        log_values = {'mean_label_negative_log_likelihood': (-label_log_likelihood).mean(),
                      'mean_data_negative_log_likelihood': -data_log_likelihood.mean(),
                      'triplet_loss': triplet_loss}

        entropy = qy_z_full.entropy()[:batch_size].mean()
        log_values['qy_z_entropy'] = entropy
        elbo = reconstruction_error - kl
        elbo = elbo.mean() - self.hparams.entropy_coef * entropy - triplet_loss
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

    def get_triplet_loss(self, batch_size, qy_z_full, pz_x, y_labeled):
        def kl_divergence(p, q):
            return torch.distributions.kl.kl_divergence(p, q).mean()

        # Jensen-Shannon Divergence using distributions
        def jensen_shannon_divergence(p, q):
            # Calculate the mean and covariance for M
            m_mean = 0.5 * (p.mean + q.mean)
            m_covariance = 0.5 * (p.covariance_matrix + q.covariance_matrix)

            # Create the multivariate normal distribution M
            m = dist.MultivariateNormal(m_mean, covariance_matrix=m_covariance)

            kl_pm = kl_divergence(p, m)
            kl_qm = kl_divergence(q, m)
            return torch.sqrt(0.5 * (kl_pm + kl_qm))

        try:
            px_z_list = np.array(
                [dist.MultivariateNormal(pz_x.mean[i], covariance_matrix=pz_x.covariance_matrix[i]) for i in
                 range(pz_x.batch_shape[0])])
        except:
            print(1)

        if self.hparams.triplet_dist == 'euclidean':
            unlabeled_z = pz_x.mean[batch_size:]
            labeled_z = pz_x.mean[:batch_size]
        else:
            unlabeled_z = px_z_list[batch_size:]
            labeled_z = px_z_list[:batch_size]

        unlabeled_mean = qy_z_full.mean[batch_size:]
        triplet_loss = torch.tensor(0.0, device=qy_z_full.mean.device)

        rows_with_values_gt = torch.any(unlabeled_mean > self.hparams.triplet_threshold, dim=1)
        row_indices = torch.nonzero(rows_with_values_gt).squeeze(-1)

        # Step 2: Check if any row meets the condition
        if row_indices.numel() != 0:
            # Get row-wise argmax for the rows where condition is met
            row_argmax = torch.argmax(unlabeled_mean, dim=1)
            filtered_argmax = row_argmax[rows_with_values_gt]

        anchors = unlabeled_z[rows_with_values_gt.cpu()]

        anchors_for_calc = []
        selected_positive_samples = []
        selected_negative_samples = []

        for class_idx, anchor, idx in zip(filtered_argmax, anchors, row_indices):
            # Filter x_labeled to get all samples with the matching class (positive samples)
            matching_indices = torch.nonzero(y_labeled == class_idx).squeeze()

            # dont choose your self
            # if i == 0:
            #     matching_indices = matching_indices[matching_indices != idx]

            if matching_indices.numel() > 0:
                # Randomly select one positive sample from the matching class
                if matching_indices.numel() > 1:
                    random_pos_idx = torch.randint(0, matching_indices.numel(), (1,))
                    selected_positive_sample = labeled_z[matching_indices[random_pos_idx]]
                else:
                    selected_positive_sample = labeled_z[matching_indices]

                if self.hparams.triplet_dist == 'euclidean' and selected_positive_sample.dim() < 3:
                    selected_positive_sample = selected_positive_sample.unsqueeze(0)

            else:
                continue

            # Filter x_labeled to get all samples with a different class (negative samples)
            non_matching_indices = torch.nonzero(y_labeled != class_idx).squeeze()

            # dont choose your self
            # if i == 0:
            #     non_matching_indices = non_matching_indices[non_matching_indices != idx]

            if non_matching_indices.numel() > 0:
                if non_matching_indices.numel() > 1:
                    # Randomly select one negative sample from a different class
                    random_neg_idx = torch.randint(0, non_matching_indices.numel(), (1,))
                    selected_negative_sample = labeled_z[non_matching_indices[random_neg_idx]]
                else:
                    selected_negative_sample = labeled_z[non_matching_indices]

                if self.hparams.triplet_dist == 'euclidean' and selected_positive_sample.dim() < 3:
                    selected_negative_sample = selected_negative_sample.unsqueeze(0)
            else:
                continue  # Handle cases where no match is found

            selected_positive_samples.append(selected_positive_sample)
            selected_negative_samples.append(selected_negative_sample)
            anchors_for_calc.append(anchor)

        if len(anchors_for_calc) == 0:
            return 0

        margin = 5  # Define margin for the triplet loss
        if self.hparams.triplet_dist == 'euclidean':
            positive = torch.stack(selected_positive_samples).squeeze()
            negative = torch.stack(selected_negative_samples).squeeze()
            anchors_for_calc = torch.stack(anchors_for_calc).squeeze()

            triplet_loss += F.triplet_margin_loss(anchors_for_calc, positive, negative, margin=margin, p=2,
                                                  reduction='sum')

        elif self.hparams.triplet_dist == 'jsd':
            for i in range(len(anchors_for_calc)):
                pos_distance = jensen_shannon_divergence(anchors_for_calc[i], selected_positive_samples[i])
                neg_distance = jensen_shannon_divergence(anchors_for_calc[i], selected_negative_samples[i])

                triplet_loss += torch.exp(pos_distance - neg_distance)
                # t_loss += F.relu(pos_distance - neg_distance + margin)

        else:
            raise Exception("wrong triplet loss distance")

        return self.hparams.triplet_loss_coef * (triplet_loss / len(anchors_for_calc))


    def get_x_y(self, batch, is_train=True):
        if is_train and self.hparams.is_ssl:
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


    def on_validation_epoch_end(self):
        # Compute and log the accuracy for the epoch
        if self.current_acc < 0.75:
            self.current_acc = self.val_metrics['ACC'].compute()

# endregion
