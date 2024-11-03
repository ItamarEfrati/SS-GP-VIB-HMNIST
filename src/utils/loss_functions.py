import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.distributions as dist


# region triplet loss
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


def get_triplet_loss(batch_size, qy_z_full, pz_x, y_labeled, triplet_dist, triplet_threshold):
    try:
        px_z_list = np.array(
            [dist.MultivariateNormal(pz_x.mean[i], covariance_matrix=pz_x.covariance_matrix[i]) for i in
             range(pz_x.batch_shape[0])])
    except:
        print(1)

    if triplet_dist == 'euclidean':
        unlabeled_z = pz_x.mean[batch_size:]
        labeled_z = pz_x.mean[:batch_size]
    else:
        unlabeled_z = px_z_list[batch_size:]
        labeled_z = px_z_list[:batch_size]

    unlabeled_mean = qy_z_full.mean[batch_size:]
    triplet_loss = torch.tensor(0.0, device=qy_z_full.mean.device)

    rows_with_values_gt = torch.any(unlabeled_mean > triplet_threshold, dim=1)
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

            if triplet_dist == 'euclidean' and selected_positive_sample.dim() < 3:
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

            if triplet_dist == 'euclidean' and selected_positive_sample.dim() < 3:
                selected_negative_sample = selected_negative_sample.unsqueeze(0)
        else:
            continue  # Handle cases where no match is found

        selected_positive_samples.append(selected_positive_sample)
        selected_negative_samples.append(selected_negative_sample)
        anchors_for_calc.append(anchor)

    if len(anchors_for_calc) == 0:
        return 0

    margin = 0.01  # Define margin for the triplet loss
    if triplet_dist == 'euclidean':
        positive = torch.stack(selected_positive_samples).squeeze()
        negative = torch.stack(selected_negative_samples).squeeze()
        anchors_for_calc = torch.stack(anchors_for_calc).squeeze()

        triplet_loss += F.triplet_margin_loss(anchors_for_calc, positive, negative, margin=margin, p=2,
                                              reduction='sum')

    elif triplet_dist == 'jsd':
        p, n = [], []
        for i in range(len(anchors_for_calc)):
            pos_distance = jensen_shannon_divergence(anchors_for_calc[i], selected_positive_samples[i])
            p.append(pos_distance.detach().cpu())
            neg_distance = jensen_shannon_divergence(anchors_for_calc[i], selected_negative_samples[i])
            n.append(neg_distance.detach().cpu())

            # triplet_loss += torch.exp(pos_distance - neg_distance)
            triplet_loss += F.relu(pos_distance - neg_distance + margin)

    else:
        raise Exception("wrong triplet loss distance")

    return triplet_loss / len(anchors_for_calc)


# endregion


class Jittering:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        return x + torch.random.normal(loc=0., scale=self.std, size=x.shape)
        # return x + torch.randn(x.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.world_size = world_size

        self.mask = None
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size=1):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size  # * self.world_size
        if (self.mask is None) or (self.mask.shape[0] != N):
            self.mask = self.mask_correlated_samples(batch_size, self.world_size)

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
