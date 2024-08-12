import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

from src.models.decoders import Decoder
from src.models.encoders import Encoder


class AbstractAdversarialVIB(ABC):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 discriminator: Decoder,
                 num_samples: int
                 ):
        super(AbstractAdversarialVIB, self).__init__()
        self.encoder = encoder
        self.label_decoder = decoder
        self.discriminator = discriminator
        self.num_samples = num_samples

        self.prior = None

    @abstractmethod
    def r_z(self):
        pass

    @abstractmethod
    def step(self, batch, stage):
        pass

    @abstractmethod
    def compute_log_likelihood(self, px_z, x):
        pass

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.label_decoder(z)

    def compute_adversarial_loss(self, q):
        # Sample from the prior distribution
        z_prior = self.r_z().sample((q.batch_shape[0], 1)) # Assume q is the latent distribution

        # Get discriminator outputs
        d_z_prior = self.discriminator(z_prior)  # Discriminator output for prior samples
        d_z_q = self.discriminator(q.sample().unsqueeze(1))  # Discriminator output for encoded samples

        # Adversarial loss for the autoencoder
        bce = torch.nn.BCELoss()

        # Labels for the loss
        real_labels = F.one_hot(torch.ones(q.batch_shape[0], device=z_prior.device, dtype=torch.long), 2).float()
        fake_labels = F.one_hot(torch.zeros(d_z_q.batch_shape[0], device=z_prior.device, dtype=torch.long), 2).float()

        # nll = torch.nn.NLLLoss(reduction='none')
        # log_likelihood = -nll(qy_z.logits, y.long())

        # Discriminator loss (real and fake)
        loss_real = bce(d_z_prior.mean, real_labels)
        loss_fake = bce(d_z_q.mean, fake_labels)
        discriminator_loss = loss_real + loss_fake

        # Generator loss (fool the discriminator)
        generator_loss = bce(d_z_q.mean, real_labels)

        # Return generator loss (which is the adversarial loss for the autoencoder)
        return generator_loss, discriminator_loss


    def forward(self, x, is_sample):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        z = z.transpose(0, 1)
        qy_z = self.decode(z)
        return pz_x, qy_z

    # endregion

    def get_latent_vectors(self, data_loader):
        labels = []
        latent_vectors = []
        for batch in data_loader:
            x, y = batch
            q_z = self.encoder(x)
            z = q_z.sample()
            latent_vectors.append(z)
            labels.append(y)
        return torch.concat(latent_vectors), torch.concat(labels)
