from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from utils.model_utils import get_linear_layers, Permute, Reshape

from torch.distributions.multinomial import Multinomial


class Decoder(nn.Module, ABC):

    def __init__(self, num_samples, output_size, is_binary=False):
        """ Decoder parent class with no specified output distribution
        """
        super(Decoder, self).__init__()
        self.num_samples = num_samples
        self.output_size = output_size
        self.is_binary = is_binary

    def get_probs(self, logits):
        logits = logits.mean(1)
        if self.is_binary:
            probs = F.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs

    @abstractmethod
    def __call__(self, x):
        pass


class DataDecoder(nn.Module, ABC):

    def __init__(self, num_samples, output_length, output_n_channels):
        """ Decoder parent class with no specified output distribution
        """
        super(DataDecoder, self).__init__()
        self.num_samples = num_samples
        self.output_length = output_length
        self.output_n_channels = output_n_channels

    @abstractmethod
    def __call__(self, x):
        pass


# region Bernoulli
class BernoulliDecoder(Decoder):

    def __init__(self,
                 num_samples,
                 output_size,
                 z_dim,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 ):
        super().__init__(num_samples, output_size, is_binary=True)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        logits = self.net(x)
        probs = self.get_probs(logits)
        return Bernoulli(probs=probs)


# endregion

class GaussianDecoder(Decoder):

    def __init__(self,
                 num_samples,
                 output_size,
                 z_dim,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 ):
        super().__init__(num_samples, output_size, is_binary=False)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        logits = self.net(x)
        mean = self.get_probs(logits)
        return Normal(loc=mean, scale=torch.ones(mean.shape, device=mean.device))


# region DataDecoders

class GaussianDataDecoder(DataDecoder):

    def __init__(self,
                 num_samples,
                 output_n_channels,
                 output_length,
                 latent_n_channels,
                 latent_length
                 ):
        super().__init__(num_samples, output_length, output_n_channels)
        self.net = nn.Sequential(Reshape((-1, latent_length, latent_n_channels)),
                                 nn.AdaptiveAvgPool1d(output_n_channels),
                                 Permute((0, 2, 1)),
                                 nn.Linear(latent_length, 128),
                                 nn.Linear(128, 128),
                                 nn.Linear(128, output_length),
                                 Permute((0, 2, 1)))

    def get_probs(self, logits):
        return logits.mean(1)

    def __call__(self, x):
        logits = self.net(x)
        logits = logits.reshape(-1, x.shape[1], self.output_length, self.output_n_channels)
        mean = self.get_probs(logits)
        return Normal(loc=mean, scale=torch.ones(mean.shape, device=mean.device))


# endregion


# region Multinomial

class MultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 num_samples,
                 output_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3
                 ):
        super(MultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, z):
        logits = self.net(z)
        probs = self.get_probs(logits)
        return Multinomial(probs=probs)


# region TimeSeriesMultinomial

class FlattenMultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 output_size,
                 num_samples,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 **kwargs):
        super(FlattenMultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim * z_dim_time_length
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        z = z.flatten(-2)
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], logits.shape[-1])
        return logits

    def __call__(self, z):
        logits = self._get_logits(z)
        probs = self.get_probs(logits)

        return Multinomial(probs=probs)

# endregion

# endregion
