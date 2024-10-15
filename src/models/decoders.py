from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from models.encoders import InceptionModule
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
                                 nn.Linear(latent_n_channels, 32),
                                 nn.Linear(32, output_n_channels),
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


class InceptionDecoder(nn.Module):
    def __init__(self,
                 encoded_n_channels,
                 encoded_series_size,
                 output_n_channels,
                 output_length,
                 number_of_filters,
                 bottleneck_size,
                 use_bottleneck,
                 num_samples,
                 kernel_size,

                 depth=1
                 ):
        super().__init__()
        self.encoded_n_channels = encoded_n_channels
        self.encoded_series_size = encoded_series_size
        self.output_n_channels = output_n_channels
        self.output_length = output_length
        self.number_of_filters = number_of_filters
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck
        self.num_samples = num_samples
        self.kernels = [kernel_size // 4, kernel_size // 2, kernel_size]
        self.depth = depth
        self.input_layer = nn.Sequential(Reshape((-1, encoded_series_size, encoded_n_channels)), Permute((0, 2, 1)))
        self.blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            current_block = self._get_next_block(d)
            self.blocks.append(current_block)
            residual_block = self._get_next_residual_block(d)
            self.residual_blocks.append(residual_block)

        self.last_block = nn.Sequential(
            nn.ConvTranspose1d(4 * self.number_of_filters, self.output_n_channels, kernel_size=1),
            nn.ConvTranspose1d(self.output_n_channels, self.output_n_channels,
                               kernel_size=self.output_length - self.encoded_series_size + 1,
                               stride=1,
                               padding=0),
            Permute((0, 2, 1)))

    def _get_next_residual_block(self, d):
        residual_block = nn.Sequential()
        in_channels = self.encoded_n_channels if d == self.depth - 1 else 4 * self.number_of_filters
        residual_block.add_module(f'residual {d + 1} Transpose CNN',
                                  nn.ConvTranspose1d(in_channels=in_channels, out_channels=4 * self.number_of_filters,
                                                     kernel_size=1, padding=0))
        residual_block.add_module(f'residual {d + 1} batch norm', nn.BatchNorm1d(4 * self.number_of_filters))
        return residual_block

    def _get_next_block(self, d):
        current_block = nn.Sequential()
        for i in range(3):
            in_size = 4 * self.number_of_filters if i != 0 else self.encoded_n_channels
            use_bottleneck = self.use_bottleneck if i == 2 else True
            inception_module = InceptionModule(in_size, self.number_of_filters, self.kernels[i],
                                               use_bottleneck=use_bottleneck,
                                               bottleneck_size=self.bottleneck_size)
            current_block.add_module(f'inception_module_{i + 1}', inception_module)
        return current_block

    def forward(self, x):
        shape = x.shape
        x = self.input_layer(x)
        residual_input = x
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            residual = self.residual_blocks[i](residual_input)
            residual_input = x
            x = F.relu(x + residual)
        x = self.last_block(x)
        logits = x.reshape(-1, shape[1], self.output_length, self.output_n_channels)
        mean = logits.mean(1)
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
