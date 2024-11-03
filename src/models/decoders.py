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
        is_binary = logits.shape[-1] == 2
        if is_binary:
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
                                 nn.Linear(latent_n_channels, output_n_channels),
                                 # nn.Linear(32, output_n_channels),
                                 Permute((0, 2, 1)),
                                 nn.Linear(latent_length, 128),
                                 # nn.Linear(128, 128),
                                 nn.Linear(128, output_length),
                                 Permute((0, 2, 1)))

    def get_probs(self, logits):
        return logits.mean(1)

    def __call__(self, x):
        logits = self.net(x)
        logits = logits.reshape(-1, x.shape[1], self.output_length, self.output_n_channels)
        mean = self.get_probs(logits)
        return Normal(loc=mean, scale=0.1 * torch.ones(mean.shape, device=mean.device))


# class InceptionDecoder(nn.Module):
#     def __init__(self,
#                  encoded_n_channels,
#                  encoded_series_size,
#                  output_n_channels,
#                  output_length,
#                  number_of_filters,
#                  bottleneck_size,
#                  use_bottleneck,
#                  num_samples,
#                  kernel_size,
#
#                  depth=1
#                  ):
#         super().__init__()
#         self.encoded_n_channels = encoded_n_channels
#         self.encoded_series_size = encoded_series_size
#         self.output_n_channels = output_n_channels
#         self.output_length = output_length
#         self.number_of_filters = number_of_filters
#         self.bottleneck_size = bottleneck_size
#         self.use_bottleneck = use_bottleneck
#         self.num_samples = num_samples
#         self.kernels = [kernel_size // 4, kernel_size // 2, kernel_size]
#         self.depth = depth
#         self.input_layer = nn.Sequential(Reshape((-1, encoded_series_size, encoded_n_channels)), Permute((0, 2, 1)))
#         self.blocks = nn.ModuleList()
#         self.residual_blocks = nn.ModuleList()
#         self._build_network()
#
#     def _build_network(self):
#         for d in range(self.depth):
#             current_block = self._get_next_block(d)
#             self.blocks.append(current_block)
#             residual_block = self._get_next_residual_block(d)
#             self.residual_blocks.append(residual_block)
#
#         self.last_block = nn.Sequential(
#             nn.ConvTranspose1d(4 * self.number_of_filters, self.output_n_channels, kernel_size=1),
#             nn.ConvTranspose1d(self.output_n_channels, self.output_n_channels,
#                                kernel_size=self.output_length - self.encoded_series_size + 1,
#                                stride=1,
#                                padding=0),
#             Permute((0, 2, 1)))
#
#     def _get_next_residual_block(self, d):
#         residual_block = nn.Sequential()
#         in_channels = self.encoded_n_channels if d == self.depth - 1 else 4 * self.number_of_filters
#         residual_block.add_module(f'residual {d + 1} Transpose CNN',
#                                   nn.ConvTranspose1d(in_channels=in_channels, out_channels=4 * self.number_of_filters,
#                                                      kernel_size=1, padding=0))
#         residual_block.add_module(f'residual {d + 1} batch norm', nn.BatchNorm1d(4 * self.number_of_filters))
#         return residual_block
#
#     def _get_next_block(self, d):
#         current_block = nn.Sequential()
#         for i in range(3):
#             in_size = 4 * self.number_of_filters if i != 0 else self.encoded_n_channels
#             use_bottleneck = self.use_bottleneck if i == 2 else True
#             inception_module = InceptionModule(in_size, self.number_of_filters, self.kernels[i],
#                                                use_bottleneck=use_bottleneck,
#                                                bottleneck_size=self.bottleneck_size)
#             current_block.add_module(f'inception_module_{i + 1}', inception_module)
#         return current_block
#
#     def forward(self, x):
#         shape = x.shape
#         x = self.input_layer(x)
#         residual_input = x
#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x)
#             residual = self.residual_blocks[i](residual_input)
#             residual_input = x
#             x = F.relu(x + residual)
#         x = self.last_block(x)
#         logits = x.reshape(-1, shape[1], self.output_length, self.output_n_channels)
#         mean = logits.mean(1)
#         return Normal(loc=mean, scale=torch.ones(mean.shape, device=mean.device))

class InceptionDecoderModule(nn.Module):
    def __init__(self, in_filters, nb_filters, kernel_size, use_bottleneck, bottleneck_size):
        super(InceptionDecoderModule, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size

        if use_bottleneck:
            # Bottleneck layer to reduce channels before transpose convolution
            self.input_inception = nn.ConvTranspose1d(in_channels=in_filters, out_channels=bottleneck_size, kernel_size=1, bias=False)
        else:
            bottleneck_size = in_filters

        # Define kernel sizes for transpose convolutions
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        self.trans_conv_list = nn.ModuleList()
        for k_size in kernel_size_s:
            self.trans_conv_list.append(
                nn.ConvTranspose1d(in_channels=bottleneck_size, out_channels=nb_filters, kernel_size=k_size, stride=1,
                                   padding=k_size//2, bias=False))

        self.max_pool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.trans_conv_6 = nn.ConvTranspose1d(in_channels=in_filters, out_channels=nb_filters, kernel_size=1, bias=False)

        # Batch normalization after combining the outputs
        self.batch_norm = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x):
        # If using bottleneck, apply the 1x1 transpose convolution
        if self.use_bottleneck and x.size(-1) > 1:
            input_inception = self.input_inception(x)
        else:
            input_inception = x

        # Apply transpose convolutions with different kernel sizes
        trans_conv_list = [trans_conv(input_inception) for trans_conv in self.trans_conv_list]
        trans_conv_list.append(self.trans_conv_6(self.max_pool_1(x)))

        # Concatenate results from different kernel sizes
        x = torch.cat(trans_conv_list, dim=1)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class InceptionDecoder(nn.Module):
    def __init__(self,
                 input_n_channels,
                 decoding_size,
                 decoding_series_size,
                 number_of_filters,
                 bottleneck_size,
                 use_bottleneck,
                 kernel_size=40,
                 depth=1):
        super().__init__()
        self.input_n_channels = input_n_channels
        self.decoding_size = decoding_size
        self.decoding_series_size = decoding_series_size
        self.number_of_filters = number_of_filters
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck
        self.kernels = [kernel_size, kernel_size // 2, kernel_size // 4]
        self.depth = depth

        self.blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            current_block = self._get_next_block(d)
            self.blocks.append(current_block)

            residual_block = self._get_next_residual_block(d)
            self.residual_blocks.append(residual_block)

        # Final transpose convolution to output the correct dimensions
        self.last_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4 * self.number_of_filters,
                               out_channels=self.input_n_channels,
                               kernel_size=1,
                               bias=False),
            nn.AdaptiveAvgPool1d(self.decoding_series_size)
        )

    def _get_next_residual_block(self, d):
        # Setup residual connection, converting input channels to the necessary output channels
        residual_block = nn.Sequential()
        in_channels = self.decoding_size if d == 0 else 4 * self.number_of_filters
        residual_block.add_module(f'residual_{d + 1}_CNN',
                                  nn.ConvTranspose1d(in_channels=in_channels,
                                                     out_channels=4 * self.number_of_filters,
                                                     kernel_size=1,
                                                     padding=0))
        residual_block.add_module(f'residual_{d + 1}_batch_norm', nn.BatchNorm1d(4 * self.number_of_filters))
        return residual_block

    def _get_next_block(self, d):
        # Each block uses the InceptionDecoderModule, which applies multi-scale transpose convolutions
        is_first_block = d == 0
        current_block = nn.Sequential()
        for i in range(3):
            in_size = self.decoding_size if i == 0 and is_first_block else 4 * self.number_of_filters
            use_bottleneck = self.use_bottleneck if i == 0 else True
            inception_decoder_module = InceptionDecoderModule(in_size, self.number_of_filters, self.kernels[i],
                                                              use_bottleneck=use_bottleneck,
                                                              bottleneck_size=self.bottleneck_size)
            current_block.add_module(f'inception_decoder_module_{i + 1}', inception_decoder_module)
        return current_block

    def forward(self, x):
        shape = x.shape
        residual_input = x
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            residual = self.residual_blocks[i](residual_input)
            residual_input = x
            x = F.relu(x + residual)  # Add residual and apply ReLU

        x = self.last_block(x)  # Final upsampling step
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
                 z_dim_time_length,
                 output_size,
                 num_samples,
                 ):
        super(MultinomialDecoder, self).__init__(num_samples, output_size)

        self.output_size = output_size
        self.num_samples = num_samples

        self.net = nn.Sequential(Reshape((-1, z_dim_time_length, z_dim)),
                                 nn.Linear(z_dim, 1),
                                 # nn.Linear(32, output_n_channels),
                                 Permute((0, 2, 1)),
                                 nn.Linear(z_dim_time_length, 32),
                                 # nn.Linear(128, 128),
                                 nn.Linear(32, output_size),
                                 Permute((0, 2, 1)))


    def __call__(self, z):
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], self.output_size)
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


class InceptionClassifier(Decoder):
    def __init__(self,
                 output_size,
                 z_dim,
                 z_dim_time_length,
                 num_samples,
                 number_of_filters,
                 bottleneck_size,
                 use_bottleneck,
                 kernel_size=40,
                 depth=1
                 ):
        super().__init__(output_size=output_size, num_samples=num_samples)
        self.output_size = output_size
        self.z_dim = z_dim
        self.z_dim_time_length = z_dim_time_length
        self.number_of_filters = number_of_filters
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck
        self.kernels = [kernel_size, kernel_size // 2, kernel_size // 4]
        self.depth = depth
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
            nn.Conv1d(in_channels=4 * self.number_of_filters,
                      out_channels=1,
                      kernel_size=1,
                      bias=False),
            nn.Linear(self.z_dim_time_length, self.output_size),
        )

    def _get_next_residual_block(self, d):
        residual_block = nn.Sequential(Permute((0, 2, 1))) if d == 0 else nn.Sequential()
        in_channels = self.z_dim if d == 0 else 4 * self.number_of_filters
        residual_block.add_module(f'residual {d + 1} CNN',
                                  nn.Conv1d(in_channels=in_channels, out_channels=4 * self.number_of_filters,
                                            kernel_size=1,
                                            padding=0))
        residual_block.add_module(f'residual {d + 1} batch norm', nn.BatchNorm1d(4 * self.number_of_filters))
        return residual_block

    def _get_next_block(self, d):
        is_first_block = d == 0
        current_block = nn.Sequential(Permute((0, 2, 1))) if is_first_block else nn.Sequential()
        for i in range(3):
            in_size = self.z_dim if i == 0 and is_first_block else 4 * self.number_of_filters
            use_bottleneck = self.use_bottleneck if i == 0 else True
            inception_module = InceptionModule(in_size, self.number_of_filters, self.kernels[i],
                                               use_bottleneck=use_bottleneck,
                                               bottleneck_size=self.bottleneck_size)
            current_block.add_module(f'inception_module_{i + 1}', inception_module)
        return current_block

    def __call__(self, z):
        shape = z.shape
        z = z.reshape(-1, self.z_dim_time_length, self.z_dim)
        residual_input = z
        for i in range(len(self.blocks)):
            z = self.blocks[i](z)
            residual = self.residual_blocks[i](residual_input)
            residual_input = z
            z = F.relu(z + residual)
        logits = self.last_block(z)
        logits = logits.reshape(-1, shape[1], self.output_size)
        # probs = self.get_probs(logits)
        return Multinomial(logits=logits.mean(1))
# endregion

# endregion
