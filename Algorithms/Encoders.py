import numpy as np
from enum import Enum
from typing import Tuple, Union, List, Optional
from math import floor

import torch
import torch.nn as nn

import Utils
from NetworkUtils import get_device
from mlagents_envs.base_env import ObservationSpec, ActionSpec
from EncoderUtils import Initialization, initialization_methods, conv_output_shape, check_input_to_kernel, \
    NetworkSettings, ActivationType, activation_methods, BodyEncoderSettings, NormalizerType

default_visual_settings = {
    'conv2d': {
        'init_channel': 16,
        'init_kernel_size': 8,
        'init_stride': 4,
        'maintain_input_size': False,
        'use_embedding': False,
        'embedding_dictionary_size': 0,
        'num_embedding_dims': 1,
        'use_visual_dense_layers': True
    },
    'conv3d': {
        'init_channel': 16,
        'init_kernel_size': 2,
        'init_stride': 1,
        'maintain_input_size': False,
        'use_embedding': False,
        'embedding_dictionary_size': 2,
        'num_embedding_dims': 16,
        'use_visual_dense_layers': True
    },
}

default_positional_settings = {
    'positional_encoding_embedding_size': 30,
    'position_encoding_max_length': 50
}


# class Swish(nn.Module):
#     """
#     Multiplies data according to swish function.
#     Beta is set to 1, therefore behaves like Sigmoid-weighted Linear Unit (SiLu).
#     Apparently better than ReLu and Sigmoid for alleviating vanishing gradient problem during backprop.
#     """
#
#     def __init__(self,
#                  beta: float = 1.0
#                  ):
#         super(Swish, self).__init__()
#         self.beta = beta
#
#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         return torch.mul(data, torch.sigmoid(self.beta * data))


def get_sinusoidal_embedding(input: torch.Tensor):
    """
    For a given input, return the alternating sin(x) & cos(x) result for each input value.
    """
    embed = torch.stack((torch.sin(input), torch.cos(input)), dim=-1)
    return torch.flatten(embed, -2, -1)


class SestiniPositionalEncoder(nn.Module):
    """
    Encodes integer position to a continuous float vector using Sine/Cosine representation.
    See Vaswani (2017), "Attention is All You Need" for details.
    Implemented for global position embedding in Sestani (2022), "CCPT: Automatic Gameplay and Validation with ..."

    Note: Different from 2/3D implementation in Wang(2019) Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training.
    As Sestini assumes a single position integer with 2*3 embedding nodes, 2 for each physical dimension.
    Can handle stack observations and any number of positions (typically agent and/or goal positions)

    See PositionalEncoder for generalized approach for 3D.
    """

    def __init__(self, input_size: int, hidden_size: int, d: int = 32, normalize: bool = False):
        super(SestiniPositionalEncoder, self).__init__()
        self.total_embedding_size = d
        self.dim_embed_size = 6
        input_modifier = 1.0 / torch.pow(10000,
                                         torch.arange(0, self.dim_embed_size, 2,
                                                      dtype=torch.float32) / self.total_embedding_size)
        self.register_buffer("input_modifier", input_modifier)
        self.device = get_device()
        self.num_positions = input_size
        embed_output_size = self.num_positions * self.dim_embed_size
        normalize = False
        self.vectorInput = VectorInput(embed_output_size, normalize=normalize)
        self.linear = LinearEncoder(input_size=embed_output_size, num_layers=1, hidden_size=hidden_size,
                                    activation_type=ActivationType.LeakyReLU)
        self.to(self.device)

    def create_encoding(self, data: torch.Tensor):
        data = data.float()
        # Data = (batch*obs_stack,1), input_mod = (3,) --> embed_input = (batch*obs_stack, 3)
        embed_input = torch.mul(data, self.input_modifier)
        embed = get_sinusoidal_embedding(embed_input)
        return embed

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor shape (batch, num_pos*obs_stack) [x0, y0, x1, y1, ..., xn, yn]
        """
        batch_size = data.shape[0]
        data = data.reshape(-1, 1)
        pos_encoding = self.create_encoding(data)
        pos_encoding = pos_encoding.reshape(batch_size, -1)

        vector_out = self.vectorInput(pos_encoding)
        linear_out = self.linear(vector_out)
        return linear_out


class PositionalEncoder(nn.Module):
    """
    Positional Encoding based on 2D version in Wang and Liu 2019 (extended to 3D).

    For 3D encoding:
    PE(x,y,2i) = sin(x/10000**(6i/D)
    PE(x,y,2i+1) = cos(x/10000**(6i/D)
    PE(x,y,2j + D/3) = sin(y/10000**(6j/D)
    PE(x,y,2j + D/3) = cos(y/10000**(6j/D)
    PE(x,y,2k + 2D/3) = sin(z/10000**(6k/D)
    PE(x,y,2k + 2D/3) = cos(z/10000**(6k/D)

    where (i,j,k) in [0,D/6) for each dimension and (x,y,z) are integer positions in the dimension.

    Position Encoder structure: PositionalEncoder->VectorInput(normalizes if need be)->FC->SiLu activation
    #     """

    def __init__(self, input_size: int, hidden_size: int, d: int = 36, pos_dim: Tuple[int, int, int] = (10, 10, 10),
                 normalize: bool = False):
        super(PositionalEncoder, self).__init__()
        self.total_embedding_size = d
        dim_embed_size = int(np.ceil(self.total_embedding_size / 6) * 2)
        self.dim_embed_size = dim_embed_size
        self.total_embedding_size = 3 * self.dim_embed_size
        self.num_positions = input_size // 3
        input_modifier = 1.0 / torch.pow(10000,
                                         torch.arange(0, dim_embed_size, 2, dtype=torch.float32) / dim_embed_size)
        self.register_buffer("input_modifier", input_modifier)

        self.device = get_device()
        embed_output_size = self.num_positions * self.total_embedding_size
        self.vectorInput = VectorInput(embed_output_size, normalize)
        self.linear = LinearEncoder(input_size=embed_output_size, num_layers=1, hidden_size=hidden_size,
                                    activation_type=ActivationType.LeakyReLU)

        # self.ijk = pos_dim
        # self.cached_pos_encoding = self.cache_encoding()
        # self.ijk_convert = torch.as_tensor([self.ijk[1] * self.ijk[2], self.ijk[2], 1], device=self.device,
        #                                    dtype=torch.long).reshape(1, 3)

        self.to(self.device)

    def create_encoding(self, data: torch.Tensor):
        """
        Data of shape batch*
        """
        data = data.float()
        # Data = (batch*obs_stack,1), input_mod = (3,) --> embed_input = (batch*obs_stack, 3)
        embed_input = torch.mul(data, self.input_modifier)
        embed = get_sinusoidal_embedding(embed_input)
        embed = embed.reshape(-1, self.dim_embed_size)
        return embed

    # def cache_encoding(self):
    #     """
    #     Creates 3D positional encoding.
    #     :return: Positional embedding with shape (i*j*k, D)
    #     """
    #     i, j, k = self.ijk
    #     pos_x = torch.arange(0, i, device=self.device, dtype=torch.float32)
    #     pos_y = torch.arange(0, j, device=self.device, dtype=torch.float32)
    #     pos_z = torch.arange(0, k, device=self.device, dtype=torch.float32)
    #
    #     input_x = torch.einsum("i,j->ij", pos_x, self.input_modifier)
    #     input_y = torch.einsum("i,j->ij", pos_y, self.input_modifier)
    #     input_z = torch.einsum("i,j->ij", pos_z, self.input_modifier)
    #
    #     embed_x = get_sinusoidal_embedding(input_x).unsqueeze(1).unsqueeze(1)
    #     embed_y = get_sinusoidal_embedding(input_y).unsqueeze(1)
    #     embed_z = get_sinusoidal_embedding(input_z)
    #
    #     embed = torch.zeros((i, j, k, self.total_embedding_size), device=self.device, dtype=torch.float32)
    #     embed[:, :, :, :self.dim_embed_size] = embed_x
    #     embed[:, :, :, self.dim_embed_size:2 * self.dim_embed_size] = embed_y
    #     embed[:, :, :, 2 * self.dim_embed_size:] = embed_z
    #
    #     return embed.reshape(-1, self.total_embedding_size)

    # def convert_to_index(self, index_tensor: torch.Tensor):
    #     """
    #     Converts index tensor (x,y,z) for x/y/z in [0,i/j/k) to single sampling index.
    #
    #     :param index_tensor: torch.long tensor with shape [batch_size*num_stack_layer, 3]
    #     :return: sampling index i = z + k*y + k*j*x with shape (batch_size*num_stack_layer)
    #     """
    #     index = index_tensor * self.ijk_convert
    #     index = index.sum(dim=-1)
    #     return index.type(torch.long)

    # def forward(self, tensor: torch.Tensor):
    #     if len(tensor.shape) != 2:
    #         raise RuntimeError("Tensor should be size 2D! (N,3*num_stackedframes [x0,y0,z0,x1,y1,z1,...]")
    #
    #     batch_size = tensor.shape[0]
    #     num_frame_stack = tensor.shape[1] // 3
    #     tensor_reshape = tensor.reshape(batch_size * num_frame_stack, -1).type(torch.long)
    #     indices = self.convert_to_index(tensor_reshape)
    #     positional_embedding = self.cached_pos_encoding[indices].reshape(batch_size,
    #                                                                      num_frame_stack * self.total_embedding_size)
    #     vector_output = self.vectorInput(positional_embedding)
    #     return self.linear(vector_output)

    def forward(self, tensor: torch.Tensor):
        if len(tensor.shape) != 2:
            raise RuntimeError("Tensor should be size 2D! (N,3*num_positions = [x0,y0,z0,x1,y1,z1,...]")

        batch_size = tensor.shape[0]
        tensor = tensor.reshape(-1, 1)
        pos_encoding = self.create_encoding(tensor)
        pos_encoding = pos_encoding.reshape(batch_size, -1)
        vector_output = self.vectorInput(pos_encoding)
        return self.linear(vector_output)


def visual_dense_layer(input_size, output_size):
    """
    Creates a linear layer for CNN outputs or any RelU activated layer.

    The linear layer has Kaiming Initialization applied according to
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", K. He, (2015).
    """
    return linear_layer(input_size, output_size, Initialization.KaimingHeNormal, 1.41, Initialization.Zero)


def linear_layer(
        input_size: int,
        output_size: int,
        kernel_init: Initialization = Initialization.XavierGlorotUniform,
        kernel_gain: float = 1.0,
        bias_init: Initialization = Initialization.Zero,
        bias_gain: float = 1.0) -> nn.Module:
    """
    Creates a linear layer with input initialization weights and bias.
    """
    layer = nn.Linear(input_size, output_size)

    # Apply initialization method to weights.
    # In case of Kaiming Initialization, enforce linear non-linearity type.
    if kernel_init == Initialization.KaimingHeUniform or kernel_init == Initialization.KaimingHeNormal:
        initialization_methods[kernel_init](layer.weight.data, nonlinearity='linear')
    else:
        initialization_methods[kernel_init](layer.weight.data)

    # Apply kernel gain and bias initialization.
    layer.weight.data *= kernel_gain
    if bias_init == Initialization.Constant:
        initialization_methods[bias_init](layer.bias.data, 1.0)
    else:
        initialization_methods[bias_init](layer.bias.data)
    layer.bias.data *= bias_gain

    if kernel_init == Initialization.Uniform:
        initialization_methods[kernel_init](layer.weight.data, -kernel_gain, kernel_gain)

    if bias_init == Initialization.Uniform:
        initialization_methods[kernel_init](layer.bias.data, -bias_gain, bias_gain)
    return layer


def get_encoder_for_obs(obs_spec: ObservationSpec, hidden_size: int,
                        normalize: bool, norm_type: NormalizerType, encode_vector_inputs: bool,
                        encoder_settings: dict) -> Tuple[nn.Module, int]:
    """
    Creates an encoder for the given ObservationSpec.
    """
    shape = obs_spec.shape

    if len(shape) == 1:
        input_size = shape[0]
        if obs_spec.name.__contains__("Positional"):
            pos_embed = encoder_settings['positional_embedding']
            d = pos_embed['positional_encoding_embedding_size']
            pos = pos_embed['position_encoding_max_length']
            return PositionalEncoder(input_size, hidden_size, d, (pos, pos, pos), normalize), hidden_size
        elif obs_spec.name.__contains__("Sestini"):
            pos_embed = encoder_settings['positional_embedding']
            d = pos_embed['positional_encoding_embedding_size']
            return SestiniPositionalEncoder(input_size, hidden_size, d, normalize), hidden_size

        if encode_vector_inputs:
            return LinearVectorInput(input_size, normalize, norm_type, hidden_size), hidden_size

        return VectorInput(input_size=input_size, normalize=normalize, norm_type=norm_type), input_size

    if len(shape) == 3:
        input_shape = (shape[0], shape[1], shape[2])
        encoder = Conv2DEncoder(input_shape, hidden_size, **encoder_settings['conv2d'])
        return encoder, encoder.output_size

    if len(shape) == 4:
        input_shape = (shape[0], shape[1], shape[2], shape[3])
        encoder = Conv3DEncoder(input_shape, hidden_size, **encoder_settings['conv3d'])
        return encoder, encoder.output_size


def create_input_encoders(observation_specs: List[ObservationSpec], hidden_size: int,
                          normalize: bool, norm_type:NormalizerType, encode_vector_inputs: bool, encoder_settings: dict) \
        -> Tuple[nn.ModuleList, List[int]]:
    """
    Creates encoders for all observation specs.
    Returns a list of encoders and embedding sizes.
    """
    encoders: List[nn.Module] = []
    embedding_sizes: List[int] = []

    for obs_spec in observation_specs:
        encoder, embed_size = get_encoder_for_obs(obs_spec, hidden_size,
                                                  normalize, norm_type, encode_vector_inputs,
                                                  encoder_settings)
        encoders.append(encoder)
        embedding_sizes.append(embed_size)

    return nn.ModuleList(encoders), embedding_sizes


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.register_buffer("normalization_steps", torch.tensor(1))
        self.register_buffer("running_mean", torch.zeros(vec_obs_size))
        self.register_buffer("running_variance", torch.ones(vec_obs_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_state = (inputs - self.running_mean) / torch.sqrt(self.running_variance / self.normalization_steps)
        normalized_state = torch.clamp(normalized_state, -5.0, 5.0)
        # if torch.isnan(normalized_state).__contains__(True):
        #     print("NaN detected")
        #     self.print_stats()
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        with torch.no_grad():
            step_increment = vector_input.size()[0]
            total_new_steps = self.normalization_steps + step_increment

            input_to_old_mean = vector_input - self.running_mean
            new_mean = self.running_mean + (input_to_old_mean / total_new_steps).sum(0)

            input_to_new_mean = vector_input - new_mean
            new_variance = self.running_variance + (input_to_new_mean * input_to_old_mean).sum(0)

            self.running_mean: torch.Tensor = new_mean
            self.running_variance: torch.Tensor = new_variance
            self.normalization_steps: torch.Tensor = total_new_steps

    def copy_from(self, other_normalizer):
        self.normalization_steps.data.copy_(other_normalizer.normalization_steps.data)
        self.running_mean.data.copy_(other_normalizer.running_mean.data)
        self.running_variance.data.copy_(other_normalizer.running_variance.data)

    def print_stats(self):
        print(f"\nNormalizer Stats: \nMean: {self.running_mean}, \nVariance: {self.running_variance}, \nSteps: {self.normalization_steps}")


class WelfordNormalizer(Normalizer):
    def __init__(self, vec_obs_size: int):
        super().__init__(vec_obs_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_state = (inputs - self.running_mean) / torch.sqrt(self.running_variance + 1e-8)
        normalized_state = torch.clamp(normalized_state, -5.0, 5.0)
        # if torch.isnan(normalized_state).__contains__(True):
        #     print("NaN detected")
        #     self.print_stats()
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        """
        Parallel variance update version of Welford's Online Algorithm by Chan et al. 1979.
        By computing the batch statistics, we can update the population statistics.
        The parallel version allows for two sets to be combined so long as they are not the same size and big.
        """
        with torch.no_grad():
            n_b = vector_input.size()[0]
            mean_a = self.running_mean
            var_a = self.running_variance
            n_a = self.normalization_steps
            mean_b = vector_input.mean(0)
            # use biased variance to avoid nan var for single sample updates. Bias correction divides by n-1.
            var_b = vector_input.var(0, unbiased=False)
            delta = mean_b - mean_a
            n_ab = n_a + n_b
            mean_ab = mean_a + delta * n_b / n_ab
            m_2_a = var_a * n_a
            m_2_b = var_b * n_b
            m_2_ab = m_2_a + m_2_b + torch.square(delta) * n_a * n_b / n_ab
            var_ab = m_2_ab / n_ab

            self.running_mean: torch.Tensor = mean_ab
            self.running_variance: torch.Tensor = var_ab
            self.normalization_steps: torch.Tensor = n_ab

        # if n_a // 10000 != n_ab // 10000:
        #     self.print_stats()

    def copy_from(self, other_normalizer):
        super().copy_from(other_normalizer)

    def print_stats(self):
        super().print_stats()



class BatchNormalizer(nn.Module):
    def __init__(self, num_features: int = 1, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.momentum = momentum
        self.EPS = 1e-5
        self.running_momentum = 0.1

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        As going to use this for 1D, input is either (batch,num_features) or (batch, sequence, features).
        Produces same results as pytorch implementation.
        """

        is_training = not (Utils.exporting_to_onnx.is_exporting() or not torch.is_grad_enabled())
        x = inputs
        # print(f"Is Eval? {is_evaluation}")

        if is_training:
            # sum over the zeroth dimension if 2D input, over 0 and 1 if 3D input
            sum_dimensions = [i for i in range(len(x.shape) - 1)]
            x_mean = x.mean(dim=sum_dimensions, keepdim=True)
            x_var = x.var(dim=sum_dimensions, keepdim=True, unbiased=False)
            with torch.no_grad():
                x_var_unbiased = x.var(dim=sum_dimensions, keepdim=True, unbiased=True)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.flatten()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var_unbiased.flatten()
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x - x_mean) / torch.sqrt(x_var + self.EPS)
        y_out = self.weight * x_hat + self.bias

        return y_out

    def get_stats(self):
        print(f"\nBatchNormalizer Running Stats: \nMean: {self.running_mean}, \nVariance: {self.running_var}, "
              f"\nweight: {self.weight.detach()}, \nbias: {self.bias.detach()}")


class BatchRenormalizer(nn.Module):
    def __init__(self, num_features: int = 1, momentum: float = 0.01):
        super().__init__()
        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.momentum = momentum
        self.EPS = 1e-5

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_std", torch.ones(num_features))
        self.register_buffer("running_batch_count", torch.tensor(0, dtype=torch.long))

    @property
    def r_max(self) -> torch.Tensor:
        """
        Anneal r_max from 1 to 3 according Ioffe 2017. From less than 5000 training steps, r_max = 1. Then it linearly anneals from 1 to 3 over steps 5K to 40K.
        """
        return torch.clamp(1.0 + 2.0 * ((self.running_batch_count - 5000.0) / 35000.0), min=1.0, max=3.0)

    @property
    def d_max(self) -> torch.Tensor:
        """
        Anneal d_max from 0 to 5 according Ioffe 2017. From less than 5000 training steps, d_max = 0. Then it linearly anneals from 0 to 5 over steps 5K to 25K.
        """
        return torch.clamp(0 + 5 * (self.running_batch_count - 5000.0) / 20000.0, min=0.0, max=5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        As going to use this for 1D, input is either (batch,num_features) or (batch, sequence, features).
        Follows implementation described by Ioffe 2017. Should prove better for low batch size and non-IID samples than batch normalization.
        """
        is_training = not (Utils.exporting_to_onnx.is_exporting() or not torch.is_grad_enabled())
        x = inputs
        # assert x.shape[2] == self.num_features

        if is_training:
            # sum over the zeroth dimension if 2D input, over 0 and 1 if 3D input
            sum_dimensions = [i for i in range(len(x.shape) - 1)]
            x_mean = x.mean(dim=sum_dimensions, keepdim=True)
            x_var = x.var(dim=sum_dimensions, keepdim=True, unbiased=False)
            x_std = torch.sqrt(x_var + self.EPS)

            r = (x_std.detach() / self.running_std.view_as(x_std)).clamp(1/self.r_max, self.r_max)
            d = ((x_mean.detach()-self.running_mean.view_as(x_mean)) / self.running_std.view_as(x_std)).clamp(-self.d_max, self.d_max)

            with torch.no_grad():
                x_var_unbiased = x.var(dim=sum_dimensions, keepdim=True, unbiased=True)
                x_std_unbiased = torch.sqrt(x_var_unbiased + self.EPS)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.flatten()
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * x_std_unbiased.flatten()

            self.running_batch_count += 1
        else:
            x_mean = self.running_mean
            x_std = self.running_std
            r = 1
            d = 0

        x_hat = ((x - x_mean) / x_std) * r + d
        y_out = self.weight * x_hat + self.bias

        return y_out

    def get_stats(self):
        print(f"\nBatchRenormalizer Running Stats: \nBatch Count: {self.running_batch_count}, "
              f"\nMean: {self.running_mean}, \nStd. Dev: {self.running_std}, "
              f"\nweight: {self.weight.detach()}, \nbias: {self.bias.detach()},"
              f" \nr_max: {self.r_max}, \nd_max: {self.d_max}")

class LayerNormalizer(nn.Module):
    def __init__(self, num_features:int, eps:float=1e-5):
        super().__init__()
        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.EPS = 1e-5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        As going to use this for 1D, input is either (batch,num_features) or (batch, sequence, features).
        Follows implementation described by Ioffe 2017. Should prove better for low batch size and non-IID samples than batch normalization.
        """
        x = inputs
        assert x.shape[-1] == self.num_features

        x_mean = x.mean(dim=[-1], keepdim=True)
        x_var = x.var(dim=[-1], keepdim=True, unbiased=False)
        x_std = torch.sqrt(x_var + self.EPS)

        x_hat = ((x - x_mean) / x_std)
        y_out = self.weight * x_hat + self.bias

        return y_out

    def get_stats(self):
        print(f"\nLayerNormalizer Running Stats: "
              f"\nweight: {self.weight.detach()}, \nbias: {self.bias.detach()}"
              )


class VectorInput(nn.Module):
    def __init__(self, input_size: int, normalize: bool = False, norm_type: NormalizerType = NormalizerType.MLAgents):
        super(VectorInput, self).__init__()
        self.normalizer = None
        # print(norm_type)
        if normalize:
            if norm_type is NormalizerType.MLAgents:
                self.normalizer = Normalizer(input_size)
            elif norm_type is NormalizerType.BatchNorm:
                self.normalizer = BatchNormalizer(input_size)
            elif norm_type is NormalizerType.BatchRenorm:
                self.normalizer = BatchRenormalizer(input_size)
            elif norm_type is NormalizerType.LayerNorm:
                self.normalizer = LayerNormalizer(input_size)
            else:
                self.normalizer = WelfordNormalizer(input_size)
            # print(type(self.normalizer))

    def forward(self, inputs: torch.Tensor):
        if self.normalizer is not None:
            return self.normalizer(inputs)
        return inputs

    def update_normalization(self, inputs: torch.Tensor) -> None:
        if self.normalizer is not None:
            if type(self.normalizer) is Normalizer or WelfordNormalizer:
                self.normalizer.update(inputs)

    def copy_normalization(self, other_vector_input):
        if self.normalizer is not None and other_vector_input.normalizer is not None:
            if type(self.normalizer) is Normalizer or WelfordNormalizer:
                self.normalizer.copy_from(other_vector_input.normalizer)

    def print_stats(self):
        if self.normalizer is not None:
            return self.normalizer.print_stats()


class LinearVectorInput(VectorInput):
    def __init__(self, input_size: int, normalize: bool = False, norm_type: NormalizerType = NormalizerType.MLAgents,
                 hidden_size: int = 64):
        super().__init__(input_size, normalize, norm_type)
        self.linear = LinearEncoder(input_size, 1, hidden_size, Initialization.KaimingHeNormal, 1.0,
                                    Initialization.Zero, 1.0, ActivationType.LeakyReLU)

    def forward(self, inputs: torch.Tensor):
        x = super(LinearVectorInput, self).forward(inputs)
        x = self.linear(x)
        return x


class LinearEncoder(nn.Module):
    """
    Creates a sequence of linear layers for feature approximation.
    """

    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 hidden_size: int,
                 kernel_init: Initialization = Initialization.KaimingHeNormal,
                 kernel_gain: float = 1.0,
                 bias_init: Initialization = Initialization.Zero,
                 bias_gain: float = 1.0,
                 activation_type: ActivationType = ActivationType.Swish
                 ):
        super(LinearEncoder, self).__init__()

        input_layer = linear_layer(input_size,
                                   hidden_size,
                                   kernel_init,
                                   kernel_gain,
                                   bias_init,
                                   bias_gain)
        activation = activation_methods[activation_type]
        hidden_layer = linear_layer(hidden_size,
                                    hidden_size,
                                    kernel_init,
                                    kernel_gain,
                                    bias_init,
                                    bias_gain)
        self.layers = [input_layer]
        self.layers.append(activation)
        for _ in range(num_layers - 1):
            self.layers.append(hidden_layer)
            self.layers.append(activation)
        self.sequence = nn.Sequential(*self.layers)

    def forward(self, obs: torch.Tensor):
        return self.sequence(obs)


class Conv2DEncoder(nn.Module):
    """
    Encoder for 2D convolutional layer.
    Input observation with shape (H,W,C).
    Note input_shape is (H,W,C).
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_size: int = 128,
                 init_channel: int = 16,
                 init_kernel_size: int = 4,
                 init_stride: int = 2,
                 init_padding: int = 0,
                 num_convs: int = 2,
                 use_embedding: bool = False,
                 embedding_dictionary_size: int = 0,
                 num_embedding_dims: int = 0,
                 use_visual_dense_layers: bool = True
                 ):
        super(Conv2DEncoder, self).__init__()

        self.use_visual_dense_layers = use_visual_dense_layers

        num_convs = max(num_convs, 1)
        conv_layers = []

        initial_channel_size = init_channel

        input_channel = input_shape[2]
        input_hw = (input_shape[0], input_shape[1])
        output_hw = (0, 0)
        output_channel = initial_channel_size
        kernel = init_kernel_size
        stride = init_stride
        padding = init_padding
        dilation = 1

        for i in range(num_convs):
            # create conv settings
            conv_settings = [input_channel, output_channel, kernel, stride, padding, dilation]

            # create conv and append to list
            conv = nn.Conv2d(*conv_settings)
            conv_layers.append(conv)
            conv_layers.append(nn.LeakyReLU())

            # what is the output shape of the conv layer?
            output_hw = conv_output_shape(input_hw, kernel, stride, padding, dilation)

            if i < num_convs - 1:
                # check if output fits into kernel for next conv
                check_input_to_kernel(output_hw, kernel)
                input_channel = output_channel
                if i + 1 > (num_convs - 1) // 2:
                    output_channel = 2 * initial_channel_size
                else:
                    output_channel = initial_channel_size
                input_hw = output_hw

        self.final_flat = torch.prod(torch.as_tensor(output_hw)).item() * output_channel

        self.output_size = output_size if self.use_visual_dense_layers else self.final_flat

        self.conv_layers = nn.Sequential(*conv_layers)
        self.dense_layers = nn.Sequential(
            visual_dense_layer(self.final_flat, self.output_size),
            nn.LeakyReLU()) if self.use_visual_dense_layers else None

    def forward(self, visual_obs: torch.Tensor):
        """
        Input shape: (N,H,W,C) or (N,S,H,W,C) if for LSTM.
        ONNX input: (N,C,H,W) or (N,S,C,H,W) for LSTM?
        """

        input_tensor = visual_obs

        # Reshape LSTM shaped input if necessary
        if len(input_tensor.shape) == 5:
            batch_size, seq_length = input_tensor.shape[0], input_tensor.shape[1]
            input_tensor = input_tensor.reshape(
                (-1, input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1]))
        else:
            batch_size, seq_length = 0, 0

        # permute tensor from (N, H, W, C) -> (N, C, H, W) if not exporting to onnx
        if not Utils.exporting_to_onnx.is_exporting():
            input_tensor = input_tensor.permute([0, 3, 1, 2])
        hidden = self.conv_layers(input_tensor)
        # ensure shape is (N, final_flat) for dense layer input
        hidden = hidden.reshape(-1, self.final_flat)
        if self.use_visual_dense_layers:
            hidden = self.dense_layers(hidden)

        # Reshape for LSTM usage (batch, seq, -1)
        if seq_length > 0:
            hidden = hidden.reshape(batch_size, seq_length, -1)

        return hidden


class Conv3DEncoder(nn.Module):
    """
        Encoder for 3D convolutional layer.
        Input observation with shape (D,H,W,C).
        """

    def __init__(self,
                 input_shape: Tuple[int, int, int, int],
                 output_size: int = 128,
                 init_channel: int = 16,
                 init_kernel_size: int = 8,
                 init_stride: int = 4,
                 init_padding: int = 0,
                 num_convs: int = 2,
                 use_embedding: bool = False,
                 embedding_dictionary_size: int = 2,
                 num_embedding_dims: int = 16,
                 use_visual_dense_layers: bool = True
                 ):
        super(Conv3DEncoder, self).__init__()

        self.use_visual_dense_layers = use_visual_dense_layers

        self.use_embedding = use_embedding and num_embedding_dims > 0
        self.num_embedding_dims = num_embedding_dims
        embedding_settings = [embedding_dictionary_size, num_embedding_dims]
        self.embedding_layers = nn.Sequential(
            nn.Embedding(*embedding_settings),
            nn.Tanh()
        ) if use_embedding else None

        num_convs = max(num_convs, 1)
        conv_layers = []
        initial_channel_size = init_channel

        input_channel = input_shape[3] if not self.use_embedding else input_shape[3] * num_embedding_dims
        input_dhw = (input_shape[0], input_shape[1], input_shape[2])
        output_dhw = (0, 0, 0)
        output_channel = initial_channel_size
        kernel = init_kernel_size
        stride = init_stride
        padding = init_padding
        dilation = 1

        for i in range(num_convs):
            # create conv settings
            conv_settings = [input_channel, output_channel, kernel, stride, padding, dilation]

            # create conv and append to list
            conv = nn.Conv3d(*conv_settings)
            conv_layers.append(conv)
            conv_layers.append(nn.LeakyReLU())

            # what is the output shape of the conv layer?
            output_dhw = conv_output_shape(input_dhw, kernel, stride, padding, dilation)

            if i < num_convs - 1:
                # check if output fits into kernel for next conv
                check_input_to_kernel(output_dhw, kernel)
                input_channel = output_channel
                if i + 1 > (num_convs - 1) // 2:
                    output_channel = 2 * initial_channel_size
                else:
                    output_channel = initial_channel_size

                input_dhw = output_dhw

        self.final_flat = torch.prod(torch.as_tensor(output_dhw)).item() * output_channel

        self.output_size = output_size if self.use_visual_dense_layers else self.final_flat

        self.conv_layers = nn.Sequential(*conv_layers)

        self.dense_layers = nn.Sequential(
            visual_dense_layer(self.final_flat, self.output_size),
            nn.LeakyReLU()) if self.use_visual_dense_layers else None

    def forward(self, visual_obs: torch.Tensor):
        """
        Input has shape (N, D, H, W, C) or (N, S, D, H, W, C) for recurrent sampling during training.

        When input from ONNX/Barracuda, input has shape (N, C, D, H, W) or (N,1,C,D,H,W) for recurrent?
        """
        input_tensor = visual_obs

        if len(input_tensor.shape) == 6:
            batch_size, seq_length = input_tensor.shape[0], input_tensor.shape[1]
            input_tensor = input_tensor.reshape(
                (-1, input_tensor.shape[-4], input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1]))
        else:
            batch_size, seq_length = 0, 0

        if self.use_embedding:
            if Utils.exporting_to_onnx.is_exporting():
                # permute tensor from (N, C, D, H, W) -> (N, D, H, W, C) if exporting to onnx for embedding
                input_tensor = input_tensor.permute([0, 2, 3, 4, 1])
            input_tensor = self.embedding_layers(input_tensor.int()).flatten(-2)

            # permute tensor from (N, D, H, W, C) -> (N, C, D, H, W) if not exporting to onnx
            input_tensor = input_tensor.permute([0, 4, 1, 2, 3])

        else:
            if not Utils.exporting_to_onnx.is_exporting():
                # Conv3D expects shape (N,C,D,H,W)
                input_tensor = input_tensor.permute([0, 4, 1, 2, 3])

        hidden = self.conv_layers(input_tensor)

        hidden = hidden.reshape(-1, self.final_flat)
        if self.use_visual_dense_layers:
            hidden = self.dense_layers(hidden)

        if seq_length > 0:
            hidden = hidden.reshape((batch_size, seq_length, -1))

        return hidden


class ObservationEncoder(nn.Module):
    """
    Combine the linear inputs, conv2d, and conv3d inputs and embed them into a singular relu output.
    Uses the same number of hidden layers and neurons as the actor/critic networks.
    Follows the method of Alonso 2020 Figure 3.
    Not Implemented: Linear encoder for absolute positions (paper notes it is worse), LSTM (maybe add later)

    Assume that there can be multiple sources of vector_sensors (1D), 1 source for 2D conv, 1 source for 3D conv.
    """

    def __init__(self, observation_specs: List[ObservationSpec], hidden_size: int,
                 normalize: bool = False, norm_type: NormalizerType = NormalizerType.MLAgents, encode_vector_inputs: bool = False,
                 conv_2d_dict=None, conv_3d_dict=None, positional_embedding_dict=None):
        """
        Creates observation embedding network.

        Pass in spec.observation_specs.
        Linear dict should have in this order: encoding size, output_size
        conv_2d_dict should have in this order: encoding size, output_size, init_channel, init_kernel_size, init_stride
        conv_3d_dict same as conv_2d_dict.
        """
        super(ObservationEncoder, self).__init__()
        self.encoder_settings = {
            'conv2d': conv_2d_dict if conv_2d_dict is not None else default_visual_settings['conv2d'],
            'conv3d': conv_3d_dict if conv_3d_dict is not None else default_visual_settings['conv3d'],
            'positional_embedding': positional_embedding_dict if positional_embedding_dict is not None else default_positional_settings
        }
        self.encoders, self.embedding_sizes = create_input_encoders(observation_specs, hidden_size,
                                                                    normalize, norm_type,
                                                                    encode_vector_inputs,
                                                                    self.encoder_settings)
        self.total_encoding_size = sum(self.embedding_sizes)
        self.device = get_device()
        self.to(self.device)

        self.device = get_device()
        self.normalize = normalize

    def forward(self, observations: List[torch.tensor]) -> torch.Tensor:
        """
        Encode observations using the list of encoders.
        """
        assert len(observations) == len(self.encoders)

        encoded_observations = []

        for i, encoder in enumerate(self.encoders):
            obs = observations[i]
            enc_obs = encoder(obs)
            encoded_observations.append(enc_obs)

        if len(encoded_observations) != 0:
            output_encode = torch.cat(encoded_observations, dim=-1)
        else:
            raise Exception("No observations were passed into the encoder. "
                            "Make sure that there is at least one sensor in the environment.")
        return output_encode

    def update_normalization(self, obs: List[torch.Tensor]) -> None:
        if not self.normalize:
            return
        for i, enc in enumerate(self.encoders):
            if isinstance(enc, VectorInput):
                enc.update_normalization(obs[i])

    def copy_normalization(self, other_obs_encoder):
        if not self.normalize:
            return
        for enc1, enc2 in zip(self.encoders, other_obs_encoder.encoders):
            if isinstance(enc1, VectorInput) and isinstance(enc2, VectorInput):
                enc1.copy_normalization(enc2)


class NetworkBody(nn.Module):
    """
    The main network body for either an actor or critic.
    Network comprised of observation encoder and body encoder, the latter comprised of many linear FC layers.
    """

    def __init__(self,
                 observation_specs: List[ObservationSpec],
                 network_settings: NetworkSettings,
                 encoded_act_size: int = 0,
                 body_encoder_settings: BodyEncoderSettings = BodyEncoderSettings,
                 ):
        super(NetworkBody, self).__init__()

        self.hidden_size = network_settings.hidden_size
        self.normalize = network_settings.normalize
        self.norm_type = network_settings.norm_type
        self.encode_vector_inputs = network_settings.encode_vector_inputs

        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.hidden_size,
            self.normalize,
            self.norm_type,
            self.encode_vector_inputs,
            network_settings.conv2d_settings,
            network_settings.conv3d_settings,
            network_settings.positional_embedding_settings
        )

        total_encoding_size = self.observation_encoder.total_encoding_size + encoded_act_size

        self.body_encoder = LinearEncoder(
            total_encoding_size,
            network_settings.num_layers,
            self.hidden_size,
            **body_encoder_settings
        )

        self.memory_size = 0

    def forward(self,
                states: List[torch.Tensor],
                actions: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For given input observations and optional actions, encode inputs and extract body encoding.
        """
        # print(f"input shape: {inputs[0].shape}")
        encoded_inputs = self.observation_encoder(states)

        if actions is not None:
            encoded_inputs = torch.cat([encoded_inputs, actions], dim=1)

        encoding = self.body_encoder(encoded_inputs)

        return encoding

    # @property
    # def memory_size(self) -> int:
    #     return self.lstm.memory_size if self.use_lstm else 0

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalize:
            return
        self.observation_encoder.update_normalization(obs)

    def copy_normalization(self, network_body):
        if not self.normalize:
            return
        self.observation_encoder.copy_normalization(network_body.observation_encoder)

    # @memory_size.setter
    # def memory_size(self, value):
    #     self._memory_size = value


class LSTMNetworkBody(nn.Module):
    """
    Network body for recurrent actor/critic.
    Network comes after observation encoder and has a structure: FC/ReLU/.../LSTM
    """

    def __init__(self,
                 observation_specs: List[ObservationSpec],
                 network_settings: NetworkSettings,
                 encoded_act_size: int = 0,
                 body_encoder_settings: BodyEncoderSettings = BodyEncoderSettings,
                 ):
        super(LSTMNetworkBody, self).__init__()

        self.hidden_size = network_settings.hidden_size
        self.normalize = network_settings.normalize
        self.norm_type = network_settings.norm_type
        self.encode_vector_inputs = network_settings.encode_vector_inputs

        print(self.norm_type)

        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.hidden_size,
            self.normalize,
            self.norm_type,
            self.encode_vector_inputs,
            network_settings.conv2d_settings,
            network_settings.conv3d_settings,
            network_settings.positional_embedding_settings
        )

        total_encoding_size = self.observation_encoder.total_encoding_size + encoded_act_size

        self.body_encoder = LinearEncoder(
            total_encoding_size,
            network_settings.num_layers,
            self.hidden_size,
            **body_encoder_settings
        )

        assert network_settings.memory_settings is not None
        self.memory_size = network_settings.memory_settings.memory_size
        self.sequence_length = network_settings.memory_settings.sequence_length
        self.lstm = LSTM(self.hidden_size, self.memory_size)

    def forward(self,
                states: List[torch.Tensor],
                actions: Optional[torch.Tensor] = None,
                memories_in: Optional[torch.Tensor] = None,
                sequence_length: int = 1
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For given input observations and optional actions, encode inputs and extract body encoding.
        """

        assert memories_in is not None

        encoded_inputs = self.observation_encoder(states)

        if actions is not None:
            encoded_inputs = torch.cat([encoded_inputs, actions], dim=-1)

        encoding = self.body_encoder(encoded_inputs)

        encoding = encoding.reshape((-1, sequence_length, self.hidden_size))

        encoding, memories_out = self.lstm(encoding, memories_in)

        return encoding, memories_out

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalize:
            return
        self.observation_encoder.update_normalization(obs)

    def copy_normalization(self, network_body):
        if not self.normalize:
            return
        self.observation_encoder.copy_normalization(network_body.observation_encoder)


class LSTM(nn.Module):
    """
    LSTM memory module inspired by MLAgents implementation.
    """

    def __init__(self,
                 input_size: int,
                 memory_size: int,
                 num_layers: int = 1,
                 forget_bias: float = 1.0,
                 kernel_init: Initialization = Initialization.XavierGlorotUniform,
                 bias_init: Initialization = Initialization.Zero
                 ):
        super(LSTM, self).__init__()
        # split memory size in two. First half for initial memory, second half for initial cell state.
        self.hidden_size = memory_size // 2
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.set_lstm_initialization(forget_bias, kernel_init, bias_init)
        self.lstm.flatten_parameters()

    @property
    def memory_size(self) -> int:
        return 2 * self.hidden_size

    def forward(self, input_encoding: torch.Tensor, memories: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
        x: shape (batch_size, sequence_length, hidden_size)
        memories: shape (1, batch_size, memory_size)

        Outputs:
        lstm_out: shape (batch_size, sequence, hidden_size)
        memories_out: shape (1, batch_size, memory_size)
        """

        memories_in = memories

        if Utils.exporting_to_onnx.is_exporting():
            memories_in = torch.transpose(memories_in, 0, 1)

        h0 = memories_in[:, :, :self.hidden_size].contiguous()
        c0 = memories_in[:, :, self.hidden_size:].contiguous()

        self.lstm.flatten_parameters()
        lstm_out, mem_out = self.lstm(input_encoding, (h0, c0))

        memories_out = torch.cat(mem_out, dim=-1)

        if Utils.exporting_to_onnx.is_exporting():
            memories_out = torch.transpose(memories_out, 0, 1)

        return lstm_out, memories_out

    def set_lstm_initialization(self,
                                forget_bias: float = 1.0,
                                kernel_init: Initialization = Initialization.XavierGlorotUniform,
                                bias_init: Initialization = Initialization.Zero):
        for name, param in self.lstm.named_parameters():
            # Each weight and bias is a concatenation of 4 matrices
            if "weight" in name:
                for idx in range(4):
                    block_size = param.shape[0] // 4
                    initialization_methods[kernel_init](
                        param.data[idx * block_size: (idx + 1) * block_size]
                    )
            if "bias" in name:
                for idx in range(4):
                    block_size = param.shape[0] // 4
                    initialization_methods[bias_init](
                        param.data[idx * block_size: (idx + 1) * block_size]
                    )
                    if idx == 1:
                        param.data[idx * block_size: (idx + 1) * block_size].add_(
                            forget_bias
                        )
