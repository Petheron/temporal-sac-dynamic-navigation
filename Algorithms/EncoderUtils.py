from enum import Enum
import torch
from typing import Dict, Tuple
from math import floor
import numpy as np

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Module", "# Parameters"])
    total_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        param = p.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Parameters: {total_params}")


class NormalizerType(Enum):
    MLAgents = "mlagents"
    Welford = "welford"
    BatchNorm = "batch_norm"
    BatchRenorm = "batch_renorm"
    LayerNorm = "layer_norm"


class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5
    Uniform = 6
    Orthogonal = 7
    Constant = 8


initialization_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
    Initialization.Uniform: torch.nn.init.uniform_,
    Initialization.Orthogonal: torch.nn.init.orthogonal_,
    Initialization.Constant: torch.nn.init.constant_
}


class ActivationType(Enum):
    ReLU = 0
    LeakyReLU = 1
    Tanh = 2
    Sigmoid = 3
    Swish = 4


activation_methods = {
    ActivationType.ReLU: torch.nn.ReLU(),
    ActivationType.LeakyReLU: torch.nn.LeakyReLU(),
    ActivationType.Tanh: torch.nn.Tanh(),
    ActivationType.Sigmoid: torch.nn.Sigmoid(),
    ActivationType.Swish: torch.nn.SiLU(),
}


class MemorySettings:
    def __init__(self,
                 sequence_length: int = 64,
                 memory_size: int = 128,
                 burn_in: int = 0,
                 overlap_length: int = 0
                 ):
        self.sequence_length = sequence_length
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.overlap_length = overlap_length

        self.validate_memory_settings()

    def validate_memory_settings(self):
        """
        Validate that memory_size is even and that burn_in is less than sequence_length.
        Memory size is evenly split between hidden state and cell state of the recurrent memory.
        """
        assert self.memory_size > 0 and self.sequence_length > 0 and self.burn_in >= 0 and self.overlap_length >= 0
        assert self.memory_size % 2 == 0


class NetworkSettings:
    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 normalize: bool = False,
                 norm_type: NormalizerType = NormalizerType.MLAgents,
                 encode_vector_inputs: bool = False,
                 conv2d_settings: dict = None,
                 conv3d_settings: dict = None,
                 positional_embedding_settings: dict = None,
                 memory_settings: MemorySettings = None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.normalize = normalize
        self.norm_type = norm_type
        self.encode_vector_inputs = encode_vector_inputs
        self.conv2d_settings = conv2d_settings
        self.conv3d_settings = conv3d_settings
        self.positional_embedding_settings = positional_embedding_settings
        self.memory_settings = memory_settings


class BodyEncoderSettings(Dict):
    def __init__(self,
                 kernel_init: Initialization = Initialization.KaimingHeNormal,
                 kernel_gain: float = 1.0,
                 bias_init: Initialization = Initialization.Zero,
                 bias_gain: float = 1.0,
                 activation_type: ActivationType = ActivationType.Swish):
        super().__init__()
        self.set_values(kernel_init,
                        kernel_gain,
                        bias_init,
                        bias_gain,
                        activation_type)

    def set_values(self,
                   kernel_init: Initialization = Initialization.KaimingHeNormal,
                   kernel_gain: float = 1.0,
                   bias_init: Initialization = Initialization.Zero,
                   bias_gain: float = 1.0,
                   activation_type: ActivationType = ActivationType.Swish
                   ):
        self.setdefault("kernel_init", kernel_init)
        self.setdefault("kernel_gain", kernel_gain)
        self.setdefault("bias_init", bias_init)
        self.setdefault("bias_gain", bias_gain)
        self.setdefault("activation_type", activation_type)


class VisualSettings(Dict):
    def __init__(self,
                 init_channel=16,
                 init_kernel_size=8,
                 init_stride=2,
                 init_padding=1,
                 num_convs=2,
                 use_embedding=False,
                 embedding_dictionary_size=0,
                 num_embedding_dims=0,
                 use_visual_dense_layers=True):
        super(VisualSettings, self).__init__()
        self.settings = {
            'init_channel': init_channel,
            'init_kernel_size': init_kernel_size,
            'init_stride': init_stride,
            'init_padding': init_padding,
            'num_convs': num_convs,
            'use_embedding': use_embedding,
            'embedding_dictionary_size': embedding_dictionary_size,
            'num_embedding_dims': num_embedding_dims,
            'use_visual_dense_layers': use_visual_dense_layers
        }


def PositionalEmbeddingSettings(positional_encoding_embedding_size: int = 30,
                                position_encoding_max_length: int = 50):
    return {
        'positional_encoding_embedding_size': positional_encoding_embedding_size,
        'position_encoding_max_length': position_encoding_max_length
    }


def conv_output_shape(hw: Tuple[int, ...],
                      kernel_size: int = 1, stride: int = 1,
                      pad: int = 0,
                      dilation: int = 1):
    """
    Computes the height and width (and depth if 3D texture) of convolutional layer.
    """
    new_hw = tuple([floor((i + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1) for i in hw])
    checked_hw = tuple([v if v > 1 else 1 for v in new_hw])
    return checked_hw


def check_input_to_kernel(input: Tuple[int, ...], kernel_size: int):
    """
    Raises exception if input for conv layer is smaller than kernel size.
    """
    if (np.array(input) < kernel_size).any():
        raise Exception(
            f'Input shape {input} into conv2d too small for kernel size {kernel_size}. Either decrease kernel or stride size or increase initial input dimensions.')
    else:
        return
