import torch
import numpy as np
from NetworkUtils import *


class ActionSpace:
    def __init__(self, min: list, max: list):
        self.min = to_torch(min)
        self.max = to_torch(max)

    def convert_from_given_range(self, old_tensor, old_min: list, old_max: list):
        old_min, old_max = to_torch(old_min), to_torch(old_max)
        old_range = old_max - old_min
        new_range = self.max - self.min
        assert old_range.shape == new_range.shape == old_tensor[0].shape
        new_tensor = (old_tensor - old_min) * (new_range / old_range) + self.min
        assert new_tensor.shape == old_tensor.shape
        return new_tensor

    def get_min_max_np(self):
        return torch_to_numpy(self.min), torch_to_numpy(self.max)


def test_action_space():
    x = to_torch([[-1, -1], [0, 0], [1, 1]])
    x_min = [-1, -1]
    x_max = [1, 1]

    a = ActionSpace([-1, 0], [1, 1])
    y = a.convert_from_given_range(x, x_min, x_max)

    print(f"old tensor: {x}")
    print(f"new tensor: {y}")

    print(f"numpy output: {a.get_min_max_np()}")
