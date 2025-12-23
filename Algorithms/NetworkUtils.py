from typing import List

import numpy as np
import os
import torch
import torch.nn as nn


def get_device():
    """
    Obtain the torch device for tensor calculations.
    Use to obtain cuda device where possible
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def ensure_obs_dimensionality(obs: List[np.ndarray], isLSTM:bool = False):
    observation = obs.copy()
    for i, o in enumerate(observation):
        if len(o.shape) == 1:
            observation[i] = o.reshape((-1, 1))
        if isLSTM:
            observation[i] = np.expand_dims(o, axis=1)
    return observation


def list_to_torch(x, requires_grad=False, device=None):
    if device is not None:
        return [torch.tensor(i, dtype=torch.float32, device=device, requires_grad=requires_grad) for i in x]
    else:
        [torch.tensor(i, dtype=torch.float32, requires_grad=requires_grad, device=get_device()) for i in x]


def list_to_numpy(list:List[torch.Tensor]):
    return [torch_to_numpy(l) for l in list]


def numpy_to_torch(x: np.ndarray, requires_grad=False, device=None) -> torch.tensor:
    """
    Convert a numpy array into a pytorch tensor on the gpu.
    """
    if device is None:
        device = get_device()
    tensor = torch.from_numpy(x).float().requires_grad_(requires_grad)
    tensor = tensor.to(device)
    return tensor


def to_torch(x, requires_grad=False) -> torch.tensor:
    t = torch.tensor(x, dtype=torch.float32, device=get_device(), requires_grad=requires_grad)
    return t


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a pytorch tensor to a numpy array on the cpu.
    """
    x = t.detach().cpu().numpy()
    return x


def save_model_state_dict(net: nn.Module, model_dir: str, model_name: str, losses: dict):
    """
    Save model weights to a pytorch file.
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    net.eval()
    file_name = f"{model_dir}/{model_name}.pt"
    save_dict = {"state_dict": net.state_dict(),
                 "losses": losses}
    torch.save(save_dict, file_name)
    print("Model Saved as " + file_name)


def load_model_state_dict(net: nn.Module, model_dir: str, model_name: str):
    if not os.path.exists(model_dir):
        print(f"No directory at {model_dir}")
        return None
    path = f"{model_dir}/{model_name}.pt"
    loaded_dict = torch.load(path)
    net.load_state_dict(loaded_dict["state_dict"])
    losses = loaded_dict["losses"]
    return losses


def assign_params_to(params, flat_params):
    """
    set tensor flatted parameters to model parameters
    """
    prev_ind = 0
    for param in params:
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def flatten_params(params):
    return torch.cat([param.view(-1) for param in params])
