from enum import Enum
import numpy as np
import torch.optim


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


def update_learning_rate(optim: torch.optim.Optimizer, lr: float) -> None:
    """
    Apply a learning rate to a torch optimizer.

    :param optim: the optimizer
    :param lr: the learning rate to apply
    """

    for param_group in optim.param_groups:
        param_group["lr"] = lr


def get_decayed_value(initial_value: float,
                      min_value: float,
                      max_step: int,
                      current_step: int,
                      power: float = 1.0
                      ) -> float:
    """
    Get the decayed LR value based on polynomial decay.

    v = v_min + (v_initial-v_min)*(1-step/max_step)^p

    :param initial_value: the initial LR value
    :param min_value: the goal decay value of the LR
    :param max_step: the final step count where LR value == min_value
    :param current_step: the current step of the model
    :param power: Power of the polynomial decay. The default value of 1.0 is a linear decay.
    """
    if initial_value < min_value:
        return initial_value
    step_fraction = float(min(current_step, max_step)) / max_step
    decayed_value = (initial_value - min_value) * (1 - step_fraction) ** power + min_value
    return decayed_value


class LearningRate:
    def __init__(self,
                 schedule_type: ScheduleType,
                 initial_value: float,
                 min_value: float,
                 max_step: int
                 ):
        """
        Class that encapsulates a learning rate and allows for the calculation of decay LRs over training lifetime
        :param schedule_type: Type of learning rate
        :param initial_value: the initial LR value
        :param min_value: the goal decay value of the LR
        :param max_step: the final step count where LR value == min_value
        """
        self.schedule_type = schedule_type
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_step = max_step

    def get_value(self, current_step: int) -> float:
        if self.schedule_type == ScheduleType.CONSTANT:
            return self.initial_value
        elif self.schedule_type == ScheduleType.LINEAR:
            return get_decayed_value(self.initial_value, self.min_value, self.max_step, current_step)


class RunningMeanStd:
    """
    Used for normalizing value function for PPO. Keeps a running mean/variance of the input values.
    """

    def __init__(self, input_shape, device, eps: float = 1e-6):
        self.mean = torch.zeros(input_shape, dtype=torch.float32, requires_grad=False, device=device)
        self.var = torch.ones(input_shape, dtype=torch.float32, requires_grad=False, device=device)
        self.count = eps
        self.EPS = eps * torch.ones(1, dtype=torch.float32, requires_grad=False, device=device)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_running_stats_from_moments(self.mean, self.var, self.count,
                                                                                  batch_mean, batch_var, batch_count)

    def _update_running_stats_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = count + batch_count

        new_mean = mean + delta * batch_count / total_count
        moment_a = var * count
        moment_b = batch_var * batch_count
        new_var = (moment_a + moment_b + torch.square(delta) * count * batch_count / total_count) / total_count

        new_count = total_count

        return new_mean, new_var, new_count

    def get_normalize_input(self, x):
        std = torch.sqrt(self.var)
        return (x - self.mean) / torch.max(std, self.EPS)

    def get_denormalized_outputs(self, x):
        std = torch.sqrt(self.var)
        return self.mean + x * torch.max(std, self.EPS)


def PAL(x, min_priority=1.0, alpha=0.4, averaged=True):
    """
    Prioritized_Approximate_Loss From Fujimoto 2020. https://github.com/sfujim/LAP-PAL/blob/master/continuous/PAL_TD3.py
    Don't understand why min_prio**alpha in true statement, but as always 1.0 by default, irrelevant.
    """
    pal = torch.where(
        x.abs() < min_priority,
        0.5 * (min_priority ** alpha) * x.pow(2),
        min_priority * x.abs().pow(1 + alpha) / (1 + alpha)
    )
    return pal.mean() if averaged else pal


def Huber_Loss(x, min_priority=1.0, averaged=True):
    """
    From Fujimoto 2020. https://github.com/sfujim/LAP-PAL/blob/master/continuous/PAL_TD3.py
    """
    huber = torch.where(
        x < min_priority,
        0.5 * x.pow(2),
        min_priority * x
    )
    return huber.mean() if averaged else huber

