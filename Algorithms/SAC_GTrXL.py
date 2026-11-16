import torch
import numpy as np
import torch.nn as nn
import os
import copy
from typing import List, Tuple, Union, Optional

import torch.nn.functional as F
from torch.distributions import Normal
from mlagents_envs.base_env import ObservationSpec, ActionSpec

from NetworkUtils import get_device, ensure_obs_dimensionality, list_to_torch, torch_to_numpy
from ModelUtils import LearningRate, ScheduleType, update_learning_rate
from Utils import ExperienceReplayType
from Encoders import TransformerNetworkBody, linear_layer
from EncoderUtils import NetworkSettings, Initialization, ActivationType, BodyEncoderSettings, count_parameters

from SAC_LSTM import compute_bellman_target


class Actor(nn.Module):

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int,
                 network_settings: NetworkSettings, transformer_settings: dict = None,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.m_size = network_settings.memory_settings.memory_size if network_settings.memory_settings is not None else 0
        self.normalized_observations = network_settings.normalize
        self.observation_specs = observation_specs
        self.act_dim = act_dim

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish)

        self.body = TransformerNetworkBody(observation_specs, network_settings, 0,
                                          body_encoder_settings, transformer_settings)

        self.mean = linear_layer(self.m_size // 2, act_dim,
                                 Initialization.Uniform, init_w,
                                 Initialization.Uniform, init_w)
        self.log_std = linear_layer(self.m_size // 2, act_dim,
                                    Initialization.Uniform, init_w,
                                    Initialization.Uniform, init_w)

        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

        print(self)
        count_parameters(self)

    def forward(self, obs: List[torch.Tensor], sequence_length: int = 1):
        encoding = self.body(obs, actions=None, sequence_length=sequence_length)
        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: List[torch.Tensor], sequence_length: int = 1):
        mean, log_std = self.forward(obs, sequence_length)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.rsample()

        log_p = normal.log_prob(action)
        corr = -(2 * (np.log(2.) - action - F.softplus(-2. * action)))
        log_p += corr
        log_p = log_p.sum(dim=-1)

        action = torch.tanh(action)
        return action, log_p

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body.update_normalization(obs)


class ONNX_Actor(nn.Module):
    """Minimal ONNX wrapper for model export.
    Processes single frames only (no context buffer in the ONNX graph)."""
    MODEL_EXPORT_VERSION = 3

    def __init__(self, policy_to_copy: Actor, is_deterministic: bool = False):
        super(ONNX_Actor, self).__init__()
        self.observation_specs = policy_to_copy.observation_specs
        self.LOG_STD_MAX = policy_to_copy.LOG_STD_MAX
        self.LOG_STD_MIN = policy_to_copy.LOG_STD_MIN
        self.body = policy_to_copy.body
        self.mean = policy_to_copy.mean
        self.log_std = policy_to_copy.log_std
        self.is_deterministic = is_deterministic
        self.m_size = policy_to_copy.m_size

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.m_size)]), requires_grad=False)
        self.continuous_action_size = torch.nn.Parameter(
            torch.Tensor([policy_to_copy.act_dim]), requires_grad=False)

        self.to(get_device())

    def copy_policy(self, policy_to_copy: Actor):
        self.body = policy_to_copy.body
        self.mean = policy_to_copy.mean
        self.log_std = policy_to_copy.log_std

    @property
    def memory_size(self) -> int:
        return self.m_size

    def forward(self, inputs: List[torch.Tensor],
                masks: Optional[torch.Tensor] = None,
                memories: Optional[torch.Tensor] = None
                ) -> Tuple[Union[int, torch.Tensor]]:
        export_output = [self.version_number, self.memory_size_vector]

        encoding = self.body(inputs, actions=None, sequence_length=1)
        if encoding.dim() == 3:
            encoding = encoding[:, -1, :]

        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        action = mean + (1.0 - self.is_deterministic) * torch.randn_like(mean) * torch.exp(log_std)
        # action = torch.tanh(action).reshape((-1, 1))
        action = torch.tanh(action)
        if action.dim() == 3:
            action = action.squeeze(1)

        export_output += [action, self.continuous_action_size]

        if self.m_size > 0 and memories is not None:
            export_output += [memories]

        return tuple(export_output)


class Critic(nn.Module):

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int,
                 network_settings: NetworkSettings, transformer_settings: dict = None,
                 init_w=3e-3):
        super(Critic, self).__init__()

        hidden_size = network_settings.memory_settings.memory_size // 2
        self.normalized_observations = network_settings.normalize

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish)

        self.body_1 = TransformerNetworkBody(observation_specs, network_settings, act_dim,
                                             body_encoder_settings, transformer_settings)
        self.q_1_head = linear_layer(hidden_size, 1,
                                     Initialization.Uniform, init_w,
                                     Initialization.Uniform, init_w)

        self.body_2 = TransformerNetworkBody(observation_specs, network_settings, act_dim,
                                             body_encoder_settings, transformer_settings)
        self.q_2_head = linear_layer(hidden_size, 1,
                                     Initialization.Uniform, init_w,
                                     Initialization.Uniform, init_w)

    def forward(self, obs: List[torch.Tensor], act: torch.Tensor,
                sequence_length: int = 1):
        q1 = self.body_1(obs, act, sequence_length)
        q1 = self.q_1_head(q1)

        q2 = self.body_2(obs, act, sequence_length)
        q2 = self.q_2_head(q2)

        return torch.squeeze(q1, -1), torch.squeeze(q2, -1)

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body_1.update_normalization(obs)
        self.body_2.update_normalization(obs)

    def copy_normalization(self, other_network_body: TransformerNetworkBody):
        if not self.normalized_observations:
            return
        self.body_1.copy_normalization(other_network_body)
        self.body_2.copy_normalization(other_network_body)


class SAC_GTrXL(object):
    def __init__(
            self,
            observation_specs: List[ObservationSpec],
            act_dim: int,
            network_settings: NetworkSettings,
            act_lr: float = 3e-4,
            crit_lr: float = 3e-4,
            learning_rate_schedule="constant",
            max_steps: int = 1e5,
            gamma=0.99,
            tau=0.005,
            policy_freq=1,
            alpha=0.2,
            alpha_lr=3e-4,
            experience_replay_type: ExperienceReplayType = ExperienceReplayType.BASIC,
            n_step: int = 1,
            per_settings: dict = None,
            transformer_settings: dict = None
    ):
        self.device = get_device()

        self.transformer_settings = transformer_settings or {}
        self.context_length = network_settings.memory_settings.sequence_length
        self.hidden_size = network_settings.hidden_size

        # Networks
        self.actor = Actor(observation_specs, act_dim, network_settings,
                           transformer_settings).to(self.device)
        self.critic = Critic(observation_specs, act_dim, network_settings,
                             transformer_settings).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=crit_lr)

        # Entropy temperature
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.log(
            (torch.ones(1, device=self.device) * alpha)).requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Learning rate schedules
        self.actor_learning_rate_schedule = LearningRate(
            ScheduleType(learning_rate_schedule), act_lr, 1e-10, max_steps)
        self.critic_learning_rate_schedule = LearningRate(
            ScheduleType(learning_rate_schedule), crit_lr, 1e-10, max_steps)
        self.alpha_learning_rate_schedule = LearningRate(
            ScheduleType(learning_rate_schedule), alpha_lr, 1e-10, max_steps)

        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0
        self.actor_loss_value = 0
        self.entropy_value = 0
        self.log_alpha_loss_values = 0

        self.experience_replay_type = experience_replay_type
        self.n_step = n_step

        self.use_per = per_settings is not None
        self.eta = per_settings["per_eta"] if self.use_per else 0

        self.q = 0
        self.entropy = 0

        # Context buffers for inference (initialised on first action)
        self._context_buffer = None
        self._context_pos = None

        # ONNX
        self.onnx_actor = ONNX_Actor(self.actor)
        self.deterministic_onnx_actor = ONNX_Actor(self.actor, is_deterministic=True)

    # ===== Context Buffer Management (inference only) =====

    def _init_context_buffers(self, num_agents):
        self._context_buffer = torch.zeros(
            num_agents, self.context_length, self.hidden_size,
            device=self.device)
        self._context_pos = torch.zeros(
            num_agents, dtype=torch.long, device=self.device)

    def _update_context(self, encoding):
        """Append new body-encoder output to each agent's context buffer."""
        num_agents = encoding.shape[0]
        for i in range(num_agents):
            pos = self._context_pos[i].item()
            if pos < self.context_length:
                self._context_buffer[i, pos] = encoding[i]
                self._context_pos[i] = pos + 1
            else:
                self._context_buffer[i, :-1] = self._context_buffer[i, 1:].clone()
                self._context_buffer[i, -1] = encoding[i]

    def reset_context(self, done_mask):
        """Clear context buffers for agents whose episodes ended."""
        if self._context_buffer is not None:
            self._context_buffer[done_mask] = 0
            self._context_pos[done_mask] = 0

    # ===== Action Selection =====

    def _get_transformer_output(self, observation):
        """Encode obs, update context buffer, run transformer, return current output."""
        num_agents = observation[0].shape[0]
        if self._context_buffer is None:
            self._init_context_buffers(num_agents)

        encoding = self.actor.body.encode(observation)
        self._update_context(encoding)

        transformer_out = self.actor.body.transformer(self._context_buffer)

        positions = (self._context_pos - 1).clamp(min=0)
        agent_indices = torch.arange(num_agents, device=self.device)
        return transformer_out[agent_indices, positions]

    def select_stochastic_action(self, obs: List[np.ndarray], memories_in: Optional[torch.Tensor] = None):
        observation = obs.copy()
        observation = ensure_obs_dimensionality(observation, isLSTM=True)
        observation = list_to_torch(observation, False, self.device)

        with torch.no_grad():
            current_encoding = self._get_transformer_output(observation)

            mean = self.actor.mean(current_encoding)
            log_std = self.actor.log_std(current_encoding)
            log_std = torch.clamp(log_std, self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)

            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)

            return torch_to_numpy(action), (memories_in.detach().cpu() if memories_in is not None else None)

    def select_deterministic_action(self, obs: List[np.ndarray], memories_in: Optional[torch.Tensor] = None):
        observation = obs.copy()
        observation = ensure_obs_dimensionality(observation, isLSTM=True)
        observation = list_to_torch(observation, False, self.device)

        with torch.no_grad():
            current_encoding = self._get_transformer_output(observation)

            mean = self.actor.mean(current_encoding)
            action = torch.tanh(mean)

            return torch_to_numpy(action), (memories_in.detach().cpu() if memories_in is not None else None)

    # ===== Learning Rate =====

    def update_learning_rate(self, current_step: int):
        update_learning_rate(self.actor_optimizer,
                             self.actor_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.critic_optimizer,
                             self.critic_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.log_alpha_optimizer,
                             self.alpha_learning_rate_schedule.get_value(current_step))

    # ===== Training =====

    def update(self, replay_buffer, batch_size=128):
        self.total_it += 1

        # Sample sequences. Burn-in and stored memories are unused
        burn_in_batch, learn_batch, burn_in_memories, learn_memories, indices, weights = \
            replay_buffer.sample(batch_size)

        critic_loss, q, priorities = self.update_critic(learn_batch, weights)

        new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = \
            self.update_actor(learn_batch, weights)

        self.soft_update()

        if self.use_per:
            replay_buffer.update_priorities(indices, priorities)

        if actor_loss_value is None:
            actor_loss_value = self.actor_loss_value
            entropy_value = self.entropy_value
            log_alpha_loss_value = self.log_alpha_loss_values
        else:
            self.actor_loss_value = actor_loss_value
            self.entropy_value = entropy_value
            self.log_alpha_loss_values = log_alpha_loss_value

        return {"actor": actor_loss_value, "critic": critic_loss,
                "alpha": log_alpha_loss_value}, \
            {"v": 0, "q": q, "log_p": new_log_prob_value,
             "entropy": entropy_value,
             "alpha": self.log_alpha.detach().exp().item()}

    def update_critic(self, minibatch, weights):
        obs, actions, rewards, next_obs, terminals = minibatch
        sequence_length = obs[0].shape[1]

        with torch.no_grad():
            new_next_action, next_log_prob = self.actor.sample(
                next_obs, sequence_length=sequence_length)

            next_target_q_1, next_target_q_2 = self.critic_target(
                next_obs, new_next_action, sequence_length)

            next_target_q = torch.min(next_target_q_1, next_target_q_2) \
                            - self.log_alpha.exp() * next_log_prob

            target_q = compute_bellman_target(
                rewards, terminals, next_target_q, self.gamma, self.n_step)
            target_q = target_q.detach()

        q_1, q_2 = self.critic(obs, actions, sequence_length)
        q = torch.min(q_1.detach(), q_2.detach()).mean().item()

        td_error_1 = target_q - q_1
        td_error_2 = target_q - q_2

        if self.use_per:
            td_error = 0.5 * (td_error_1.clone().detach() + td_error_2.clone().detach())
            td_error = torch.abs(td_error)
            priorities = self.eta * td_error.max(dim=1).values \
                         + (1 - self.eta) * td_error.mean(dim=1)
            priorities = priorities.clamp(min=0.00001, max=10000)
        else:
            priorities = None

        td_loss_1 = (td_error_1.pow(2) * weights).mean()
        td_loss_2 = (td_error_2.pow(2) * weights).mean()
        critic_loss = td_loss_1 + td_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), q, priorities

    def update_actor(self, minibatch, weights):
        obs, _, _, _, _ = minibatch
        sequence_length = obs[0].shape[1]

        new_action, new_log_prob = self.actor.sample(
            obs, sequence_length=sequence_length)

        actor_loss_value, entropy_value, log_alpha_loss_value = None, None, None

        if self.total_it % self.policy_freq == 0:
            q1, q2 = self.critic(obs, new_action, sequence_length)
            min_q = torch.min(q1, q2)
            entropy_bonus = -self.log_alpha.exp() * new_log_prob
            actor_loss = -(min_q + entropy_bonus)
            actor_loss = (actor_loss * weights).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            log_alpha_loss = -(self.log_alpha *
                               (new_log_prob + self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

            actor_loss_value = actor_loss.item()
            entropy_value = entropy_bonus.mean().item()
            log_alpha_loss_value = log_alpha_loss.item()

        return new_log_prob.mean().item(), actor_loss_value, \
            entropy_value, log_alpha_loss_value

    # ===== Utilities =====

    def soft_update(self):
        for parameter, target_parameter in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_parameter.data.copy_(
                self.tau * parameter.data + (1 - self.tau) * target_parameter.data)

    def update_normalization(self, obs: List[torch.Tensor]):
        self.actor.update_normalization(obs)
        self.critic.copy_normalization(self.actor.body)
        self.critic_target.copy_normalization(self.actor.body)

    def save(self, save_dir, suffix: str = ""):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.critic.state_dict(), f"{save_dir}/critic_dict_{suffix}")
        torch.save(self.critic_optimizer.state_dict(), f"{save_dir}/critic_optimizer_dict_{suffix}")
        torch.save(self.actor.state_dict(), f"{save_dir}/actor_dict_{suffix}")
        torch.save(self.actor_optimizer.state_dict(), f"{save_dir}/actor_optimizer_dict_{suffix}")
        torch.save(self.log_alpha, f"{save_dir}/log_alpha_dict_{suffix}")
        torch.save(self.log_alpha_optimizer.state_dict(), f"{save_dir}/log_alpha_optimizer_dict_{suffix}")
        print("\nModel saved")

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(f"{load_dir}/actor_dict"))
        self.actor_optimizer.load_state_dict(torch.load(f"{load_dir}/actor_optimizer_dict"))
        self.critic.load_state_dict(torch.load(f"{load_dir}/critic_dict"))
        self.critic_optimizer.load_state_dict(torch.load(f"{load_dir}/critic_optimizer_dict"))
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = copy.deepcopy(torch.load(f"{load_dir}/log_alpha_dict"))
        self.log_alpha_optimizer.load_state_dict(torch.load(f"{load_dir}/log_alpha_optimizer_dict"))
        print(f"\nModel loaded from {load_dir}")