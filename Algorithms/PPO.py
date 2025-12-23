import copy
import math
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from mlagents_envs.base_env import ObservationSpec, ActionSpec

from NetworkUtils import *
from ModelUtils import LearningRate, ScheduleType, update_learning_rate, RunningMeanStd
from Encoders import NetworkBody, linear_layer
from EncoderUtils import NetworkSettings, Initialization, ActivationType, BodyEncoderSettings
from DataStructures import RolloutBuffer

from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    """
    Note: act_dim and obs_dim are the number of actions/observations. Not the shape of what Unity gives back.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int, network_settings: NetworkSettings,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        hidden_size = network_settings.hidden_size

        self.m_size = network_settings.memory_settings.memory_size if network_settings.memory_settings is not None else 0
        if self.m_size > 0:
            hidden_size = self.m_size // 2

        self.normalized_observations = network_settings.normalize
        self.observation_specs = observation_specs
        self.act_dim = act_dim

        body_encoder_settings = BodyEncoderSettings(Initialization.Orthogonal, np.sqrt(2),
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Tanh
                                                    )  # from Andrychowicz 2020.

        # body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
        #                                             Initialization.Constant, 0.0,
        #                                             ActivationType.Swish
        #                                             )

        self.body = NetworkBody(observation_specs, network_settings, 0, body_encoder_settings)

        self.mean = linear_layer(hidden_size, act_dim, Initialization.Orthogonal, 0.01, Initialization.Constant, 0.0)
        # self.mean = linear_layer(hidden_size, act_dim, Initialization.KaimingHeNormal, 0.2, Initialization.Constant, 0.0)
        # self.log_std = linear_layer(hidden_size, act_dim, Initialization.Uniform, init_w, Initialization.Uniform,
        #                             init_w)
        initial_std = 1.0  # from Andrychowicz 2020, C61.
        self.log_std = nn.Parameter(
            (math.log(initial_std) * torch.ones(1, act_dim, dtype=torch.float32)).requires_grad_(True))
        # self.log_std = nn.Parameter(torch.zeros(1, act_dim, dtype=torch.float32, requires_grad=True))

        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

    def distribution(self,
                     obs: List[torch.Tensor],
                     memories_in: Optional[torch.Tensor] = None,
                     sequence_lengths: Optional[torch.Tensor] = None
                     ):
        mean, log_std, memories_out = self(obs, memories_in, sequence_lengths)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, memories_out

    def forward(self,
                obs: List[torch.Tensor],
                memories_in: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None
                ):
        """
        Calculates mean and log_std for given obs.
        """
        encoding, memories_out = self.body(obs, actions=None, memories_in=memories_in, sequence_lengths=sequence_lengths)

        mean = self.mean(encoding)
        # mean = F.tanh(mean)
        log_std = 0 * mean + self.log_std
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN)
        # log_std = self.log_std(encoding)
        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std, memories_out

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body.update_normalization(obs)

    def select_stochastic_action(self,
                                 obs: List[torch.Tensor],
                                 memories_in: Optional[torch.Tensor] = None,
                                 sequence_lengths: Optional[torch.Tensor] = None
                                 ):
        """
        Select action for environment step.
        """
        with torch.no_grad():
            dist, memories_out = self.distribution(obs, memories_in, sequence_lengths)
            action = dist.sample()
            log_p = dist.log_prob(action).sum(dim=1)
        return action, log_p, memories_out

    def select_deterministic_action(self,
                                    obs: List[torch.Tensor],
                                    memories_in: Optional[torch.Tensor] = None,
                                    sequence_lengths: Optional[torch.Tensor] = None
                                    ):
        """
        Select deterministic action for eval environment step.
        """
        with torch.no_grad():
            dist, memories_out = self.distribution(obs, memories_in, sequence_lengths)
            action = dist.mean
        return action


class ONNX_Actor(nn.Module):
    MODEL_EXPORT_VERSION = 3

    def __init__(self, policy_to_copy: Actor, is_deterministic:bool = False):
        super(ONNX_Actor, self).__init__()

        policy = policy_to_copy

        self.observation_specs = policy.observation_specs

        self.LOG_STD_MAX = policy.LOG_STD_MAX
        self.LOG_STD_MIN = policy.LOG_STD_MIN

        self.body = policy.body
        self.mean = policy.mean
        self.log_std = policy.log_std
        self.is_deterministic = is_deterministic

        self.m_size = self.body.memory_size

        self.version_number = torch.nn.Parameter(torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)
        self.memory_size_vector = torch.nn.Parameter(torch.Tensor([int(self.m_size)]), requires_grad=False)
        self.continuous_action_size = torch.nn.Parameter(torch.Tensor([policy.act_dim]), requires_grad=False)

        self.to(get_device())

    def copy_policy(self, policy_to_copy: Actor):
        policy = policy_to_copy
        self.body = policy.body
        self.mean = policy.mean
        self.log_std = policy.log_std

    def forward(self,
                obs: List[torch.Tensor],
                memories_in: Optional[torch.Tensor] = None
                ) -> Tuple[Union[int, torch.Tensor], ...]:
        export_output = [self.version_number, self.memory_size_vector]
        encoding, memories_out = self.body(obs, actions=None, memories_in=memories_in, sequence_lengths=None)
        mean = self.mean(encoding)
        log_std = 0 * mean + self.log_std
        log_std = torch.clamp(log_std, self.LOG_STD_MIN)
        action = mean + (1.0 - self.is_deterministic)*torch.randn_like(mean) * torch.exp(log_std)
        action = torch.clamp(action, -1, 1)
        export_output += [action, self.continuous_action_size]

        if self.m_size > 0:
            export_output += [memories_out]

        return tuple(export_output)


class Critic(nn.Module):
    """
    PPO critic that can take lists of observations and both vector/visual inputs.

    Forward pass returns 1 Vfunctions.

    Act_dim is the number of continuous actions.
    """

    def __init__(self, observation_specs: List[ObservationSpec], network_settings: NetworkSettings,
                 init_w=3e-3):
        super(Critic, self).__init__()

        hidden_size = network_settings.hidden_size
        if network_settings.memory_settings is not None:
            hidden_size = network_settings.memory_settings.memory_size//2

        self.normalized_observations = network_settings.normalize
        self.normalize_values = False

        # V architecture
        body_encoder_settings = BodyEncoderSettings(Initialization.Orthogonal, np.sqrt(2),
                                                    Initialization.Constant, 0,
                                                    ActivationType.Tanh
                                                    )
        # body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
        #                                             Initialization.Constant, 0.0,
        #                                             ActivationType.Swish
        #                                             )
        self.body = NetworkBody(observation_specs, network_settings, 0, body_encoder_settings)
        self.v_head = linear_layer(hidden_size, 1, Initialization.Orthogonal, 1.0, Initialization.Constant, 0.0)
        # self.v_head = linear_layer(hidden_size, 1, Initialization.KaimingHeNormal, 1.0, Initialization.Constant, 0.0)
        self.device = get_device()

        self.value_normalizer = RunningMeanStd((1,), device=self.device) if self.normalize_values else None

    def forward(self,
                obs: List[torch.Tensor],
                memories_in: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None
                ):
        encoding, _ = self.body(obs, None, memories_in=memories_in, sequence_lengths=sequence_lengths)
        head = self.get_output_values(self.v_head(encoding))
        return torch.squeeze(head, -1)

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body.update_normalization(obs)

    def copy_normalization(self, other_network_body: NetworkBody):
        if not self.normalized_observations:
            return
        self.body.copy_normalization(other_network_body)

    def get_output_values(self, head_output: torch.Tensor) -> torch.Tensor:
        if not self.normalize_values:
            return head_output
        # normalized_values = self.value_normalizer.get_normalize_input(head_output)
        denormalized_outputs = self.value_normalizer.get_denormalized_outputs(head_output)
        return denormalized_outputs

    def update_value_normalization(self, values: torch.Tensor) -> None:
        if not self.normalize_values:
            return
        self.value_normalizer.update(values)


class Agent(nn.Module):
    def __init__(self, observation_specs: List[ObservationSpec], action_dim: int, network_settings: NetworkSettings):
        super(Agent, self).__init__()
        self.actor = Actor(observation_specs, action_dim, network_settings)
        self.critic = Critic(observation_specs, network_settings)

    def get_distribution(self,
                         obs: List[torch.Tensor],
                         memories_in: Optional[torch.Tensor] = None,
                         sequence_lengths: Optional[torch.Tensor] = None
                         ) -> torch.distributions.distribution:
        mean, log_std, memories_out = self.actor(obs, memories_in, sequence_lengths)
        std = log_std.exp()
        return Normal(mean, std), memories_out

    def get_action(self,
                   obs: List[torch.Tensor],
                   memories_in: Optional[torch.Tensor] = None,
                   sequence_lengths: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, memories_out = self.get_distribution(obs, memories_in, sequence_lengths)
        return dist.sample(), memories_out

    def get_log_probs_and_entropy(self,
                                  obs: List[torch.Tensor],
                                  action: torch.Tensor,
                                  memories_in: Optional[torch.Tensor] = None,
                                  sequence_lengths: Optional[torch.Tensor] = None,
                                  dist: torch.distributions.distribution = None):
        if dist is None:
            dist, _ = self.get_distribution(obs, memories_in, sequence_lengths)
        # log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean(dim=1, keepdim=True).squeeze(-1)
        return log_prob, entropy

    def get_value(self,
                  obs: List[torch.Tensor],
                  memories_in: Optional[torch.Tensor] = None,
                  sequence_lengths: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        return self.critic(obs, memories_in, sequence_lengths)

    def get_action_and_value(self,
                             obs: List[torch.Tensor],
                             action: torch.Tensor = None,
                             memories_in: Optional[torch.Tensor] = None,
                             sequence_lengths: Optional[torch.Tensor] = None
                             ):
        dist, memories_out = self.get_distribution(obs, memories_in, sequence_lengths)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
        value = self.get_value(obs)
        log_prob, entropy = self.get_log_probs_and_entropy(obs, action, memories_in, sequence_lengths, dist)
        del dist
        return action, log_prob, entropy, value, memories_out

    def get_deterministic_action(self,
                                 obs: List[torch.Tensor],
                                 memories_in: Optional[torch.Tensor] = None
                                 ):
        with torch.no_grad():
            mean, _, memories_out = self.actor(obs, memories_in)
            mean = torch.clamp(mean, -1, 1)
        return mean, memories_out

    def update_normalization(self, obs: List[torch.Tensor]) -> None:
        self.actor.update_normalization(obs)
        self.critic.copy_normalization(self.actor.body)


class PPO(object):
    def __init__(
            self,
            observation_specs: List[ObservationSpec],
            act_dim: int,
            network_settings: NetworkSettings,
            act_lr: float = 3e-4,
            crit_lr: float = 3e-4,
            learning_rate_schedule="constant",
            max_steps: int = 1e5,
            epsilon=0.2,
            num_epoch=5,
            beta=1e-2,
            v_loss_clipping=True,
            normalize_advantages=True,
            sequence_length: int = 1
    ):
        self.device = get_device()

        self.agent = Agent(observation_specs, act_dim, network_settings).to(self.device)
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=act_lr, eps=1e-8)

        self.learning_rate_schedule = LearningRate(ScheduleType(learning_rate_schedule), act_lr, 1e-10, max_steps)
        self.beta_schedule = LearningRate(ScheduleType(learning_rate_schedule), beta, 1e-5, max_steps)
        self.epsilon_schedule = LearningRate(ScheduleType(learning_rate_schedule), epsilon, 0.1, max_steps)

        # cache variables
        self.epsilon = epsilon
        self.num_epoch = num_epoch
        self.beta = beta
        self.v_loss_clipping = v_loss_clipping
        self.normalize_advantages = normalize_advantages
        self.sequence_length = sequence_length if network_settings.memory_settings is not None else 1

        # Initialize counters & loss storage
        self.total_it = 0
        self.critic_loss_values = 0
        self.actor_loss_values = 0
        self.v = 0

        # Initialize onnx saver
        # self.onnx_actor = ONNX_Actor(self.actor)
        self.onnx_actor = ONNX_Actor(self.agent.actor)
        self.deterministic_onnx_actor = ONNX_Actor(self.agent.actor, is_deterministic=True)

    def get_action(self,
                   obs: List[np.ndarray],
                   memories_in: Optional[np.array] = None
                   ):
        obs = ensure_obs_dimensionality(obs)
        obs = list_to_torch(obs, requires_grad=False, device=self.device)

        if memories_in is not None:
            memories_in = numpy_to_torch(memories_in, False, self.device)

        actions, memories_out = self.agent.get_action(obs, memories_in)

        actions = torch_to_numpy(actions)

        if memories_out is not None:
            memories_out = torch_to_numpy(memories_out)

        return actions, memories_out

    def get_deterministic_action(self,
                                 obs: List[np.ndarray],
                                 memories_in: Optional[np.array] = None
                                 ):
        obs = ensure_obs_dimensionality(obs)
        obs = list_to_torch(obs, requires_grad=False, device=self.device)

        if memories_in is not None:
            memories_in = numpy_to_torch(memories_in, False, self.device)

        actions, memories_out = self.agent.get_deterministic_action(obs, memories_in)

        if memories_out is not None:
            memories_out = torch_to_numpy(memories_out)

        return torch_to_numpy(actions), memories_out

    # def get_action_and_value(self,
    #                          obs: List[np.ndarray],
    #                          ):
    #     obs = ensure_obs_dimensionality(obs)
    #     obs = list_to_torch(obs, requires_grad=False, device=self.device)
    #     return self.agent.get_action_and_value(obs)

    def get_log_prob_and_entropy(self, obs: List[np.ndarray], action: np.ndarray):
        obs = ensure_obs_dimensionality(obs)
        obs = list_to_torch(obs, requires_grad=False, device=self.device)
        action = numpy_to_torch(action, False, self.device)
        log_prob, entropy = self.agent.get_log_probs_and_entropy(obs, action, None)
        return log_prob, entropy.squeeze(-1)

    def get_value(self,
                  obs: List[np.ndarray],
                  memories_in: Optional[np.array] = None
                  ):
        obs = ensure_obs_dimensionality(obs)
        obs = list_to_torch(obs, requires_grad=False, device=self.device)

        if memories_in is not None:
            memories_in = numpy_to_torch(memories_in, False, self.device)


        return self.agent.get_value(obs,memories_in)

    def update_learning_rate(self, current_step: int = 0):
        new_lr = self.learning_rate_schedule.get_value(current_step)
        update_learning_rate(self.agent_optimizer, new_lr)
        self.beta = self.beta_schedule.get_value(current_step)
        self.epsilon = self.epsilon_schedule.get_value(current_step)

    def update(self, rollout_buffer: RolloutBuffer, batch_size: int = 32):
        b_obs, b_actions, b_log_probs, b_values, b_advantages, b_returns, b_memories_in, b_memories_out, b_sequence_lengths = rollout_buffer.get(self.normalize_advantages)
        buffer_size = rollout_buffer.buffer_size

        b_indices = np.arange(buffer_size)
        clip_fracs = []

        avg_v = b_values.mean().item()

        # avg_log_p, avg_entropy = self.agent.get_log_probs_and_entropy(list_to_torch(b_obs, device=self.device),
        #                                                               numpy_to_torch(b_actions, device=self.device),
        #                                                               None)
        avg_log_p = b_log_probs.mean().item()

        policy_loss = 0
        critic_loss = 0
        entropy_loss = 0

        num_minibatches = buffer_size // batch_size
        num_updates = self.num_epoch * buffer_size // batch_size

        if buffer_size % batch_size != 0:
            print(f"Batch size does not divide evenly into buffer. Currently {buffer_size % batch_size} over.")

        for _ in range(self.num_epoch):
            np.random.shuffle(b_indices)
            norm_count = 0
            for start_idx in range(0, buffer_size, batch_size):
                end_idx = start_idx + batch_size
                mb_indices = b_indices[start_idx:end_idx]
                mb_obs, mb_actions = [o[mb_indices] for o in b_obs], b_actions[mb_indices]
                mb_log_probs, mb_values, mb_advantages, mb_returns = b_log_probs[mb_indices], b_values[mb_indices], \
                                                                     b_advantages[mb_indices], b_returns[mb_indices]

                mb_obs = list_to_torch(mb_obs, device=self.device)
                mb_actions = numpy_to_torch(mb_actions, device=self.device)
                mb_log_probs = numpy_to_torch(mb_log_probs, device=self.device)
                mb_values = numpy_to_torch(mb_values, device=self.device)
                mb_advantages = numpy_to_torch(mb_advantages, device=self.device)
                mb_returns = numpy_to_torch(mb_returns, device=self.device)

                if b_memories_in is not None:
                    mb_memories_in = b_memories_in[mb_indices]
                    mb_memories_out = b_memories_out[mb_indices]
                    mb_memories_in = numpy_to_torch(mb_memories_in, device=self.device)
                    mb_memories_out = numpy_to_torch(mb_memories_out, device=self.device)
                    mb_sequence_lengths = None
                else:
                    mb_memories_in = None
                    mb_memories_out = None
                    mb_sequence_lengths = None

                _, mb_new_log_probs, mb_entropy, mb_new_values, _ = self.agent.get_action_and_value(mb_obs, mb_actions, mb_memories_in, mb_sequence_lengths)

                log_ratio = mb_new_log_probs - mb_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    # how many actions are outside clipping/trust region?
                    clip_fracs += [((ratio - 1.0).abs() > self.epsilon).float().mean().item()]

                # can normalize advantage here over minibatch, currently normalize over buffer
                # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                mb_advantages = mb_advantages.unsqueeze(-1)
                surrogate_loss_1 = mb_advantages * ratio
                surrogate_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                pg_loss_min = torch.min(surrogate_loss_1, surrogate_loss_2)
                pg_loss = -pg_loss_min.mean()

                # Value Loss (using clipped v_loss as in MLAgents)
                v_loss_unclipped = (mb_new_values - mb_returns) ** 2
                if self.v_loss_clipping:
                    v_clipped = mb_values + torch.clamp(mb_new_values - mb_values, -self.epsilon, self.epsilon)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped)).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                # Entropy Loss
                ent_loss = mb_entropy.mean()

                # Total loss and Backprop
                loss = pg_loss + v_loss * 0.5 - self.beta * ent_loss

                policy_loss += pg_loss.item()
                critic_loss += v_loss.item()
                entropy_loss += ent_loss.item()

                self.agent_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                self.agent_optimizer.step()

                if norm_count < num_minibatches:
                    num_minibatches += 1
                    self.agent.update_normalization(mb_obs)
                    self.agent.critic.update_value_normalization(mb_values)

        policy_loss /= num_updates
        critic_loss /= num_updates
        entropy_loss /= num_updates

        # self.agent.update_normalization(b_obs)
        # self.agent.critic.update_value_normalization(b_values)

        del loss, mb_new_log_probs, mb_entropy, mb_new_values
        del mb_obs, mb_actions, mb_advantages, mb_returns, mb_values, mb_log_probs

        return {"actor": policy_loss, "critic": critic_loss}, {"q": 0.0, "v": avg_v,
                                                               "log_p": avg_log_p,
                                                               "entropy": entropy_loss}

    def save(self, save_dir, suffix: str = ""):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        torch.save(self.agent.state_dict(), f"{save_dir}/agent_dict_{suffix}")
        torch.save(self.agent_optimizer.state_dict(), f"{save_dir}/agent_optimizer_dict_{suffix}")
        print("Model saved")

    def load(self, load_dir):

        self.agent.load_state_dict(torch.load(f"{load_dir}/agent_dict"))
        self.agent_optimizer.load_state_dict(torch.load(f"{load_dir}/agent_optimizer_dict"))
