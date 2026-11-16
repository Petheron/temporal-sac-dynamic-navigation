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
from Encoders import LSTMNetworkBody, linear_layer
from EncoderUtils import NetworkSettings, Initialization, ActivationType, BodyEncoderSettings, count_parameters

from DataStructures import LSTMReplayBuffer, PrioritizedLSTMReplayBuffer


class Actor(nn.Module):
    """
    Note: act_dim and obs_dim are the number of actions/observations. Not the shape of what Unity gives back.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int, network_settings: NetworkSettings,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.m_size = network_settings.memory_settings.memory_size if network_settings.memory_settings is not None else 0
        self.normalized_observations = network_settings.normalize
        self.observation_specs = observation_specs
        self.act_dim = act_dim

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish
                                                    )

        self.body = LSTMNetworkBody(observation_specs, network_settings, 0, body_encoder_settings)

        self.mean = linear_layer(self.m_size // 2, act_dim, Initialization.Uniform, init_w, Initialization.Uniform,
                                 init_w)
        self.log_std = linear_layer(self.m_size // 2, act_dim, Initialization.Uniform, init_w, Initialization.Uniform,
                                    init_w)

        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

        print(self)
        count_parameters(self)

    def forward(self,
                obs: List[torch.Tensor],
                memories_in: torch.Tensor,
                sequence_length: int = 1
                ):
        """
        Returns the actions and action's log_prob for a batch of agents for a given list of observation tensors.

        Actions in shape (n,a).

        Log_p in shape (n,).
        """
        encoding, memories_out = self.body(obs, actions=None, memories_in=memories_in, sequence_length=sequence_length)

        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std, memories_out

    def sample(self,
               obs: List[torch.Tensor],
               memories_in: torch.Tensor,
               sequence_length: int = 1
               ):
        """
        For a given observation list, return the action and log_prob.

        This method uses the reparameterization trick and therefore can be used for backprop.

        The action returned is tanh-squeezed and is in the range  [-1,1].

        Input Shape: list of (batch, seq, obs_shape)
        """

        mean, log_std, memories_out = self.forward(obs, memories_in, sequence_length)
        std = log_std.exp()

        # Construct action distribution as stochastic policy
        normal = Normal(mean, std)

        # rsample samples the normal using the reparameterization trick.
        # normal(mu,std).rsample() = mu + std*normal(0,1).sample()
        # This keeps the backprop alive unlike normal.sample()
        action = normal.rsample()

        # Need to compute log_prob for selected action for entropy calculations.
        # The action value used should be the pre-squeezed action (pre-tanh)

        # Must be summed as Normal dist returns (b,a) with log_p for each act dim \
        # We want (b,) shape as log_p should be summed over all actions.
        log_p = normal.log_prob(action)
        # Correction term: same as ( -log( 1- tanh(action)^2 + eps) in paper
        # This correction term is properly migrate log_p from gaussian action density to tanh action density
        corr = -(2 * (np.log(2.) - action - F.softplus(-2. * action)))
        log_p += corr
        log_p = log_p.sum(dim=-1)  # Need to sum over actions to ensure log_p is unit distribution

        # Squeeze action with tanh function to bound between -1 and 1
        action = torch.tanh(action)

        return action, log_p, memories_out

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body.update_normalization(obs)


class ONNX_Actor(nn.Module):
    MODEL_EXPORT_VERSION = 3

    def __init__(self, policy_to_copy: Actor, is_deterministic: bool = False):
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

    @property
    def memory_size(self) -> int:
        return self.m_size

    def forward(self,
                inputs: List[torch.Tensor],
                masks: Optional[torch.Tensor] = None,
                memories: Optional[torch.Tensor] = None
                ) -> Tuple[Union[int, torch.Tensor]]:
        export_output = [self.version_number, self.memory_size_vector]

        encoding, memories_out = self.body(inputs, actions=None, memories_in=memories, sequence_length=1)

        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        action = mean + (1.0 - self.is_deterministic) * torch.randn_like(mean) * torch.exp(log_std)
        # action = torch.tanh(action).reshape((-1, 1))
        action = torch.tanh(action)
        if action.dim() == 3:
            action = action.squeeze(1)

        export_output += [action, self.continuous_action_size]

        if self.m_size > 0:
            # similarly, memories_out also must be a single tensor
            export_output += [memories_out]

        return tuple(export_output)


class Critic(nn.Module):
    """
    SAC critic that can take lists of observations and both vector/visual inputs.

    Forward pass returns 2 q functions.

    Comprised of 2 network bodies for Q1 and Q2. Each network body has a head layer for q-value extraction.

    Act_dim is the number of continuous actions.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int, network_settings: NetworkSettings,
                 init_w=3e-3):
        super(Critic, self).__init__()

        hidden_size = network_settings.hidden_size
        if network_settings.memory_settings is not None:
            hidden_size = network_settings.memory_settings.memory_size // 2

        self.normalized_observations = network_settings.normalize

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish
                                                    )

        # Q1 architecture
        self.body_1 = LSTMNetworkBody(observation_specs, network_settings, act_dim, body_encoder_settings)
        self.q_1_head = linear_layer(hidden_size, 1, Initialization.Uniform, init_w, Initialization.Uniform, init_w)

        # Q2 architecture
        self.body_2 = LSTMNetworkBody(observation_specs, network_settings, act_dim, body_encoder_settings)
        self.q_2_head = linear_layer(hidden_size, 1, Initialization.Uniform, init_w, Initialization.Uniform, init_w)

    def forward(self,
                obs: List[torch.Tensor],
                act: torch.Tensor,
                memories_in: Optional[torch.Tensor],
                memories_in_2: Optional[torch.Tensor] = None,
                sequence_length: int = 1
                ):
        """
        Input shape: (batch, seq, obs_shape)

        """

        if memories_in_2 is None:
            memories_in_2 = memories_in

        q1, memories_out_1 = self.body_1(obs, act, memories_in, sequence_length)
        q1 = self.q_1_head(q1)

        q2, memories_out_2 = self.body_2(obs, act, memories_in_2, sequence_length)
        q2 = self.q_2_head(q2)

        return torch.squeeze(q1, -1), torch.squeeze(q2, -1), memories_out_1, memories_out_2

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body_1.update_normalization(obs)
        self.body_2.update_normalization(obs)

    def copy_normalization(self, other_network_body: LSTMNetworkBody):
        if not self.normalized_observations:
            return
        self.body_1.copy_normalization(other_network_body)
        self.body_2.copy_normalization(other_network_body)


class SAC_LSTM(object):
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
            per_settings: dict = None

    ):
        self.device = get_device()

        # Initialize networks
        self.actor = Actor(observation_specs, act_dim, network_settings).to(self.device)
        self.critic = Critic(observation_specs, act_dim, network_settings).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=crit_lr)

        # Initialize Entropy Temperature
        self.target_entropy = -float(act_dim)
        # log_alpha of 0 is same as initial alpha of 1.
        self.log_alpha = torch.log((torch.ones(1, device=self.device) * alpha)).requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Initialize Learning Rate Schedules
        self.actor_learning_rate_schedule = LearningRate(ScheduleType(learning_rate_schedule), act_lr, 1e-10, max_steps)
        self.critic_learning_rate_schedule = LearningRate(ScheduleType(learning_rate_schedule), crit_lr, 1e-10,
                                                          max_steps)
        self.alpha_learning_rate_schedule = LearningRate(ScheduleType(learning_rate_schedule), alpha_lr, 1e-10,
                                                         max_steps)

        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq

        # Initialize counters and loss storage
        self.total_it = 0
        self.actor_loss_value = 0
        self.entropy_value = 0
        self.log_alpha_loss_values = 0

        self.experience_replay_type = experience_replay_type
        self.n_step = n_step  # n=1 means value overestimation/high off-policy bias, n-->inf underestimation & high off-policy variance

        # per settings
        self.use_per = per_settings is not None
        self.eta = per_settings["per_eta"] if self.use_per else 0
        # performance outputs
        self.q = 0
        self.entropy = 0

        # onnx model parameters
        self.onnx_actor = ONNX_Actor(self.actor)
        self.deterministic_onnx_actor = ONNX_Actor(self.actor, is_deterministic=True)

    def select_stochastic_action(self, obs: List[np.ndarray], memories_in: torch.Tensor):
        """
        Takes a numpy observation from the environment to take an action.

        This method is to be used during sample collection during training.

        This action does not use the reparameterisation trick and therefore is not to be used during train method.

        Returns action.
        """
        # Ensure observation in shape (batch, seq, obs_dim)
        observation = obs.copy()
        observation = ensure_obs_dimensionality(observation, isLSTM=True)
        observation = list_to_torch(observation, False, self.device)

        with torch.no_grad():
            mean, log_std, memories_out = self.actor(observation, memories_in.to(self.device), sequence_length=1)

            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)

            # squeeze action to shape (batch, num_actions) for use in Unity/buffer storage.
            action = action.squeeze(1)

            return torch_to_numpy(action), memories_out.detach().cpu()

    def select_deterministic_action(self, obs: List[np.ndarray], memories_in: torch.Tensor):
        """
        Take a numpy observation from the environment and return a deterministic action.

        This method is used during policy evaluation/testing.
        """
        # Ensure observation in shape (batch, seq, obs_dim)
        observation = obs.copy()
        observation = ensure_obs_dimensionality(observation, isLSTM=True)
        observation = list_to_torch(observation, False, self.device)

        with torch.no_grad():
            mean, _, memories_out = self.actor(observation, memories_in.to(self.device))
            action = torch.tanh(mean)
            action = action.squeeze(1)
            return torch_to_numpy(action), memories_out.detach().cpu()

    def update_learning_rate(self, current_step: int):
        update_learning_rate(self.actor_optimizer, self.actor_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.critic_optimizer, self.critic_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.log_alpha_optimizer, self.alpha_learning_rate_schedule.get_value(current_step))

    def update(self, replay_buffer, batch_size=128):
        """
        Update optimizers by sampling a batch of sequences from the replay buffer.
        """

        self.total_it += 1

        burn_in_batch, learn_batch, burn_in_memories, learn_memories, indices, weights = replay_buffer.sample(
            batch_size)

        post_burn_in_memories = self.unroll_burn_in_sequence(burn_in_batch, burn_in_memories)

        critic_loss, q, priorities = self.update_critic(learn_batch, learn_memories, post_burn_in_memories, weights)

        new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = self.update_actor(learn_batch,
                                                                                                      learn_memories,
                                                                                                      post_burn_in_memories, weights)
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
            {"v": 0, "q": q, "log_p": new_log_prob_value, "entropy": entropy_value,
             "alpha": self.log_alpha.detach().exp().item()}

    def unroll_burn_in_sequence(self, burn_in_batch, burn_in_memories):
        """
        Generates initial memory states for actor-critic updates by rolling out over the burn-in states provided.
        Unlike R2D2 which only rollouts hidden states for the critic using stored (obs,action), SAC requires a more comprehensive rollout for every LSTM memory used in update.
        SAC uses the actor to predict new actions and next actions for q-value prediction, and each critic has two Q-networks.
        Additionally, the critic network is used for q-value prediction using both stored actions and new actions.

        Returns hidden_in & out memories for the actor's LSTM network, hidden_in for both critic q-networks, hidden_in for both critic q-networks that takes new actions, and hidden out for target critic q-networks.
        """
        if burn_in_batch is None:
            return None

        obs, actions, rewards, next_obs, terminals = burn_in_batch
        memories_in, memories_out = burn_in_memories

        sequence_length = rewards.shape[1]

        with torch.no_grad():
            new_action, _, actor_hidden_in = self.actor.sample(obs, memories_in, sequence_length=sequence_length)
            new_next_action, _, actor_hidden_out = self.actor.sample(next_obs, memories_out, sequence_length=sequence_length)

            _, _, critic_hidden_in_1, critic_hidden_in_2 = self.critic(obs, actions, memories_in, sequence_length=sequence_length)
            _, _, critic_hidden_in_pol_1, critic_hidden_in_pol_2 = self.critic(obs, new_action, memories_in, sequence_length=sequence_length)
            _, _, target_critic_hidden_out_1, target_critic_hidden_out_2 = self.critic_target(next_obs, new_next_action, memories_out, sequence_length=sequence_length)

        return actor_hidden_in, actor_hidden_out, critic_hidden_in_1, critic_hidden_in_2, critic_hidden_in_pol_1, critic_hidden_in_pol_2, target_critic_hidden_out_1, target_critic_hidden_out_2

    def update_critic(self, minibatch, learn_memories, post_burn_in_memories, weights):

        obs, actions, rewards, next_obs, terminals = minibatch
        memories_in, memories_out = learn_memories
        sequence_length = obs[0].shape[1]

        if post_burn_in_memories is not None:
            actor_memories_in, actor_memories_out, critic_memories_in_1, critic_memories_in_2, _, _, target_memories_out_1, target_memories_out_2 = post_burn_in_memories
        else:
            actor_memories_in, actor_memories_out, critic_memories_in_1, critic_memories_in_2, target_memories_out_1, target_memories_out_2 = memories_in, memories_out, memories_in, None, memories_out, None

        with torch.no_grad():
            new_next_action, next_log_prob, _ = self.actor.sample(next_obs, actor_memories_out, sequence_length=sequence_length)

            # Compute target q-values
            next_target_q_1, next_target_q_2, _, _ = self.critic_target(next_obs, new_next_action,
                                                                        target_memories_out_1, target_memories_out_2,
                                                                        sequence_length)
            next_target_q = torch.min(next_target_q_1, next_target_q_2) - self.log_alpha.exp() * next_log_prob
            target_q = compute_bellman_target(rewards, terminals, next_target_q, self.gamma, self.n_step)
            target_q = target_q.detach()

        # obtain current q-estimates
        q_1, q_2, _, _ = self.critic(obs, actions, critic_memories_in_1, critic_memories_in_2, sequence_length)
        q = torch.min(q_1.detach(), q_2.detach()).mean().item()

        td_error_1 = target_q - q_1
        td_error_2 = target_q - q_2

        # Calculate new priorities
        if self.use_per:
            td_error = 0.5*(td_error_1.clone().detach() + td_error_2.clone().detach())
            td_error = torch.abs(td_error)
            priorities = self.eta*td_error.max(dim=1).values + (1-self.eta)*td_error.mean(dim=1)
            priorities = priorities.clamp(min=0.00001, max=10000)
        else:
            priorities = None

        td_loss_1 = (td_error_1.pow(2)*weights).mean()
        td_loss_2 = (td_error_2.pow(2)*weights).mean()

        critic_loss = td_loss_1 + td_loss_2

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), q, priorities

    def update_actor(self, minibatch, learn_memories, post_burn_in_memories, weights):
        obs, _, _, _, _ = minibatch
        sequence_length = obs[0].shape[1]

        memories_in, memories_out = learn_memories

        if post_burn_in_memories is not None:
            actor_memories_in, _, _, _, critic_memories_in_1, critic_memories_in_2, _, _ = post_burn_in_memories
        else:
            actor_memories_in, critic_memories_in_1, critic_memories_in_2 = memories_in, memories_in, None

        new_action, new_log_prob, _ = self.actor.sample(obs, memories_in=actor_memories_in, sequence_length=sequence_length)
        actor_loss_value, entropy_value, log_alpha_loss_value = None, None, None

        if self.total_it % self.policy_freq == 0:
            # Calculate clipped dual q-function
            q1, q2, _, _ = self.critic(obs, new_action, critic_memories_in_1, critic_memories_in_2, sequence_length)
            min_q = torch.min(q1, q2)
            entropy_bonus = -self.log_alpha.exp() * new_log_prob
            actor_loss = -(min_q + entropy_bonus)
            actor_loss = (actor_loss*weights).mean()

            # optimize actor via gradient ascent
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            ###########################################################

            ######### UPDATE THE ALPHA VALUE ########################
            # In rlkit sac implementation, the official pytorch implementation, they use log_alpha rather than alpha as in the paper.
            log_alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()
            ############################################

            actor_loss_value = actor_loss.item()
            entropy_value = entropy_bonus.mean().item()
            log_alpha_loss_value = log_alpha_loss.item()

        return new_log_prob.mean().item(), actor_loss_value, entropy_value, log_alpha_loss_value

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
        actor_dict = torch.load(f"{load_dir}/actor_dict")
        actor_opt_dict = torch.load(f"{load_dir}/actor_optimizer_dict")

        critic_dict = torch.load(f"{load_dir}/critic_dict")
        critic_opt_dict = torch.load(f"{load_dir}/critic_optimizer_dict")

        log_alpha = torch.load(f"{load_dir}/log_alpha_dict")
        log_alpha_opt_dict = torch.load(f"{load_dir}/log_alpha_optimizer_dict")

        self.actor.load_state_dict(actor_dict)
        self.actor_optimizer.load_state_dict(actor_opt_dict)

        self.critic.load_state_dict(critic_dict)
        self.critic_optimizer.load_state_dict(critic_opt_dict)
        self.critic_target = copy.deepcopy(self.critic)

        self.log_alpha = copy.deepcopy(log_alpha)
        self.log_alpha_optimizer.load_state_dict(log_alpha_opt_dict)

        print(f"\nModel loaded from {load_dir}")


def compute_bellman_target(rewards: torch.tensor, dones: torch.tensor, next_target_q: torch.Tensor, gamma: float,
                           n_step: int):
    """
    Computes the bellman return for n-steps ahead for a recurrent learnable sequence of length T for batch size B.
    All tensor inputs should have shape (B,T).
    Calculating R_t = Sum_(k=0, n-1) [Gamma**k * r_{t+k} * (1.0 - (dones_t+k))] + gamma**n * Q(s_{t+n})
    """

    r = torch.cat([rewards] + [torch.zeros_like(rewards[:, 0:1])] * n_step, dim=1)
    d = torch.cat([dones] + [torch.zeros_like(dones[:, 0:1])] * n_step, dim=1)
    bellman_target = torch.cat(
        [torch.zeros_like(next_target_q[:, 0:1]), next_target_q]
        + [next_target_q[:, -1:] / gamma ** k for k in range(1, n_step)],
        dim=1)

    for i in range(n_step):
        r = r[:, :-1]
        d = d[:, :-1]
        bellman_target = r + gamma * (1.0 - d.float()) * bellman_target[:, 1:]

    return bellman_target


def test_bellman(batch: int = 3, seq: int = 8, gamma: float = 0.9, n_step: int = 1):
    end = seq // 2
    r = torch.ones((batch, seq)).float()
    r = 0.1 * r
    r[-1, end] = 1.0
    r[-1, end + 1:] = 0

    d = torch.zeros((batch, seq)).float()
    d[-1, end:] = 1
    d[-2, -1] = 1

    q = 2 * torch.ones((batch, seq)).float()

    print(compute_bellman_target(r, d, q, gamma, n_step))
