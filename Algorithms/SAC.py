import copy
from typing import List, Tuple, Union, Optional

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from mlagents_envs.base_env import ObservationSpec, ActionSpec

from NetworkUtils import get_device, ensure_obs_dimensionality, list_to_torch, torch_to_numpy
from ModelUtils import LearningRate, ScheduleType, update_learning_rate, PAL, Huber_Loss
from Utils import ExperienceReplayType
from Encoders import NetworkBody, linear_layer
from EncoderUtils import NetworkSettings, Initialization, ActivationType, BodyEncoderSettings, count_parameters

from DataStructures import ReplayBuffer, PrioritizedReplayBuffer, LA3PReplayBuffer


class Actor(nn.Module):
    """
    Note: act_dim and obs_dim are the number of actions/observations. Not the shape of what Unity gives back.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int, network_settings: NetworkSettings,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.normalized_observations = network_settings.normalize
        self.observation_specs = observation_specs
        self.act_dim = act_dim

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish
                                                    )

        self.body = NetworkBody(observation_specs, network_settings, 0, body_encoder_settings)


        self.mean = linear_layer(network_settings.hidden_size, act_dim,
                                 Initialization.Uniform, init_w, Initialization.Uniform, init_w)
        self.log_std = linear_layer(network_settings.hidden_size, act_dim,
                                    Initialization.Uniform, init_w, Initialization.Uniform, init_w)

        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

        print(self)
        count_parameters(self)

    def forward(self, obs: List[torch.Tensor]):
        """
        Returns the actions and action's log_prob for a batch of agents for a given list of observation tensors.

        Actions in shape (n,a).

        Log_p in shape (n,).
        """
        encoding = self.body(obs, actions=None)

        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs: List[torch.Tensor]):
        """
        For a given observation list, return the action and log_prob.

        This method uses the reparameterization trick and therefore can be used for backprop.

        The action returned is tanh-squeezed and is in the range  [-1,1].
        """

        mean, log_std = self.forward(obs)
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
        log_p = log_p.sum(dim=1)

        # Squeeze action with tanh function to bound between -1 and 1
        action = torch.tanh(action)

        return action, log_p

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

        self.m_size = policy.body.memory_size

        self.version_number = torch.nn.Parameter(torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)
        self.memory_size_vector = torch.nn.Parameter(torch.Tensor([int(self.m_size)]), requires_grad=False)
        self.continuous_action_size = torch.nn.Parameter(torch.Tensor([policy.act_dim]), requires_grad=False)

        self.to(get_device())

    def copy_policy(self, policy_to_copy: Actor):
        policy = policy_to_copy
        self.body = policy.body
        self.mean = policy.mean
        self.log_std = policy.log_std

    def forward(self, obs: List[torch.Tensor], masks: Optional[torch.Tensor], memories_in: Optional[torch.Tensor]) -> Tuple[Union[int, torch.Tensor], ...]:

        export_output = [self.version_number, self.memory_size_vector]
        encoding = self.body(obs, actions=None)
        mean = self.mean(encoding)
        log_std = self.log_std(encoding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        action = mean + (1.0 - self.is_deterministic)*torch.randn_like(mean) * torch.exp(log_std)
        action = torch.tanh(action)
        export_output += [action, self.continuous_action_size]

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

        self.normalized_observations = network_settings.normalize

        body_encoder_settings = BodyEncoderSettings(Initialization.KaimingHeNormal, 1.0,
                                                    Initialization.Constant, 0.0,
                                                    ActivationType.Swish
                                                    )

        # Q1 architecture
        self.body_1 = NetworkBody(observation_specs, network_settings, act_dim, body_encoder_settings)
        self.q_1_head = linear_layer(network_settings.hidden_size, 1, Initialization.Uniform, init_w, Initialization.Uniform, init_w)

        # Q2 architecture
        self.body_2 = NetworkBody(observation_specs, network_settings, act_dim, body_encoder_settings)
        self.q_2_head = linear_layer(network_settings.hidden_size, 1, Initialization.Uniform, init_w, Initialization.Uniform, init_w)

    def forward(self,obs: List[torch.Tensor], act: torch.Tensor):

        q1 = self.body_1(obs, act)
        q1 = self.q_1_head(q1)

        q2 = self.body_2(obs, act)
        q2 = self.q_2_head(q2)

        return torch.squeeze(q1, -1), torch.squeeze(q2, -1)

    def update_normalization(self, obs: List[torch.Tensor]):
        if not self.normalized_observations:
            return
        self.body_1.update_normalization(obs)
        self.body_2.update_normalization(obs)

    def copy_normalization(self, other_network_body: NetworkBody):
        if not self.normalized_observations:
            return
        self.body_1.copy_normalization(other_network_body)
        self.body_2.copy_normalization(other_network_body)


class SAC(object):
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
            per_alpha=0.4,  # 0.6 for per, 0.4 for la3p
            per_beta=0.4,
            la3p_frac=0.5,
            experience_replay_type: ExperienceReplayType = ExperienceReplayType.BASIC,
            n_step: int = 1
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

        # # Must detach the grad to prevent extra gradient information to be passed to log_alpha
        # # note: detach() keeps the tensor on the GPU, unlike .data or .item().
        # # Transfer between cpu and gpu should be done as little as possible.
        # self.alpha = (self.log_alpha.detach()).exp()
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq

        # Initialize counters and loss storage
        self.total_it = 0
        self.actor_loss_value = 0
        self.entropy_value = 0
        self.log_alpha_loss_values = 0

        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.la3p_frac = la3p_frac
        self.experience_replay_type = experience_replay_type
        self.n_step = n_step  # n=1 means value overestimation/high off-policy bias, n-->inf underestimation & high off-policy variance

        self.q = 0
        self.entropy = 0

        self.onnx_actor = ONNX_Actor(self.actor)
        self.deterministic_onnx_actor = ONNX_Actor(self.actor, is_deterministic=True)

    def select_stochastic_action(self, obs: List[np.ndarray]):
        """
        Takes a numpy observation from the environment to take an action.

        This method is to be used during sample collection during training.

        This action does not use the reparameterisation trick and therefore is not to be used during train method.

        Returns action.
        """
        # Ensure observation in shape (batch, obs_dim)
        observation = obs.copy()
        observation = ensure_obs_dimensionality(observation,isLSTM=False)
        observation = list_to_torch(observation, False, self.device)

        with torch.no_grad():

            mean, log_std = self.actor(observation)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)

            return torch_to_numpy(action)

    def select_deterministic_action(self, obs: List[np.ndarray]):
        """
        Take a numpy observation from the environment and return a deterministic action.

        This method is used during policy evaluation/testing.
        """
        obs = ensure_obs_dimensionality(obs)
        obs = list_to_torch(obs, False, self.device)

        with torch.no_grad():
            mean = self.actor(obs)
            action = torch.tanh(mean)
            return torch_to_numpy(action)

    def update_learning_rate(self, current_step: int):
        update_learning_rate(self.actor_optimizer, self.actor_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.critic_optimizer, self.critic_learning_rate_schedule.get_value(current_step))
        update_learning_rate(self.log_alpha_optimizer, self.alpha_learning_rate_schedule.get_value(current_step))

    def update(self, replay_buffer, batch_size=128):
        self.total_it += 1

        if self.experience_replay_type == ExperienceReplayType.PER:
            # Standard PER ala Schaul (Reduces variance of sampling)
            loss_values = self.per_update(replay_buffer, batch_size)
        elif self.experience_replay_type == ExperienceReplayType.LA3P:
            # Loss-Adjusted Approximate Actor Prioritized Experience Replay
            # Combines LAP and PAL along with inverse prioritization for actor update
            # ABout 2x slower than base
            # https://github.com/baturaysaglam/LA3P
            loss_values = self.la3p_update(replay_buffer, batch_size)
        elif self.experience_replay_type == ExperienceReplayType.LAP:
            # Loss-Adjusted Priortized ER (PER with Huber Loss --> unbiased sampling)
            # https://github.com/sfujim/LAP-PAL
            loss_values = self.per_update(replay_buffer, batch_size)
        elif self.experience_replay_type == ExperienceReplayType.PAL:
            # Prioritized Approximation Loss ER (Basic ER with PAL loss --> Use where variance reduction minimal, gives same loss gradient as LAP)
            # https://github.com/sfujim/LAP-PAL
            loss_values = self.basic_update(replay_buffer, batch_size)
        elif self.experience_replay_type == ExperienceReplayType.RELO:
            # Reducible Loss Prioritized Experience Replay (Use ReLO to assess "learnability" of transition. Low TD ignored, High non-noisy TD favored)
            # https://github.com/shivakanthsujit/reducible-loss
            loss_values = self.per_update(replay_buffer, batch_size)
        else:
            loss_values = self.basic_update(replay_buffer, batch_size)

        return loss_values

    def basic_update(self, replay_buffer: ReplayBuffer, batch_size: int):
        minibatch = replay_buffer.sample(batch_size, self.n_step, self.gamma)

        _, _, critic_loss, q = self.update_critic(minibatch)

        new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = self.update_actor(minibatch)
        self.soft_update()

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

    def per_update(self, replay_buffer: PrioritizedReplayBuffer, batch_size):
        minibatch = replay_buffer.prioritized_sample(batch_size)
        indices, priorities, critic_loss, q = self.update_critic(minibatch)
        replay_buffer.update_priority(indices, priorities)
        new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = self.update_actor(minibatch)
        self.soft_update()

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

    def la3p_update(self, replay_buffer: LA3PReplayBuffer, batch_size):
        critic_losses, qs, new_log_prob_values, actor_losses, entropy_values, log_alpha_losses = [], [], [], [], [], []

        uniform_batch_size = int((1 - self.la3p_frac) * batch_size)
        prioritized_batch_size = int(self.la3p_frac * batch_size)

        if uniform_batch_size > 0:
            uniform_minibatch = replay_buffer.uniform_sample(uniform_batch_size)
            # Update critic with uniform sampling
            indices, priorities, critic_loss, q = self.update_critic(uniform_minibatch, uniform_sampling=True)
            replay_buffer.update_priority(indices, priorities)
            # update actor with shared uniform sampling
            new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = self.update_actor(
                uniform_minibatch)
            self.soft_update()
            # append values for plotting
            critic_losses.append(critic_loss)
            qs.append(q)
            new_log_prob_values.append(new_log_prob_value)
            actor_losses.append(actor_loss_value)
            entropy_values.append(entropy_value)
            log_alpha_losses.append(log_alpha_loss_value)

        prioritized_critic_batch = replay_buffer.sample_critic(prioritized_batch_size)
        indices, priorities, critic_loss, q = self.update_critic(prioritized_critic_batch, uniform_sampling=False)
        replay_buffer.update_priority(indices, priorities)
        self.soft_update()

        inverse_prio_actor_batch = replay_buffer.sample_actor(prioritized_batch_size)
        new_log_prob_value, actor_loss_value, entropy_value, log_alpha_loss_value = self.update_actor(
            inverse_prio_actor_batch)

        # handle plotting values
        critic_losses.append(critic_loss)
        qs.append(q)
        new_log_prob_values.append(new_log_prob_value)
        actor_losses.append(actor_loss_value)
        entropy_values.append(entropy_value)
        log_alpha_losses.append(log_alpha_loss_value)

        critic_loss = np.mean(critic_losses)
        q = np.mean(qs)
        new_log_prob_value = np.mean(new_log_prob_values)

        if actor_losses[0] is not None:
            actor_loss = np.mean(actor_losses)
            entropy_value = np.mean(entropy_values)
            log_alpha_loss = np.mean(log_alpha_losses)
            self.actor_loss_value = actor_loss
            self.entropy_value = entropy_value
            self.log_alpha_loss_values = log_alpha_loss
        else:
            actor_loss = self.actor_loss_value
            entropy_value = self.entropy_value
            log_alpha_loss = self.log_alpha_loss_values

        return {"actor": actor_loss, "critic": critic_loss,
                "alpha": log_alpha_loss}, \
               {"v": 0, "q": q, "log_p": new_log_prob_value, "entropy": entropy_value,
                "alpha": self.log_alpha.detach().exp().item()}

    def update_critic(self, minibatch, uniform_sampling: bool = True):
        obs, actions, rewards, next_obs, terminals, sample_indices, sample_weights, n_power = minibatch

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)

            # Compute target q-values
            next_target_q_1, next_target_q_2 = self.critic_target(next_obs, next_action)
            next_entropy = -self.log_alpha.exp() * next_log_prob
            next_target_q = torch.min(next_target_q_1, next_target_q_2) + next_entropy
            target_q = rewards + (1 - terminals) * (self.gamma ** n_power) * next_target_q
            target_q = target_q.detach()

        # obtain current q-estimates
        q_1, q_2 = self.critic(obs, actions)
        q = torch.min(q_1.detach(), q_2.detach()).mean().item()

        td_loss_1 = target_q - q_1
        td_loss_2 = target_q - q_2

        if self.experience_replay_type == ExperienceReplayType.LA3P:
            if uniform_sampling:
                critic_loss = PAL(td_loss_1) + PAL(td_loss_2)
                critic_loss /= torch.max(td_loss_1.abs(), td_loss_2.abs()).clamp(min=1.0).pow(
                    self.per_alpha).mean().detach()
            else:
                critic_loss = Huber_Loss(td_loss_1.abs()) + Huber_Loss(td_loss_2.abs())

        elif self.experience_replay_type == ExperienceReplayType.PER:
            # weighted MSE loss
            element_critic_loss = td_loss_1.pow(2) + td_loss_2.pow(2)
            critic_loss = (sample_weights * element_critic_loss).mean()

        elif self.experience_replay_type == ExperienceReplayType.PAL:
            critic_loss = PAL(td_loss_1) + PAL(td_loss_2)
            critic_loss /= torch.max(td_loss_1.abs(), td_loss_2.abs()).clamp(min=1.0).pow(
                self.per_alpha).mean().detach()

        elif self.experience_replay_type == ExperienceReplayType.LAP:
            critic_loss = Huber_Loss(td_loss_1.abs()) + Huber_Loss(td_loss_2.abs())

        elif self.experience_replay_type == ExperienceReplayType.RELO:
            element_critic_loss = td_loss_1.pow(2) + td_loss_2.pow(2)
            critic_loss = (sample_weights * element_critic_loss).mean()

        else:
            # MSE loss
            critic_loss = td_loss_1.pow(2).mean() + td_loss_2.pow(2).mean()

        relo_loss = None
        if self.experience_replay_type == ExperienceReplayType.RELO:
            with torch.no_grad():
                orig_loss = element_critic_loss / 2
                target_q_1, target_q_2 = self.critic_target(obs, actions)
                irr_loss_1 = (target_q_1 - target_q).pow(2)
                irr_loss_2 = (target_q_2 - target_q).pow(2)
                irr_loss = (irr_loss_1 + irr_loss_2) / 2
                relo_loss = orig_loss - irr_loss
                # how much more loss remaining compared to old model?
                # large pos == much more to learn or current model forgotten --> high prio
                # small pos means not much more or noisy/unlearnable--> low prio
                # negative means new model is confident, and has learnt it --> lowest prio

        if relo_loss is None:
            # clipping min at 1 to reduce likelihood of dead transitions
            priorities = torch.max(td_loss_1.abs().detach(), td_loss_2.abs().detach()).clamp(min=1.0).pow(
                self.per_alpha)
        else:
            priorities = ((F.relu(relo_loss) + 1e-5).pow(self.per_alpha))

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return sample_indices, torch_to_numpy(priorities), critic_loss.item(), q

    def update_actor(self, minibatch):
        obs, _, _, _, _, _, weights, _ = minibatch
        new_actions, new_log_prob = self.actor.sample(obs)
        actor_loss_value, entropy_value, log_alpha_loss_value = None, None, None

        if self.total_it % self.policy_freq == 0:
            # Calculate clipped dual q-function
            q1, q2 = self.critic(obs, new_actions)
            min_q = torch.min(q1, q2)
            entropy_bonus = -self.log_alpha.exp() * new_log_prob
            actor_loss = -(min_q + entropy_bonus)
            # calculate actor loss (Q in expectation, negative to perform gradient ascent)
            if self.experience_replay_type == ExperienceReplayType.RELO:
                actor_loss = (actor_loss * weights).mean()
            actor_loss = actor_loss.mean()

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
