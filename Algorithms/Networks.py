import abc
import itertools

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal, Normal
from torch._six import inf

from NetworkUtils import *
from ActionSpace import ActionSpace
from typing import Union
from copy import deepcopy

Activation = Union[str, nn.Module]

str_to_activation_dict = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def ppo_init_method(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)


def ddpg_init_method(layer):
    # As defined in Lillicrap 2015
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        nn.init.uniform_(layer.weight, -lim, lim)
        nn.init.uniform_(layer.bias, -lim, lim)


def ddpg_output_method(layer):
    if isinstance(layer, nn.Linear):
        lim = 0.003
        nn.init.uniform_(layer.weight, -lim, lim)
        nn.init.uniform_(layer.bias, -lim, lim)


def build_mlp(input_size: int, output_size: int, hidden_size: int, num_hidden_layers: int,
              activation: Activation = 'relu', output_activation: Activation = 'identity',
              init_method=None, output_method=None
              ) -> nn.Sequential:
    if isinstance(activation, str):
        activation = str_to_activation_dict[activation]
    if isinstance(output_activation, str):
        output_activation = str_to_activation_dict[output_activation]

    layers = []
    size_list = [input_size] + num_hidden_layers * [hidden_size]
    for i in range(num_hidden_layers):
        current_layer = nn.Linear(size_list[i], size_list[i + 1])
        if init_method is not None:
            current_layer.apply(init_method)
        layers.append(current_layer)
        layers.append(activation)
    output_layer = nn.Linear(size_list[-1], output_size)
    if output_method is not None:
        output_layer.apply(output_method)
    layers.append(output_layer)
    layers.append(output_activation)
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 obs_size: int, action_size: int, hidden_size: int, num_hidden_layers: int, device: torch.device,
                 discrete: bool = True, learning_rate: float = 1e-4, training: bool = True,
                 nn_baseline: bool = False, baseline_learning_rate: float = 1e-4,
                 activation='relu', output_activation='identity'
                 ):
        super(MLPPolicy, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.baseline_learning_rate = baseline_learning_rate
        self.activation = activation
        self.output_activation = output_activation

        self.policy_loss = None
        self.baseline_loss = None

        if self.device is None:
            self.device = get_device()

        if self.discrete:
            self.discrete_logits = build_mlp(input_size=self.obs_size,
                                             output_size=self.action_size,
                                             hidden_size=self.hidden_size,
                                             num_hidden_layers=self.num_hidden_layers,
                                             activation=self.activation,
                                             output_activation=self.output_activation,
                                             ).to(self.device)

            self.mean_net = None
            self.log_std = None
        else:
            self.discrete_logits = None
            self.mean_net = build_mlp(input_size=self.obs_size,
                                      output_size=self.action_size,
                                      hidden_size=self.hidden_size,
                                      num_hidden_layers=self.num_hidden_layers,
                                      activation=self.activation,
                                      output_activation=self.output_activation,
                                      ).to(self.device)
            self.log_std = nn.Parameter(torch.zeros(self.action_size, dtype=torch.float, device=self.device))

        self.optimizer = optim.Adam(self.policy_parameters(), lr=self.learning_rate)

        if self.nn_baseline:
            self.baseline = build_mlp(input_size=self.obs_size,
                                      output_size=1,
                                      num_hidden_layers=self.num_hidden_layers,
                                      hidden_size=self.hidden_size).to(self.device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=self.baseline_learning_rate)
        else:
            self.baseline = None

    def policy_parameters(self):
        if self.discrete:
            return self.discrete_logits.parameters()
        else:
            return itertools.chain(self.mean_net.parameters(), [self.log_std])

    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            return Categorical(logits=self.discrete_logits(observation))
        else:
            batch_mean = self.mean_net(observation)
            batch_log_std = self.log_std.expand_as(batch_mean)
            batch_std = torch.exp(batch_log_std)
            action_dist = Normal(loc=batch_mean, scale=batch_std)
            return action_dist

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        For a given numpy observation (batch_size,num_obs), return an action (batch_size,)
        """
        # Ensure obs shape is (batch,obs)
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = numpy_to_torch(observation)

        with torch.no_grad():
            if self.discrete:
                batch_actions = self(observation).sample()
            else:
                batch_actions = self(observation).rsample()
        return torch_to_numpy(batch_actions)

    def log_probs(self, obs, actions):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if isinstance(observation, np.ndarray):
            observation = numpy_to_torch(observation)

        if isinstance(actions, np.ndarray):
            actions = numpy_to_torch(actions)

        if self.discrete:
            log_probs = self(observation).log_prob(actions)
            return log_probs
        else:
            # Must sum over last dimension for pytorch Normal distribution
            log_probs = self(observation).log_prob(actions).sum(dim=-1)
            return log_probs

    def save(self, save_dir, exp_name):
        name = f"{exp_name}"
        losses = {"policy_loss": self.policy_loss, "baseline_loss": self.baseline_loss}
        save_model_state_dict(self, save_dir, name, losses)

    def load(self, model_dir, model_name):
        # print(self.state_dict())
        losses = load_model_state_dict(self, model_dir, model_name)
        self.policy_loss = losses["policy_loss"]
        self.baseline_loss = losses["baseline_loss"]

        if self.training:
            self.train()
        else:
            self.eval()


class MLPPolicyPG(MLPPolicy):
    def __init__(self,
                 obs_size: int, action_size: int, hidden_size: int, num_hidden_layers: int, device: torch.device,
                 discrete: bool = True, learning_rate: float = 1e-4, training: bool = True,
                 nn_baseline: bool = False, baseline_learning_rate: float = 1e-4
                 ):
        super(MLPPolicyPG, self).__init__(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            device=device,
            discrete=discrete, learning_rate=learning_rate, training=training, nn_baseline=nn_baseline,
            baseline_learning_rate=baseline_learning_rate
        )
        self.baseline_loss_critereon = nn.MSELoss()

    def update(self, observations: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
               q_values: np.ndarray = None):
        """
        Update the policy for a given batch of observations and actions.
        """
        observations = numpy_to_torch(observations)
        actions = numpy_to_torch(actions)
        advantages = numpy_to_torch(advantages)

        # Update the actor network
        action_dist = self(observations)
        log_probs = action_dist.log_prob(actions)
        policy_loss = -torch.sum(advantages * log_probs)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.policy_loss = policy_loss

        # Update the critic network
        if self.nn_baseline:
            # Normalize target q-values to 0 mean and 1 std
            target = normalize(data=q_values, mean=np.mean(q_values), std=np.std(q_values))
            target = numpy_to_torch(target)
            # Obtain predicted state-values for batch
            prediction = torch.squeeze(self.baseline(observations))
            assert prediction.shape == target.shape
            baseline_loss = self.baseline_loss_critereon(prediction, target)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            self.baseline_loss = baseline_loss

        train_log = {
            'training_loss': torch_to_numpy(policy_loss)
        }
        return train_log

    def run_baseline_prediction(self, observations: numpy.ndarray):
        """
        Runs a baseline prediction on an observation tensor.
        Input: observations: Tensor of size [N,obs]
        Output: predictions: Tensor of size [N,]
        """

        observations = numpy_to_torch(observations)
        predication = self.baseline(observations)
        return torch_to_numpy(predication.squeeze())


class MLPPolicyAC(MLPPolicy):
    def update(self, observations: np.ndarray, actions: np.ndarray, advantages: np.ndarray):
        observations = numpy_to_torch(observations)
        actions = numpy_to_torch(actions)
        advantages = numpy_to_torch(advantages)

        # Update the actor network
        action_dist = self(observations)
        log_probs = action_dist.log_prob(actions)
        policy_loss = -torch.sum(advantages * log_probs)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.policy_loss = policy_loss
        return policy_loss.item()


class MLPPolicyTRPO(MLPPolicy):
    def __init__(self,
                 obs_size: int, action_size: int, hidden_size: int, num_hidden_layers: int, device: torch.device,
                 cg_steps: int, damping: float, max_backtracks: int, max_kl: float,
                 discrete: bool = True, learning_rate: float = 1e-4, training: bool = True,
                 nn_baseline: bool = False, baseline_learning_rate: float = 1e-4,
                 ):
        super(MLPPolicyTRPO, self).__init__(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            device=device,
            discrete=discrete, learning_rate=learning_rate, training=training, nn_baseline=nn_baseline,
            baseline_learning_rate=baseline_learning_rate
        )
        self.baseline_loss_critereon = nn.MSELoss()

        # number of conjugate grad steps
        self.cg_steps = cg_steps
        #
        self.cg_damping = damping
        # max number of backtrack steps during line search
        self.max_backtracks = max_backtracks
        # Max KL distance
        self.max_kl = max_kl

    def policy_parameters(self):
        if self.discrete:
            return self.discrete_logits.parameters()
        else:
            return itertools.chain(self.mean_net.parameters(), [self.log_std])

    def update(self, observations, actions, advantages, q_value=None):
        """
        TRPO policy update function
        """
        observations = numpy_to_torch(observations)
        actions = numpy_to_torch(actions)
        advantages = numpy_to_torch(advantages)

        action_dist = self(observations)
        log_probs = action_dist.log_prob(actions)

        with torch.no_grad():
            old_log_probs = self(observations).log_prob(actions)

        loss = self.surrogate_reward(advantages, log_probs, old_log_probs)

        # Compute the grad of the loss manually
        grads = torch.autograd.grad(loss, self.policy_parameters())
        pol_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # determine the step direction for the policy gradient via Conjugate Gradient Descent
        step_dir = self.conjugate_gradient(-pol_grad, observations)

        max_step = torch.sqrt(2 * self.max_kl / torch.dot(step_dir, self.fisher_vector_product(observations, step_dir)))
        full_step = max_step * step_dir
        expected_improve = torch.dot(-pol_grad, full_step)

        prev_params = flatten_params(self.policy_parameters()).clone()
        success, new_params = self.line_search(observations, actions, advantages, old_log_probs, prev_params, full_step,
                                               expected_improve)
        assign_params_to(self.policy_parameters(), new_params)

        return loss.item()

    def run_baseline_prediction(self, observations: numpy.ndarray):
        """
        Runs a baseline prediction on an observation tensor.
        Input: observations: Tensor of size [N,obs]
        Output: predictions: Tensor of size [N,]
        """

        observations = numpy_to_torch(observations)
        predication = self.baseline(observations)
        return torch_to_numpy(predication.squeeze())

    def surrogate_reward(self, advantages, log_probs, old_log_probs):
        """
        Compute the surrogate reward objective for TRPO.

        L = -E[ A*(pi_new / pi_old) ] using data from old policy (old advantage)
        """
        return -(advantages * torch.exp(log_probs - old_log_probs)).mean()

    def conjugate_gradient(self, b, observations, residual_tolerance=1e-10):
        """
        Conjugate gradient descent algorithm.

        Compute x_k = H_k^-1 g_k.
        The Hessian difficult to compute and store -> determine way to compute H^-1g.
        """
        x_k = torch.zeros(b.size()).to(self.device)
        d_k = b.clone()
        g_k = b.clone()
        g_dot_g = torch.dot(g_k, g_k)

        for _ in range(self.cg_steps):
            fvp = self.fisher_vector_product(observations, d_k)
            alpha = g_dot_g / torch.dot(d_k, fvp)
            x_k += alpha * d_k

            g_k -= alpha * fvp
            new_g_dot_g = torch.dot(g_k, g_k)

            beta = new_g_dot_g / g_dot_g
            d_k = g_k + beta * d_k
            g_dot_g = new_g_dot_g

            if g_dot_g < residual_tolerance:
                break

        return x_k

    def line_search(self, observations, actions, advantages, old_log_probs, prev_params, full_step,
                    expected_improvement, acceptance_ratio=0.1):
        """
        Use line search to find optimal parameters in trust region.
        """

        log_probs = self(observations).log_prob(actions).detach()
        prev_surr_reward = self.surrogate_reward(advantages, log_probs, old_log_probs)

        for stepfrac in [0.5 ** x for x in range(self.max_backtracks)]:
            new_params = prev_params + stepfrac * full_step
            assign_params_to(self.policy_parameters(), new_params)

            log_probs = self(observations).log_prob(actions).detach()
            surr_reward = self.surrogate_reward(advantages, log_probs, old_log_probs)
            improved = prev_surr_reward - surr_reward
            expected_improvement = expected_improvement * stepfrac
            ratio = improved / expected_improvement

            if ratio.item() > acceptance_ratio:
                return True, new_params

        return False, prev_params

    def fisher_vector_product(self, observations, vector):
        """
        Helper function to compute Hessian vector product for use in Conjugate Gradient Descent.

        FVP = Grad( Grad(D_KL)*x ) = Hx
        """

        # Compute KL divergence between action distributions
        dist_old = self(observations)
        dist_new = self(observations)
        kl_loss = self.kl_divergence(dist_new, dist_old)

        # Compute grad of KL div
        grads = torch.autograd.grad(kl_loss, self.policy_parameters(), create_graph=True)
        grad_vector = torch.cat([grad.view(-1) for grad in grads])
        grad_vector_product = torch.sum(grad_vector * vector)
        grad_grads = torch.autograd.grad(grad_vector_product, self.policy_parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).detach()

        return fisher_vector_product + self.cg_damping * vector

    def kl_divergence(self, dist_p, dist_q):
        """
        KL-Divergence between two policy distributions as shown in
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

        Discrete version:
        D_KL(P||Q) = -sum( P(x) * (log( P(x) ) - log( Q(x) ) )

        Continuous version with Mean and std dev:
        D_KL (P||Q) = sum ( -log(std_q/std_p) + (std_p^2 + (mean_p - mean_q)^2) / (2 * std_p^2 ) - 0.5 )
        """
        if self.discrete:
            # Perform KL divergence between two categorical distributions
            t = dist_p.probs * (dist_p.logits - dist_q.logits)
            t[(dist_q.probs == 0).expand_as(t)] = inf
            t[(dist_p.probs == 0).expand_as(t)] = 0
            return t.sum(-1).mean()
        else:
            mean_old, std_old = dist_p.mean.detach(), dist_p.stddev.detach()
            mean, std = dist_q.mean, dist_q.stddev
            kl = -torch.log(std / std_old) + (std.pow(2) + (mean - mean_old).pow(2)) / (2 * std_old.pow(2)) - 0.5
            kl_f = kl.sum(-1, keepdim=True)
            return kl_f.mean()


class MLPPolicyPPO(MLPPolicy):
    def __init__(self,
                 obs_size: int, action_size: int, hidden_size: int, num_hidden_layers: int, device: torch.device,
                 clip_eps: float, ent_coeff: float, max_grad_norm: float,
                 discrete: bool = True, learning_rate: float = 1e-4, training: bool = True,
                 nn_baseline: bool = False, baseline_learning_rate: float = 1e-4,
                 ):
        super(MLPPolicyPPO, self).__init__(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            device=device,
            discrete=discrete, learning_rate=learning_rate, training=training, nn_baseline=nn_baseline,
            baseline_learning_rate=baseline_learning_rate
        )
        self.baseline_loss_critereon = nn.MSELoss()

        # The clipping range of how far the new policy can change relative to the old policy
        self.clip_eps = clip_eps
        # Entropy coefficient
        self.ent_coeff = ent_coeff
        # max step size of the gradient norm
        self.max_grad_norm = max_grad_norm

    def update(self, observations, actions, advantages, old_log_probs):
        """
        TRPO policy update function
        """
        observations = numpy_to_torch(observations)
        actions = numpy_to_torch(actions)
        advantages = numpy_to_torch(advantages)
        old_log_probs = numpy_to_torch(old_log_probs).detach()

        log_probs = self.log_probs(observations, actions)

        loss = self.surrogate_reward(advantages, log_probs, old_log_probs)

        entropy_bonus = self(observations).entropy().mean()
        loss -= self.ent_coeff * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        # Clip the gradients of the parameters to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(self.policy_parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def surrogate_reward(self, advantages, log_probs, old_log_probs):
        """
        Compute the clipped surrogate reward objective for PPO.

        L = -Expectation[ Min (A*(pi_new / pi_old), Clip (pi_new/pi_old, 1 - eps, 1 + eps)*A ]
        Note: Negative as gradient ascent
        """
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss


class MLPPolicyDDPG(MLPPolicy):
    def __init__(self,
                 obs_size: int, action_size: int, hidden_size: int, num_hidden_layers: int, device: torch.device,
                 polyak: float, act_limit: float, action_space: ActionSpace,
                 discrete: bool = False, learning_rate: float = 1e-4, training: bool = True,
                 nn_baseline: bool = False, baseline_learning_rate: float = 1e-4,
                 activation='relu', output_activation='tanh'
                 ):
        super(MLPPolicyDDPG, self).__init__(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            device=device,
            discrete=discrete, learning_rate=learning_rate, training=training, nn_baseline=nn_baseline,
            baseline_learning_rate=baseline_learning_rate,
            activation=activation, output_activation=output_activation
        )
        self.mean_net = build_mlp(input_size=self.obs_size,
                                  output_size=self.action_size,
                                  hidden_size=self.hidden_size,
                                  num_hidden_layers=self.num_hidden_layers,
                                  activation=self.activation,
                                  output_activation=self.output_activation,
                                  init_method=ddpg_init_method,
                                  output_method=ddpg_output_method
                                  ).to(self.device)
        self.optimizer = optim.Adam(
            self.mean_net.parameters(), lr=self.learning_rate
        )

        # Copy network to target network and freeze parameters of target
        self.mean_net_target = deepcopy(self.mean_net)

        for p in self.mean_net_target.parameters():
            p.requires_grad = False

        self.act_limit = act_limit
        self.action_space = action_space
        # min_max is (-1,1) as output activation function is Tanh.
        self.min_list = [-1.0] * self.action_size
        self.max_list = [1.0] * self.action_size
        self.polyak = polyak
        self.loss = None

        assert self.discrete is False

    def forward(self, observation: torch.FloatTensor):
        mu = self.mean_net(observation)
        range_adjusted_mu = self.action_space.convert_from_given_range(mu, self.min_list, self.max_list)
        return range_adjusted_mu
        # return self.mean_net(observation) * self.act_limit

    def forward_target(self, observation: torch.FloatTensor):
        mu_target = self.mean_net_target(observation)
        adjusted_mu_target = self.action_space.convert_from_given_range(mu_target, self.min_list, self.max_list)
        return adjusted_mu_target
        # return self.mean_net_target(observation) * self.act_limit

    def get_action(self, obs: np.ndarray, noise=None) -> np.ndarray:
        """
        Get action from actor.

        This method takes gaussian noise from the agent to add to the action. Noise is a numpy array of shape (n,a).
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = numpy_to_torch(observation).detach()

        with torch.no_grad():
            action = self(observation)
            action = torch_to_numpy(action)
            if noise is not None:
                action += noise
        mins, maxs = self.action_space.get_min_max_np()

        return action.clip(mins, maxs)
        # return action.clip(-self.act_limit, self.act_limit)

    def update(self, q_values: torch.tensor):
        """
        DDPG policy update.

        q_values assumed to be tensor.
        """

        loss = -q_values.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy_loss = loss

        return loss.item()

    def update_target_network(self):
        """
        Perform polyak update to target parameters
        """
        for target_param, param in zip(
                self.mean_net_target.parameters(), self.mean_net.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

    def load(self, model_dir, model_name):
        super(MLPPolicyDDPG, self).load(model_dir, model_name)
        self.mean_net_target.eval()


class BootstrappedContinuousCritic(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params):
        super().__init__()
        print(params)
        self.obs_size = params['obs_size']
        self.hidden_size = params['hidden_size']
        self.num_hidden_layers = params['num_hidden_layers']
        self.learning_rate = params['critic_lr']
        self.training = params['training']

        # Critic parameters
        self.num_target_updates = params['num_target_updates']
        self.num_grad_steps_per_target_update = params['num_grad_steps_per_target_update']
        self.gamma = params['gamma']

        self.critic_network = build_mlp(self.obs_size,
                                        1,
                                        self.hidden_size,
                                        self.num_hidden_layers)
        self.loss_critereon = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic_network.parameters(),
                                    self.learning_rate
                                    )
        self.loss = None
        self.to(get_device())

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs: np.ndarray):
        obs = numpy_to_torch(obs)
        prediction = self(obs)
        return torch_to_numpy(prediction)

    def update(self, last_obs, obs, reward, terminal, gae_lambda=None):
        """
        Update parameters of the critic.

        For each update of the target value, perform many updates of the critic gradient.

        Note n is batch size being passed into network.

        Arguments:
            last_obs: shape: (n, obs_size)
            actions: shape: (n, action_size)
            obs: shape: (n, obs_size)
            reward: shape: (n,)
            terminal: shape: (n,)
        """

        for _ in range(self.num_target_updates):
            if gae_lambda is None:
                v_s_prime = self.forward_np(obs)
                target = reward + self.gamma * v_s_prime * (1 - terminal)
            else:
                target = self.compute_gae_target(last_obs, obs, reward, terminal, gae_lambda)
            target = numpy_to_torch(target).detach()

            for _ in range(self.num_grad_steps_per_target_update):
                rand_indices = torch.randperm(target.shape[0])
                v_s = self(numpy_to_torch(last_obs))[rand_indices]
                v_target = target[rand_indices]

                loss = self.loss_critereon(v_s, v_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.loss = loss
        return loss.item()

    def compute_gae_target(self, last_obs, obs, reward, terminal, gae_lambda):
        v_s = self.forward_np(last_obs)
        v_s_prime = self.forward_np(obs)
        batch_size = obs.shape[0]
        last_gae = 0
        generalized_advantage_estimates = np.zeros(batch_size)

        for i in range(batch_size - 1, -1, -1):
            v_prime = v_s_prime[i]
            v = v_s[i]
            done = 1 - terminal[i]
            r = reward[i]
            delta = r + self.gamma * v_prime * done - v
            last_gae = delta + self.gamma * gae_lambda * last_gae * done
            generalized_advantage_estimates[i] = last_gae
        critic_target_function = generalized_advantage_estimates + v_s
        return critic_target_function

    def save(self, save_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = {"loss": self.loss}
        save_model_state_dict(self, save_dir, name, losses)

    def load(self, exp_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = load_model_state_dict(self, exp_dir, name)
        self.loss = losses["loss"]

        if self.training:
            self.train()
        else:
            self.eval()


class TRPOCritic(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params):
        super().__init__()
        print(params)
        self.obs_size = params['obs_size']
        self.hidden_size = params['hidden_size']
        self.num_hidden_layers = params['num_hidden_layers']
        self.learning_rate = params['critic_lr']
        self.training = params['training']

        # Critic parameters
        self.num_target_updates = params['num_target_updates']
        self.num_grad_steps_per_target_update = params['num_grad_steps_per_target_update']
        self.gamma = params['gamma']
        self.l2_reg = params['l2_reg']

        self.critic_network = build_mlp(self.obs_size,
                                        1,
                                        self.hidden_size,
                                        self.num_hidden_layers)
        self.loss_critereon = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic_network.parameters(),
                                    self.learning_rate
                                    )
        self.loss = None
        self.to(get_device())

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs: np.ndarray):
        obs = numpy_to_torch(obs)
        prediction = self(obs)
        return torch_to_numpy(prediction)

    def update(self, last_obs, obs, reward, terminal, gae_lambda=None):
        """
        Update parameters of the critic.

        For each update of the target value, perform many updates of the critic gradient.

        Note n is batch size being passed into network.

        Arguments:
            last_obs: shape: (n, obs_size)
            actions: shape: (n, action_size)
            obs: shape: (n, obs_size)
            reward: shape: (n,)
            terminal: shape: (n,)
        """

        for _ in range(self.num_target_updates):
            if gae_lambda is None:
                v_s_prime = self.forward_np(obs)
                target = reward + self.gamma * v_s_prime * (1 - terminal)
            else:
                target = self.compute_gae_target(last_obs, obs, reward, terminal, gae_lambda)
            target = numpy_to_torch(target).detach()

            for _ in range(self.num_grad_steps_per_target_update):
                rand_indices = torch.randperm(target.shape[0])
                v_s = self(numpy_to_torch(last_obs))[rand_indices]
                v_target = target[rand_indices]

                loss = self.loss_critereon(v_s, v_target)

                # Apply l2 regularization to loss function
                for param in self.critic_network.parameters():
                    loss += param.pow(2).sum() * self.l2_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.loss = loss
        return loss.item()

    def compute_gae_target(self, last_obs, obs, reward, terminal, gae_lambda):
        v_s = self.forward_np(last_obs)
        v_s_prime = self.forward_np(obs)
        batch_size = obs.shape[0]
        last_gae = 0
        generalized_advantage_estimates = np.zeros(batch_size)

        for i in range(batch_size - 1, -1, -1):
            v_prime = v_s_prime[i]
            v = v_s[i]
            done = 1 - terminal[i]
            r = reward[i]
            delta = r + self.gamma * v_prime * done - v
            last_gae = delta + self.gamma * gae_lambda * last_gae * done
            generalized_advantage_estimates[i] = last_gae
        critic_target_function = generalized_advantage_estimates + v_s
        return critic_target_function

    def save(self, save_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = {"loss": self.loss}
        save_model_state_dict(self, save_dir, name, losses)

    def load(self, exp_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = load_model_state_dict(self, exp_dir, name)
        self.loss = losses["loss"]

        if self.training:
            self.train()
        else:
            self.eval()


class PPOCritic(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params):
        super().__init__()
        print(params)
        self.obs_size = params['obs_size']
        self.hidden_size = params['hidden_size']
        self.num_hidden_layers = params['num_hidden_layers']
        self.learning_rate = params['critic_lr']
        self.training = params['training']

        # Critic parameters
        self.num_target_updates = params['num_target_updates']
        self.gamma = params['gamma']
        self.l2_reg = params['l2_reg']

        self.critic_network = build_mlp(self.obs_size,
                                        1,
                                        self.hidden_size,
                                        self.num_hidden_layers,
                                        init_method=ppo_init_method)
        self.loss_critereon = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic_network.parameters(),
                                    self.learning_rate,
                                    weight_decay=self.l2_reg
                                    )
        self.loss = None
        self.to(get_device())

    def forward(self, obs):
        """
        Returns the value function in tensor
        """
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs: np.ndarray):
        """
        For a given numpy observation, returns a ndarray version of the value function.
        """
        obs = numpy_to_torch(obs)
        prediction = self(obs)
        return torch_to_numpy(prediction)

    def update(self, last_obs, targets=None):
        """
        Update parameters of the critic.

        For each update of the target value, perform many updates of the critic gradient.

        Note n is batch size being passed into network.

        Arguments:
            last_obs: shape: (n, obs_size)
            actions: shape: (n, action_size)
            obs: shape: (n, obs_size)
            reward: shape: (n,)
            terminal: shape: (n,)
        """

        if targets is not None:
            targets = numpy_to_torch(targets).detach()

            for _ in range(self.num_target_updates):
                v_s = self(numpy_to_torch(last_obs))
                v_target = targets

                loss = self.loss_critereon(v_s, v_target)

                # # Apply l2 regularization to loss function
                # for param in self.critic_network.parameters():
                #     loss += param.pow(2).sum() * self.l2_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.loss = loss
            return loss.item()

    def save(self, save_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = {"loss": self.loss}
        save_model_state_dict(self, save_dir, name, losses)

    def load(self, exp_dir, exp_name):
        name = f"{exp_name}_critic"
        losses = load_model_state_dict(self, exp_dir, name)
        self.loss = losses["loss"]

        if self.training:
            self.train()
        else:
            self.eval()


class DDPGCritic(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params):
        super().__init__()
        self.obs_size = params['obs_size']  # int
        self.act_size = params['num_actions']  # int
        self.hidden_size = params['hidden_size']
        self.num_hidden_layers = params['num_hidden_layers']
        self.learning_rate = params['critic_lr']
        self.training = params['training']

        # Critic parameters
        self.l2_reg = params['l2_reg']
        self.polyak = params['polyak']

        self.q_net = build_mlp(self.obs_size + self.act_size,
                               1,
                               self.hidden_size,
                               self.num_hidden_layers,
                               init_method=ddpg_init_method,
                               output_method=ddpg_output_method
                               )

        self.q_net_target = deepcopy(self.q_net)

        # Freeze target network w.r.t optimizers
        for p in self.q_net_target.parameters():
            p.requires_grad = False

        self.loss_critereon = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    self.learning_rate,
                                    weight_decay=self.l2_reg
                                    )
        self.loss = None
        self.device = get_device()
        self.to(self.device)

    def forward(self, obs, act):
        input = torch.cat([obs, act], dim=-1)
        q = self.q_net(input)
        return torch.squeeze(q, -1)

    def forward_target(self, obs, act):
        with torch.no_grad():
            input = torch.cat([obs, act], dim=-1)
            q = self.q_net_target(input)
            return torch.squeeze(q, -1)

    def update(self, obs_no, ac_na, targets=None, perform_backprop=True):
        """
        Update critic weights via DDPG update.

        Inputs are assumed to be tensors via the DDPGReplayBuffer.sample() method.

        Targets are to be tensors computed in DDPG agent.
        """

        q = self.forward(obs_no, ac_na)

        # Calculate the BSME loss via torch.MSE()
        loss = self.loss_critereon(q, targets)

        if perform_backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.loss = loss

        if perform_backprop:
            return loss.item()
        else:
            return loss

    def update_target_network(self):
        """
        Perform polyak update to target parameters
        """
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

    def save(self, save_dir, exp_name, suffix=""):
        name = f"{exp_name}_critic{suffix}"
        losses = {"loss": self.loss}
        save_model_state_dict(self, save_dir, name, losses)

    def load(self, exp_dir, exp_name, suffix=""):
        name = f"{exp_name}_critic{suffix}"
        losses = load_model_state_dict(self, exp_dir, name)
        self.loss = losses["loss"]

        if self.training:
            self.train()
            self.q_net_target.eval()
        else:
            self.eval()


class MLP(nn.Module):
    def __init__(self,
                 input_size: int, output_size: int, hidden_size: int, num_hidden_layers: int,
                 ):
        super(MLP, self).__init__()
        size_list = [input_size] + num_hidden_layers * [hidden_size] + [output_size]
        self.linear_list = nn.ModuleList(
            [nn.Linear(size_list[i], size_list[i + 1]) for i in range(num_hidden_layers + 1)])
        self.initialize_weights_bias()

    def initialize_weights_bias(self):
        """
        Initialize weights/biases of layers to samplings from a Normal distribution with Mean = 0, std = sqrt(2/out_features).
        """
        for layer in self.linear_list:
            if type(layer) == nn.Linear:
                std = np.sqrt(2 / layer.out_features)
                nn.init.normal_(layer.weight, 0, std)
                nn.init.normal_(layer.bias, 0, std)

    def forward(self, x: torch.tensor):
        """
        Forward pass through network with ReLu activation for hidden layers.
        """
        for i, l in enumerate(self.linear_list):
            if i != len(self.linear_list) - 1:
                x = F.relu(l(x))
            else:
                x = l(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, actor_output: int, critic_output: int,
                 hidden_size: int, num_hidden_layers: int, device: torch.device
                 ):
        super(ActorCritic, self).__init__()
        hidden_size_list = [input_size] + num_hidden_layers * [hidden_size]
        self.hidden_list = nn.ModuleList(
            [nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]) for i in range(num_hidden_layers)]
        )
        self.actor_head = nn.Linear(hidden_size, actor_output)
        self.critic_head = nn.Linear(hidden_size, critic_output)
        self.to(device)

    def forward(self, x: torch.tensor):
        for layer in self.hidden_list:
            x = F.relu(layer(x))

        action_prob = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)

        return action_prob, state_value

    def get_action(self, obs: torch.tensor):
        with torch.no_grad():
            probs, _ = self(obs)
            action_dist = Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
