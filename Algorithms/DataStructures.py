from collections import deque
from typing import NamedTuple, Optional, Union, List, Tuple

import numpy as np
import torch

from NetworkUtils import get_device, numpy_to_torch, list_to_numpy, list_to_torch
from mlagents_envs.base_env import ObservationSpec
from math import floor, ceil

from EncoderUtils import MemorySettings
from SegmentTree import SumSegmentTree, MinSegmentTree


def combined_shape(buffer_size: int, data_shape):
    """
    For a data_shape tuple, create a (buffer,shape) tuple
    """
    if np.isscalar(data_shape):
        return (buffer_size, data_shape)
    return (buffer_size, *data_shape)


def create_buffer(buffer_size: int, data_shape, includes_num_agents: bool = False):
    """
    Returns buffer with shape (B, *shape)
    """
    buffer_type = np.float32
    if len(data_shape) == 4 + includes_num_agents:
        buffer_type = np.int8
    buffer = np.zeros(combined_shape(buffer_size, data_shape), dtype=buffer_type)
    # print(f"Buffer with shape {buffer.shape} and type {buffer.dtype} requires {buffer.nbytes/1e6} MB of RAM.")
    return buffer


def create_obs_buffers(buffer_size: int, observation_specs: List[ObservationSpec], num_agents: int = 1):
    """
    Creates a list of observation buffers, one for each observation source.

    buffers = [array(buff,obs_1), array(buff,obs_2),...]
    """
    if len(observation_specs) == 0:
        return np.array([])

    if num_agents <= 1:
        buffers = [create_buffer(buffer_size, o.shape) for o in observation_specs]
    else:
        buffers = [create_buffer(buffer_size, (num_agents,) + o.shape, includes_num_agents=True) for o in
                   observation_specs]
    return buffers


def flatten_obs_buffer(buffer: List[np.ndarray], observation_specs):
    return [flatten_buffer(buffer[i], obs.shape) for i, obs in enumerate(observation_specs)]


def flatten_buffer(buffer: np.ndarray, original_shape=None):
    new_shape = (-1,) + original_shape if original_shape is not None else (-1,)
    return buffer.reshape(new_shape)


def discounted_return(rewards: np.array, gamma: float):
    """
    Takes a list of rewards [r_0,r_1,...,r_t',...,r_T] from a single trajectory

    Returns:
        A list where each index t' contains sum_{t'=0}^T gamma^t' * r_{t'}
    """

    discounted_returns = np.array([(gamma ** t) * rewards[t] for t in range(len(rewards))])
    total_return = np.sum(discounted_returns)
    discounted_returns_list = [total_return] * len(rewards)
    return discounted_returns_list


def discounted_cum_sum(rewards: np.array, gamma: float):
    """
    Takes a list of rewards [r_0,r_1,...,r_t',...,r_T] from a single trajectory.

    Use for reward to go.

    Returns:
        A list where each index t' contains sum_{t'=t}^T gamma^{t'-t} * r_{t'}
    """
    discounted_returns = np.array([(gamma ** t) * rewards[t] for t in range(len(rewards))])
    discounted_returns_list = np.array(
        [np.sum(discounted_returns[t::] / (gamma ** t)) for t in range(len(rewards))])
    return discounted_returns_list


class SumTree:
    def __init__(self, max_size: int):
        # levels is list of ndarrays for nodes on each level of tree.
        # 0th entry is top of tree. -1 are leaf nodes.
        # 0th entry by construction is sum of all leaf nodes.
        self.levels = [np.zeros(1)]
        self.max_priority = 1.0
        # construct empty tree, double nodes per level as binary until nodes >= max_size
        nodes_per_level = 1
        while nodes_per_level < max_size:
            nodes_per_level *= 2
            self.levels.append(np.zeros(nodes_per_level))

    def sample(self, batch_size: int):
        """
        Sample the tree based on priority.
        """
        total_priority = self.get_total_priority()
        priority_segment = total_priority / batch_size
        value = np.arange(batch_size) * priority_segment
        value += np.random.uniform(0, priority_segment, size=batch_size)
        idx = np.zeros(batch_size, dtype=np.int32)

        for level in self.levels[1:]:
            idx *= 2
            left_sum = level[idx]
            v_go_right = np.greater(value, left_sum)
            # if v > left_sum, go to right, else go to left
            idx += v_go_right
            # if going to right, remove left_sum from value as only want right part of tree.
            value -= left_sum * v_go_right
        return idx

    def set(self, indices, new_priorities):
        """
        Set the value of a many indices, ensure they are unique, and propagate through the sum tree.
        """
        indices, unique_indices = np.unique(indices, return_index=True)
        priority_diff = new_priorities[unique_indices] - self.levels[-1][indices]

        for level in self.levels[::-1]:
            np.add.at(level, indices, priority_diff)
            indices //= 2

    def set_inverse(self, ind, new_priority):
        max_ind_value = ind[-1]

        if len(ind) % 2 == 0:
            loop_counter = len(self.levels[::-1])

            for i in range(loop_counter):
                if i == 0:
                    self.levels[::-1][i][:len(new_priority)] = new_priority

                    max_ind_value //= 2

                else:
                    check_cond_1 = max_ind_value + 1

                    if i == 1:
                        len_priorities = len(new_priority)
                    else:
                        len_priorities = len(self.levels[::-1][i - 1][0:dummy])

                    if ceil(len_priorities / 2) == check_cond_1:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] = new_priority[0:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1] = self.levels[::-1][i - 1][0:dummy][0:len_priorities:2]
                    else:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] = new_priority[0:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] = self.levels[::-1][i - 1][0:dummy][
                                                                      0:len_priorities:2]

                    if floor(len_priorities / 2) == check_cond_1:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] += new_priority[1:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1] += self.levels[::-1][i - 1][0:dummy][1:len_priorities:2]
                    else:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] += new_priority[1:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] += self.levels[::-1][i - 1][0:dummy][
                                                                       1:len_priorities:2]

                    dummy = len_priorities // 2

                    if dummy == 1 or dummy == 0:
                        dummy = 2

                    max_ind_value //= 2

    def insert(self, ptr: int, new_ptr: int, max_size: int, split_index: int):
        indices = np.array(list(range(ptr, split_index)) + list(range(split_index % max_size, new_ptr)))
        values = np.ones_like(indices) * self.max_priority
        self.set(indices, values)

    def set_max_priority(self, priorities: np.ndarray):
        self.max_priority = max(self.max_priority, priorities.max())

    def update_priority(self, indices, priorities):
        self.set_max_priority(priorities)
        self.set(indices, priorities)

    def get_total_priority(self):
        return self.levels[0].item()

    def get_values(self, indices):
        return self.levels[-1][indices]


class LA3PReplayBuffer:
    def __init__(self, observation_specs: List[ObservationSpec], act_shape: int, max_size=100000, num_agents: int = 1,
                 rand_gen=None,
                 alpha: float = 0.4, beta_start: float = 0.4, beta_frames: int = 100000, use_cer=False):
        self.obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.next_obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.actions = create_buffer(buffer_size=max_size, data_shape=(act_shape,))
        self.rewards = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.terminals = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.policy_ages = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.rand_gen = rand_gen
        self.device = get_device()
        self.num_agents = num_agents

        self.alpha = alpha  # power for sampling probabilities. 0 means uniform selection
        self.frame = num_agents
        self.beta_start = beta_start  # importance sampling correction exponent
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.critic_tree = SumTree(max_size=max_size)
        self.actor_tree = SumTree(max_size=max_size)

        self.use_cer = use_cer
        self.current_policy_age = 0

    def update_beta(self, frame_index: int = 1):
        return min(1.0, self.beta_start + frame_index * (1.0 - self.beta_start) / self.beta_frames)

    def store(self,
              obs: List[np.ndarray],
              act: np.ndarray,
              reward: np.ndarray,
              next_obs: List[np.ndarray],
              terminal: np.ndarray):
        """
        Store experiences in the replay buffer from each agent this step.

        Ensures that when the buffer is full, the overflowing experiences wrap to the beginning of the buffer.
        """
        num_samples = act.shape[0]
        new_ptr = (self.ptr + num_samples) % self.max_size
        new_size = min(self.size + num_samples, self.max_size)
        # Index to split the buffer at
        split_index = self.max_size if new_ptr < self.ptr else new_ptr
        # index to split the inserted sample at
        item_split_index = self.max_size - self.ptr if new_ptr < self.ptr else num_samples

        for i in range(len(self.obs)):
            self.insert(self.obs[i], obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
            self.insert(self.next_obs[i], next_obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.actions, act, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.rewards, reward, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.terminals, terminal, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.policy_ages, np.ones_like(terminal) * self.current_policy_age, self.ptr, new_ptr,
                    self.max_size, split_index, item_split_index)

        # set priority of new samples to best stored so far.
        self.critic_tree.insert(self.ptr, new_ptr, self.max_size, split_index)

        self.ptr = new_ptr
        self.size = new_size

    def insert(self, buffer: np.ndarray, data: np.ndarray, ptr: int, new_ptr: int, max_size: int, split_index: int,
               item_split_index: int):
        """
        Inserts data into a given buffer and ensures data wraps around end of buffer.
        """
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        buffer[ptr:split_index] = data[:item_split_index]
        buffer[split_index % max_size:new_ptr] = data[item_split_index:]

    def uniform_sample(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        if self.rand_gen is not None:
            indices = self.rand_gen.choice(self.size, size=batch_size)
        else:
            indices = np.random.choice(self.size, size=batch_size)

        rewards = self.get_n_step_return(indices, n_step, gamma) if n_step > 1 else self.rewards[indices].flatten()

        return (
            [numpy_to_torch(o[indices], device=self.device) for o in self.obs],
            numpy_to_torch(self.actions[indices], device=self.device),
            numpy_to_torch(rewards, device=self.device),
            [numpy_to_torch(o[indices], device=self.device) for o in self.next_obs],
            numpy_to_torch(self.terminals[indices].flatten(), device=self.device),
            indices,
            None
        )

    def prioritized_sample(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        """
        Takes a prioritized sampling from the replay buffer.

        Returns a tuple of GPU tensors.
        """

        indices = self.critic_tree.sample(batch_size)

        rewards = self.get_n_step_return(indices, n_step, gamma) if n_step > 1 else self.rewards[indices].flatten()

        importance_sampling_weights = self.critic_tree.get_values(indices) ** (-self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()
        importance_sampling_weights = np.array(importance_sampling_weights, dtype=np.float32)

        self.beta = self.update_beta(self.frame)
        self.frame += self.num_agents

        return (
            [numpy_to_torch(o[indices], device=self.device) for o in self.obs],
            numpy_to_torch(self.actions[indices], device=self.device),
            numpy_to_torch(rewards, device=self.device),
            [numpy_to_torch(o[indices], device=self.device) for o in self.next_obs],
            numpy_to_torch(self.terminals[indices].flatten(), device=self.device),
            indices,
            numpy_to_torch(importance_sampling_weights.flatten(), device=self.device)
        )

    def sample_critic(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        return self.prioritized_sample(batch_size, n_step, gamma)

    def sample_actor(self, batch_size=32):
        # top_value = self.critic_tree.get_total_priority()
        top_value = self.critic_tree.max_priority
        reversed_indices = np.arange(self.ptr)
        reversed_priorities = top_value / (self.critic_tree.get_values(reversed_indices) + 1e-6)
        if self.ptr != 0:
            self.actor_tree.set_inverse(reversed_indices, reversed_priorities)
        indices = self.actor_tree.sample(batch_size)

        return (
            [numpy_to_torch(o[indices], device=self.device) for o in self.obs],
            numpy_to_torch(self.actions[indices], device=self.device),
            numpy_to_torch(self.rewards[indices].flatten(), device=self.device),
            [numpy_to_torch(o[indices], device=self.device) for o in self.next_obs],
            numpy_to_torch(self.terminals[indices].flatten(), device=self.device),
            indices,
            numpy_to_torch(reversed_priorities[indices].flatten(), device=self.device)
        )

    def update_priority(self, indices: np.ndarray, new_priorities: np.ndarray):
        self.critic_tree.update_priority(indices, new_priorities)

    def get_trajectory(self, agent_index: int = 0, trajectory_length: int = 0) -> List[torch.Tensor]:
        if trajectory_length <= 0 or trajectory_length > self.size:
            return []
        final_index = ((self.ptr + self.max_size) - self.num_agents + agent_index) % self.max_size
        start_index = (final_index - self.num_agents * (
                trajectory_length - 1) + self.num_agents * self.max_size) % self.max_size
        buffer_split_index = ((
                                      self.max_size - start_index) // self.num_agents) * self.num_agents + start_index + self.num_agents
        front_end_index = buffer_split_index if start_index > final_index else final_index
        back_start_index = final_index if start_index < final_index else front_end_index % self.max_size

        obs_front = [o[start_index:front_end_index:self.num_agents] for o in self.obs]
        obs_back = [o[back_start_index:final_index:self.num_agents] for o in self.obs]
        terminal_obs = [o[final_index].reshape(1, *o[final_index].shape) for o in self.next_obs]
        obs_traj = [np.concatenate((o1, o2, o3)) for o1, o2, o3 in zip(obs_front, obs_back, terminal_obs)]

        return list_to_torch(obs_traj, device=self.device)

    def get_n_step_return(self, sample_indices: np.ndarray, n: int = 1, gamma: float = 0.99):
        r = np.zeros(len(sample_indices), dtype=np.float32)

        for i in range(n):
            indices = (sample_indices - i * self.num_agents) % self.max_size
            r += (gamma ** i) * self.rewards[indices].flatten() * (1 - self.terminals[indices]).flatten()
        return r


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer based on Schaul 2016 "PRIORITIZED EXPERIENCE REPLAY".
    Prioritization follows "proportional" style.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_shape: int, max_size=100000, num_agents: int = 1,
                 rand_gen=None, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000, use_cer=False,
                 memory_settings: Optional[MemorySettings] = None):
        self.obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.next_obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.actions = create_buffer(buffer_size=max_size, data_shape=(act_shape,))
        self.rewards = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.terminals = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.episode_number = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.use_lstm = memory_settings is not None
        if self.use_lstm:
            self.memories_in = create_buffer(buffer_size=max_size,
                                             data_shape=(1, memory_settings.memory_size))
            self.memories_out = create_buffer(buffer_size=max_size,
                                              data_shape=(1, memory_settings.memory_size))
            self.sequence_length = memory_settings.sequence_length
        else:
            self.memories_in = None
            self.memories_out = None
            self.sequence_length = 1
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.rand_gen = rand_gen
        self.device = get_device()
        self.num_agents = num_agents
        self.current_episode_nums = np.arange(num_agents)

        self.alpha = alpha  # power for sampling probabilities. 0 means uniform selection
        self.frame = num_agents
        self.beta_start = beta_start  # importance sampling correction exponent
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.tree = SumTree(max_size=max_size)
        self.max_priority = 1.0

        self.use_cer = use_cer

    def update_beta(self, frame_index: int = 1):
        return min(1.0, self.beta_start + frame_index * (1.0 - self.beta_start) / self.beta_frames)

    def store(self,
              obs: List[np.ndarray],
              act: np.ndarray,
              reward: np.ndarray,
              next_obs: List[np.ndarray],
              terminal: np.ndarray,
              memories_in: Optional[np.ndarray] = None,
              memories_out: Optional[np.ndarray] = None
              ):
        """
        Store experiences in the replay buffer from each agent this step.

        Ensures that when the buffer is full, the overflowing experiences wrap to the beginning of the buffer.
        """
        num_samples = act.shape[0]
        new_ptr = (self.ptr + num_samples) % self.max_size
        new_size = min(self.size + num_samples, self.max_size)
        # Index to split the buffer at
        split_index = self.max_size if new_ptr < self.ptr else new_ptr
        # index to split the inserted sample at
        item_split_index = self.max_size - self.ptr if new_ptr < self.ptr else num_samples

        for i in range(len(self.obs)):
            self.insert(self.obs[i], obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
            self.insert(self.next_obs[i], next_obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.actions, act, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.rewards, reward, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.terminals, terminal, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.episode_number, self.current_episode_nums, self.ptr, new_ptr, self.max_size, split_index,
                    item_split_index)

        if self.use_lstm:
            self.insert(self.memories_in, memories_in, self.ptr, new_ptr, self.max_size, split_index,
                        item_split_index)
            self.insert(self.memories_out, memories_out, self.ptr, new_ptr, self.max_size, split_index,
                        item_split_index)

        # set priority of new samples to best stored so far.
        self.tree.insert(self.ptr, new_ptr, self.max_size, split_index)

        self.ptr = new_ptr
        self.size = new_size

    def insert(self, buffer: np.ndarray, data: np.ndarray, ptr: int, new_ptr: int, max_size: int, split_index: int,
               item_split_index: int):
        """
        Inserts data into a given buffer and ensures data wraps around end of buffer.
        """
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        buffer[ptr:split_index] = data[:item_split_index]
        buffer[split_index % max_size:new_ptr] = data[item_split_index:]

    def prioritized_sample(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        """
        Takes a prioritized sampling from the replay buffer.

        Returns a tuple of GPU tensors.
        """

        indices = self.tree.sample(batch_size)

        if self.use_cer:
            start_idx = (self.ptr - self.num_agents) % self.max_size
            end_idx = self.ptr
            new_indices = list(range(start_idx, self.max_size)) + list(range(0, end_idx)) if start_idx > end_idx \
                else list(range(start_idx, end_idx))
            indices = np.array(list(indices) + new_indices)

        if self.use_lstm:
            # As LSTM use N sequences of exactly L length, find sequences of that size from the random index buffer.
            start_indices = indices
            indices, sequence_lengths = self.get_sequence_indices_lengths(start_indices, self.sequence_length)

            # Obtain the indices of the first step of the sequence.
            mem_in = self.memories_in[start_indices]
            mem_out = self.memories_out[start_indices]
        else:
            mem_in = None
            mem_out = None
            sequence_lengths = None

        obs = [o[indices] for o in self.obs]
        actions = self.actions[indices]
        if n_step > 1:
            highest_n, rewards, next_obs, terminals = self.n_step_bootstrapping(indices, n_step, gamma)
        else:
            next_obs = [next_o[indices] for next_o in self.next_obs]
            rewards = self.rewards[indices].flatten()
            terminals = self.terminals[indices].flatten()
            highest_n = np.ones(len(indices))

        importance_sampling_weights = self.tree.get_values(indices) ** (-self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()
        importance_sampling_weights = np.array(importance_sampling_weights, dtype=np.float32)

        self.beta = self.update_beta(self.frame)
        self.frame += self.num_agents

        return (
            list_to_torch(obs, False, device=self.device),
            numpy_to_torch(actions, False, device=self.device),
            numpy_to_torch(rewards, False, device=self.device),
            list_to_torch(next_obs, False, device=self.device),
            numpy_to_torch(terminals, False, device=self.device),
            numpy_to_torch(mem_in, False, device=self.device) if self.use_lstm else None,
            numpy_to_torch(mem_out, False, device=self.device) if self.use_lstm else None,
            numpy_to_torch(sequence_lengths, False, device=torch.device("cpu")) if self.use_lstm else None,
            indices,
            numpy_to_torch(importance_sampling_weights, False, device=self.device),
            numpy_to_torch(highest_n, False, device=self.device)
        )

    def uniform_sample(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        """
                Uniform random sampling from the replay buffer.

                Returns a tuple of GPU tensors.
                """
        if self.rand_gen is not None:
            indices = self.rand_gen.choice(self.size, size=batch_size)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)

        if self.use_cer:
            # CER appends the last batch of experiences to the minibatch
            # Used to limit negative effect of very large replay buffer
            # Makes learning more on-policy by favoring recent transitions
            start_idx = (self.ptr - self.num_agents) % self.max_size
            end_idx = self.ptr
            new_indices = list(range(start_idx, self.max_size)) + list(range(0, end_idx)) if start_idx > end_idx \
                else list(range(start_idx, end_idx))
            indices = np.array(list(indices) + new_indices)

        if self.use_lstm:
            # As LSTM use N sequences of exactly L length, find sequences of that size from the random index buffer.
            start_indices = indices
            indices, sequence_lengths = self.get_sequence_indices_lengths(start_indices, self.sequence_length)

            # Obtain the indices of the first step of the sequence.
            mem_in = self.memories_in[start_indices] if self.use_lstm else None
            mem_out = self.memories_out[start_indices] if self.use_lstm else None
        else:
            mem_in = None
            mem_out = None
            sequence_lengths = None

        obs = [o[indices] for o in self.obs]
        actions = self.actions[indices]
        weights = np.ones(len(indices)) / self.size

        if n_step > 1:
            highest_n, rewards, next_obs, terminals = self.n_step_bootstrapping(indices, n_step, gamma)
        else:
            next_obs = [next_o[indices] for next_o in self.next_obs]
            rewards = self.rewards[indices].flatten()
            terminals = self.terminals[indices].flatten()
            highest_n = np.ones(len(indices))

        return (
            list_to_torch(obs, False, device=self.device),
            numpy_to_torch(actions, False, device=self.device),
            numpy_to_torch(rewards, False, device=self.device),
            list_to_torch(next_obs, False, device=self.device),
            numpy_to_torch(terminals, False, device=self.device),
            numpy_to_torch(mem_in, False, device=self.device) if self.use_lstm else None,
            numpy_to_torch(mem_out, False, device=self.device) if self.use_lstm else None,
            numpy_to_torch(sequence_lengths, False, device=torch.device("cpu")) if self.use_lstm else None,
            indices,
            numpy_to_torch(weights, False, device=self.device),
            numpy_to_torch(highest_n, False, device=self.device)
        )

    def update_priority(self, indices: np.ndarray, new_priorities: np.ndarray):
        self.tree.update_priority(indices, new_priorities)

    def get_sequence_indices_lengths(self, indices: np.ndarray, max_sequence_len: int = 1):
        indices_list = []  # should be length indices*max_seq_len
        sequence_len_list = []  # should be length indices

        for i in indices:
            ep_num = self.episode_number[i]
            seq_len = 0
            for j in range(max_sequence_len):
                idx = i + j * self.num_agents
                if idx >= self.size and self.max_size != self.size:
                    idx = 0
                else:
                    idx = idx % self.max_size
                    seq_ep_num = self.episode_number[idx]
                    if ep_num != seq_ep_num:
                        idx = 0
                    else:
                        seq_len += 1
                indices_list.append(idx)
            sequence_len_list.append(seq_len)

        return np.array(indices_list), np.array(sequence_len_list)

    def get_trajectory(self, agent_index: int = 0, trajectory_length: int = 0) -> List[torch.Tensor]:
        if trajectory_length <= 0 or trajectory_length > self.size:
            return []
        final_index = ((self.ptr + self.max_size) - self.num_agents + agent_index) % self.max_size
        start_index = (final_index - self.num_agents * (
                trajectory_length - 1) + self.num_agents * self.max_size) % self.max_size
        buffer_split_index = ((
                                      self.max_size - start_index) // self.num_agents) * self.num_agents + start_index + self.num_agents
        front_end_index = buffer_split_index if start_index > final_index else final_index
        back_start_index = final_index if start_index < final_index else front_end_index % self.max_size

        obs_front = [o[start_index:front_end_index:self.num_agents] for o in self.obs]
        obs_back = [o[back_start_index:final_index:self.num_agents] for o in self.obs]
        terminal_obs = [o[final_index].reshape(1, *o[final_index].shape) for o in self.next_obs]
        obs_traj = [np.concatenate((o1, o2, o3)) for o1, o2, o3 in zip(obs_front, obs_back, terminal_obs)]

        return list_to_torch(obs_traj, device=self.device)

    def get_n_step_return(self, sample_indices: np.ndarray, n: int = 1, gamma: float = 0.99):
        r = np.zeros(len(sample_indices), dtype=np.float32)

        for i in range(n):
            indices = (sample_indices - i * self.num_agents) % self.max_size
            r += (gamma ** i) * self.rewards[indices].flatten() * (1 - self.terminals[indices]).flatten()
        return r

    def n_step_bootstrapping(self, sample_indices: np.ndarray, n: int = 1, gamma: float = 0.99):
        """
        Calculates n-step return.

        Outputs return and n-step observation for TD-error.
        Highest n is for Q-value weighting to varying with highest n-step reached for that sample. Ranges from 1-n.

        Should step forward upto n-steps from the provided indices.

        Prevent further steps if either:
        1. Next step is not in initial episode
        2. A terminal state has been reached
        """
        n_step_r = np.zeros(len(sample_indices), dtype=np.float32)
        current_indices = sample_indices.copy()
        initial_ep_num = self.episode_number[sample_indices].flatten()
        has_not_terminated = np.ones_like(n_step_r) > 0
        highest_n = np.zeros_like(n_step_r)

        for i in range(n):
            # consider ith step of n-step for each sample (loops around end of buffer)
            ith_indices = (sample_indices + i * self.num_agents) % self.max_size
            episode_num = self.episode_number[ith_indices].flatten()
            # is step in same episode?
            is_same_ep = episode_num == initial_ep_num
            # is n-step valid? Same ep and not already past terminal?
            is_valid_n_step = np.logical_and(is_same_ep, has_not_terminated)
            is_new_state_terminal = self.terminals[ith_indices].flatten() > 0
            is_valid_terminal_step = np.logical_and(is_valid_n_step, is_new_state_terminal)
            # update current index if both same ep
            current_indices[is_valid_n_step] = ith_indices[is_valid_n_step]
            # store which samples are now terminal for valid n-step checking
            has_not_terminated[is_valid_terminal_step] = False
            # n-step return : R = R_1 + g*R_2 + g**2*R_3 + ... + g**(n-1)*R_n
            n_step_r += (gamma ** i) * self.rewards[current_indices].flatten() * is_valid_n_step
            highest_n[is_valid_n_step] += 1

        return highest_n, n_step_r, [next_o[current_indices] for next_o in self.next_obs], self.terminals[
            current_indices].flatten()


class LSTMReplayBuffer:
    """
    A replay buffer that stores sequences of size sequence_length. Sequences do not extend over terminal states / between trajectories.
    Batch size in this case is the number of sampled trajectories (mini-batch size = batch_size * sequence_length).
    Best to store recurrent state AND use burn-in period (Kapturowski 2019). They also use PER, but we opt not to use it here.

    Idea: Sample a sequence, use the recurrent state of the first transition. Then update recurrent state by the next n burn-in states.
    From this point, actually calculate the Q-value.
    """

    def __init__(self, act_shape: int = 1, max_size=100000, num_agents: int = 1,
                 rand_gen=None, use_cer=False, memory_settings: MemorySettings = None):

        # Handle LSTM settings
        assert memory_settings is not None
        self.burn_in = max(memory_settings.burn_in, 0)
        self.overlap_length = max(memory_settings.overlap_length, 0)
        self.sequence_length = max(memory_settings.sequence_length, 0) + self.burn_in
        self.memory_size = memory_settings.memory_size

        # How many sequences to store
        self.capacity: int = int(np.ceil(max_size / self.sequence_length))
        self.buffer: deque = deque(maxlen=self.capacity)

        self.num_agents: int = num_agents
        self.sequences = [[] for _ in range(self.num_agents)]

        self.current_trajectory_observations = [[] for _ in range(self.num_agents)]
        self.complete_trajectory_observations = []

        self.num_actions = act_shape

        self.rand_gen = rand_gen
        self.device = get_device()

    def store(self,
              obs: List[np.ndarray],
              act: np.ndarray,
              reward: np.ndarray,
              next_obs: List[np.ndarray],
              terminal: np.ndarray,
              hidden_in: torch.Tensor,
              hidden_out: torch.Tensor
              ):
        """
        Add individual transition to the current trajectories list.
        If sequence is complete or terminal state reached, then add the sequence to the buffer.
        If sequence is complete, but not terminal, copy last burn-in transitions to the next sequence for overlap.
        If sequence is terminal, fill with empty experiences until complete.
        """
        assert act.shape[0] == self.num_agents

        for i in range(self.num_agents):
            # create transition tuple and add to current trajectory
            transition = (
                [o[i] for o in obs], act[i], reward[i], [o[i] for o in next_obs], terminal[i],
                hidden_in[:, i, :], hidden_out[:, i, :]
            )
            self.current_trajectory_observations[i].append(transition[0])
            self.sequences[i].append(transition)

            is_terminal = terminal[i] > 0

            # Fill sequence with dummy info if terminal
            if is_terminal:
                while len(self.sequences[i]) < self.sequence_length:
                    self.sequences[i].append(self.make_empty_experience(transition))
                self.complete_trajectory_observations.append(self.current_trajectory_observations[i])
                self.current_trajectory_observations[i] = []

            # if sequence complete, add to buffer
            if len(self.sequences[i]) == self.sequence_length:
                # Note: Deque removes oldest element when appending new one if full
                self.buffer.append(self.sequences[i])

                if is_terminal or self.overlap_length <= 0:
                    self.sequences[i] = []
                else:
                    self.sequences[i] = self.sequences[i][-self.overlap_length:]

    def make_empty_experience(self, transition):
        """
        Empty experience for post-terminal transition to pad sequence.
        Terminal state is set to 1 to prevent unnecessary computation for bellman return.
        """
        o = [np.zeros_like(obs, dtype=np.float32) for obs in transition[0]]
        a = np.zeros_like(transition[1], dtype=np.float32)
        r = np.zeros_like(transition[2], dtype=np.float32)
        next_o = [np.zeros_like(obs, dtype=np.float32) for obs in transition[3]]
        t = np.ones_like(transition[4], dtype=np.float32)
        h_in = torch.zeros_like(transition[5], dtype=torch.float32)
        h_out = torch.zeros_like(transition[6], dtype=torch.float32)
        return o, a, r, next_o, t, h_in, h_out

    def sample(self, batch_size=32):
        """
        Randomly sample a stored sequence from the buffer. Each minibatch consists of N sequences of M length.
        Batch contains (obs, action, rew, next_obs, terminals).
        Also returns tuple(mem_in, mem_out) for burn-in batch (i=0) and learn batch (i = burn-in)
        """

        # batch_size = int(np.ceil(batch_size / self.sequence_length))

        # Sample buffer for batch of N sequences
        if self.rand_gen is not None:
            indices = self.rand_gen.choice(len(self.buffer), size=batch_size)
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size)
        weights = np.ones((len(indices), 1), dtype=np.float32)

        batch = [self.buffer[i] for i in indices]

        # batch is a list of lists.
        # Each sub-list is a list of experience tuples.
        # Need to extract each tuple part into own list for tensor.
        # Input tensor shape (batch_size, sequence_length, feature_shape)
        # For observation, it is a List of Tensors. Each tensor is (batch, sequence, feature)

        o_list = [[] for _ in batch[0][0][0]]
        n_o_list = [[] for _ in batch[0][0][0]]
        a_list, r_list, t_list = [], [], []
        burn_h_in_list, burn_h_out_list = [], []
        learn_h_in_list, learn_h_out_list = [], []

        for sequence in batch:
            o_sublist, n_o_sublist = [[] for _ in o_list], [[] for _ in n_o_list]
            a_sublist, r_sublist, t_sublist = [], [], []

            for i, transition in enumerate(sequence):
                o, a, r, n_o, t, h_in, h_out = transition
                for j in range(len(o_sublist)):
                    o_sublist[j].append(o[j])
                    n_o_sublist[j].append(n_o[j])
                a_sublist.append(a)
                r_sublist.append(r)
                t_sublist.append(t)

                if i == 0:
                    burn_h_in_list.append(h_in)
                    burn_h_out_list.append(h_out)

                if i == self.burn_in:
                    learn_h_in_list.append(h_in)
                    learn_h_out_list.append(h_out)

            for i in range(len(o_list)):
                o_list[i].append(o_sublist[i])
                n_o_list[i].append(n_o_sublist[i])
            a_list.append(a_sublist)
            r_list.append(r_sublist)
            t_list.append(t_sublist)

        batch = (list_to_torch(o_list, False, self.device),
                 torch.tensor(a_list, dtype=torch.float32, requires_grad=False, device=self.device),
                 torch.tensor(r_list, dtype=torch.float32, requires_grad=False, device=self.device),
                 list_to_torch(n_o_list, False, self.device),
                 torch.tensor(t_list, dtype=torch.float32, requires_grad=False, device=self.device))

        burn_in_memories = (
            torch.stack(burn_h_in_list, dim=1).float().requires_grad_(False).to(self.device),
            torch.stack(burn_h_out_list, dim=1).float().requires_grad_(False).to(self.device),
        )

        learn_memories = (
            torch.stack(learn_h_in_list, dim=1).float().requires_grad_(False).to(self.device),
            torch.stack(learn_h_out_list, dim=1).float().requires_grad_(False).to(self.device),
        )

        weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False, device=self.device)

        burn_batch, learn_batch = self.split_minibatch_by_burn_in(batch)
        return burn_batch, learn_batch, burn_in_memories, learn_memories, indices, weights

    def split_minibatch_by_burn_in(self, batch):
        """
        Splits batch tuple into two tuples: Burn-in tuple for initializing memory state
        and learn tuple for updating recurrent state.
        """

        if self.burn_in <= 0:
            return None, batch

        burn_in_batch = []
        learn_batch = []

        burn_indices = torch.arange(self.burn_in).to(self.device)
        learn_indices = torch.arange(self.burn_in, self.sequence_length).to(self.device)

        for element in batch:
            # Handle observation lists
            if type(element) is list:
                burn_in_batch.append([o.index_select(1, burn_indices) for o in element])
                learn_batch.append([o.index_select(1, learn_indices) for o in element])
            else:
                burn_in_batch.append(element.index_select(1, burn_indices))
                learn_batch.append(element.index_select(1, learn_indices))

        return tuple(burn_in_batch), tuple(learn_batch)

    def get_complete_trajectory_observations(self, num_transitions: int = -1):
        """
        Obtains a list of complete trajectory observations. Output has form List[Obs] where Obs is List[Tensor]. Each tensor of shape (Traj_Length, Obs_shape).
        """
        num_transitions = max(num_transitions, -1)
        # Convert trajectory of transitions into observation list with shape (traj_length, obs_shape)
        # trajectory_observations = [[np.stack(y, axis=0)[:num_transitions, :] for y in zip(*[x for x in zip(*traj)][0])] for
        #                            traj in self.complete_trajectory_observations]
        traj_obs = [[np.stack(x, axis=0) for x in [y for y in zip(*traj)]] for traj in self.complete_trajectory_observations]
        self.complete_trajectory_observations = []
        # Concatenate observations from each sequence together for shape (num_comp_traj*traj_length, obs_shape)
        # trajectory_observations = [np.concatenate(x, axis=0) for x in zip(*traj_obs)]
        return [list_to_torch(obs, False, self.device) for obs in traj_obs]
        # return list_to_torch(trajectory_observations, False, self.device)

    def get_complete_trajectory_observations_old(self, num_transitions: int = -1):
        num_transitions = max(num_transitions, -1)
        trajectory_observations = []
        for traj in self.complete_trajectory_observations:
            z = zip(*traj)
            trajectory_observations.append([xs for xs in z][0])
            trajectory_observations[-1] = trajectory_observations[-1][:num_transitions]
        self.complete_trajectory_observations = []
        return list_to_torch(trajectory_observations, False, self.device)

    def dummy_transition(self):
        o = [np.arange(self.num_agents*9*9*9).reshape((self.num_agents, 9, 9, 9, 1)), np.arange(self.num_agents*9*9*1).reshape((self.num_agents, 9, 9, 1)), np.arange(self.num_agents*18).reshape((self.num_agents, 18))]
        # o = [np.arange(self.num_agents * 4).reshape((self.num_agents, 4))]
        a = np.random.rand(self.num_agents, self.num_actions)
        r = np.random.rand(self.num_agents, 1)
        t = np.zeros((self.num_agents, 1))
        h = torch.rand([1, self.num_agents, self.memory_size]).float()
        return o, a, r, o, t, h, h

    def test(self):
        for i in range(3 * self.sequence_length):
            self.store(*self.dummy_transition())
        term_transition = self.dummy_transition()
        term_transition[4][0] = 1
        self.store(*term_transition)

        for i in range(self.sequence_length):
            self.store(*self.dummy_transition())

        print(f"Buffer Size: {len(self.buffer)}")
        print(self.sample(len(self.buffer) // 2))

        for i, seq in enumerate(self.buffer):
            print(f"\nSequence {i}: {seq}")


class PrioritizedLSTMReplayBuffer:
    """
    Structurally the same as the LSTMReplayBuffer but utilizes Prioritized Experience Replay to sample buffer.
    Each sequence will have a priority, with higher priority increasing sampling probability.
    Newly added sequences have a priority of infinite to ensure they are sampled at least once.

    Initial implementation will use a standard dequeue for priority storage, but then will update to SumTree.
    Initial implementation based on Deep RL Zoo, Sum Tree based on StableBaseline3.
    """

    def __init__(self, act_shape: int = 1, max_size=100000, num_agents: int = 1,
                 rand_gen=None, use_cer=False, memory_settings: MemorySettings = None,
                 priority_exponent: float = 0, importance_sampling_exponent: float = 0,
                 use_sum_tree: bool = False):

        # Handle LSTM settings
        assert memory_settings is not None
        self.burn_in = max(memory_settings.burn_in, 0)
        self.overlap_length = max(memory_settings.overlap_length, 0)
        self.sequence_length = max(memory_settings.sequence_length, 0) + self.burn_in
        self.memory_size = memory_settings.memory_size

        # How many sequences to store
        self.capacity: int = int(np.ceil(max_size / self.sequence_length))
        self.buffer: deque = deque(maxlen=self.capacity)

        self.num_agents: int = num_agents
        self.sequences = [[] for _ in range(self.num_agents)]

        self.current_trajectories = [[] for _ in range(self.num_agents)]
        self.complete_trajectories = []

        self.num_actions = act_shape

        self.rand_gen = rand_gen
        self.device = get_device()

        # Handle PER settings
        self.priorities: deque = deque(maxlen=self.capacity)
        self.per_alpha: float = priority_exponent
        self.per_beta: float = importance_sampling_exponent
        self.per_beta_0: float = importance_sampling_exponent
        self.max_priority: float = 1.0

        ## Trees
        self.use_sum_tree = use_sum_tree
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.buffer_index: int = 0

    def store(self,
              obs: List[np.ndarray],
              act: np.ndarray,
              reward: np.ndarray,
              next_obs: List[np.ndarray],
              terminal: np.ndarray,
              hidden_in: torch.Tensor,
              hidden_out: torch.Tensor,
              ):
        """
        Add individual transition to the current trajectories list.
        If sequence is complete or terminal state reached, then add the sequence to the buffer.
        If sequence is complete, but not terminal, copy last burn-in transitions to the next sequence for overlap.
        If sequence is terminal, fill with empty experiences until complete.
        """
        assert act.shape[0] == self.num_agents

        for i in range(self.num_agents):
            # create transition tuple and add to current trajectory
            transition = (
                [o[i] for o in obs], act[i], reward[i], [o[i] for o in next_obs], terminal[i],
                hidden_in[:, i, :], hidden_out[:, i, :]
            )
            self.current_trajectories[i].append(transition)
            self.sequences[i].append(transition)

            is_terminal = terminal[i] > 0

            # Fill sequence with dummy info if terminal
            if is_terminal:
                while len(self.sequences[i]) < self.sequence_length:
                    self.sequences[i].append(self.make_empty_experience(transition))
                self.complete_trajectories.append(self.current_trajectories[i])
                self.current_trajectories[i] = []

            # if sequence complete, add to buffer
            if len(self.sequences[i]) == self.sequence_length:
                # Note: Deque removes oldest element when appending new one if full
                self.buffer.append(self.sequences[i])

                # new transition priority is max to ensure sampled at least once
                if self.use_sum_tree:
                    # insert priority term into tree and update buffer index.
                    self.sum_tree[self.buffer_index] = self.max_priority ** self.per_alpha
                    self.min_tree[self.buffer_index] = self.max_priority ** self.per_alpha
                    self.buffer_index = (self.buffer_index + 1) % self.capacity
                else:
                    self.priorities.append(self.max_priority ** self.per_alpha)

                if is_terminal or self.overlap_length <= 0:
                    self.sequences[i] = []
                else:
                    self.sequences[i] = self.sequences[i][-self.overlap_length:]

    def make_empty_experience(self, transition):
        """
        Empty experience for post-terminal transition to pad sequence.
        Terminal state is set to 1 to prevent unnecessary computation for bellman return.
        """
        o = [np.zeros_like(obs, dtype=np.float32) for obs in transition[0]]
        a = np.zeros_like(transition[1], dtype=np.float32)
        r = np.zeros_like(transition[2], dtype=np.float32)
        next_o = [np.zeros_like(obs, dtype=np.float32) for obs in transition[3]]
        t = np.ones_like(transition[4], dtype=np.float32)
        h_in = torch.zeros_like(transition[5], dtype=torch.float32)
        h_out = torch.zeros_like(transition[6], dtype=torch.float32)
        return o, a, r, next_o, t, h_in, h_out

    def get_prioritized_indices_and_weights(self, batch_size: int):
        if self.use_sum_tree:
            # tree method as in StableBaselines
            indices = []
            p_total = self.sum_tree.sum(0, len(self.buffer) - 1)
            range_len = p_total / batch_size

            for i in range(batch_size):
                mass = (np.random.rand(1) + i) * range_len
                indices.append(self.sum_tree.find_prefix_sum_idx(mass))
            p_min = self.min_tree.min() / self.sum_tree.sum()
            max_weight = (p_min * len(self.buffer)) ** (-self.per_beta) + 1e-8

            weights = []
            for idx in indices:
                p_sample = self.sum_tree[idx] / self.sum_tree.sum()
                weight = (p_sample * len(self.buffer)) ** (-self.per_beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            # No Tree method as in DeepRL R2D2 implementation
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()

            if self.rand_gen is not None:
                indices = self.rand_gen.choice(len(self.buffer), size=batch_size, replace=True, p=probs)
            else:
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=True, p=probs)

            weights = (1.0 / (len(self.buffer) * np.take(probs, indices))) ** self.per_beta
            weights /= np.max(weights) + 1e-8

        return indices, weights

    def sample(self, batch_size=32):
        """
        Randomly sample a stored sequence from the buffer. Each minibatch consists of N sequences of M length.
        Batch contains (obs, action, rew, next_obs, terminals).
        Also returns tuple(mem_in, mem_out) for burn-in batch (i=0) and learn batch (i = burn-in)
        """

        # batch_size = int(np.ceil(batch_size / self.sequence_length))

        if self.per_alpha == 0:
            if self.rand_gen is not None:
                indices = self.rand_gen.choice(len(self.buffer), size=batch_size)
            else:
                indices = np.random.choice(len(self.buffer), size=batch_size)
            importance_weights = np.ones_like(indices, dtype=np.float32)
        else:

            # Large priority = large change in NN weights.
            # Large prio = Large Prob of sample = Low weight to constrain rapid change in weights.
            indices, importance_weights = self.get_prioritized_indices_and_weights(batch_size)

        batch = [self.buffer[i] for i in indices]

        # batch is a list of lists.
        # Each sub-list is a list of experience tuples.
        # Need to extract each tuple part into own list for tensor.
        # Input tensor shape (batch_size, sequence_length, feature_shape)
        # For observation, it is a List of Tensors. Each tensor is (batch, sequence, feature)

        o_list = [[] for _ in batch[0][0][0]]
        n_o_list = [[] for _ in batch[0][0][0]]
        a_list, r_list, t_list = [], [], []
        burn_h_in_list, burn_h_out_list = [], []
        learn_h_in_list, learn_h_out_list = [], []

        for sequence in batch:
            o_sublist, n_o_sublist = [[] for _ in o_list], [[] for _ in n_o_list]
            a_sublist, r_sublist, t_sublist = [], [], []

            for i, transition in enumerate(sequence):
                o, a, r, n_o, t, h_in, h_out = transition
                for j in range(len(o_sublist)):
                    o_sublist[j].append(o[j])
                    n_o_sublist[j].append(n_o[j])
                a_sublist.append(a)
                r_sublist.append(r)
                t_sublist.append(t)

                if i == 0:
                    burn_h_in_list.append(h_in)
                    burn_h_out_list.append(h_out)

                if i == self.burn_in:
                    learn_h_in_list.append(h_in)
                    learn_h_out_list.append(h_out)

            for i in range(len(o_list)):
                o_list[i].append(o_sublist[i])
                n_o_list[i].append(n_o_sublist[i])
            a_list.append(a_sublist)
            r_list.append(r_sublist)
            t_list.append(t_sublist)

        batch = (list_to_torch(o_list, False, self.device),
                 torch.tensor(a_list, dtype=torch.float32, requires_grad=False, device=self.device),
                 torch.tensor(r_list, dtype=torch.float32, requires_grad=False, device=self.device),
                 list_to_torch(n_o_list, False, self.device),
                 torch.tensor(t_list, dtype=torch.float32, requires_grad=False, device=self.device))

        burn_in_memories = (
            torch.stack(burn_h_in_list, dim=1).float().requires_grad_(False).to(self.device),
            torch.stack(burn_h_out_list, dim=1).float().requires_grad_(False).to(self.device),
        )

        learn_memories = (
            torch.stack(learn_h_in_list, dim=1).float().requires_grad_(False).to(self.device),
            torch.stack(learn_h_out_list, dim=1).float().requires_grad_(False).to(self.device),
        )

        weights = torch.tensor(importance_weights.reshape((-1, 1)), dtype=torch.float32, requires_grad=False,
                               device=self.device)

        burn_batch, learn_batch = self.split_minibatch_by_burn_in(batch)
        return burn_batch, learn_batch, burn_in_memories, learn_memories, indices, weights

    def update_priorities(self, indices, priorities):
        if priorities is None:
            return
        assert len(indices) == len(priorities)
        priorities = np.asarray(priorities.detach().cpu())
        priorities = np.abs(priorities)
        self.max_priority = np.max([self.max_priority, np.max(priorities)])

        for idx, p in zip(indices, priorities):
            assert p > 0
            assert 0 <= idx <= len(self.buffer)
            if self.use_sum_tree:
                self.sum_tree[idx] = p ** self.per_alpha
                self.min_tree[idx] = p ** self.per_alpha
            else:
                self.priorities[idx] = p ** self.per_alpha

    def update_importance_sampling_exponent(self, current_training_step: int, final_training_step: int):
        self.per_beta = self.per_beta_0 + (1.0 - self.per_beta_0) * float(current_training_step) / final_training_step

    def print_stats(self):
        array = np.array(self.priorities)
        print(
            f"\nReplay Buffer Priority Stats: Min: {array.min():.3e}, Max: {array.max():.3e}, Avg: {array.mean():.3e}, Max Seen: {self.max_priority:.3e}. Importance Expo: {self.per_beta}")

    def split_minibatch_by_burn_in(self, batch):
        """
        Splits batch tuple into two tuples: Burn-in tuple for initializing memory state
        and learn tuple for updating recurrent state.
        """

        if self.burn_in <= 0:
            return None, batch

        burn_in_batch = []
        learn_batch = []

        b_in_indices = torch.arange(self.burn_in).to(self.device)
        learn_indices = torch.arange(self.burn_in, self.sequence_length).to(self.device)

        for element in batch:
            # Handle observation lists
            if type(element) is list:
                burn_in_batch.append([o.index_select(1, b_in_indices) for o in element])
                learn_batch.append([o.index_select(1, learn_indices) for o in element])
            else:
                burn_in_batch.append(element.index_select(1, b_in_indices))
                learn_batch.append(element.index_select(1, learn_indices))

        return tuple(burn_in_batch), tuple(learn_batch)

    def get_complete_trajectory_observations(self):
        # Convert trajectory of transitions into observation list with shape (traj_length, obs_shape)
        trajectory_observations = [[np.stack(y, axis=0) for y in zip(*[x for x in zip(*traj)][0])]
                                   for traj in self.complete_trajectories]
        self.complete_trajectories = []
        # Concatenate observations from each sequence together for shape (num_comp_traj*traj_length, obs_shape)
        # trajectory_observations = [np.concatenate(x, axis=0) for x in zip(*trajectory_observations)]
        return [list_to_torch(obs, False, self.device) for obs in trajectory_observations]

    def dummy_transition(self):
        o = [np.arange(12).reshape(self.num_agents, -1), np.arange(12, 15).reshape(self.num_agents, -1)]
        a = np.random.rand(self.num_agents, self.num_actions)
        r = np.random.rand(self.num_agents, 1)
        t = np.zeros((self.num_agents, 1))
        h = torch.rand([1, self.num_agents, self.memory_size]).float()
        return o, a, r, o, t, h, h

    def test(self):
        for i in range(3 * self.sequence_length):
            self.store(*self.dummy_transition())
        term_transition = self.dummy_transition()
        term_transition[4][0] = 1
        self.store(*term_transition)

        for i in range(self.sequence_length):
            self.store(*self.dummy_transition())

        print(f"Buffer Size: {len(self.buffer)}")
        print(self.sample(len(self.buffer) // 2))

        for i, seq in enumerate(self.buffer):
            print(f"\nSequence {i}: {seq}")


class ReplayBuffer:
    """
    A basic FIFO replay buffer. Stores SARST experiences one at a time.

    Note: size is the current size of the buffer or how many experiences have been stored so far.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_shape: int, max_size=100000, num_agents: int = 1,
                 rand_gen=None, use_cer=False,
                 memory_settings: Optional[MemorySettings] = None):
        self.obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.next_obs = create_obs_buffers(buffer_size=max_size, observation_specs=observation_specs)
        self.actions = create_buffer(buffer_size=max_size, data_shape=(act_shape,))
        self.rewards = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.terminals = create_buffer(buffer_size=max_size, data_shape=(1,))
        self.episode_number = create_buffer(buffer_size=max_size, data_shape=(1,))

        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.rand_gen = rand_gen
        self.device = get_device()
        self.num_agents = num_agents
        self.current_episode_nums = np.arange(num_agents)

        self.use_cer = use_cer

    def store(self,
              obs: List[np.ndarray],
              act: np.ndarray,
              reward: np.ndarray,
              next_obs: List[np.ndarray],
              terminal: np.ndarray
              ):
        """
        Store experiences in the replay buffer from each agent this step.

        Ensures that when the buffer is full, the overflowing experiences wrap to the beginning of the buffer.
        """
        num_samples = act.shape[0]
        new_ptr = (self.ptr + num_samples) % self.max_size
        new_size = min(self.size + num_samples, self.max_size)
        # Index to split the buffer at
        split_index = self.max_size if new_ptr < self.ptr else new_ptr
        # index to split the inserted sample at
        item_split_index = self.max_size - self.ptr if new_ptr < self.ptr else num_samples

        for i in range(len(self.obs)):
            self.insert(self.obs[i], obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
            self.insert(self.next_obs[i], next_obs[i], self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.actions, act, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.rewards, reward, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.terminals, terminal, self.ptr, new_ptr, self.max_size, split_index, item_split_index)
        self.insert(self.episode_number, self.current_episode_nums, self.ptr, new_ptr, self.max_size, split_index,
                    item_split_index)
        self.ptr = new_ptr
        self.size = new_size

    def insert(self, buffer: np.ndarray, data: np.ndarray, ptr: int, new_ptr: int, max_size: int, split_index: int,
               item_split_index: int):
        """
        Inserts data into a given buffer and ensures data wraps around end of buffer.
        """
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        buffer[ptr:split_index] = data[:item_split_index]
        buffer[split_index % max_size:new_ptr] = data[item_split_index:]

    def sample(self, batch_size=32, n_step: int = 1, gamma: float = 0.99):
        """
        Uniform random sampling from the replay buffer.

        Returns a tuple of GPU tensors.
        """
        if self.rand_gen is not None:
            indices = self.rand_gen.choice(self.size, size=batch_size)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)

        if self.use_cer:
            # CER appends the last batch of experiences to the minibatch
            # Used to limit negative effect of very large replay buffer
            # Makes learning more on-policy by favoring recent transitions
            start_idx = (self.ptr - self.num_agents) % self.max_size
            end_idx = self.ptr
            new_indices = list(range(start_idx, self.max_size)) + list(range(0, end_idx)) if start_idx > end_idx \
                else list(range(start_idx, end_idx))
            indices = np.array(list(indices) + new_indices)

        obs = [o[indices] for o in self.obs]
        actions = self.actions[indices]

        if n_step > 1:
            highest_n, rewards, next_obs, terminals = self.n_step_bootstrapping(indices, n_step, gamma)
        else:
            next_obs = [next_o[indices] for next_o in self.next_obs]
            rewards = self.rewards[indices].flatten()
            terminals = self.terminals[indices].flatten()
            highest_n = np.ones(len(indices))

        return (
            list_to_torch(obs, False, device=self.device),
            numpy_to_torch(actions, False, device=self.device),
            numpy_to_torch(rewards, False, device=self.device),
            list_to_torch(next_obs, False, device=self.device),
            numpy_to_torch(terminals, False, device=self.device),
            indices,
            numpy_to_torch(np.ones(len(indices)) / self.size, False, device=self.device),
            numpy_to_torch(highest_n, False, device=self.device)
        )

    def get_trajectory(self, agent_index: int = 0, trajectory_length: int = 0) -> List[torch.Tensor]:
        if trajectory_length <= 0 or trajectory_length > self.size:
            return []
        final_index = ((self.ptr + self.max_size) - self.num_agents + agent_index) % self.max_size
        start_index = (final_index - self.num_agents * (
                trajectory_length - 1) + self.num_agents * self.max_size) % self.max_size
        buffer_split_index = ((
                                      self.max_size - start_index) // self.num_agents) * self.num_agents + start_index + self.num_agents
        front_end_index = buffer_split_index if start_index > final_index else final_index
        back_start_index = final_index if start_index < final_index else front_end_index % self.max_size

        obs_front = [o[start_index:front_end_index:self.num_agents] for o in self.obs]
        obs_back = [o[back_start_index:final_index:self.num_agents] for o in self.obs]
        terminal_obs = [o[final_index].reshape(1, *o[final_index].shape) for o in self.next_obs]
        obs_traj = [np.concatenate((o1, o2, o3)) for o1, o2, o3 in zip(obs_front, obs_back, terminal_obs)]

        return list_to_torch(obs_traj, device=self.device)

    def n_step_bootstrapping(self, sample_indices: np.ndarray, n: int = 1, gamma: float = 0.99):
        """
        Calculates n-step return.

        Outputs return and n-step observation for TD-error.
        Highest n is for Q-value weighting to varying with highest n-step reached for that sample. Ranges from 1-n.

        Should step forward upto n-steps from the provided indices.

        Prevent further steps if either:
        1. Next step is not in initial episode
        2. A terminal state has been reached
        """
        n_step_r = np.zeros(len(sample_indices), dtype=np.float32)
        current_indices = sample_indices.copy()
        initial_ep_num = self.episode_number[sample_indices].flatten()
        has_not_terminated = np.ones_like(n_step_r) > 0
        highest_n = np.zeros_like(n_step_r)

        for i in range(n):
            # consider ith step of n-step for each sample (loops around end of buffer)
            ith_indices = (sample_indices + i * self.num_agents) % self.max_size
            episode_num = self.episode_number[ith_indices].flatten()
            # is step in same episode?
            is_same_ep = episode_num == initial_ep_num
            # is n-step valid? Same ep and not already past terminal?
            is_valid_n_step = np.logical_and(is_same_ep, has_not_terminated)
            is_new_state_terminal = self.terminals[ith_indices].flatten() > 0
            is_valid_terminal_step = np.logical_and(is_valid_n_step, is_new_state_terminal)

            # update current index if both same ep
            current_indices[is_valid_n_step] = ith_indices[is_valid_n_step]
            # store which samples are now terminal for valid n-step checking
            has_not_terminated[is_valid_terminal_step] = False
            # n-step return : R = R_1 + g*R_2 + g**2*R_3 + ... + g**(n-1)*R_n
            n_step_r += (gamma ** i) * self.rewards[current_indices].flatten() * is_valid_n_step
            highest_n[is_valid_n_step] += 1

        return highest_n, n_step_r, [next_o[current_indices] for next_o in self.next_obs], self.terminals[
            current_indices].flatten()


class RolloutBuffer:
    """
    A rollout buffer for vectorized environments in MLAgents.
    """

    def __init__(self, observation_specs: List[ObservationSpec], act_dim: int, buffer_size: int = 2560,
                 gamma: float = 0.99, lambd: float = 0.95, num_agents: int = 1,
                 memory_settings: Optional[MemorySettings] = None):
        self.observation_specs = observation_specs
        self.act_shape = (act_dim,)
        self.buffer_size = buffer_size
        self.device = get_device()
        self.gamma, self.lambd = gamma, lambd
        self.num_agents = num_agents
        self.buffer_size_per_agent = self.buffer_size // self.num_agents
        assert self.buffer_size == self.buffer_size_per_agent * self.num_agents

        self.obs, self.actions, self.rewards, self.terminals = None, None, None, None
        self.values, self.log_ps = None, None
        self.advantages, self.returns = None, None
        self.ptr = None

        self.use_lstm = memory_settings is not None
        if self.use_lstm:
            self.memories_in = create_buffer(buffer_size=self.buffer_size_per_agent,
                                             data_shape=(num_agents, memory_settings.memory_size))
            self.memories_out = create_buffer(buffer_size=self.buffer_size_per_agent,
                                              data_shape=(num_agents, memory_settings.memory_size))
            self.sequence_length = memory_settings.sequence_length
        else:
            self.memories_in = None
            self.memories_out = None
            self.sequence_length = 1

        self.initialize()

    def initialize(self):
        """
        Reset buffers to 0.
        Typical buffer shape is (Buffer_size_per_agent,num_agents,*data_shape)
        """
        self.obs = create_obs_buffers(buffer_size=self.buffer_size_per_agent, observation_specs=self.observation_specs,
                                      num_agents=self.num_agents)
        self.actions = create_buffer(buffer_size=self.buffer_size_per_agent,
                                     data_shape=(self.num_agents,) + self.act_shape)
        self.rewards = create_buffer(buffer_size=self.buffer_size_per_agent, data_shape=(self.num_agents,))
        self.terminals = create_buffer(buffer_size=self.buffer_size_per_agent, data_shape=(self.num_agents,))
        self.truncated = create_buffer(buffer_size=self.buffer_size_per_agent, data_shape=(self.num_agents,))
        self.log_ps = create_buffer(buffer_size=self.buffer_size_per_agent,
                                    data_shape=(self.num_agents,) + self.act_shape, includes_num_agents=True)
        self.values = create_buffer(buffer_size=self.buffer_size_per_agent, data_shape=(self.num_agents,))

        self.ptr = 0

    def reset(self):
        """
        Reset buffers to 0.
        Typical buffer shape is (Buffer_size_per_agent,num_agents,*data_shape)
        """
        for o in self.obs:
            o.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.terminals.fill(0)
        self.truncated.fill(0)
        self.log_ps.fill(0)
        self.values.fill(0)
        self.ptr = 0

        if self.use_lstm:
            self.memories_in.fill(0)
            self.memories_out.fill(0)

    def store(self,
              obs: List[np.ndarray],
              actions: np.ndarray,
              rewards: np.ndarray,
              terminals: np.ndarray,
              truncated: np.ndarray,
              log_ps: np.ndarray,
              values: np.ndarray,
              memories_in: Optional[np.ndarray] = None,
              memories_out: Optional[np.ndarray] = None
              ):
        """
        Store one experience per agent into the rollout buffer from each agent this step.
        """
        assert self.ptr < self.buffer_size_per_agent
        for i in range(len(obs)):
            self.obs[i][self.ptr] = obs[i].copy()
        self.actions[self.ptr] = actions.copy()
        self.rewards[self.ptr] = rewards.copy()
        self.terminals[self.ptr] = terminals.copy()
        self.truncated[self.ptr] = truncated.copy()
        self.log_ps[self.ptr] = log_ps.copy()
        self.values[self.ptr] = values.copy()

        if self.use_lstm:
            self.memories_in[self.ptr] = memories_in.copy()
            self.memories_out[self.ptr] = memories_out.copy()

        self.ptr += 1

    def compute_advantages_and_returns(self, last_values: np.ndarray, next_terminals: np.ndarray):
        assert self.ptr == self.buffer_size_per_agent
        assert last_values.shape == next_terminals.shape

        self.advantages = np.zeros_like(self.values, dtype=np.float32)
        last_delta = np.zeros((1, self.num_agents), dtype=np.float32)

        for t in reversed(range(self.buffer_size_per_agent)):
            if t == self.buffer_size_per_agent - 1:
                is_next_non_terminal = 1.0 - next_terminals
                next_values = last_values
            else:
                is_next_non_terminal = 1.0 - self.terminals[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * is_next_non_terminal - self.values[t]
            last_delta = delta + self.gamma * self.lambd * is_next_non_terminal * last_delta
            self.advantages[t] = last_delta
        self.returns = self.advantages + self.values

    def get_episode_returns_lengths_per_agent(self):
        assert self.ptr == self.buffer_size_per_agent
        returns, lengths = [[] for _ in range(self.num_agents)], [[] for _ in range(self.num_agents)]
        any_dones = [0 for _ in range(self.num_agents)]
        for a in range(self.num_agents):
            returns_for_agent, lengths_for_agent = [], []
            current_return, current_length = 0.0, 0
            dones = self.terminals[:, a] + self.truncated[:, a]
            rewards = self.rewards[:, a]
            for i, d in enumerate(dones):
                current_return += rewards[i]
                current_length += 1
                if d >= 1:
                    returns_for_agent.append(current_return)
                    lengths_for_agent.append(current_length)
                    current_return = 0
                    current_length = 0
                    any_dones[a] = 1
            returns_for_agent.append(current_return)
            lengths_for_agent.append(current_length)
            returns[a].extend(returns_for_agent)
            lengths[a].extend(lengths_for_agent)

        return returns, lengths, any_dones

    def get(self, normalize_advantage: bool = False, output_as_tensor: bool = False):
        assert self.ptr == self.buffer_size_per_agent
        assert self.advantages is not None

        # # normalize advantages
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean(axis=0)) / (self.advantages.std(axis=0) + 1e-8)

        obs = flatten_obs_buffer(self.obs, self.observation_specs)
        actions = flatten_buffer(self.actions, self.act_shape)
        log_ps = flatten_buffer(self.log_ps, self.act_shape)
        values = flatten_buffer(self.values)
        advantages = flatten_buffer(self.advantages)
        returns = flatten_buffer(self.returns)

        if self.use_lstm:
            memories_in = flatten_buffer(self.memories_in)
            memories_out = flatten_buffer(self.memories_out)
        else:
            memories_in = None
            memories_out = None

        if output_as_tensor:
            return (
                list_to_torch(obs, False, device=self.device),
                numpy_to_torch(actions, False, device=self.device),
                numpy_to_torch(log_ps, False, device=self.device),
                numpy_to_torch(values, False, device=self.device),
                numpy_to_torch(advantages, False, device=self.device),
                numpy_to_torch(returns, False, device=self.device),
                numpy_to_torch(memories_in, False, device=self.device) if self.use_lstm else None,
                numpy_to_torch(memories_out, False, device=self.device) if self.use_lstm else None,
                None
            )

        return (
            obs,
            actions,
            log_ps,
            values,
            advantages,
            returns,
            memories_in,
            memories_out,
            None
        )
