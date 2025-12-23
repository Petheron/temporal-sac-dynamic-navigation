from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from collections import defaultdict

import platform
import numpy as np

from mlagents_envs.environment import UnityEnvironment, ActionTuple, BehaviorSpec
from mlagents_envs.base_env import ObservationSpec
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid

class TrajectoryPositionChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        """
        Trajectories are a dictionary of lists. Key is episode ID.
        List is of float for position [x0,x1,...,xT, y0, y1, ..., yT, z0, z1, ..., zT]
        """
        self.trajectories = None
        self.end_conditions = None
        self.goal_positions = None
        self.goal_distances = None
        self.reset()

    def on_message_received(self, msg: IncomingMessage) -> None:
        episode_id = msg.read_int32()
        split_trajectory = msg.read_float32_list()
        end_condition = msg.read_string()
        goal_position = msg.read_float32_list()
        initial_position = split_trajectory[:3]
        goal_distance = np.sqrt(np.sum((np.array(goal_position) - np.array(initial_position)) ** 2))
        x_y_z_traj = np.array(split_trajectory).reshape(-1, 3)
        xyz_list = [x_y_z_traj[:, i] for i in range(3)]
        self.trajectories[episode_id] = xyz_list
        self.end_conditions[episode_id] = end_condition
        self.goal_positions[episode_id] = goal_position
        self.goal_distances[episode_id] = goal_distance

    def get_and_reset_trajectory_positions(self) -> Tuple[defaultdict, defaultdict, defaultdict, defaultdict]:
        positions = self.trajectories
        end_conditions = self.end_conditions
        goal_positions = self.goal_positions
        goal_distances = self.goal_distances
        self.reset()
        return positions, end_conditions, goal_positions, goal_distances

    def reset(self):
        self.trajectories = defaultdict(list)
        self.end_conditions = defaultdict(str)
        self.goal_positions = defaultdict(list)
        self.goal_distances = defaultdict(float)


def get_system_type() -> str:
    return platform.system()


class BaseEnv:
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Params:
            env: The unity environment to be initialized
            stats: the stats side channel for the unity environment
            engine: the engine side channel for the unity environment
            env_params: the environment parameters side channel for the unity environment
            behaviour_name: the name of the behavior, used to obtain spec
            spec: the specs of the environment, used for unity actions and obs
            num_decision_agents: the number of agents awaiting a decision
            seed: the random seed to pass to the UnityEnvironment
            discrete_action_shape: a tuple of discrete actions for an agent
                e.g. (5,3) is two-actions, one with 5 actions, other with 3 actions
            continuous_action_size: an int of the number of float actions an agent has
            is_terminal: numpy array of len(num_agents) to keep track of terminal agent count
        """
        self.env = None
        self.stats = None
        self.engine = None
        self.env_params = None
        self.behavior_name = None
        self.spec = None
        self.num_agents = None
        self.seed = None
        self.discrete_action_shape = None
        self.continuous_action_size = None

        self.trajectory_position_channel = None

        # Store the current action to repeat for each agent.
        # Until action repeat count reaches num_action_repeat.
        # All stored in agent id order.
        self.repeating_actions = None
        self.action_repeat_count = None
        self.num_action_repeat = None
        self.deciding_agent_ids = None

        self.observation_specs = None

        # Use to reorder agent indexed arrays by agent id
        self.current_decision_agent_id = None
        # Use to reorder agent id'ed arrays by agent index
        self.current_decision_agent_index = None

        self.use_hidden_states = True

    def env_init(self, env_info: dict) -> None:
        """
        Initializes the Unity Environment using the given env_info args.
        If Unity Env is initialized, then at start of each run, set all params for RL.

        Assumed env_info contains:
        env_name: the directory+name of the environment
        seed: (int) the RandomState seed for the environment
        time_scale: float, the time_scale of the environment

        """
        self.seed = int(env_info.get("seed", 0))
        if self.env is None:
            self.create_unity_env(env_info)
        if self.env is not None:
            for name, value in env_info.get("env_params").items():
                if isinstance(value, float) or isinstance(value, int):
                    self.env_params.set_float_parameter(name, value)
            self.env_params.set_float_parameter("seed", self.seed)
            self.env.reset()
            self.behavior_name = list(self.env.behavior_specs)[0]
            self.spec = self.env.behavior_specs[self.behavior_name]
            dec_steps, term_steps = self.env.get_steps(self.behavior_name)
            # Make sure to count all agents, as may be possible for initial terminal agent
            # if agent is terminal, it will want new action (unless action repetition).
            # So possible for it to be in both decision and terminal id lists.
            self.num_agents = len(np.unique(np.concatenate((dec_steps.agent_id, term_steps.agent_id))))
            self.action_repeat_count = np.zeros(self.num_agents, dtype=np.int32)
            self.env.reset()
            self.discrete_action_shape = self.spec.action_spec.discrete_branches
            self.continuous_action_size = self.spec.action_spec.continuous_size

            self.observation_specs = self.spec.observation_specs
            self.num_action_repeat = env_info.get("env_params").get("num_action_repeat", 1)
            self.use_hidden_states = env_info.get("env_params").get("use_hidden_states", True)

    def create_unity_env(self, env_info: dict) -> None:
        self.stats = StatsSideChannel()
        self.engine = EngineConfigurationChannel()
        self.env_params = EnvironmentParametersChannel()
        self.trajectory_position_channel = TrajectoryPositionChannel() if env_info.get("env_params").get(
            "store_positions", 0) > 0 else None
        side_channels = [self.stats, self.engine, self.env_params]
        if self.trajectory_position_channel is not None:
            side_channels.append(self.trajectory_position_channel)

        root_folder = get_system_type()
        env_name = env_info.get("env_name")
        env_dir = env_info.get("env_dir")
        load_name = f"{env_dir}{root_folder}/{env_name}"

        self.env = UnityEnvironment(file_name=load_name, seed=self.seed,
                                    no_graphics=env_info['no_graphics'],
                                    side_channels=side_channels,
                                    worker_id=env_info.get("worker_id", 0)
                                    )
        self.engine.set_configuration_parameters(time_scale=env_info.get("time_scale", 20.0))

    def get_trajectory_positions(self):
        if self.trajectory_position_channel is None:
            return None, None, None, None
        return self.trajectory_position_channel.get_and_reset_trajectory_positions()

    @abstractmethod
    def env_start(self):
        """
        The starting of the environment called at the start of an episode and before agent_start.

        Returns:
            The first observation (np.ndarray) of the environment.

        """

    @abstractmethod
    def construct_action(self, action):
        """
        For a given action, construct an ActionTuple for all agents awaiting a decision.
        Returns:
            An action tuple that sets the action of all deciding agents
        """

    @abstractmethod
    def env_step(self, action):
        """
        For a given action, the environment takes a step.
        Args:
            action: (int): the action from the agent
        Returns:
            (float,observation,bool): a tuple of the reward, state observation, isTerminal after the step
        """

    def close(self) -> None:
        """
        Close the Unity Environment and set it to None after all runs are done
        """
        print("\nEnvironment Closing\n")
        if self.env is not None:
            self.env.close()
            self.env = None

    @abstractmethod
    def env_message(self, message):
        """
        For a given message to the environment, return an answer.
        Args:
            message: (str): a message
        Returns:
            An answer to the message
        """

    def get_stats(self):
        """
        Returns dictionary of stored stats.
        """
        return self.stats.get_and_reset_stats()


class Environment(BaseEnv):
    def __init__(self):
        super().__init__()

    def env_init(self, env_info: dict):
        super(Environment, self).env_init(env_info)

    def env_start(self):
        """
        Starts the environment and returns the first observation.
        Returns:
        observation - List[np.ndarray] - the observation vector of the first observation of the environment. Dim [num_agent,obs_size]
        """
        self.env.reset()
        observation, reward = self.env_step(action=None)

        return observation, reward

    def construct_action(self, action):
        """
        For a given action integer, construct an action tuple for each agent awaiting a decision.
        returns:
        action_tuple - ActionTuple - The chosen action for all deciding agents for the environment
        """
        action_tuple: ActionTuple = self.spec.action_spec.empty_action(action.shape[0])
        if len(self.discrete_action_shape) > 0:
            assert action.dtype == int or action.dtype == np.int64 or action.dtype == np.int32
            action_tuple.add_discrete(action.reshape((-1, len(self.discrete_action_shape))))
        else:
            action_tuple.add_continuous(action.reshape((-1, self.continuous_action_size)))
        return action_tuple

    def env_step(self, action: np.ndarray = None, observation: List[np.ndarray] = None):
        """
        For a batch of actions (n,a), take an environmental step.

        Observations and actions supplied are in decision_id order.

        Returns Next Observation (for deciding agents), Rewards, Terminals, Truncated, & The Experiences to store.

        Terminals are states which have ended, but not through an arbitrary time-limit (truncated) or epoch end.

        Experiences to store are (Obs, Action, Rewards, Next_Obs, Terminals). Note Next Obs here combined terminal and deciding states.

        All output values to be stored are in agent_id order.
        """

        # Initialize experience to store to None
        obs_to_store = None
        action_to_store = None
        rewards_to_store = None
        next_obs_to_store = None
        terminals_to_store = None

        # handle things from previous observation
        # sort it into agent id
        if observation is not None:
            obs_to_store = sort_observation(observation, self.current_decision_agent_id)

        # Create action and take environment step
        if action is not None:
            # store actions
            action_to_store = action.copy()
            # sort action into agent id order
            action_to_store = action_to_store[self.current_decision_agent_id]

            # handle repeating actions & their counters
            if self.repeating_actions is None:
                self.repeating_actions = action_to_store.copy()
            self.reset_repeating_action_counter()

            # contruct, set, and take actions
            action_tuple = self.construct_action(action)
            self.env.set_actions(self.behavior_name, action_tuple)
            self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Handle storage of early terminating states (those that terminate during action repetition)
        is_early_termination = len(terminal_steps.agent_id) > 0 and len(decision_steps.agent_id) < self.num_agents

        early_terminal_experience_tuple = None
        if is_early_termination:
            e_t_id = np.zeros(self.num_agents, dtype=bool)
            e_t_rewards = np.zeros(self.num_agents)
            e_t_next_obs = get_zeros_observation_list(self.num_agents, self.observation_specs)
            e_t_terminals = np.zeros(self.num_agents, dtype=np.int)

            while is_early_termination:
                t_id = terminal_steps.agent_id.copy()
                e_t_id[t_id] = True
                for i in range(len(e_t_next_obs)):
                    e_t_next_obs[i][t_id] = terminal_steps.obs[i]
                e_t_rewards[t_id] = terminal_steps.reward
                e_t_terminals[t_id] = 1 - terminal_steps.interrupted

                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                is_early_termination = len(terminal_steps.agent_id) > 0 and len(decision_steps.agent_id) < self.num_agents

            early_terminal_ids = np.arange(self.num_agents)[e_t_id]
            early_terminal_next_obs = [o[early_terminal_ids] for o in e_t_next_obs]
            early_terminal_rewards = e_t_rewards[early_terminal_ids]
            early_terminal_terminals = e_t_terminals[early_terminal_ids]
            early_terminal_experience_tuple = (early_terminal_ids,
                                               early_terminal_rewards,
                                               early_terminal_next_obs,
                                               early_terminal_terminals
                                               )

        # agent id array stores agent ids in agent index order
        new_decision_ids = decision_steps.agent_id
        new_terminal_ids = terminal_steps.agent_id

        # Store next obs for future action decisions (agent index order)
        next_obs = decision_steps.obs.copy()

        # update counter for any agent that took a step this action
        if action is not None:
            # self.action_repeat_count += 1
            self.action_repeat_count[new_terminal_ids] = 0
            if early_terminal_experience_tuple is not None:
                self.action_repeat_count[early_terminal_ids] = 0

        # store new agent decision ids, decision indexes, & current ids.
        self.current_decision_agent_id = new_decision_ids
        self.current_decision_agent_index = np.zeros(len(self.current_decision_agent_id), dtype=np.int)
        if len(self.current_decision_agent_id) != self.num_agents:
            for idx, id in enumerate(self.current_decision_agent_id):
                self.current_decision_agent_index[id] = idx

        # copy reward array, sort and insert terminal rewards as step 0 reward should be 0.
        reward = np.zeros(self.num_agents, dtype=np.float32)
        reward[new_decision_ids] = decision_steps.reward[new_decision_ids]
        reward[new_terminal_ids] = terminal_steps.reward

        # end early as starting environment
        if action is None:
            return next_obs, reward

        # terminals is 0 for deciding agent, 0 for interrupted agent, 1 for terminating agent
        terminals = np.zeros(self.num_agents, dtype=np.int32)
        terminals[new_terminal_ids] = 1 - terminal_steps.interrupted

        # dones is 0 for deciding, 1 otherwise
        dones = np.zeros(self.num_agents, dtype=np.int32)
        dones[new_terminal_ids] = 1

        # Create next_obs for replay buffer storage (use copy to avoid list mutability)
        terminal_obs = terminal_steps.obs

        # incorporate early terminal stats into the record statistics for this step.
        if early_terminal_experience_tuple is not None:
            reward[early_terminal_ids] = early_terminal_rewards
            terminals[early_terminal_ids] = early_terminal_terminals
            dones[early_terminal_ids] = 1

            new_terminal_ids = np.concatenate([new_terminal_ids, early_terminal_ids])
            for i in range(len(terminal_obs)):
                terminal_obs[i] = np.vstack([terminal_obs[i], early_terminal_next_obs[i]])

        # truncated is 1 for interrupted agents, 0 otherwise
        truncated = dones - terminals
        rewards_to_store = reward.copy()
        terminals_to_store = terminals.copy()

        # For each obs, sort into agent id order, then replace with terminal obs according to terminal ids
        next_obs_to_store = self.get_sorted_next_obs(next_obs, terminal_obs, new_decision_ids,
                                                     new_terminal_ids)

        experience_to_store = (obs_to_store, action_to_store, rewards_to_store, next_obs_to_store, terminals_to_store)

        return next_obs, reward, terminals, truncated, experience_to_store

    def get_deciding_agent_observations(self, obs: List[np.ndarray]):
        """
        Return a modified observation to only include those for agents that require new actions.
        Returned observations are in index order for use in network.
        """
        agent_ids_deciding = self.is_agent_action_resetting()
        self.deciding_agent_ids = agent_ids_deciding.copy()
        agent_ids_deciding = agent_ids_deciding[self.current_decision_agent_id]
        agent_index_deciding = agent_ids_deciding[self.current_decision_agent_index]

        obs_index = obs.copy()
        for i, o in enumerate(obs_index):
            o = o[agent_index_deciding]
            obs_index[i] = o
        return obs_index

    def combine_repeating_action_with_new_actions(self, new_actions: np.ndarray):
        """
        For an array of new actions size (num_deciding_agent, action_size), insert into currently repeating actions.
        Returns actions to take in agent index order and stores new actions in env.repeating_actions.
        """
        if self.repeating_actions is None:
            return new_actions

        # Order repeating actions and deciding agents by agent index
        repeat_action_ids = self.repeating_actions[self.current_decision_agent_id]
        repeating_actions_idx = repeat_action_ids[self.current_decision_agent_index]
        agent_ids_deciding = self.is_agent_action_resetting()[self.current_decision_agent_id]
        agent_index_deciding = agent_ids_deciding[self.current_decision_agent_index]

        # Insert new actions into repeating actions, store in agent_id sorted variable, and return.
        repeating_actions_idx[agent_index_deciding] = new_actions
        self.repeating_actions[self.current_decision_agent_id] = repeating_actions_idx[self.current_decision_agent_id]
        return repeating_actions_idx

    def is_agent_action_resetting(self) -> np.ndarray:
        return self.action_repeat_count % self.num_action_repeat == 0

    def reset_repeating_action_counter(self):
        """
        Handle action storage, resetting of counter if mod 0.
        Sorts actions into action_id order for replay buffer.
        If repeating actions null, copies sorted actions.

        """
        self.action_repeat_count[self.is_agent_action_resetting()] = 0

    def close(self):
        super(Environment, self).close()

    def env_message(self, message):
        if message == "get discrete action shape":
            return self.discrete_action_shape
        if message == "get continuous action shape":
            return self.continuous_action_size
        if message == "get observation shape":
            return self.observation_shape
        return None

    def get_sorted_next_obs(self, decision_obs: List[np.ndarray], terminal_obs: List[np.ndarray], decision_ids,
                            terminal_ids):
        """
        For given decision and terminal observations, uses the supplied decision and terminal ids to create a next_obs list
        """
        next_obs = get_zeros_observation_list(self.num_agents, self.observation_specs)
        for i, obs in enumerate(next_obs):
            # insert decision obs
            d_obs = decision_obs[i]
            obs[decision_ids] = d_obs
            # insert terminal obs
            t_obs = terminal_obs[i]
            obs[terminal_ids] = t_obs
            # replace observation in list with processed obs.
            next_obs[i] = obs
        return next_obs


def get_zeros_observation_list(num_agents: int, observation_specs: ObservationSpec) -> List[np.ndarray]:
    return [np.zeros((num_agents, *o.shape)) for o in observation_specs]


def sort_observation(obs: List[np.ndarray], sort_order: np.ndarray):
    """
    Sorts the arrays of a given list of observations by a supplied sort order.
    Returns a list of sorted arrays.
    """
    observations = obs.copy()
    for i, o in enumerate(observations):
        o = o[sort_order]
        observations[i] = o
    return observations
