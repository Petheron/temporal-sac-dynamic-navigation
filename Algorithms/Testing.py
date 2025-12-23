import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

import numpy as np
from tqdm import tqdm
from Encoders import *
from mlagents.trainers.torch.encoders import SmallVisualEncoder

#
stats = StatsSideChannel()
engine = EngineConfigurationChannel()
engine.set_configuration_parameters(width=512, height=512)
env_params = EnvironmentParametersChannel()

#
# env_dir = "../Environments/ContinuousCartpole/Build/"
# name="ContinuousCartpole"
# env_dir = "../Environments/RaycastNav/Build/"
name = 'RaycastNav'
# name = '3DBall'
# name = 'ContinuousCartpole'
env_dir = f"../Environments/{name}/Build/Windows/"
env_name = env_dir + name
time_scale = 20.0
epochs = 1
num_agents = 8
num_action_repeat = 1
steps_per_epoch = 5
engine.set_configuration_parameters(time_scale=time_scale)
env_params.set_float_parameter("num_agents", num_agents)
env_params.set_float_parameter("num_action_repeat", num_action_repeat)
env_params.set_float_parameter("level_type", 3)
#
env = UnityEnvironment(file_name=env_name, seed=1, side_channels=[stats, engine, env_params], worker_id=0)
env.reset()
#
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
decision_steps, terminal_steps = env.get_steps(behavior_name)
total_agents = len(decision_steps.agent_id)
#
# hidden_size = 4
# conv_2d_dict = {'init_channel': 16, 'init_kernel_size': 3, 'init_stride': 1}
# conv_3d_dict = {'init_channel': 16, 'init_kernel_size': 3, 'init_stride': 1}
# network_settings = NetworkSettings(hidden_size, 2, conv_2d_dict, conv_3d_dict)
# obs_enc = ObservationEncoder(spec.observation_specs, hidden_size, conv_2d_dict=conv_2d_dict, conv_3d_dict=None)
# obs = [numpy_to_torch(i, False, True) for i in decision_steps.obs]
# x = obs_enc(obs)
# network = NetworkBody(spec.observation_specs, network_settings, 0)
# y = network(obs)
#
#

for episode in range(epochs):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1  # -1 indicates not yet tracking
    done = False  # For the tracked_agent
    episode_rewards = 0  # For the tracked_agent
    for _ in tqdm(range(steps_per_epoch)):
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]
        # Generate an action for all agents
        action = spec.action_spec.random_action(len(decision_steps))
        #
        # For specific action
        # action = spec.action_spec.empty_action(len(decision_steps))
        # action.add_discrete(np.array([2]).reshape(len(decision_steps),1))
        #
        # Set the actions
        env.set_actions(behavior_name, action)
        # Move the simulation forward
        env.step()
        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if tracked_agent in decision_steps:  # The agent requested a decision
            episode_rewards += decision_steps[tracked_agent].reward
        if tracked_agent in terminal_steps:  # The agent terminated its episode
            episode_rewards += terminal_steps[tracked_agent].reward
            done = True
    print(f"Total rewards for episode {episode} is {episode_rewards}")

# # test terminalsteps
# is_terminal = False
# env_params.set_float_parameter("max_steps", 50)
# env.reset()
# decision_steps, terminal_steps = env.get_steps(behavior_name)
# print(decision_steps[0])
# while not is_terminal:
#     decision_steps, terminal_steps = env.get_steps(behavior_name)
#     # if len(terminal_steps.agent_id) > 0:
#     #     is_terminal = True
#     # else:
#     #     env.step()
#     if len(terminal_steps.interrupted) > 0 and terminal_steps.interrupted.any():
#         is_terminal = True
#     else:
#         env.step()
#
# t = terminal_steps.agent_id
# print(t)
# for i in decision_steps.agent_id:
#     if i not in t:
#         print(i)



env.close()
