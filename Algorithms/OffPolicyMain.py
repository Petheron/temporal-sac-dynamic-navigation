import os
# use to make LSTM deterministic (bug in cudnn 10.2 onwards)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import distutils.util
import random
import argparse
import time
import torch.backends.cudnn
import yaml
import gc

from SAC_LSTM import SAC_LSTM
from SAC import SAC
from DataStructures import ReplayBuffer, PrioritizedReplayBuffer, LA3PReplayBuffer, LSTMReplayBuffer, \
    PrioritizedLSTMReplayBuffer
from Environment import *
from Utils import *
from EncoderUtils import NormalizerType, NetworkSettings, VisualSettings, PositionalEmbeddingSettings, MemorySettings
from NetworkUtils import list_to_torch


def eval_policy(policy, env: Environment, num_eval_episodes=10):
    """
    Runs a given policy on a provided evaluation environment for num_eval_episodes episodes per agent.
    """
    # # set env seed
    # env.env_params.set_float_parameter("seed", np.random.randint(0, 10000))

    complete_eps = 0
    episode_returns = []
    episode_lens = []
    o, r = env.env_start()
    num_agents = env.num_agents
    agent_ret = np.zeros(num_agents)
    agent_ep_len = np.zeros(num_agents, dtype=np.int32)

    while complete_eps < num_eval_episodes * num_agents:
        new_actions, _ = policy.select_deterministic_action(o)
        o, r, terminals, truncated, _ = env.env_step(new_actions, o)
        is_dones = (terminals + truncated) > 0
        agent_ret += r
        agent_ep_len += 1
        complete_eps += is_dones.sum()

        if is_dones.any():
            episode_returns.extend(list(agent_ret[is_dones]))
            episode_lens.extend(list(agent_ep_len[is_dones]))
            agent_ret[is_dones] = 0
            agent_ep_len[is_dones] = 0

    return np.mean(episode_returns), np.mean(episode_lens)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='SAC')
    parser.add_argument('--env_name', '-env', type=str, default='ContinuousCartpole')
    parser.add_argument('--env_dir', '-dir', type=str, default='../Environments/ContinuousCartpole/Build/')
    parser.add_argument('--results_dir', '-r_dir', type=str, default='../Environments/ContinuousCartpole/Build/results')
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--time_scale', type=float, default=20.)
    parser.add_argument('--no_graphics', action='store_true')

    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--total_run_steps', type=int, default=100000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--num_test_eps', type=int, default=10)

    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--buffer_size', '-buff', type=int, default=1000000)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)

    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--learning_rate_schedule', type=str, default='constant')

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--normalize', type=distutils.util.strtobool, default=False)

    parser.add_argument('--start_steps', '-ss', type=int, default=1000,
                        help='Number of experiences to collect randomly before following policy.')
    parser.add_argument('--training_frequency', type=int, default=1,
                        help='How often to train the agent. The agent is updated training_frequency times per update to ensure step to update ratio is 1:1.')

    parser.add_argument('--checkpoint_interval', type=int, default=50000)
    parser.add_argument('--load_seeds', action='store_true')
    parser.add_argument('--load_model', type=str,
                        default="")  # the directory to load an existing model from. Leave blank to not load. "default" to load from results dir.

    parser.add_argument('--environment_parameters', type=yaml.safe_load, action="store")

    # LSTM Parameters
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--sequence_length', type=int, default=0)

    # DDPG and TD3 parameters
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Polyak averaging coefficient for target network updates. Should be near 0.')
    parser.add_argument("--exploration_noise", type=float,
                        default=0.1)  # Noise added to target policy during critic update

    # TD3 Parameters
    parser.add_argument("--policy_noise", type=float, default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", type=float, default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=1, type=int)  # Frequency of delayed policy updates

    # SAC Parameters
    parser.add_argument("--alpha", type=float, default=0.01)  # Entropy Regularization Coefficient
    parser.add_argument("--alpha_lr", type=float, default=3e-4)  # Range to clip target policy noise

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    # clear gpu/RAM memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)

    params = arg_parser()
    env_params = params['environment_parameters']
    env_dir = params['env_dir']
    env_name = params['env_name']
    results_dir = params['results_dir']
    policy_name = params["policy"]
    exp_name = params['exp_name']

    runs = params['runs']
    buffer_size = params["buffer_size"]
    training_frequency = params['training_frequency']
    total_run_steps = params['total_run_steps']

    num_test_eps = params['num_test_eps']
    eval_freq = params['eval_freq']
    checkpoint_interval = params['checkpoint_interval']
    num_evals = total_run_steps // eval_freq + 1  # extra 1 for initial test

    start_steps = params['start_steps']

    # set worker_id for environment and assert it is even to allow for test env on (id + 1)
    worker_id = env_params.get("env_id", 0)
    if worker_id % 2 != 0:
        worker_id += 1

    env_info = {
        "env_name": env_name,
        "env_dir": env_dir,
        "time_scale": params['time_scale'],
        "no_graphics": params['no_graphics'],
        "worker_id": worker_id,
        "env_params": env_params
    }

    eval_env_info = env_info.copy()
    eval_env_info["worker_id"] = worker_id + 1
    eval_env_info["no_graphics"] = True

    # set seeds
    if params["load_seeds"]:
        seeds_dir = results_dir
        if params["load_model"] != "":
            # load model here
            if params["load_model"] != "default":
                seeds_dir = params["load_model"]
        seeds = load_seeds(exp_name, seeds_dir)
    else:
        # seeds = create_seeds(num_runs=runs)
        seeds = env_params.get("seeds", [])
        seeds = np.array(seeds)
    if len(seeds) < runs:
        seeds = list(seeds)
        seeds += list(create_seeds(runs - len(seeds)))
        seeds = np.array(seeds)

    experience_replay_type = ExperienceReplayType(env_params.get('replay_buffer_type', 'basic'))
    per_alpha = env_params.get('per_alpha', 0.9)
    per_beta = env_params.get('per_beta', 0.9)
    per_eta = env_params.get('per_eta', 0.9)
    use_cer = env_params.get('use_cer', False)
    n_step = env_params.get('n_step', 1)

    use_curriculum = env_params.get('use_curriculum', False)
    curriculum_name = 'max_goal_distance'
    max_goal_distance = env_params.get(curriculum_name, 10.0)
    num_curriculum_steps = int(env_params.get('num_curriculum_steps', 10.0))
    curriculum_success_rate = env_params.get('curriculum_success_rate', 0.9)

    goal_curriculum = Curriculum(curriculum_name,
                                 max_goal_distance,
                                 use_curriculum,
                                 num_curriculum_steps,
                                 curriculum_success_rate
                                 )

    current_goal_value = goal_curriculum.get_value()
    env_info["env_params"][curriculum_name] = current_goal_value
    eval_env_info["env_params"][curriculum_name] = current_goal_value

    env = Environment()
    env.env_init(env_info)
    num_agents = env.num_agents

    eval_env = Environment() if num_test_eps > 0 else None
    if eval_env is not None:
        eval_env.env_init(eval_env_info)

    training_frequency *= num_agents
    total_run_steps = total_run_steps - (
            total_run_steps % num_agents - num_agents) if total_run_steps % num_agents > 0 else total_run_steps

    # print(f"Num Agents: {num_agents}, \nTraining Frequency: {training_frequency}, \ntotal_run_steps: {total_run_steps}")

    # discrete implementation only supports mono-discrete action space.
    observation_specs = env.observation_specs
    act_dim = env.continuous_action_size if env.continuous_action_size > 0 else env.discrete_action_shape[0]

    visual_settings = VisualSettings(env_params.get("2D_init_channel", 16),
                                     env_params.get("2D_kernel_size", 3),
                                     env_params.get("2D_stride", 1),
                                     env_params.get("2D_padding", 0),
                                     env_params.get("2D_num_convs", 2),
                                     use_visual_dense_layers=env_params.get("use_visual_dense_layer", True)
                                     )

    boxcast_settings = VisualSettings(env_params.get("3D_init_channel", 16),
                                      env_params.get("3D_kernel_size", 3),
                                      env_params.get("3D_stride", 1),
                                      env_params.get("3D_padding", 0),
                                      env_params.get("3D_num_convs", 2),
                                      env_params.get("3D_use_embedding", 0) == 1,
                                      env_params.get("3D_embedding_dictionary_size", 2),
                                      env_params.get("3D_num_embedding_dims", 0),
                                      env_params.get("use_visual_dense_layers", True)
                                      )

    positional_embedding_settings = PositionalEmbeddingSettings(
        env_params.get("positional_encoding_embedding_size", 30),
        env_params.get("occupancy_map_grid_size", 41)
    )

    sequence_length = params.get("sequence_length", 0)
    burn_in_length = env_params.get("burn_in_length", 0)
    overlap_length = env_params.get("overlap_length", 0)
    memory_size = params.get("memory_size", 0)
    memory_settings = MemorySettings(sequence_length=sequence_length,
                                     memory_size=memory_size,
                                     burn_in=burn_in_length,
                                     overlap_length=overlap_length
                                     ) if sequence_length > 0 and memory_size > 0 else None
    use_lstm = memory_settings is not None

    batch_size = params["batch_size"]

    network_settings = NetworkSettings(hidden_size=params["hidden_size"],
                                       num_layers=params["num_layers"],
                                       normalize=params["normalize"],
                                       norm_type=NormalizerType(env_params.get("norm_type", "mlagents")),
                                       encode_vector_inputs=env_params.get('encode_vector_inputs', False),
                                       conv2d_settings=visual_settings.settings,
                                       conv3d_settings=boxcast_settings.settings,
                                       positional_embedding_settings=positional_embedding_settings,
                                       memory_settings=memory_settings
                                       )

    per_settings = {
        "per_alpha": per_alpha,
        "per_beta": per_beta,
        "per_eta": per_eta,
    } if experience_replay_type == ExperienceReplayType.PER else None

    kwargs = {
        "observation_specs": observation_specs,
        "act_dim": act_dim,
        "network_settings": network_settings,
        "act_lr": params["actor_lr"],
        "crit_lr": params["critic_lr"],
        "learning_rate_schedule": params["learning_rate_schedule"],
        "max_steps": total_run_steps
    }

    # off-policy parameters
    kwargs = {**kwargs,
              **{
                  "gamma": params["gamma"],
                  "tau": params["tau"],
                  "experience_replay_type": experience_replay_type,
                  "n_step": n_step,
              }
              }

    if policy_name == "SAC":
        kwargs = {**kwargs,
                  **{
                      "alpha": params["alpha"],
                      "alpha_lr": params["alpha_lr"]
                  }
                  }
        if use_lstm:
            kwargs = {**kwargs,
                      "per_settings": per_settings,
                      }
        Policy = SAC_LSTM if use_lstm else SAC
    else:
        Policy = SAC

    alpha = params["alpha"]

    # Results storage
    test_returns = np.zeros((runs, num_evals))
    test_lengths = np.zeros((runs, num_evals))
    success_rate = np.zeros((runs, num_evals))
    fail_rate = np.zeros((runs, num_evals))
    euclidean_fail_rate = np.zeros((runs, num_evals))
    timeout_fail_rate = np.zeros((runs, num_evals))

    max_goal_distances = np.zeros((runs, num_evals))
    avg_goal_distances = np.zeros((runs, num_evals))

    trajectory_positions_dict = defaultdict(lambda: defaultdict(list))
    end_conditions_dict = defaultdict(lambda: defaultdict(str))
    goal_positions_dict = defaultdict(lambda: defaultdict(list))

    steps_so_far = np.ones((runs, num_evals), dtype=np.int32) * -1
    time_so_far = np.zeros((runs, num_evals))
    avg_ep_return_array = np.zeros((runs, num_evals))
    avg_ep_length_array = np.zeros((runs, num_evals))

    actor_loss_array = np.zeros((runs, num_evals))
    critic_loss_array = np.zeros((runs, num_evals))
    alpha_loss_array = np.zeros((runs, num_evals))

    q_array = np.zeros((runs, num_evals))
    v_array = np.zeros((runs, num_evals))
    log_p_array = np.zeros((runs, num_evals))
    entropy_array = np.zeros((runs, num_evals))
    alpha_array = np.zeros((runs, num_evals))

    # for run in tqdm(range(runs), leave=True):
    for run in range(runs):
        # set the seed parameters for all rng mechanics
        start_time = time.time()
        seed = seeds[run]
        env_info["seed"] = seed
        eval_env_info["seed"] = seed
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # initialize environment
        env.env_init(env_info)
        if eval_env is not None:
            eval_env.env_init(eval_env_info)

        print(f"\nRun: {run}, Seeds:: Env: {env.seed}\n")

        if not use_lstm:
            if experience_replay_type == ExperienceReplayType.PER or experience_replay_type == ExperienceReplayType.LAP or experience_replay_type == ExperienceReplayType.RELO:
                replay_buffer = PrioritizedReplayBuffer(observation_specs, act_dim, buffer_size, num_agents,
                                                        alpha=per_alpha, beta_start=per_beta,
                                                        beta_frames=total_run_steps, use_cer=use_cer,
                                                        memory_settings=memory_settings)
            elif experience_replay_type == ExperienceReplayType.LA3P:
                replay_buffer = LA3PReplayBuffer(observation_specs, act_dim, buffer_size, num_agents,
                                                 alpha=per_alpha, beta_start=per_beta, beta_frames=total_run_steps,
                                                 use_cer=use_cer)
            else:
                replay_buffer = ReplayBuffer(observation_specs, act_dim, buffer_size, num_agents, use_cer=use_cer,
                                             memory_settings=None)
        else:  # BASIC or PAL
            if experience_replay_type == ExperienceReplayType.PER:
                replay_buffer = PrioritizedLSTMReplayBuffer(act_dim, buffer_size, num_agents, use_cer=use_cer,
                                                            memory_settings=memory_settings,
                                                            priority_exponent=per_settings["per_alpha"],
                                                            importance_sampling_exponent=per_settings["per_beta"])
            else:
                replay_buffer = LSTMReplayBuffer(act_dim, buffer_size, num_agents, use_cer=use_cer,
                                                 memory_settings=memory_settings)

        policy = Policy(**kwargs)

        if params["load_model"] != "":
            # load model here
            if params["load_model"] == "default":
                policy.load(results_dir)
            else:
                policy.load(params["load_model"])
        onnx_saver = OnnxSaver(policy, observation_specs)

        # run initial deterministic evaluation before any training
        if num_test_eps > 0:
            test_return, test_length = eval_policy(policy, eval_env, num_test_eps)
            print('\n')
            print(f"Untrained Test Return: {test_return:.3f}")
            test_returns[run, 0] = test_return
            test_lengths[run, 0] = test_length
        ###

        # initialize episode length and returns
        episode_timesteps = np.zeros(num_agents)
        episode_return = np.zeros(num_agents)

        episode_returns = []
        episode_lengths = []

        eval_losses = {
            "actor": [],
            "critic": [],
            "alpha": []
        }
        eval_values = {
            "v": [],
            "q": [],
            "log_p": [],
            "entropy": [],
            "alpha": []
        }

        current_step = 0

        num_complete_update_steps = 0
        steps_per_update = params['training_frequency']
        # steps_per_update = training_frequency

        # initialize hidden state for lstm (h0, c0), array is shape [num_agents, 1, memory_size//2]
        memories_out = torch.zeros([1, num_agents, params.get("memory_size")], dtype=torch.float32, requires_grad=False) \
            if use_lstm else None

        # Start environment and get initial state
        print(f"\nEnvironment Started for current run.")
        obs, r = env.env_start()

        while current_step <= total_run_steps:

            memories_in = memories_out

            current_step += num_agents

            i_min, i_max = current_step - num_agents, current_step

            # select action either using policy or random if warming up buffer
            if current_step < start_steps:
                action = np.random.uniform(-1.0, 1.0, (num_agents, act_dim))
                alpha_array[run, i_min:i_max] = alpha
                eval_values["alpha"].extend([alpha] * num_agents)
            else:
                # determine action via policy
                if use_lstm:
                    action, memories_out = policy.select_stochastic_action(obs=obs, memories_in=memories_in)
                else:
                    action = policy.select_stochastic_action(obs)

            # perform action
            next_obs, rewards, terminals, truncated, experiences_to_store = env.env_step(
                action=action, observation=obs)

            dones = (terminals + truncated) > 0

            episode_timesteps += 1
            episode_return += rewards.flatten()

            if use_lstm:
                replay_buffer.store(*experiences_to_store, memories_in, memories_out)
            else:
                replay_buffer.store(*experiences_to_store)
                replay_buffer.current_episode_nums[dones.flatten()] += 1

            obs = next_obs

            # Update observation normalization for each complete trajectory.
            if not use_lstm:
                complete_traj_length = episode_timesteps * dones.flatten()
                for i, l in enumerate(complete_traj_length):
                    if l > 0:
                        traj_obs = replay_buffer.get_trajectory(i, int(l))
                        policy.actor.update_normalization(traj_obs)
            else:
                if len(replay_buffer.complete_trajectory_observations) > 0:
                    complete_trajectory_obs = replay_buffer.get_complete_trajectory_observations(num_transitions=-1)
                    # print(f"traj obs: {len(complete_trajectory_obs)}")
                    for complete_obs in complete_trajectory_obs:
                        # print(f"trying to update normalization: {complete_obs[0].shape}")
                        # complete_obs = [complete_obs.squeeze(1)]
                        # print(f"Complete Obs: {complete_obs.shape}")
                        policy.actor.update_normalization(complete_obs)

            # append for storage
            episode_returns.extend(list(episode_return[dones.flatten()]))
            episode_lengths.extend(list(episode_timesteps[dones.flatten()]))
            # reset to 0
            episode_return *= (1 - dones).flatten()
            episode_timesteps *= (1 - dones).flatten()

            # reset memory to zero for done agents
            if memories_out is not None:
                memories_out[:, dones.flatten(), :] = 0

            if current_step >= start_steps:
                # Does a training update happen this step?

                perform_update = (
                                             current_step - start_steps) > steps_per_update * num_complete_update_steps and training_frequency > 0

                # Perform training update and store relevant values
                if perform_update:
                    actor_l, critic_l, alpha_l = 0.0, 0.0, 0.0
                    v, q, log_p, entropy, alpha = 0, 0, 0, 0, 0

                    # update the learning rates according to training progress
                    policy.update_learning_rate(current_step - start_steps)

                    old_total_updates = num_complete_update_steps

                    policy.critic.copy_normalization(policy.actor.body)
                    policy.critic_target.copy_normalization(policy.actor.body)

                    if type(replay_buffer) is PrioritizedLSTMReplayBuffer:
                        replay_buffer.update_importance_sampling_exponent(current_step - start_steps,
                                                                          total_run_steps - start_steps)

                    while perform_update:
                        loss, values = policy.update(replay_buffer, batch_size=batch_size)

                        num_complete_update_steps += 1
                        perform_update = (current_step - start_steps) > steps_per_update * num_complete_update_steps

                        actor_l += loss.get("actor")
                        critic_l += loss.get("critic")
                        alpha_l += loss.get("alpha")
                        q += values.get("q")
                        log_p += values.get("log_p")
                        entropy += values.get("entropy")
                        alpha += values.get("alpha")

                    updates_done = float(num_complete_update_steps - old_total_updates)

                    eval_losses["actor"].append(actor_l / updates_done)
                    eval_losses["critic"].append(critic_l / updates_done)
                    eval_losses["alpha"].append(alpha_l / updates_done)
                    eval_values["v"].append(v / updates_done)
                    eval_values["q"].append(q / updates_done)
                    eval_values["log_p"].append(log_p / updates_done)
                    eval_values["entropy"].append(entropy / updates_done)
                    eval_values["alpha"].append(alpha / updates_done)

            # Store array information
            if (current_step % eval_freq) // num_agents == 0 or current_step % total_run_steps == 0:

                storage_index = current_step // eval_freq

                # store what the current step is for plotting
                time_for_eval = time.time() - start_time
                steps_so_far[run, storage_index] = current_step
                time_so_far[run, storage_index] = time_for_eval

                if len(episode_returns) > 0:
                    mean_return = np.mean(episode_returns).item()
                    mean_length = np.mean(episode_lengths).item()
                    episode_returns.clear()
                    episode_lengths.clear()
                else:
                    # if no episode completed, then use mean return/length of current episodes
                    mean_return = np.mean(episode_return).item()
                    mean_length = np.mean(episode_timesteps).item()

                # store average return/length for the evaluation period
                avg_ep_return_array[run, storage_index] = mean_return
                avg_ep_length_array[run, storage_index] = mean_length

                # handle side stats of the environment (success_rate, fail_rate, euclidean_fail_rate, timeout_fail_rate)
                side_stats = env.get_stats()
                s, f, e_f, t_f = 0, 0, 0, 0
                if len(side_stats) > 0:
                    successes = [v[0] for v in side_stats.get("success_rate", 0)]
                    fails = [v[0] for v in side_stats.get("fail_rate", 0)]
                    euc_fails = [v[0] for v in side_stats.get("euclidean_fail_rate", 0)]
                    timeouts = [v[0] for v in side_stats.get("timeout_fail_rate", 0)]
                    s = np.mean(successes) if len(successes) > 0 else 0
                    f = np.mean(fails) if len(fails) > 0 else 0
                    e_f = np.mean(euc_fails) if len(euc_fails) > 0 else 0
                    t_f = np.mean(timeouts) if len(timeouts) > 0 else 0

                success_rate[run, storage_index] = s
                fail_rate[run, storage_index] = f
                euclidean_fail_rate[run, storage_index] = e_f
                timeout_fail_rate[run, storage_index] = t_f
                max_goal_distances[run, storage_index] = goal_curriculum.get_value()

                print(
                    f"Decision Step {current_step} ({num_complete_update_steps}): Time Elapsed: {time_for_eval:.3f} s Mean Return {mean_return:.3f}. Mean Length {mean_length:.3f}. Mean Success Rate {s:.3f}.\n", flush=True)

                # handle trajectory positions for heatmap plotting
                trajectories, end_conditions, goal_positions, goal_distances = env.get_trajectory_positions()
                if trajectories is not None:
                    trajectory_positions_dict[run][current_step].append(trajectories)
                    np.save(results_dir + "/trajectory_positions.npy", np.array(dict(trajectory_positions_dict)))
                if end_conditions is not None:
                    end_conditions_dict[run][current_step] = end_conditions
                    np.save(results_dir + "/end_conditions.npy", np.array(dict(end_conditions_dict)))
                if goal_positions is not None:
                    goal_positions_dict[run][current_step].append(goal_positions)
                    np.save(results_dir + "/goal_positions.npy", np.array(dict(goal_positions_dict)))

                if goal_distances is not None:
                    dists = list(goal_distances.values())
                    avg_dist = np.mean(dists) if len(dists) > 0 else 0
                    avg_goal_distances[run][storage_index] = avg_dist

                # save losses/update values
                actor_loss_array[run, current_step // eval_freq] = np.mean(eval_losses["actor"]) if len(
                    eval_losses["actor"]) > 0 else 0
                critic_loss_array[run, current_step // eval_freq] = np.mean(eval_losses["critic"]) if len(
                    eval_losses["critic"]) > 0 else 0
                if (len(eval_losses.get("alpha")) > 0):
                    alpha_loss_array[run, current_step // eval_freq] = np.mean(eval_losses["alpha"])

                v_array[run, current_step // eval_freq] = np.mean(eval_values["v"]) if len(eval_values["v"]) > 0 else 0
                q_array[run, current_step // eval_freq] = np.mean(eval_values["q"]) if len(eval_values["q"]) > 0 else 0
                log_p_array[run, current_step // eval_freq] = np.mean(eval_values["log_p"]) if len(
                    eval_values["log_p"]) > 0 else 0
                entropy_array[run, current_step // eval_freq] = np.mean(eval_values["entropy"]) if len(
                    eval_values["entropy"]) > 0 else 0
                alpha_array[run, current_step // eval_freq] = np.mean(eval_values["alpha"]) if len(
                    eval_values["alpha"]) > 0 else 0

                for k in eval_losses.keys():
                    eval_losses[k].clear()
                for k in eval_values.keys():
                    eval_values[k].clear()

                # run policy evaluation with deterministic policy
                if num_test_eps > 0:
                    iteration = current_step // eval_freq
                    test_return, test_length = eval_policy(policy, eval_env, num_test_eps)
                    test_returns[run, iteration] = test_return
                    test_lengths[run, iteration] = test_length
                    print(
                        f"\nStep {current_step}: Time Elapsed: {time_for_eval:.3f} s Mean Return {test_return:.3f}. Mean Length {test_lengths:.3f}. Mean Success Rate {s:.3f}.")

                # update curriculum value to current one in the environment
                goal_curriculum.update(s)
                env.env_params.set_float_parameter(curriculum_name, goal_curriculum.get_value())
                if eval_env is not None:
                    eval_env.env_params.set_float_parameter(curriculum_name, goal_curriculum.get_value())

                # save numpy array results and model at this checkpoint
                step_data = [steps_so_far, time_so_far]
                return_data = [avg_ep_return_array, test_returns]
                length_data = [avg_ep_length_array, test_lengths]
                loss_data = [actor_loss_array, critic_loss_array, alpha_loss_array]
                value_data = [v_array, q_array, log_p_array, entropy_array, alpha_array]
                side_stats_data = [success_rate, fail_rate, euclidean_fail_rate, timeout_fail_rate]
                goal_distance_data = [max_goal_distances, avg_goal_distances]

                data_to_save = [step_data, return_data, loss_data, value_data, length_data, side_stats_data,
                                goal_distance_data]
                save_data(data_to_save, exp_name, results_dir, seeds)

            if current_step % checkpoint_interval // num_agents == 0 or current_step % total_run_steps == 0:
                policy.save(save_dir=results_dir, suffix=str(run))
                policy.onnx_actor.copy_policy(policy.actor)
                policy.deterministic_onnx_actor.copy_policy(policy.actor)
                onnx_saver.export_onnx_model(results_dir, str(run))

    if env is not None:
        env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
