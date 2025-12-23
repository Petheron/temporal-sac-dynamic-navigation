import argparse
import math
import os
import time
from typing import Optional

import torch.backends.cudnn

from PPO import PPO
from DataStructures import RolloutBuffer
from Environment import *
from Utils import *
import yaml
import random
import gc
import distutils.util

from EncoderUtils import NetworkSettings, VisualSettings, PositionalEmbeddingSettings, MemorySettings


def eval_policy(policy, env: Environment, num_eval_episodes=10):
    # set env seed
    env.env_params.set_float_parameter("seed", np.random.randint(0, 10000))

    complete_eps = 0
    episode_returns = []
    episode_lens = []
    o, r = env.env_start()
    num_agents = env.num_agents
    agent_ret = np.zeros(num_agents)
    agent_ep_len = np.zeros(num_agents, dtype=np.int32)

    while complete_eps < num_eval_episodes * num_agents:
        action, memories_out = policy.get_deterministic_action(o)
        o, r, terminals, trunc, _ = env.env_step(action, o)
        agent_ret += r
        agent_ep_len += 1
        is_dones = (terminals + trunc)>0
        complete_eps += is_dones.sum()

        if is_dones.any():
            episode_returns.extend(list(agent_ret[is_dones]))
            episode_lens.extend(list(agent_ep_len[is_dones]))
            agent_ret[is_dones] = 0
            agent_ep_len[is_dones] = 0

    return np.mean(episode_returns), np.mean(episode_lens)


def rollout(policy,
            rollout_buffer: RolloutBuffer,
            env: Environment,
            current_step: int,
            num_steps: int,
            starting_obs: List[np.ndarray],
            starting_terminals: np.ndarray,
            starting_trunc: np.ndarray,
            starting_memories_in: Optional[np.ndarray] = None
            ):
    assert num_steps > 0
    current_obs = starting_obs
    current_terminals = starting_terminals
    current_truncated = starting_trunc
    obs_for_value = None
    terminals_for_value = None
    memories_out = starting_memories_in

    for i in range(0, num_steps):
        memories_in = memories_out
        current_step += 1 * env.num_agents
        with torch.no_grad():
            new_actions, memories_out = policy.get_action(current_obs, memories_in)
        next_obs, rewards, terminals, truncated, experience_to_store = env.env_step(new_actions, current_obs)
        # note a_to_store, o_to_store in agent_id order unlike obs/action/next_obs which are in decision id order
        # rewards, terminals, truncated, etc are already in agent id order
        obs_to_store, a_to_store, _, next_obs_for_val, _ = experience_to_store
        with torch.no_grad():
            log_probs_to_store, _ = policy.get_log_prob_and_entropy(obs_to_store, a_to_store)
            values_to_store = policy.get_value(obs_to_store)
        log_probs_to_store, values_to_store = torch_to_numpy(log_probs_to_store), torch_to_numpy(values_to_store)
        rollout_buffer.store(obs_to_store, a_to_store, rewards, current_terminals, current_truncated, log_probs_to_store, values_to_store, memories_in, memories_out)
        current_obs = next_obs
        current_terminals = terminals
        current_truncated = truncated

        if memories_out is not None:
            dones = (terminals + truncated) > 0
            memories_out[dones.flatten(), :, :] = 0

        if i == num_steps-1:
            # cache the obs & done for the next state
            obs_for_value = next_obs_for_val
            terminals_for_value = current_terminals
    return current_obs, current_terminals, current_truncated, obs_for_value, terminals_for_value, memories_out


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--env_name', '-env', type=str, default='ContinuousCartpole')
    parser.add_argument('--env_dir', '-dir', type=str, default='../Environments/ContinuousCartpole/Build/')
    parser.add_argument('--results_dir', '-r_dir', type=str, default='../Environments/ContinuousCartpole/Build/results')
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--time_scale', type=float, default=20.)
    parser.add_argument('--no_graphics', action='store_true')

    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--total_run_steps', type=int, default=50000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--num_test_eps', type=int, default=10)

    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--buffer_size', '-buff', type=int, default=1000000)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--normalize', type=distutils.util.strtobool, default=False)

    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--learning_rate_schedule', type=str, default='constant')

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--training_frequency', type=int, default=1,
                        help='How often to train the agent. If set to 0, agent is trained at the end of each episode.')
    parser.add_argument('--num_training_updates', type=int, default=1,
                        help='Number of times to update the network per training step. If set to 0, train training_frequency times.')

    parser.add_argument('--checkpoint_interval', type=int, default=50000)
    parser.add_argument('--load_seeds', action='store_true')
    parser.add_argument('--load_model', type=str, default="")

    parser.add_argument('--environment_parameters', type=yaml.safe_load, action="store")

    # LSTM Parameters
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--sequence_length', type=int, default=0)

    # PPO parameters
    parser.add_argument("--beta", type=float,
                        default=0.01)  # The entropy bonus coefficient.
    parser.add_argument("--epsilon", type=float,
                        default=0.2)  # The clipping range for the surrogate update.
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument("--num_epoch", type=int,
                        default=5)  # Number of update epochs per update. How many loops through the rollout buffer.
    parser.add_argument("--time_horizon", type=int,
                        default=20)  # Number of steps per agent before adding to replay buffer.

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    # clear gpu/RAM memory
    gc.collect()
    torch.cuda.empty_cache()

    params = arg_parser()
    env_params = params['environment_parameters']
    env_dir = params['env_dir']
    env_name = params['env_name']
    results_dir = params['results_dir']
    policy_name = params["policy"]
    exp_name = params['exp_name']

    # set worker_id for environment and assert it is even to allow for test env on (id + 1)
    worker_id = env_params.get("env_id")
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
    eval_env_info["worker_id"] = worker_id+1
    eval_env_info["no_graphics"] = True

    env = Environment()
    env.env_init(env_info)
    num_agents = env.num_agents

    eval_env = Environment()
    eval_env.env_init(eval_env_info)

    observation_specs = env.observation_specs
    act_dim = env.continuous_action_size if env.continuous_action_size > 0 else env.discrete_action_shape[0]

    runs = params['runs']
    batch_size = params['batch_size']
    buffer_size = params['buffer_size']

    #ensure buffer_size is divisible by batch_size AND num_agents, increase buffer_size until achieved.
    # simplest way seems to be adjust batch_size to be divisible by num_agents, then buffersize by batch_size
    batch_size = math.ceil(batch_size/float(num_agents))*num_agents
    buffer_size = math.ceil(buffer_size/float(batch_size))*batch_size

    # print(f"\n PPO with batch size {batch_size} and buffer size {buffer_size} for {num_agents} agents")

    total_run_steps = params['total_run_steps']
    total_run_steps = total_run_steps - (
            total_run_steps % buffer_size - buffer_size) if total_run_steps % buffer_size > 0 else total_run_steps

    num_test_eps = params['num_test_eps']
    # eval_freq = params['eval_freq']
    eval_freq = buffer_size
    checkpoint_interval = params['checkpoint_interval']
    num_evals = total_run_steps // eval_freq + 1  # extra 1 for initial test

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

    visual_settings = VisualSettings(16,
                                     params["environment_parameters"].get("2D_kernel_size",3),
                                     params["environment_parameters"].get("2D_stride",1),
                                     params["environment_parameters"].get("2D_maintain_input_size",0)
                                     )
    boxcast_settings = VisualSettings(16,
                                      params["environment_parameters"].get("3D_kernel_size",3),
                                      params["environment_parameters"].get("3D_stride",1),
                                      params["environment_parameters"].get("3D_maintain_input_size",0),
                                      params["environment_parameters"].get("3D_embedding_dictionary_size",2),
                                      params["environment_parameters"].get("3D_num_embedding_dims",0)
                                      )
    positional_embedding_settings = PositionalEmbeddingSettings(
        params["environment_parameters"].get("positional_encoding_embedding_size",30),
        params["environment_parameters"].get("position_encoding_max_length",41)
    )

    sequence_length = params.get("sequence_length", 0)
    memory_size = params.get("memory_size", 0)
    memory_settings = MemorySettings(sequence_length, memory_size) if sequence_length > 0 and memory_size > 0 else None
    use_lstm = memory_settings is not None

    network_settings = NetworkSettings(hidden_size=params["hidden_size"],
                                       num_layers=params["num_layers"],
                                       normalize=params["normalize"],
                                       conv2d_settings=visual_settings.settings,
                                       conv3d_settings=boxcast_settings.settings,
                                       positional_embedding_settings=positional_embedding_settings,
                                       memory_settings=memory_settings
                                       )

    kwargs = {
        "observation_specs": observation_specs,
        "act_dim": act_dim,
        "network_settings": network_settings,
        "act_lr": params["actor_lr"],
        "crit_lr": params["critic_lr"],
        "learning_rate_schedule": params["learning_rate_schedule"],
        "max_steps": total_run_steps,
        "sequence_length": sequence_length
    }

    # Policy Setup
    if policy_name == "PPO":
        kwargs = {**kwargs,
                  **{
                      "epsilon": params["epsilon"],
                      "num_epoch": params["num_epoch"],
                      "beta": params["beta"]
                  }
                  }
        Policy = PPO

    # variables for rollout buffer
    gamma = params.get("gamma")
    lambd = params.get("lambd")

    # Result Storage
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

    steps_so_far = np.ones((runs, num_evals), dtype=np.int32)*-1
    time_so_far = np.zeros((runs, num_evals))
    avg_ep_return_array = np.zeros((runs, num_evals))
    avg_ep_length_array = np.zeros((runs, num_evals))

    actor_loss_array = np.zeros((runs, num_evals))
    critic_loss_array = np.zeros((runs, num_evals))

    q_array = np.zeros((runs, num_evals))
    v_array = np.zeros((runs, num_evals))
    log_p_array = np.zeros((runs, num_evals))
    entropy_array = np.zeros((runs, num_evals))



    # 1 iteration is buffer_size steps
    # for run in tqdm(range(runs), leave=True):
    for run in range(runs):
        # set the seed parameters for all rng mechanics
        start_time = time.time()
        seed = seeds[run]
        env_info["seed"] = seed
        eval_env_info["seed"] = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # initialize environment
        env.env_init(env_info)
        if eval_env is not None:
            eval_env.env_init(eval_env_info)

        # set up rollout buffer for this run
        rollout_buffer = RolloutBuffer(observation_specs, act_dim, buffer_size, gamma, lambd,
                                            num_agents=num_agents)

        # load up the policy
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

        episode_returns = []  # complete episode returns list
        episode_lengths = []  # complete episode lengths list

        eval_losses = {
            "actor": [],
            "critic": [],
        }
        eval_values = {
            "v": [],
            "q": [],
            "log_p": [],
            "entropy": [],
        }

        current_step = 0
        # pbar = tqdm(total=total_run_steps, leave=False)
        updates_per_run = total_run_steps // buffer_size
        steps_per_update_per_agent = buffer_size // num_agents

        # return/length to roll-over between updates
        rolling_over_returns, rolling_over_lengths = np.zeros(num_agents), np.zeros(num_agents)

        # initialize hidden state for lstm (h0, c0), array is shape [batch, sequence_length*num_lstm_layers, memory_size]
        # c0 is concatenated to h0, each with size hidden_size = memory_size//2
        memories_out = np.zeros(
            [num_agents, 1, params.get("memory_size")], dtype=np.float32) if use_lstm \
            else None

        # env does not need to restart after rollout, next rollout continues environment but with fresh trajectory
        obs, r = env.env_start()
        dones = np.zeros(num_agents)
        trunc = np.zeros(num_agents)

        for update in range(1, updates_per_run + 1):
            memories_in = memories_out
            current_step += steps_per_update_per_agent * num_agents

            obs, dones, trunc, obs_for_val, dones_for_val, memories_out = rollout(policy, rollout_buffer, env, current_step, steps_per_update_per_agent, obs, dones, trunc, memories_in)
            next_value = torch_to_numpy(policy.get_value(obs_for_val, memories_out))
            rollout_buffer.compute_advantages_and_returns(next_value, dones_for_val)
            update_returns, update_lengths, any_dones = rollout_buffer.get_episode_returns_lengths_per_agent()
            for i in range(num_agents):
                d = any_dones[i] == 1
                if d:
                    update_returns[i][0] += rolling_over_returns[i]
                    update_lengths[i][0] += rolling_over_lengths[i]
                    rolling_over_returns[i] = update_returns[i][-1]
                    rolling_over_lengths[i] = update_lengths[i][-1]
                else:
                    rolling_over_returns[i] += update_returns[i][-1]
                    rolling_over_lengths[i] += update_lengths[i][-1]
                episode_returns.extend(update_returns[i][:-1])
                episode_lengths.extend(update_lengths[i][:-1])

            # update policy using experiences from rollout buffer
            policy.update_learning_rate(current_step)
            loss, values = policy.update(rollout_buffer, batch_size)
            rollout_buffer.reset()

            eval_losses["actor"].append(loss["actor"])
            eval_losses["critic"].append(loss["critic"])
            eval_values["v"].append(values["v"])
            eval_values["q"].append(values["q"])
            eval_values["log_p"].append(values["log_p"])
            eval_values["entropy"].append(values["entropy"])

            # Store array information
            if (current_step % eval_freq) // buffer_size == 0 or current_step % total_run_steps == 0:

                storage_index = current_step // eval_freq

                # store what the current step is for plotting
                time_for_eval = time.time() - start_time
                steps_so_far[run, storage_index] = int(current_step)
                time_so_far[run, storage_index] = time_for_eval

                # handle storage of average returns/lengths
                r = np.mean(episode_returns)
                l = np.mean(episode_lengths)
                episode_returns.clear()
                episode_lengths.clear()
                # if no episode completed, then use mean return/length of current episodes
                r = np.nan_to_num(r, False, np.mean(rolling_over_returns).item())
                l = np.nan_to_num(l, False, np.mean(rolling_over_returns).item())

                avg_ep_return_array[run, storage_index] = r
                avg_ep_length_array[run, storage_index] = l

                # handle side stats of the environment (success_rate, fail_rate, euclidean_fail_rate, timeout_fail_rate)
                side_stats = env.get_stats()
                s, f, e_f, t_f = 0, 0, 0, 0
                if len(side_stats) > 0:
                    s = np.mean([v[0] for v in side_stats.get("success_rate", 0)])
                    f = np.mean([v[0] for v in side_stats.get("fail_rate", 0)])
                    e_f = np.mean([v[0] for v in side_stats.get("euclidean_fail_rate", 0)])
                    t_f = np.mean([v[0] for v in side_stats.get("timeout_fail_rate", 0)])

                success_rate[run, storage_index] = s
                fail_rate[run, storage_index] = f
                euclidean_fail_rate[run, storage_index] = e_f
                timeout_fail_rate[run, storage_index] = t_f

                print(
                    f"\nStep {current_step}: Time Elapsed: {time_for_eval:.3f} s Mean Return {r:.3f}. Mean Length {l:.3f}. Mean Success Rate {s:.3f}.")

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
                    np.save(results_dir+"/goal_positions.npy", np.array(dict(goal_positions_dict)))

                if goal_distances is not None:
                    dists = list(goal_distances.values())
                    avg_dist = 0
                    if len(dists) > 0:
                        avg_dist = np.mean(dists)
                    avg_goal_distances[run][storage_index] = avg_dist

                actor_loss_array[run, storage_index] = np.mean(eval_losses["actor"]) if len(
                    eval_losses["actor"]) > 0 else 0
                critic_loss_array[run, storage_index] = np.mean(eval_losses["critic"]) if len(
                    eval_losses["critic"]) > 0 else 0

                v_array[run, storage_index] = np.mean(eval_values["v"])
                q_array[run, storage_index] = np.mean(eval_values["q"])
                log_p_array[run, storage_index] = np.mean(eval_values["log_p"])
                entropy_array[run, storage_index] = np.mean(eval_values["entropy"])

                for k in eval_losses.keys():
                    eval_losses[k].clear()
                for k in eval_values.keys():
                    eval_values[k].clear()

                # run policy evaluation with deterministic policy
                if num_test_eps > 0:
                    test_return, test_length = eval_policy(policy, eval_env, num_test_eps)
                    test_returns[run, storage_index] = test_return
                    test_lengths[run, storage_index] = test_length
                    print(
                        f"\nStep {current_step}: Test Return: {test_return:.3f} with episode length {test_length:.3f}")

                # save numpy array results and model at this checkpoint
                step_data = [steps_so_far, time_so_far]
                return_data = [avg_ep_return_array, test_returns]
                length_data = [avg_ep_length_array, test_lengths]
                loss_data = [actor_loss_array, critic_loss_array]
                value_data = [v_array, q_array, log_p_array, entropy_array]
                side_stats_data = [success_rate, fail_rate, euclidean_fail_rate, timeout_fail_rate]

                data_to_save = [step_data, return_data, loss_data, value_data, length_data, side_stats_data]
                save_data(data_to_save, exp_name, results_dir, seeds)

            if current_step % checkpoint_interval // buffer_size == 0 or current_step % total_run_steps == 0:
                policy.save(results_dir, str(run))
                policy.onnx_actor.copy_policy(policy.agent.actor)
                policy.deterministic_onnx_actor.copy_policy(policy.agent.actor)
                onnx_saver.export_onnx_model(results_dir, str(run))

            # pbar.update(buffer_size)
        # pbar.close()


if __name__ == "__main__":
    main()
