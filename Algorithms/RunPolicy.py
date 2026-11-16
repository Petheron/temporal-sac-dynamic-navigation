import torch
import os


from Utils import plot_from_file, load_data, plot_trajectory_positions
from config_loader import load_yaml, get_behavior_settings, get_network_settings, get_hyperparameters, get_max_steps, get_trainer_type, get_reward_signals, get_environment_parameters
import shutil

def main():
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=int, default=0)
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--plot_trajectories', '-pt', type=int, default=0)
    parser.add_argument('--show_plots', '-sp', action='store_true')
    parser.add_argument('--config', '-c', type=str, default="sac.yaml")
    parser.add_argument('--batch_config_dir', '-bc', type=str, default="")
    parser.add_argument('--env', '-e', type=str, default="ContinuousCartpole")
    parser.add_argument('--time_scale', '-t', type=int, default=20)
    parser.add_argument("--initialize_from", '-i', type=str, default="")  # the name of the file to initialize model from
    parser.add_argument("--name_suffix", '-n', type=str, default="")  # name of directory to save/load results to
    parser.add_argument('--load_seeds', action='store_true')
    parser.add_argument('--no_graphics', '-ng', action='store_true')
    parser.add_argument('--env_id', '-id', type=int, default=0)
    parser.add_argument('--dir', '-d', type=str, default="Results")

    args = parser.parse_args()

    default_results_dir = f'../{args.dir}/'
    env_dir = f'../Environments/{args.env}/'
    build_dir = f'{env_dir}Build/'
    config_dir = f'{env_dir}Config/{args.batch_config_dir}/'
    config_yaml = f'{config_dir}{args.config}'

    config = load_yaml(config_yaml)
    behavior_settings = get_behavior_settings(config)
    policy = get_trainer_type(config).upper()
    environment_parameters = get_environment_parameters(config)
    hyper_params = get_hyperparameters(behavior_settings)
    network_params = get_network_settings(behavior_settings)
    reward_signal_params = get_reward_signals(behavior_settings)

    # Common parameters
    runs = args.run
    total_steps_per_run = int(get_max_steps(behavior_settings))
    time_scale = args.time_scale

    buffer_size = int(hyper_params.get('buffer_size', 1000000))
    batch_size = int(hyper_params.get('batch_size', 128))
    gamma = reward_signal_params['extrinsic'].get('gamma', 0.99)
    learning_rate = hyper_params.get('learning_rate', 3.0e-4)
    learning_rate_schedule = hyper_params.get("learning_rate_schedule", "constant")
    act_lr = learning_rate
    crit_lr = learning_rate
    max_steps = environment_parameters.get('max_steps', 1000)
    environment_parameters["env_id"] = args.env_id

    print("-" * 72, flush=True)
    print(f"[RunPolicy] Config File           : {config_yaml}", flush=True)
    print(f"[RunPolicy] env_params Keys        : {sorted(environment_parameters.keys())}", flush=True)
    print(f"[RunPolicy] temporal_architecture  : "
          f"{environment_parameters.get('temporal_architecture', '<ABSENT>')!r}", flush=True)
    print(f"[RunPolicy] environment_parameters : {environment_parameters}", flush=True)
    print("-" * 72, flush=True)

    # Neural Network Parameters
    hidden_size = int(network_params.get('hidden_units', 512))
    num_layers = int(network_params.get('num_layers', 2))
    normalize = network_params.get('normalize', False)

    # Memory Settings
    memory = network_params.get('memory')
    sequence_length = memory.get('sequence_length', 0) if memory is not None else 0
    memory_size = memory.get('memory_size', 0) if memory is not None else 0

    # Model saving & evaluation parameters
    eval_freq = int(behavior_settings.get('summary_freq', 1000))
    checkpoint_interval = int(behavior_settings.get('checkpoint_interval', 50000))
    num_test_eps = int(environment_parameters.get("num_test_eps", 0))

    ## Off-policy parameters
    starting_steps = int(hyper_params.get('buffer_init_steps', 10000))
    init_alpha = hyper_params.get('init_entcoef', 0.5)
    alpha_lr = learning_rate
    tau = hyper_params.get('tau', 0.005)
    policy_freq = 2 if policy != "SAC" else 1
    train_freq = int(hyper_params.get('steps_per_update', 1))
    num_train_updates = int(environment_parameters.get('num_train_updates', 0))

    # on-policy parameters
    beta = hyper_params.get("beta", 1.0e-3)
    epsilon = hyper_params.get("epsilon", 0.2)
    lambd = hyper_params.get("lambd", 0.95)
    num_epoch = int(hyper_params.get("num_epoch", 10))
    time_horizon = int(behavior_settings.get("time_horizon", 10))

    algo_name = "OffPolicyMain" if (policy == "SAC" or policy == "TD3" or policy == "DDPG") else "OnPolicyMain"
    env_name = args.env

    exp_name = f"{policy}_{env_name}"+args.name_suffix

    # directory to plot from
    results_dir = default_results_dir+exp_name
    load_dir = default_results_dir+args.initialize_from

    # construct directory for new experiment
    new_run_results_dir = results_dir
    if args.initialize_from != "default":
        counter = 0
        while os.path.exists(new_run_results_dir):
            new_run_results_dir = results_dir + f"_{counter}"
            counter += 1
        if args.run > 0:
            if new_run_results_dir != results_dir:
                print(f"\nDirectory exists for {exp_name}, new directory at {new_run_results_dir}")
            os.mkdir(new_run_results_dir)

    plot_name = exp_name

    cmd = f"python {algo_name}.py --policy {policy} --exp_name {exp_name} -dir {build_dir} -r_dir {new_run_results_dir} \
    --env_name {env_name} --time_scale {time_scale} --max_steps {max_steps} \
    -r {runs} --total_run_steps {total_steps_per_run} \
    --actor_lr {act_lr} --critic_lr {crit_lr} --learning_rate_schedule {learning_rate_schedule} \
    --batch_size {batch_size} --buffer_size {buffer_size} --gamma {gamma} \
    --hidden_size {hidden_size} --num_layers {num_layers} --normalize {normalize}\
    --training_frequency {train_freq} \
    --checkpoint_interval {checkpoint_interval} --eval_freq {eval_freq} --num_test_eps {num_test_eps} \
    --sequence_length {sequence_length} --memory_size {memory_size} \
    --environment_parameters \"{environment_parameters}\""

    if algo_name == "OffPolicyMain":
        cmd += f" -ss {starting_steps} --tau {tau} \
        --exploration_noise 0.1 \
        --policy_noise 0.2 --noise_clip 0.5 --policy_freq {policy_freq} \
        --alpha {init_alpha} --alpha_lr {alpha_lr}"
    else:
        cmd += f" --beta {beta} --epsilon {epsilon} --lambd {lambd} \
        --num_epoch {num_epoch} --time_horizon {time_horizon}"

    if len(args.initialize_from) > 0:
        cmd += f" --load_model {load_dir}"

    if args.load_seeds:
        cmd += f" --load_seeds"

    if args.no_graphics:
        cmd += f" --no_graphics"

    if args.run > 0:
        if os.path.isfile(config_yaml):
            shutil.copy(config_yaml, new_run_results_dir)
        print(f"[RunPolicy] launching: {cmd}", flush=True)
        os.system(cmd)

    # optional plotting argument. Use to refresh plots.
    if args.plot:
        plot_from_file(plot_name, results_dir, True, True, True, True, None, eval_freq=eval_freq,
                       smoothing=0, show=args.show_plots)

    if args.plot_trajectories>0:
        num_trajs = args.plot_trajectories
        positions_data = load_data("trajectory_positions", results_dir)
        end_conditions_data = load_data("end_conditions", results_dir)
        goal_positions_data = load_data("goal_positions", results_dir)
        if positions_data is not None:
            plot_trajectory_positions(results_dir, positions_data, end_conditions_data, goal_positions_data, num_trajs, 50, 10, 0, show=args.show_plots)


if __name__ == '__main__':
    main()
