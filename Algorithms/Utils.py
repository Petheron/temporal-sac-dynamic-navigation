import os
import threading
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from NetworkUtils import get_device
from enum import Enum

class ExperienceReplayType(Enum):
    BASIC = 'basic'
    PER = 'per' # Schaul 2016 http://arxiv.org/abs/1511.05952
    LA3P = 'la3p' # Saglam 2022 http://arxiv.org/abs/2209.00532
    PAL = 'pal' # Fujimoto 2020 http://arxiv.org/abs/2007.06049
    LAP = 'lap' # Fujimoto 2020 http://arxiv.org/abs/2007.06049
    RELO = 'relo' # Sujit 2022 https://arxiv.org/abs/2208.10483


class Curriculum:
    def __init__(self,
                 name: str = "",
                 final_value: float = 10,
                 use_curriculum: bool = False,
                 num_steps: int = 1,
                 advancement_condition: float = 0.0
                 ):
        self.use_curriculum = use_curriculum
        self.final_value = final_value
        self.num_steps = num_steps
        self.advancement_condition = advancement_condition
        self.current_step = 0
        self.current_value = 0
        self.name = name

        self.update(1.0)

    def update(self, condition_value: float):
        if not self.use_curriculum:
            return

        if self.current_step >= self.num_steps:
            return

        if condition_value < self.advancement_condition:
            return

        self.update_current_value()

    def update_current_value(self):
        self.current_step += 1
        if self.current_step > self.num_steps:
            self.current_step = self.num_steps

        self.current_value = self.final_value*self.current_step/self.num_steps

    def get_value(self) -> float:
        return self.current_value if self.use_curriculum else self.final_value


def create_name(exp_name, nn_baseline=False, initial_model=None, eval=False) -> str:
    load_name = exp_name
    if nn_baseline:
        load_name += "_with_baseline"
    if initial_model is not None:
        load_name += "_with_init"
    if eval:
        load_name += "_eval"
    return load_name


def save_data(data: list, exp_name: str, save_dir: str, seeds=None, params=None):
    """
    data is a list [[list of data names], [data0, data1, data2, ...]]
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = exp_name
    save_name = "results"
    filename = f"{save_dir}/{save_name}.npy"
    data = np.array(data, dtype=object)
    np.save(filename, data)
    # print(f"Data saved under {filename}")

    if seeds is not None:
        save_seeds(seeds, exp_name, save_dir)

    if params is not None:
        save_params(params, exp_name, save_dir)


def load_data(exp_name: str, result_dir: str):
    """
    Returns a data list in the form [ [names], [data_sets]]
    """
    if not os.path.exists(result_dir):
        print(f"No directory at {result_dir}")
        return None
    load_path = f"{result_dir}/{exp_name}.npy"
    data = np.load(load_path, allow_pickle=True)
    return data


def plot_from_file(exp_name: str, result_dir: str, plot_all_runs=False, plot_test=False,
                   plot_loss=False, plot_values=False, iterations_to_average_returns=None,
                   eval_freq=5000, smoothing=0, show = False):
    data = load_data("results", result_dir)
    step_data = data[0]
    steps_so_far = step_data[0]
    time_so_far = step_data[1]
    return_data = data[1]
    loss_data = data[2]
    loss_data = np.nan_to_num(loss_data)
    value_data = data[3]
    value_data = np.nan_to_num(value_data)
    length_data = data[4]
    side_stats_data = None if len(data) < 6 else data[5]
    goal_dist_data = None if len(data) < 7 else data[6]

    plot_fill_between(return_data[0], "Return", plot_all_runs=plot_all_runs,
                      window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                      save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
    plot_fill_between(length_data[0], "Episode Length", plot_all_runs=plot_all_runs,
                      window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                      save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
    if side_stats_data:
        names = ["Success Rate", "Fail Rate", "Euclidean Fail Rate", "Timeout Rate"]
        plot_many_fill_betweens_on_one_plot(side_stats_data, names, "Completion Rates", iterations_to_average_returns,
                                            eval_freq, smoothing, y_limits=(0.0, 1.0), save_dir=result_dir,
                                            steps_array=steps_so_far, time_array=time_so_far)

    if goal_dist_data:
        names = ["Curriculum Radius", "Average Goal Distance"]
        for i, name in enumerate(names):
            plot_fill_between(goal_dist_data[i], data_name=name, plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)

    if plot_test:
        plot_fill_between(return_data[1], "Test Return", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        plot_fill_between(length_data[1], "Test Episode Length", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
    if plot_loss:
        plot_fill_between(loss_data[0], "Policy Loss", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        plot_fill_between(loss_data[1], "Q Loss", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        if len(loss_data) > 2:
            plot_fill_between(loss_data[2], "Alpha Loss", plot_all_runs=plot_all_runs,
                              window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                              save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
    if plot_values:
        plot_fill_between(value_data[0], "V value estimate", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        plot_fill_between(value_data[1], "Q value estimate", plot_all_runs=plot_all_runs,
                          window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                          save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        if len(value_data) > 2:
            plot_fill_between(value_data[2], "Log Pi", plot_all_runs=plot_all_runs,
                              window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                              save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
            plot_fill_between(value_data[3], "Entropy", plot_all_runs=plot_all_runs,
                              window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                              save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
        if len(value_data) > 4:
            plot_fill_between(value_data[4], "Entropy Coefficient (alpha)", plot_all_runs=plot_all_runs,
                              window_size=iterations_to_average_returns, step_size=eval_freq, smoothing=smoothing,
                              save_dir=result_dir, steps_array=steps_so_far, time_array=time_so_far)
    if show:
        show_plots()


def plot_many_fill_betweens_on_one_plot(datas, names, plot_name, window_size, step_size, smoothing, y_limits,
                                        save_dir="", save_name: str = "", steps_array: np.ndarray = None,
                                        time_array: np.ndarray = None):
    runs = datas[0].shape[0]

    if steps_array is not None:
        x_range = steps_array[0]
    else:
        x_range = np.arange(datas[0].shape[1]) * step_size

    truth_array = x_range.copy() > -1
    x_range = x_range[truth_array]

    fig, ax = plt.subplots()

    ax2 = None
    x2_range = None
    if time_array is not None:
        ax2 = ax.twiny()
        x2_range = time_array[0]
        x2_range = x2_range[truth_array]
        x2_range = [int(x) for x in x2_range]

    for i, data in enumerate(datas):
        data = data
        correct_nans_in_data(data)
        if window_size is not None:
            apply_moving_window_average(data, window_size)
        if smoothing > 0:
            apply_tensorboard_smoothing(data, smoothing)

        data_mean = data.mean(axis=0).flatten()
        data_std_err = data.std(axis=0).flatten() / np.sqrt(runs)

        data_mean = data_mean[truth_array]
        data_std_err = data_std_err[truth_array]

        ax.fill_between(x_range, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
        ax.plot(x_range, data_mean, label=f'{names[i]} Average', )

    ax.set_title(f" {plot_name} averaged over {runs} runs")
    if y_limits is not None:
        ax.set_ylim(y_limits)
        ax.set_yticks(np.linspace(0, 1.0, 11))
    ax.set_xlim(0, x_range[-1])
    ax.set_xticks(np.linspace(0, x_range[-1], 6, dtype=np.int32))
    ax.legend(loc='best')

    if x2_range is not None:
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.linspace(0, x_range[-1], 6, dtype=np.int32))
        ax2.set_xticklabels(np.linspace(0, x2_range[-1], 6, dtype=np.int32))

    plt.savefig(f"{save_dir}/{plot_name}.pdf")
    plt.savefig(f"{save_dir}/{plot_name}.eps")


def plot_fill_between(data, data_name: str, plot_all_runs=False, window_size=None,
                      step_size: int = 1, smoothing: float = 0.0, y_limits: tuple = None, save_dir: str = "",
                      steps_array: np.ndarray = None, time_array: np.ndarray = None):
    """
    Plot data with shape (runs,iterations). Resultant plot shown mean and std_error of data over runs.

    If mean and std_err provided, find the average of both over runs. Otherwise calcualte the mean and std of given data.
    """

    data = data

    # correct any nans in data
    correct_nans_in_data(data)

    # smooth data using moving window average
    if window_size is not None:
        apply_moving_window_average(data, window_size)

    # Smooth data using Tensorboard smoothing method
    if smoothing > 0:
        apply_tensorboard_smoothing(data, smoothing)

    fig, ax = plt.subplots()
    ax2 = ax.twiny()

    runs = data.shape[0]
    data_mean = data.mean(axis=0).flatten()
    data_std_err = data.std(axis=0).flatten() / np.sqrt(runs)

    if steps_array is not None:
        x_range = steps_array[0]
    else:
        x_range = np.arange(data.shape[1]) * step_size

    truth_array = x_range.copy() > -1
    x_range = x_range[truth_array]

    x2_range = None
    if time_array is not None:
        x2_range = time_array[0]
        x2_range = x2_range[truth_array]
        x2_range = [int(x) for x in x2_range]

    data_mean = data_mean[truth_array]
    data_std_err = data_std_err[truth_array]

    if plot_all_runs:
        for i in range(runs):
            if steps_array is not None:
                x_range_run = steps_array[i]
            else:
                x_range_run = x_range
            truth_array_run = x_range_run.copy() > -1
            ax.plot(x_range_run[truth_array_run], data[i][truth_array_run], label=f"Run {i}", linestyle='solid',
                    alpha=0.5)
    ax.fill_between(x_range, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    ax.plot(x_range, data_mean, label=f'Average', color='blue')
    ax.set_title(f"{data_name} averaged over {runs} runs")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.set_xlim(0, x_range[-1])
    ax.set_xticks(np.linspace(0, x_range[-1], 6, dtype=np.int32))
    ax.legend(loc='best')
    if x2_range is not None:
        ax2.set_xticks(np.linspace(0, x_range[-1], 6, dtype=np.int32))
        ax2.set_xticklabels(np.linspace(x2_range[0], x2_range[-1], 6, dtype=np.int32))

    plt.savefig(f"{save_dir}/{data_name}.pdf")
    plt.savefig(f"{save_dir}/{data_name}.eps")


def plot_trajectory_positions(dir, traj_data, end_cond_data, goal_pos_data, num_eval_freqs=1, num_bins=10,
                              num_trajectories=10, min_traj_length=0, show=False):
    # use .item() to extract dictionary from np.load()
    traj_dict = dict(traj_data.item())
    end_cond_dict = dict(end_cond_data.item())
    goal_pos_dict = dict(goal_pos_data.item())

    # ensure num_bins is even, so bins centred around 0
    num_bins = num_bins + (num_bins % 2)

    run_list = list(traj_dict.keys())
    for run in run_list:
        run_traj_dict = traj_dict[run]
        run_end_cond_dict = end_cond_dict[run]
        run_goal_pos_dict = goal_pos_dict[run]
        run_steps_list = list(run_traj_dict.keys())
        step_range = (run_steps_list[-num_eval_freqs], run_steps_list[-1])
        # lists that contain all visited points in given step_range
        x_pos, y_pos, z_pos = [], [], []
        # lists for containing lists of steps per trajectory
        x_pos_per_traj, y_pos_per_traj, z_pos_per_traj = [], [], []
        end_conds = []
        goal_pos_x, goal_pos_y, goal_pos_z = [], [], []
        for step in run_steps_list:
            if step >= step_range[0] and step <= step_range[1]:
                id_traj_dict = run_traj_dict[step][0]
                id_end_cond_dict = run_end_cond_dict[step]
                id_goal_pos_dict = run_goal_pos_dict[step][0]
                episode_id_list = list(id_traj_dict.keys())
                for id in episode_id_list:
                    x_y_z_list = id_traj_dict[id]
                    x_pos.extend(x_y_z_list[0])
                    y_pos.extend(x_y_z_list[1])
                    z_pos.extend(x_y_z_list[2])
                    x_pos_per_traj.append(x_y_z_list[0])
                    y_pos_per_traj.append(x_y_z_list[1])
                    z_pos_per_traj.append(x_y_z_list[2])
                    end_conds.append(id_end_cond_dict[id])
                    goal_pos_x.append(id_goal_pos_dict[id][0])
                    goal_pos_y.append(id_goal_pos_dict[id][1])
                    goal_pos_z.append(id_goal_pos_dict[id][2])

        # create bins for x/y/z pos for heat map
        xyz_bins = np.zeros((num_bins, num_bins, num_bins))

        # Find the max range for each position dimension
        x_limit = np.amax(np.ceil(np.abs(x_pos)))
        y_limit = np.amax(np.ceil(np.abs(y_pos)))
        z_limit = np.amax(np.ceil(np.abs(z_pos)))

        # determine bin size
        bin_size_x = 2 * x_limit / float(num_bins)
        bin_size_y = 2 * y_limit / float(num_bins)
        bin_size_z = 2 * z_limit / float(num_bins)

        # place each position into a 3d bin (centred around (0,0,0))
        for i in range(len(x_pos)):
            bin_index_x = int(np.floor((num_bins / 2) + (x_pos[i] - 0) / bin_size_x))
            bin_index_y = int(np.floor((num_bins / 2) + (y_pos[i] - 0) / bin_size_y))
            bin_index_z = int(np.floor((num_bins / 2) + (z_pos[i] - 0) / bin_size_z))
            xyz_bins[bin_index_x, bin_index_y, bin_index_z] += 1

        # normalize positions for line plotting to be in [0,num_bins] range (maybe change to place in bins?)
        normalized_x_pos_per_traj = [num_bins * (x + x_limit) / (2 * x_limit) for x in x_pos_per_traj]
        normalized_y_pos_per_traj = [num_bins * (y + y_limit) / (2 * y_limit) for y in y_pos_per_traj]
        normalized_z_pos_per_traj = [num_bins * (z + z_limit) / (2 * z_limit) for z in z_pos_per_traj]

        normalized_goal_pos_x = [num_bins * (x + x_limit) / (2 * x_limit) for x in goal_pos_x]
        normalized_goal_pos_y = [num_bins * (y + y_limit) / (2 * y_limit) for y in goal_pos_y]
        normalized_goal_pos_z = [num_bins * (z + z_limit) / (2 * z_limit) for z in goal_pos_z]

        # compute heat map of xz (average over vertical occupancy)
        xz_view = xyz_bins.mean(axis=1)

        fig = plt.figure(figsize=(13, 13))
        fig.suptitle(f"Step Range {step_range}, num bins {num_bins}")
        fig.add_subplot(2, 2, 1)
        fig.add_subplot(2, 2, 2, projection='3d')
        fig.add_subplot(2, 2, 3, projection='3d')
        fig.add_subplot(2, 2, 4, projection='3d')
        axes = fig.get_axes()

        ax = axes[0]
        cp = ax.imshow(xz_view, cmap='gist_gray')
        ax.set_title("xz-view positions heatmap")
        ax.set_xlabel("x position")
        ax.set_ylabel("z position")
        fig.colorbar(cp, ax=ax)

        ax = axes[1]
        xyz_3d = xyz_bins > 0
        alpha = 0.05
        cp = ax.voxels(np.swapaxes(xyz_3d, 1, 2), facecolor=np.array([1, 0, 0, alpha]))
        ax.set_title(f"Trajectory positions for {num_trajectories} trajectories with end_cond success")
        ax.w_xaxis.set_pane_color([0, 0, 0, 1])
        ax.w_yaxis.set_pane_color([0, 0, 0, 1])
        ax.w_zaxis.set_pane_color([0, 0, 0, 1])
        num_traj_counter = 0
        for i in range(len(normalized_x_pos_per_traj)):
            if end_conds[i] == "success" and num_traj_counter < num_trajectories:
                col = np.random.rand(3)
                ax.plot(normalized_x_pos_per_traj[i], normalized_z_pos_per_traj[i], normalized_y_pos_per_traj[i],
                        color=col,
                        alpha=1.0)
                num_traj_counter += 1
                ax.scatter(normalized_x_pos_per_traj[i][0], normalized_z_pos_per_traj[i][0],
                           normalized_y_pos_per_traj[i][0], color=col, marker='o')
                ax.scatter(normalized_goal_pos_x[i], normalized_goal_pos_z[i], normalized_goal_pos_y[i], color=col,
                           marker='*')
        # fig.subplots_adjust(wspace=0.5)

        ax = axes[2]
        cp = ax.voxels(np.swapaxes(xyz_3d, 1, 2), facecolor=np.array([1, 0, 0, alpha]))
        ax.set_title(f"Trajectory positions for {num_trajectories} trajectories with end_cond fail.")
        ax.w_xaxis.set_pane_color([0, 0, 0, 1])
        ax.w_yaxis.set_pane_color([0, 0, 0, 1])
        ax.w_zaxis.set_pane_color([0, 0, 0, 1])
        num_traj_counter = 0
        for i in range(len(normalized_x_pos_per_traj)):
            if end_conds[i] == "fail" and num_traj_counter < num_trajectories:
                col = np.random.rand(3)
                ax.plot(normalized_x_pos_per_traj[i], normalized_z_pos_per_traj[i], normalized_y_pos_per_traj[i],
                        color=col,
                        alpha=1.0)
                num_traj_counter += 1
                ax.scatter(normalized_x_pos_per_traj[i][0], normalized_z_pos_per_traj[i][0],
                           normalized_y_pos_per_traj[i][0], color=col, marker='o')
                ax.scatter(normalized_goal_pos_x[i], normalized_goal_pos_z[i], normalized_goal_pos_y[i], color=col,
                           marker='*')
                ax.scatter(normalized_x_pos_per_traj[i][-1], normalized_z_pos_per_traj[i][-1],
                           normalized_y_pos_per_traj[i][-1], color=col, marker='x')

        ax = axes[3]
        cp = ax.voxels(np.swapaxes(xyz_3d, 1, 2), facecolor=np.array([1, 0, 0, alpha]))
        ax.set_title(f"Trajectory positions for {num_trajectories} trajectories with end_cond timout.")
        ax.w_xaxis.set_pane_color([0, 0, 0, 1])
        ax.w_yaxis.set_pane_color([0, 0, 0, 1])
        ax.w_zaxis.set_pane_color([0, 0, 0, 1])
        num_traj_counter = 0
        for i in range(len(normalized_x_pos_per_traj)):
            if end_conds[i] == "timeout" and num_traj_counter < num_trajectories:
                col = np.random.rand(3)
                ax.plot(normalized_x_pos_per_traj[i], normalized_z_pos_per_traj[i], normalized_y_pos_per_traj[i],
                        color=col,
                        alpha=1.0)
                num_traj_counter += 1
                ax.scatter(normalized_x_pos_per_traj[i][0], normalized_z_pos_per_traj[i][0],
                           normalized_y_pos_per_traj[i][0], color=col, marker='o')
                ax.scatter(normalized_goal_pos_x[i], normalized_goal_pos_z[i], normalized_goal_pos_y[i], color=col,
                           marker='*')
                ax.scatter(normalized_x_pos_per_traj[i][-1], normalized_z_pos_per_traj[i][-1],
                           normalized_y_pos_per_traj[i][-1], color=col, marker='x')
                print(f"{goal_pos_x[i], goal_pos_z[i], goal_pos_y[i]}")

        plt.savefig(f"{dir}/trajectory_positions_run_{run}.pdf")
    if show:
        show_plots()


def correct_nans_in_data(data):
    """
    Corrects any NaNs in the data by averaging from surrounding values.
    """
    for j, row in enumerate(data):
        for i, d in enumerate(row):
            if np.isnan(d):
                avg = []
                if i > 0: avg.append(row[i - 1])
                if i < len(row) - 1: avg.append(row[i + 1])
                avg = np.mean(avg).item()
                row[i] = avg
        data[j] = row


def apply_moving_window_average(data, window_size: int = 2):
    average_over = max(2, window_size)
    for i in range(data.shape[1]):
        lower_idx = max(i - average_over // 2, 0)
        upper_idx = min(i + average_over // 2, data.shape[1] - 1)
        data[:, i] = data[:, lower_idx:upper_idx].mean(axis=1)


def apply_tensorboard_smoothing(data, smoothing: float = 0.0):
    for c in range(1, data.shape[1]):
        data[:, c] = data[:, c - 1] * smoothing + (1 - smoothing) * data[:, c]


def create_seeds(num_runs, sample_range=(2**32)-1):
    seeds = np.random.default_rng().integers(0, sample_range, size=num_runs)
    return seeds


def save_seeds(seeds, exp_name, dir):
    filename = f"{dir}/seeds.npy"
    np.save(filename, seeds)


def load_seeds(exp_name, dir):
    filename = f"{dir}/seeds.npy"
    seeds = np.load(filename)
    return seeds


def save_params(save_dict, exp_name, dir):
    """
    Saves exp params
    """
    filename = f"{dir}/model_params.npy"
    np.save(filename, save_dict)


def load_params(exp_name, dir):
    filename = f"{dir}/model_params.npy"
    params_dict = np.load(filename, allow_pickle=True)
    return params_dict.item()


def show_plots():
    plt.show()


class exporting_to_onnx:
    """
    Set this context by calling
    ```
    with exporting_to_onnx():
    ```
    Within this context, the variable exporting_to_onnx.is_exporting() will be true.
    This implementation is thread safe.
    """

    # local is_exporting flag for each thread
    _local_data = threading.local()
    _local_data._is_exporting = False

    # global lock shared among all threads, to make sure only one thread is exporting at a time
    _lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self._local_data._is_exporting = True

    def __exit__(self, *args):
        self._local_data._is_exporting = False
        self._lock.release()

    @staticmethod
    def is_exporting():
        if not hasattr(exporting_to_onnx._local_data, "_is_exporting"):
            return False
        return exporting_to_onnx._local_data._is_exporting


class OnnxSaver:
    def __init__(self, policy, observation_specs):
        self.policy = policy
        observation_specs = observation_specs
        num_obs = len(observation_specs)

        batch_dim = [1]
        seq_len_dim = [1]
        memory_size = self.policy.onnx_actor.m_size

        dummy_obs = [torch.zeros(batch_dim + list(OnnxSaver._get_onnx_shape(obs_spec.shape)), device=get_device())
                     for obs_spec in observation_specs]

        dummy_masks = torch.ones(batch_dim + [0])

        dummy_memories = torch.zeros(batch_dim + seq_len_dim + [memory_size], device=get_device())

        self.dummy_input = (dummy_obs, dummy_masks, dummy_memories)

        self.input_names = [f"obs_{i}" for i in range(num_obs)]
        self.input_names += ["action_masks", "recurrent_in"]

        self.output_names = ["version_number", "memory_size", "continuous_actions", "continuous_action_output_shape"]

        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}
        self.dynamic_axes.update({name: {0: "batch"} for name in self.output_names})

        if memory_size > 0:
            self.output_names += ["recurrent_out"]

    @staticmethod
    def _get_onnx_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Convert shape of a visual observation from Pytorch HWC -> CHW for ONNX.
        """
        if len(shape) == 3:
            return shape[2], shape[0], shape[1]
        elif len(shape) == 4:
            return shape[3], shape[0], shape[1], shape[2]
        return shape

    @staticmethod
    def _get_pytorch_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Convert shape of a visual observation from ONNX CHW -> HWC for pytorch.
        """
        if len(shape) == 3:
            return shape[1], shape[2], shape[0]
        elif len(shape) == 4:
            return shape[1], shape[2], shape[3], shape[0]
        return shape

    def export_onnx_model(self, save_dir, suffix: str = ""):
        filename = f"{save_dir}/model_{suffix}.onnx"
        with exporting_to_onnx():
            torch.onnx.export(
                self.policy.onnx_actor,
                self.dummy_input,
                filename,
                opset_version=9,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes
            )

        filename = f"{save_dir}/deterministic_model_{suffix}.onnx"
        with exporting_to_onnx():
            torch.onnx.export(
                self.policy.deterministic_onnx_actor,
                self.dummy_input,
                filename,
                opset_version=9,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes
            )
