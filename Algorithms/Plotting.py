import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

from matplotlib.ticker import FormatStrFormatter


def main():
    num_colors = 10
    colors = {
        "default":[f'C{i}' for i in range(10)],
        "blue": [(i/float(num_colors-1), 0, 1.0) for i in range(num_colors)]
    }
    default_names = ['Base', 'Without DM', 'Without OGM', 'Without DM and OGM']
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', '-n', nargs='+', default=default_names)
    parser.add_argument('--color_offset', '-c', type=int, default=0)
    parser.add_argument('--color_type', '-ct', type=str, default="default")
    parser.add_argument('--dirName', '-d', type=str, default="Navigation")
    parser.add_argument('--rootDir', '-r', type=str, default="OutputResults")
    parser.add_argument('--smoothing', '-s', type=float, default=0.0)
    parser.add_argument('--title', '-t', type=str, default="")
    args = parser.parse_args()

    rootDir = ".."
    dirToLoadFrom = f"{rootDir}/{args.rootDir}/{args.dirName}"
    plot(args.names, dirToLoadFrom, args.smoothing, colors[args.color_type][args.color_offset % num_colors:], args.title)

def plot(result_names:list, mainDir:str, smoothing:float, colors:list, title:str):
    """
    Loop through each directory in the given directory (in alphabetical order), load results from each.
    Then plot same data for all runs on same plots.
    """
    data_list = []
    folders = [f.name for f in os.scandir(mainDir) if f.is_dir()]
    for f in folders:
        results_dir = f"{mainDir}/{f}"
        data_list.append(load_data("results", results_dir))

    while len(data_list)>len(result_names):
        result_names.append("DummyName")
    while len(data_list)>len(colors):
        colors.append(f"{tuple(np.random.rand(3))}")

    plot_names = ["Episode Returns", "Test Returns", "Episode Lengths", "Test Lengths", "Success Rate", "Failure Rate", "Timeout Rate", "Curriculum Radius", "Goal Distance"]
    loss_plot_names = ["Policy Loss", "Critic Loss", "Alpha Loss"]
    value_plot_names = ["State Value (V)", "State-Action Value (Q)", "Log Pi", "Entropy", "Alpha"]
    plot_names = plot_names + loss_plot_names + value_plot_names
    plot_dictionary = {}

    for i, data in enumerate(data_list):
        data_dict = {}
        steps = data[0][0]
        time_taken = data[0][1]
        return_data = data[1]
        ep_ret = return_data[0]
        t_ret = return_data[1]
        loss_data = np.nan_to_num(data[2])
        value_data = np.nan_to_num(data[3])
        length_data = data[4]
        ep_len = length_data[0]
        t_len = length_data[1]
        data_dict['steps'] = steps
        data_dict['time_taken'] = time_taken
        data_dict[plot_names[0]]=ep_ret
        data_dict[plot_names[1]]=t_ret
        data_dict[plot_names[2]]=ep_len
        data_dict[plot_names[3]]=t_len
        if loss_data is not None:
            for j, name in enumerate(loss_plot_names):
                data_dict[name] = loss_data[j]
        if value_data is not None:
            for j, name in enumerate(value_plot_names):
                data_dict[name] = value_data[j]
        side_stats_data = None if len(data) < 6 else data[5]
        if side_stats_data is not None:
            success_rate = side_stats_data[0]
            fail_rate = side_stats_data[1]
            timeout_rate = side_stats_data[3]
            data_dict[plot_names[4]] = success_rate
            data_dict[plot_names[5]] = fail_rate
            data_dict[plot_names[6]] = timeout_rate
        goal_dist_data = None if len(data) < 7 else data[6]
        if goal_dist_data is not None:
            curric_R = goal_dist_data[0]
            goal_dist = goal_dist_data[1]
            data_dict[plot_names[7]]=curric_R
            data_dict[plot_names[8]]=goal_dist

        plot_dictionary[result_names[i]] = data_dict
    MakePlots(plot_dictionary, plot_names, mainDir, smoothing, colors, title)
    


def MakePlots(dict_of_dicts: dict, plot_names: list, save_dir: str, smoothing: float, colors: list, title: str):
    # loop through each result data
    for p_name in plot_names:
        name_list = []
        steps_list = []
        time_list = []
        data_list = []
        for r_name, sub_dict in dict_of_dicts.items():
            name_list.append(r_name)
            steps_list.append(sub_dict["steps"])
            time_list.append(sub_dict["time_taken"])
            data_list.append(sub_dict[p_name])
        y_lim = (0, 1.0) if p_name.__contains__("Rate") else None
        plot_many_fill_betweens_on_one_plot(steps_list, time_list, data_list, name_list, p_name, smoothing, colors, title, y_lim, save_dir)

def plot_many_fill_betweens_on_one_plot(steps_list:list, time_list:list, data_list:list, result_names:list, plot_name:str,
                                        smoothing:float, colors:list, title:str, y_limits:tuple=None, save_dir:str=""):
    runs = data_list[0].shape[0]

    # use experiment with most steps (assuming equal step size) for x_range
    max_step_idx = np.argmax([len(s[0]) for s in steps_list])
    x_range = steps_list[max_step_idx][0]

    x_range = x_range[x_range.copy() > -1]

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    x2_range = None

    output_means = {}

    if time_list is not None:
        # Find the experiment with the largest average run time
        max_idx = np.argmax([np.mean(t[:,-1]) for t in time_list]).item()

        # Set time axis to the mean time of that experiment
        x2_range = np.mean(time_list[max_idx], axis=0)
        x2_range = x2_range[x2_range.copy() > -1]
        x2_range = [int(x) for x in x2_range]


    for i, data in enumerate(data_list):
        data = data
        name = result_names[i]
        steps = steps_list[i]
        steps = steps.mean(axis=0).flatten()
        time = time_list[i]
        time = time.mean(axis=0).flatten()

        truth_array = steps.copy() > -1

        color = colors[i]

        if smoothing > 0:
            apply_tensorboard_smoothing(data, smoothing)

        data_mean = data.mean(axis=0).flatten()
        data_std_err = data.std(axis=0).flatten() / np.sqrt(runs)

        data_mean = data_mean[truth_array]
        data_std_err = data_std_err[truth_array]

        steps = steps[truth_array]
        time = time[truth_array]

        line_width = 2.0
        alpha=0.2

        ax.fill_between(steps, data_mean - 2*data_std_err, data_mean + 2*data_std_err, color=color, alpha=alpha)
        ax.plot(steps, data_mean, color=color, label=f'{name}',linewidth=line_width)

        ax2.fill_between(time, data_mean - 2*data_std_err, data_mean + 2*data_std_err, color=color, alpha=alpha)
        ax2.plot(time, data_mean, color=color, label=f'{name}',linewidth=line_width)

        output_means["name"]=[steps,data_mean,data_std_err]

    tick_label_size = 12
    legend_label_size = 12
    num_ticks = 6

    if y_limits is not None:
        ax.set_ylim(y_limits)
        ax2.set_ylim(y_limits)
    y_ticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 11)
    delta_y = y_ticks[1]-y_ticks[0]

    # format string according to scale of y-value delta
    format_string = '%.1f'
    if delta_y >= 1:
        format_string = '%.0f'
    elif delta_y < 1:
        log_value = int(-np.floor(np.log10(delta_y)).item())
        format_string = f'%.{log_value}f'

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks,fontsize=tick_label_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter(format_string))
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_ticks,fontsize=tick_label_size)
    ax2.yaxis.set_major_formatter(FormatStrFormatter(format_string))

    ax.set_xlim(0, x_range[-1])
    ax.set_xticks(np.linspace(0, x_range[-1], num_ticks, dtype=np.int32))
    ax.set_xticklabels(np.linspace(0, x_range[-1], num_ticks, dtype=np.int32), fontsize=tick_label_size)

    title_fontsize = 20
    title_vert_pos = 0.95
    title_weight = "semibold"
    ax.legend(loc='best', fontsize=legend_label_size)
    fig.suptitle(title, size=title_fontsize, weight=title_weight, y=title_vert_pos)
    ax2.legend(loc='best',fontsize=legend_label_size)
    fig2.suptitle(title, size=title_fontsize, weight=title_weight, y=title_vert_pos)

    if x2_range is not None:
        ax2.set_xlim(x2_range[0], x2_range[-1])
        ax2.set_xticks(np.linspace(0, x2_range[-1], num_ticks, dtype=np.int32))
        ax2.set_xticklabels(np.linspace(0, x2_range[-1], num_ticks, dtype=np.int32), fontsize=tick_label_size)

    label_font_size = 16
    label_weight="semibold"

    ax.set_xlabel("Environment Decision Steps", fontsize=label_font_size, weight=label_weight)
    ax2.set_xlabel("Wall-Clock Training Time (seconds)", fontsize=label_font_size, weight=label_weight)

    ax.set_ylabel(f"{plot_name}",fontsize=label_font_size, weight=label_weight)
    ax2.set_ylabel(f"{plot_name}",fontsize=label_font_size, weight=label_weight)

    fig.savefig(f"{save_dir}/{plot_name}.pdf")
    fig.savefig(f"{save_dir}/{plot_name}.png")
    fig2.savefig(f"{save_dir}/{plot_name}2.pdf")
    fig2.savefig(f"{save_dir}/{plot_name}2.png")

    plt.close(fig)
    plt.close(fig2)

    return output_means


def load_data(exp_name: str, result_dir: str):
    """
    Returns the data of a saved numpy file.
    """
    if not os.path.exists(result_dir):
        print(f"No directory at {result_dir}")
        return None
    load_path = f"{result_dir}/{exp_name}.npy"
    data = np.load(load_path, allow_pickle=True)
    return data

def apply_moving_window_average(data, window_size: int = 2):
    average_over = max(2, window_size)
    for i in range(data.shape[1]):
        lower_idx = max(i - average_over // 2, 0)
        upper_idx = min(i + average_over // 2, data.shape[1] - 1)
        data[:, i] = data[:, lower_idx:upper_idx].mean(axis=1)


def apply_tensorboard_smoothing(data, smoothing: float = 0.0):
    for c in range(1, data.shape[1]):
        data[:, c] = data[:, c - 1] * smoothing + (1 - smoothing) * data[:, c]


if __name__ == "__main__":
    main()