import numpy as np
from collections import defaultdict
import os
import csv
"""
#########################################################
##                       Ploting                       ##
#########################################################
"""

def read_csv_file(file_path, type=float):
    verify_key = 'MetaTest/Average/AverageReturn'
    data_dict = defaultdict(lambda: [])
    with open(file_path, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                try:
                    if v == '':
                        data_dict[k].append(0.0)
                    else:
                        data_dict[k].append(type(v))
                except:
                    print("problem: {}".format(v))
    delete_index = []
    return data_dict

def plot_curve_avg(matplot, exps, format='-',
                   title = 'MetaTest/Average/AverageReturn', x_title = 'TotalEnvSteps',
                   legend = None, limit = -1, line_width=2, opacity=0.2, env_step_limit=2500000, backward_smooth_window=1, plot=True):
    if exps == None or len(exps) == 0:
        return
    result_paths = [os.path.join(name, 'progress.csv') for name in exps]
    data_dicts = [read_csv_file(result_path) for result_path in result_paths]
    min_step_length = min([len(data_dict[x_title]) for data_dict in data_dicts])
    if limit != -1:
        min_step_length = min(min_step_length, limit)
    label = legend if legend != None else'{}_{}'.format(result_paths[0], title)
    x = data_dicts[0][x_title][:min_step_length]
    y = [list(map(lambda x: float(x), data_dict[title][:min_step_length])) for data_dict in data_dicts]
    y_ave = np.average(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    # Cutting the tail by env step limit
    x_new, y_avg_new, y_min_new, y_max_new = [],[],[],[]
    last_visited_index = 0
    if env_step_limit != -1:
        for i in range(len(x)):
            if x[i] > env_step_limit:
                break
            last_visited_index = i
            x_new.append(x[i])
            y_avg_new.append(y_ave[i])
            y_min_new.append(y_min[i])
            y_max_new.append(y_max[i])
    else:
        x_new = x
        y_avg_new = y_ave
        y_min_new = y_min
        y_max_new = y_max
    y_stats_window = y_ave[last_visited_index-backward_smooth_window: last_visited_index]
    if plot:
        matplot.plot(x_new, y_avg_new, format, lw=line_width,label=label)
        matplot.fill_between(x_new, y_min_new, y_max_new, alpha=opacity)
        matplot.legend()

    # Prepare infos for report table
    return y_stats_window


def print_hyper_tests(axs, dir_path, exp_name, label, num_seeds=1,
                      title = 'MetaTest/Average/AverageReturn', x_title = 'TotalEnvSteps',
                      limit = -1, env_step_limit=2500000, backward_smooth_window=1, plot=True):
    results_dirs = []
    valid_seed_cnt = 0
    valid_stats = []
    for i in range(num_seeds):
        task_name = '{}{}'.format(exp_name, '_{}'.format(i) if i > 0 else '')
        if 'meta-q-learning' in dir_path:
            direct_dir_path = '{}/metaworld-ml1-{}_mql_dummy'.format(dir_path, task_name)
            title = 'episode_reward'
            x_title = 'total_timesteps'
        else:
            direct_dir_path = '{}/{}'.format(dir_path, task_name)
        if os.path.exists(direct_dir_path):
            results_dirs.append(direct_dir_path)
    try:
        stats_window = plot_curve_avg(axs, results_dirs, '-',
                                      legend=label, title=title, x_title=x_title, limit=limit,
                                      env_step_limit=env_step_limit, backward_smooth_window=backward_smooth_window, plot=plot)
        valid_seed_cnt += 1
        valid_stats.extend(stats_window)
    except:
        print("Failed to Locate {}".format(exp_name))

    if valid_seed_cnt == 0:
        return {'env_name': exp_name,
                'seed_cnt': 0}
    else:
        return {'env_name': exp_name,
                'average': np.average(valid_stats),
                'std': np.std(valid_stats),
                'seed_cnt': valid_seed_cnt}

