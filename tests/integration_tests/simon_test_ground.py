import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import csv

"""
#########################################################
##                       Ploting                       ##
#########################################################
"""
def read_progress_file(file_path):
    data_dict = defaultdict(lambda: [])
    with open(file_path, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                data_dict[k].append(v)
    return data_dict

def plot_curve(matplot, path, exp_name, format='-', label='', success_rate = True, limit = -3):
    result_path = os.path.join(path, exp_name, 'progress.csv')
    data_dict = read_progress_file(result_path)
    x = data_dict['TotalEnvSteps']
    if success_rate:
        y = data_dict['MetaTest/Average/SuccessRate']
    else:
        y = data_dict['MetaTest/Average/AverageReturn'] # MetaTest/Average/SuccessRate

    if limit != -1:
        cap = min(len(x), limit)
        x = x[:cap]
        y = y[:cap]
    matplot.plot(x, y, format, label=label)
    matplot.legend()

def plot_curve_avg(matplot, root_path, exps, format='-', label='', plot_train = True, avg = True):
    result_paths = [os.path.join(root_path, exp_name, 'progress.csv') for exp_name in exps]
    data_dicts = [read_progress_file(result_path) for result_path in result_paths]

    step_length = [len(data_dict['Number of env steps total']) for data_dict in data_dicts]
    print("Step length [{}]: \n{}".format(root_path, step_length))
    min_step = min(step_length)

    x = data_dicts[0]['Number of env steps total'][:min_step]
    # x_avg = np.average(np.array([data_dict['Number of env steps total'] for data_dict in data_dicts]))
    # assert x == x_avg # otherwise something is wrong

    if plot_train:
        y_avg = np.average([list(map(lambda x: float(x), data_dict['AverageReturn_all_train_tasks'][:min_step])) for data_dict in data_dicts], axis=0)
        y_med = np.median([list(map(lambda x: float(x), data_dict['AverageReturn_all_train_tasks'][:min_step])) for data_dict in data_dicts], axis=0)
    else:
        y_avg = np.average([list(map(lambda x: float(x), data_dict['AverageReturn_all_test_tasks'][:min_step])) for data_dict in data_dicts], axis=0)
        y_med = np.median([list(map(lambda x: float(x), data_dict['AverageReturn_all_test_tasks'][:min_step])) for data_dict in data_dicts], axis=0)

    if avg:
        matplot.plot(x, y_avg, format, label=label)
    else:
        matplot.plot(x, y_med, format, label=label)
    matplot.legend()

# def generate_report(dirs):
#     assert dirs != None and isinstance(dirs[0], list)
#     rows, cols = len(dirs), len(dirs[0])
#     fig, axs = plt.subplots(rows, cols)
#     for i, row_dirs in enumerate(dirs):
#         for j, exp_dirs in enumerate(row_dirs):
#             axs[i][j].set_title('Meta Test Avg Return Push')


def plot_exp():
    reprodction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(2, 3)

    axs[0][0].set_title('Meta Test Avg Return Push')
    axs[0][0].set_xlabel('Total Env Steps')
    axs[0][0].set_ylabel('Avg Test Return')
    plot_curve(axs[0][0], reprodction_path, 'pearl_metaworld_ml1_push_1', '-',
               'pearl_metaworld_ml1_push', success_rate=False, limit=500)
    plot_curve(axs[0][0], reprodction_path, 'multitask_oracle_metaworld_ml1_push_1', '-',
               'multitask_oracle_metaworld_ml1_push_1', success_rate=False, limit=500)
    plot_curve(axs[0][0], reprodction_path,
               'deeper_multitask_oracle_metaworld_ml1_push', '-',
               'deeper_multitask_oracle_metaworld_ml1_push', success_rate=False, limit=500)
    plot_curve(axs[0][0], reprodction_path,
               'multitask_emphasized_oracle_metaworld_ml1_push', '-',
               'multitask_emphasized_oracle_metaworld_ml1_push',
               success_rate=False, limit=500)

    axs[0][1].set_title('Meta Test Avg Return Reach')
    axs[0][1].set_xlabel('Total Env Steps')
    axs[0][1].set_ylabel('Avg Test Return')
    plot_curve(axs[0][1], reprodction_path, 'pearl_metaworld_ml1_reach_1', '-',
               'pearl_metaworld_ml1_reach', success_rate=False, limit=500)
    plot_curve(axs[0][1], reprodction_path, 'multitask_oracle_metaworld_ml1_reach_1', '-',
               'multitask_oracle_metaworld_ml1_reach_1', success_rate=False, limit=500)
    plot_curve(axs[0][1], reprodction_path,
               'deeper_multitask_oracle_metaworld_ml1_reach_1', '-',
               'deeper_multitask_oracle_metaworld_ml1_reach', success_rate=False, limit=500)
    plot_curve(axs[0][1], reprodction_path,
               'multitask_emphasized_oracle_metaworld_ml1_reach', '-',
               'multitask_emphasized_oracle_metaworld_ml1_reach',
               success_rate=False, limit=500)


    axs[0][2].set_title('Meta Test Avg Return Pick Place')
    axs[0][2].set_xlabel('Total Env Steps')
    axs[0][2].set_ylabel('Avg Test Return')
    plot_curve(axs[0][2], reprodction_path, 'pearl_metaworld_ml1_pick-place', '-',
               'pearl_metaworld_ml1_pick-place', success_rate=False, limit=500)
    plot_curve(axs[0][2], reprodction_path,
               'multitask_oracle_metaworld_ml1_pick_place_1', '-',
               'multitask_oracle_metaworld_ml1_pick_place_1', success_rate=False, limit=500)
    plot_curve(axs[0][2], reprodction_path,
               'deeper_multitask_oracle_metaworld_ml1_pick_place_1', '-',
               'deeper_multitask_oracle_metaworld_ml1_pick_place', success_rate=False, limit=500)
    plot_curve(axs[0][2], reprodction_path,
               'multitask_emphasized_oracle_metaworld_ml1_pick_place', '-',
               'multitask_emphasized_oracle_metaworld_ml1_pick_place', success_rate=False, limit=500)

    plt.show()


def sim_policy():
    from garage.experiment import Snapshotter
    exp_root = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment/'
    snapshotter = Snapshotter()
    data = snapshotter.load(os.path.join(exp_root, 'multitask_emphasized_oracle_metaworld_ml1_reach'))
    policy = data['algo'].policy

    # You can also access other components of the experiment
    env = data['env']

    steps, max_steps = 0, 150
    done = False
    obs = env.reset()  # The initial observation
    # policy.reset()

    while steps < max_steps and not done:
        # context = env._env.env.active_env.goal
        # obs_in = np.concatenate([obs, env._env.env.active_env.goal])
        obs, rew, done, _ = env.step(policy.get_action(obs))
        env.render()  # Render the environment to see what's going on (optional)
        steps += 1

    env.close()

if __name__ == '__main__':
    plot_exp()
    # sim_policy()
