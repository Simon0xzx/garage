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

def plot_curve(matplot, path, exp_name, format='-', label='', plot_train = True):
    result_path = os.path.join(path, exp_name, 'progress.csv')
    data_dict = read_progress_file(result_path)
    x = data_dict['TotalEnvSteps']
    if plot_train:
        y = data_dict['MetaTest/Average/SuccessRate']
    else:
        y = data_dict['MetaTest/Average/AverageReturn'] # MetaTest/Average/SuccessRate
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

def plot_exp():
    reprodction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(2, 2)

    axs[0][0].set_title('Meta Test Avg Return Push')
    plot_curve(axs[0][0], reprodction_path, 'pearl_metaworld_ml1_push', '-',
               'pearl_metaworld_ml1_push', plot_train=False)
    axs[0][1].set_title('Meta Test Avg Return Reach')
    plot_curve(axs[0][1], reprodction_path, 'pearl_metaworld_ml1_reach', '-',
               'pearl_metaworld_ml1_push', plot_train=False)

    axs[1][0].set_title('Meta Test Avg Success Rate Push')
    plot_curve(axs[1][0], reprodction_path, 'pearl_metaworld_ml1_push', '-',
               'pearl_metaworld_ml1_push', plot_train=True)
    axs[1][1].set_title('Meta Test Avg Success Rate Reach')
    plot_curve(axs[1][1], reprodction_path, 'pearl_metaworld_ml1_reach', '-',
               'pearl_metaworld_ml1_push', plot_train=True)

    plt.show()


if __name__ == '__main__':
    # plot_report()
    plot_exp()
    # plot_exp3(train=False, avg=False)
    # run_experiment()
