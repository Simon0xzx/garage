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
    data_dict = defaultdict(lambda: [])
    with open(file_path, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                data_dict[k].append(type(v))
    return data_dict

def plot_curve_avg(matplot, exps, format='-',
                   title = 'MetaTest/Average/AverageReturn', x_title = 'TotalEnvSteps', legend = None, limit = -1, line_width=2, opacity=0.2):
    if exps == None or len(exps) == 0:
        return
    result_paths = [os.path.join(name, 'progress.csv') for name in exps]
    data_dicts = [read_csv_file(result_path) for result_path in result_paths]
    min_step_length = min([len(data_dict[x_title]) for data_dict in data_dicts])
    if limit != -1:
        min_step_length = min(min_step_length, limit)
    label = legend if legend != None else'{}_{}'.format(result_paths[0], title)
    x = data_dicts[0][x_title][:min_step_length]
    y_ave = np.average([list(map(lambda x: float(x), data_dict[title][:min_step_length])) for data_dict in data_dicts], axis=0)
    y_min = np.min(
        [list(map(lambda x: float(x), data_dict[title][:min_step_length])) for
         data_dict in data_dicts], axis=0)
    y_max = np.max(
        [list(map(lambda x: float(x), data_dict[title][:min_step_length])) for
         data_dict in data_dicts], axis=0)
    matplot.plot(x, y_ave, format, lw=line_width,label=label)
    matplot.fill_between(x, y_min, y_max, alpha=opacity)
    matplot.legend()

def print_hyper_tests(axs, dir_path, exp_name, labels,
                      title = 'MetaTest/Average/AverageReturn', x_title = 'TotalEnvSteps', limit = -1):
    for i, label in enumerate(labels):
        task_name = '{}{}'.format(exp_name, '_{}'.format(i) if i > 0 else '')
        try:
            plot_curve_avg(axs, ['{}/{}'.format(dir_path, task_name)], '-',
                           legend=label, title=title, x_title=x_title, limit=limit)
        except:
            print("Failed to Locate {}".format(task_name))
