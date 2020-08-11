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

def plot_curve(matplot, path, exp_name, format='-', title = 'MetaTest/Average/SuccessRate', limit = -3):
    result_path = os.path.join(path, exp_name, 'progress.csv')
    data_dict = read_progress_file(result_path)
    x = data_dict['TotalEnvSteps']
    y = data_dict[title]

    if limit != -1:
        cap = min(len(x), limit)
        x = x[:cap]
        y = y[:cap]
    matplot.plot(x, y, format, label='{}_{}'.format(exp_name, title))
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

def generate_report(exp_path_root, exps_matrix, success_rate=False, limit=500):
    rows, cols = len(exps_matrix), len(exps_matrix[0])
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 or cols == 1:
        for i, exp_dirs in enumerate(exps_matrix):
            axs[i].set_title('Meta Test Avg Return')
            axs[i].set_xlabel('Total Env Steps')
            axs[i].set_ylabel('Avg Test Return')
            for exp_name in exp_dirs:
                plot_curve(axs[i], exp_path_root, exp_name, '-', exp_name, limit=limit)
    else:
        for i, row_dirs in enumerate(exps_matrix):
            for j, exp_dirs in enumerate(row_dirs):
                axs[i][j].set_title('Meta Test Avg Return')
                axs[i][j].set_xlabel('Total Env Steps')
                axs[i][j].set_ylabel('Avg Test Return')
                for exp_name in exp_dirs:
                    plot_curve(axs[i][j], exp_path_root, exp_name, '-', exp_name, limit=limit)


def plot_exp():
    namazu_reprodction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/old_formulation/namazu/experiment'
    local_reporduction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/old_formulation/experiment'
    new_reproduction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 3)

    axs[0].set_title('Push Meta Test Avg Return Push')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')

    plot_curve(axs[0], namazu_reprodction_path,
               'multitask_oracle_metaworld_ml1_push', '-', limit=500)
    plot_curve(axs[0], local_reporduction_path,
               'pearl_metaworld_ml1_push_1', '-', limit=500)
    plot_curve(axs[0], local_reporduction_path,
               'multitask_emphasized_oracle_metaworld_ml1_push_1',
               '-', limit=500)
    plot_curve(axs[0], local_reporduction_path,
               'pearl_emphasized_metaworld_ml1_push',
               '-', limit=500)


    axs[1].set_title('Reach Meta Test Avg Return Reach')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    # plot_curve(axs[1], namazu_reprodction_path,
    #            'multitask_oracle_metaworld_ml1_reach', '-', limit=500)
    plot_curve(axs[1], local_reporduction_path,
               'pearl_metaworld_ml1_reach_1', '-', title='MetaTest/reach-v1/AverageReturn',
               limit=500)
    plot_curve(axs[1], local_reporduction_path,
               'multitask_emphasized_oracle_metaworld_ml1_reach_1', '-', title='MetaTest/reach-v1/AverageReturn',
               limit=500)
    plot_curve(axs[1], local_reporduction_path,
               'pearl_emphasized_metaworld_ml1_reach', '-', title='MetaTest/reach-v1/AverageReturn',
               limit=500)
    plot_curve(axs[1], new_reproduction_path,
               'curl_metaworld_ml1_reach_22', '-', title='MetaTest/reach-v1/AverageReturn',
               limit=500)



    axs[2].set_title('Pick Place Meta Test Avg Return Pick Place')
    axs[2].set_xlabel('Total Env Steps')
    axs[2].set_ylabel('Avg Test Return')

    plot_curve(axs[2], namazu_reprodction_path,
               'multitask_oracle_metaworld_ml1_pick_place', '-', limit=500)
    plot_curve(axs[2], local_reporduction_path,
               'pearl_metaworld_ml1_pick-place', '-',
               limit=500)
    plot_curve(axs[2], local_reporduction_path,
               'multitask_emphasized_oracle_metaworld_ml1_pick_place_1', '-',
               limit=500)
    plot_curve(axs[2], local_reporduction_path,
               'pearl_emphasized_metaworld_ml1_pick_place', '-',
               limit=500)

    plt.show()


def plot_exp2():
    namazu_reprodction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/namazu/experiment'
    local_reporduction_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title('multitask SAC MT10 Avg Training Return')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Training Return')
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='reach-v1/AverageReturn', limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='push-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='pick-place-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='door-open-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='drawer-open-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='drawer-close-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='button-press-topdown-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='peg-insert-side-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='window-open-v1/AverageReturn',
               limit=500)
    plot_curve(axs[0], namazu_reprodction_path,
               'mtsac_metaworld_mt10', '-', title='window-close-v1/AverageReturn',
               limit=500)



    axs[1].set_title('multitask SAC MT10 Avg Success Rate')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Return')
    # plot_curve(axs[1], namazu_reprodction_path,
    #            'mtsac_metaworld_mt10', '-', title='Average/AverageReturn', limit=500)

    plot_curve(axs[1], local_reporduction_path,
               'pearl_metaworld_ml10', '-',
               title='MetaTest/Average/AverageReturn', limit=500)
    plot_curve(axs[1], local_reporduction_path,
               'pearl_metaworld_mlsp', '-',
               title='MetaTest/Average/AverageReturn', limit=500)



    # axs[2].set_title('SAC lever_pull Task (Export Agent)')
    # axs[2].set_xlabel('Total Env Steps')
    # axs[2].set_ylabel('Avg Training Return')
    # plot_curve(axs[2], local_reporduction_path,
    #            'sac_metaworld_ml1_lever_pull', '-',title='Average/TrainAverageReturn', limit=500)




    plt.show()


class CameraWrapper(object):

    def __init__(self, env,  *args, **kwargs):
        self._wrapped_env = env
        self.initialize_camera()

    def get_image(self, width=256, height=256, camera_name=None):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def initialize_camera(self):
        import mujoco_py
        # set camera parameters for viewing
        sim = self.sim
        viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.elevation = -20
        sim.add_render_context(viewer)

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

def sim_policy():
    import metaworld.benchmarks as mwb
    from garage.envs import GarageEnv, normalize
    from garage.experiment.task_sampler import EnvPoolSampler
    # create multi-task environment and sample tasks
    ml_train_envs = [
        GarageEnv(normalize(mwb.MLSP.from_task(task_name)))
        for task_name in mwb.MLSP.get_train_tasks().all_task_names
    ]

    ml_test_envs = [
        GarageEnv(normalize(mwb.MLSP.from_task(task_name)))
        for task_name in mwb.MLSP.get_test_tasks().all_task_names
    ]

    env_sampler = EnvPoolSampler(ml_train_envs)
    env_sampler.grow_pool(10)
    # env = env_sampler.sample(10)
    test_env_sampler = EnvPoolSampler(ml_test_envs)
    test_env_sampler.grow_pool(5)
    env = test_env_sampler.sample(5)
    rander_env = env[0]._env._env.env.active_env
    rander_env._task = {'partially_observable':None}
    rander_env.reset_model()
    viewer = rander_env._get_viewer('human')

    for i in range(1000000):
        action = eval(input("step_action 4D: \n"))
        viewer.render()





if __name__ == '__main__':
    plot_exp()
    # sim_policy()
