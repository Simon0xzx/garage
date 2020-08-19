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

def plot_curve(matplot, path, exp_name, format='-', title = 'MetaTest/Average/AverageReturn', limit = -3):
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

'''
ML1 category
MetaTest/Average/StdReturn,
MetaTest/Average/AverageDiscountedReturn,
MetaTest/Average/TerminationRate,
MetaTest/push-v1/Iteration,
MetaTest/push-v1/NumTrajs,
MetaTest/push-v1/MinReturn,
MetaTest/Average/AverageReturn,
MetaTest/push-v1/AverageDiscountedReturn,
MetaTest/Average/MinReturn,
MetaTest/push-v1/MaxReturn,
MetaTest/Average/MaxReturn,
MetaTest/push-v1/StdReturn,
MetaTest/push-v1/SuccessRate,
MetaTest/Average/SuccessRate,
MetaTest/push-v1/AverageReturn,
MetaTest/push-v1/TerminationRate,
MetaTest/Average/NumTrajs,
MetaTest/Average/Iteration,TotalEnvSteps
'''
def ml1_exp_plot():
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/experiment'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 3)

    axs[0].set_title('ML1 Push Avg Return')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')
    plot_curve(axs[0], local_path,
               'pearl_metaworld_ml1_push', '-', limit=250)
    plot_curve(axs[0], local_path,
               'pearl_emphasized_metaworld_ml1_push', '-', limit=250)

    plot_curve(axs[0], namazu_path,
               'multitask_oracle_metaworld_ml1_push', '-', limit=250)
    plot_curve(axs[0], leviathan_path,
               'multitask_emphasized_oracle_metaworld_ml1_push', '-', limit=250)

    plot_curve(axs[0], local_path,
               'curl_metaworld_ml1_push', '-', limit=250)


    axs[1].set_title('ML1 Reach Avg Return')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    plot_curve(axs[1], local_path,
               'pearl_metaworld_ml1_reach', '-', limit=250)
    plot_curve(axs[1], local_path,
               'pearl_emphasized_metaworld_ml1_reach', '-', limit=250)

    plot_curve(axs[1], namazu_path,
               'multitask_oracle_metaworld_ml1_reach', '-', limit=250)
    plot_curve(axs[1], leviathan_path,
               'multitask_emphasized_oracle_metaworld_ml1_reach', '-', limit=250)

    plot_curve(axs[1], local_path,
               'curl_metaworld_ml1_reach', '-', limit=250)


    axs[2].set_title('ML1 Pick Place Avg Return')
    axs[2].set_xlabel('Total Env Steps')
    axs[2].set_ylabel('Avg Test Return')
    plot_curve(axs[2], namazu_path,
               'pearl_metaworld_ml1_pick_place', '-', limit=250) # YES
    # plot_curve(axs[2], leviathan_path,
    #            'pearl_metaworld_ml1_pick_place_1', '-', limit=250) # Keep
    plot_curve(axs[2], local_path,
               'pearl_emphasized_metaworld_ml1_pick_place', '-', limit=250)

    plot_curve(axs[2], namazu_path,
               'multitask_oracle_metaworld_ml1_pick_place', '-', limit=250)
    plot_curve(axs[2], leviathan_path,
               'multitask_emphasized_oracle_metaworld_ml1_pick_place', '-', limit=250)

    plot_curve(axs[2], namazu_path,
               'curl_metaworld_ml1_pick_place', '-', limit=250)

    plt.show()

'''
MLSP category
MetaTest/shelf-place-v1/NumTrajs,
MetaTest/shelf-place-v1/SuccessRate,
MetaTest/shelf-place-v1/StdReturn,
MetaTest/shelf-place-v1/AverageReturn,
MetaTest/shelf-place-v1/MaxReturn,
MetaTest/shelf-place-v1/Iteration,
MetaTest/shelf-place-v1/TerminationRate,
MetaTest/shelf-place-v1/AverageDiscountedReturn,
MetaTest/shelf-place-v1/MinReturn,
MetaTest/bin-picking-v1/Iteration,
MetaTest/bin-picking-v1/SuccessRate,
MetaTest/bin-picking-v1/StdReturn,
MetaTest/bin-picking-v1/MaxReturn,
MetaTest/bin-picking-v1/NumTrajs,
MetaTest/bin-picking-v1/TerminationRate,
MetaTest/bin-picking-v1/AverageReturn,
MetaTest/bin-picking-v1/AverageDiscountedReturn,
MetaTest/bin-picking-v1/MinReturn
MetaTest/push-wall-v1/MaxReturn,
MetaTest/push-wall-v1/NumTrajs,
MetaTest/push-wall-v1/StdReturn,
MetaTest/push-wall-v1/AverageDiscountedReturn,
MetaTest/push-wall-v1/SuccessRate,
MetaTest/push-wall-v1/TerminationRate,
MetaTest/push-wall-v1/AverageReturn,
MetaTest/push-wall-v1/MinReturn,
MetaTest/push-wall-v1/Iteration,
MetaTest/button-press-wall-v1/Iteration,
MetaTest/button-press-wall-v1/MinReturn,
MetaTest/button-press-wall-v1/MaxReturn,
MetaTest/button-press-wall-v1/AverageDiscountedReturn,
MetaTest/button-press-wall-v1/SuccessRate,
MetaTest/button-press-wall-v1/NumTrajs,
MetaTest/button-press-wall-v1/AverageReturn,
MetaTest/button-press-wall-v1/TerminationRate,
MetaTest/button-press-wall-v1/StdReturn,
MetaTest/box-close-v1/NumTrajs,
MetaTest/box-close-v1/TerminationRate,
MetaTest/box-close-v1/StdReturn,
MetaTest/box-close-v1/MaxReturn,
MetaTest/box-close-v1/Iteration,
MetaTest/box-close-v1/AverageDiscountedReturn,
MetaTest/box-close-v1/SuccessRate,
MetaTest/box-close-v1/MinReturn,
MetaTest/box-close-v1/AverageReturn,
MetaTest/Average/AverageDiscountedReturn,
MetaTest/Average/NumTrajs,
MetaTest/Average/AverageReturn,
MetaTest/Average/TerminationRate,
MetaTest/Average/Iteration,
MetaTest/Average/SuccessRate,
MetaTest/Average/MinReturn,
MetaTest/Average/MaxReturn,
MetaTest/Average/StdReturn,

TotalEnvSteps,
'''
def mlsp_plot():
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/experiment'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    num_plot = 5
    fig, axs = plt.subplots(1, num_plot)
    plot_order = ['MetaTest/shelf-place-v1/AverageReturn',
                  'MetaTest/bin-picking-v1/AverageReturn',
                  'MetaTest/push-wall-v1/AverageReturn',
                  'MetaTest/button-press-wall-v1/AverageReturn',
                  'MetaTest/box-close-v1/AverageReturn']

    for i in range(num_plot):
        axs[i].set_title('MLSP {} Avg Return'.format(plot_order[i]))
        axs[i].set_xlabel('Total Env Steps')
        axs[i].set_ylabel('Avg Test Return')
        plot_curve(axs[i], local_path,
                   'pearl_metaworld_mlsp', '-', title=plot_order[i], limit=250)
        plot_curve(axs[i], local_path,
                   'curl_metaworld_mlsp', '-', title=plot_order[i], limit=250)
        plot_curve(axs[i], local_path,
                   'curl_emphasized_metaworld_mlsp', '-', title=plot_order[i], limit=250)


    plt.show()


'''
SAC Category
TotalEnvSteps,
Evaluation/MaxReturn,
Evaluation/SuccessRate,
Evaluation/MinReturn,
QF/Qf1Loss,
QF/Qf2Loss,
Evaluation/AverageReturn,
Policy/Loss,
Evaluation/StdReturn,
Evaluation/Iteration,
AlphaTemperature/mean,
ReplayBuffer/buffer_size,
Average/TrainAverageReturn,
Evaluation/NumTrajs,
Evaluation/TerminationRate,
Evaluation/AverageDiscountedReturn
'''
def expert_plot():
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    num_plot = 1
    fig, axs = plt.subplots(1, num_plot)

    axs.set_title('SAC Expert Avg Return')
    axs.set_xlabel('Total Env Steps')
    axs.set_ylabel('Avg Test Return')
    plot_curve(axs, leviathan_path,
               'sac_metaworld_ml1_bin_picking', '-', title='Average/TrainAverageReturn', limit=250)
    plot_curve(axs, leviathan_path,
               'sac_metaworld_ml1_box_close', '-', title='Average/TrainAverageReturn', limit=250)
    plot_curve(axs, leviathan_path,
               'sac_metaworld_ml1_button_press_wall', '-', title='Average/TrainAverageReturn', limit=250) # Check this
    plot_curve(axs, leviathan_path,
               'sac_metaworld_ml1_push_wall', '-', title='Average/TrainAverageReturn', limit=250)
    plot_curve(axs, leviathan_path,
               'sac_metaworld_ml1_shelf_place', '-', title='Average/TrainAverageReturn',limit=250)
    plt.show()

def mtsac_mt10_plot():
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
    import pickle
    import metaworld.benchmarks as mwb
    from garage.envs import GarageEnv, normalize
    from garage.experiment.task_sampler import EnvPoolSampler
    # create multi-task environment and sample tasks
    ml_test_envs = [
        GarageEnv(normalize(mwb.MLSP.from_task('button-press-wall-v1')))
    ]

    env_sampler = EnvPoolSampler(ml_test_envs)
    env_sampler.grow_pool(1)
    envs = env_sampler.sample(1)
    rander_env = envs[0]._env._env.env.active_env
    rander_env._task = {'partially_observable': None}
    rander_env.reset_model()
    rander_env.frame_skip=200
    viewer = rander_env._get_viewer('human')
    env = envs[0]._env
    obs = env.reset()
    env.render()
    expert_path = {'observations': [],
                   'actions': [],
                   'rewards': [],
                   'next_observations': [],
                   'dones': []}

    expert_actions = expert_action()
    for i in range(150):
        action = expert_actions[i]
        # action[0] += np.random.rand(1)*0.04-0.
        # action[1] += np.random.rand(1) * 0.04 - 0.02
        # action[2] += np.random.rand(1) * 0.04 - 0.02
        expert_path['observations'].append(obs)
        expert_path['actions'].append(action)
        obs, rew, done, env_infos = env.step(action)
        expert_path['rewards'].append(rew)
        expert_path['next_observations'].append(obs)
        expert_path['dones'].append(done)
        print('Step: {}\nObs:\n {}, \nAction: \n{}\n Reward: \n{}'.format(i, obs, action, rew))
        env.render()
        if done:
            break

    expert_path['observations'] = np.array(expert_path['observations'])
    expert_path['actions'] = np.array(expert_path['actions'])
    expert_path['rewards'] = np.array(expert_path['rewards'])
    expert_path['next_observations'] = np.array(expert_path['next_observations'])
    expert_path['dones'] = np.array(expert_path['dones'])

    with open('exper_traj_1.pickle', 'wb') as handle:
        pickle.dump(expert_path, handle)
    print(expert_path)

def expert_action():
    return np.array([[0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0.2, 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0., 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.2, -0.2, 0.2],
           [0, 0.1, 0., 0.2],
           [0, 0.1, 0., 0.2],
           [0, 0.1, 0., 0.2],
           [0, 0.1, 0., 0.2],
           [0, 0.1, 0., 0.2],
           [0, 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.2],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., 0.],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, 0., -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0.2, -0.2, -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., -0.2],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

def button_press_wall_expert_traj1():
    trajectories = []
    traj1 = {}

    trajectories.append(traj1)

    return trajectories


def load_pearl_mlsp_policy():
    from garage.experiment import Snapshotter
    exp_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    base_agent_path = '{}/pearl_metaworld_mlsp'.format(exp_path)
    expert_path = '{}/sac_metaworld_ml1_button_press_wall'.format(exp_path)

    # Load the policy
    snapshotter = Snapshotter()
    snapshot = snapshotter.load(expert_path)

    exp_policy = snapshot['algo'].policy
    env = snapshot['env']  # We assume env is the same
    steps, max_steps = 0, 150
    done = False
    obs = env.reset()  # The initial observation
    exp_policy.reset()

    while steps < max_steps and not done:
        action = exp_policy.get_action(obs)
        obs, rew, done, _ = env.step(action)
        env.render()  # Render the environment to see what's going on (optional)
        steps += 1

    env.close()

if __name__ == '__main__':
    # ml1_exp_plot()
    # mlsp_plot()
    # expert_plot()
    # mtsac_mt10_plot()
    sim_policy()

    # load_pearl_mlsp_policy()