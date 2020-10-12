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
                data_dict[k].append(float(v))
    return data_dict

def plot_curve(matplot, path, exp_name, format='-',
               title = 'MetaTest/Average/AverageReturn',  legend = None, limit = -1):
    result_path = os.path.join(path, exp_name, 'progress.csv')
    data_dict = read_progress_file(result_path)
    # x = data_dict['MetaTest/Average/Iteration']
    x = data_dict['TotalEnvSteps']
    y = data_dict[title]
    label = '{}_{}'.format(exp_name, title)
    if legend != None:
        label = legend
    if limit != -1:
        cap = min(len(x), limit)
        x = x[:cap]
        y = y[:cap]

    max_ret = -1000
    max_itr = 0
    for i in range(len(y)):
        if i % 10 == 0:
            if y[i]> max_ret:
                max_ret = max(max_ret, y[i])
                max_itr = i

    print("{} max iter: {}, max return: {}".format(exp_name, max_itr, max_ret))
    matplot.plot(x, y, format, label=label)
    matplot.legend()

def plot_curve_avg(matplot, exps, format='-',
                   title = 'MetaTest/Average/AverageReturn',  legend = None, limit = -1):
    if exps == None or len(exps) == 0:
        return
    result_paths = [os.path.join(name, 'progress.csv') for name in exps]
    data_dicts = [read_progress_file(result_path) for result_path in result_paths]
    min_step_length = min([len(data_dict['TotalEnvSteps']) for data_dict in data_dicts])
    if limit != -1:
        min_step_length = min(min_step_length, limit)
    print("{}: \nMinimum step length: {}".format(exps[0], min_step_length))
    label = '{}_{}'.format(result_paths[0], title)
    if legend != None:
        label = legend
    x = data_dicts[0]['TotalEnvSteps'][:min_step_length]
    y_ave = np.average([list(map(lambda x: float(x), data_dict[title][:min_step_length])) for data_dict in data_dicts], axis=0)
    y_min = np.min(
        [list(map(lambda x: float(x), data_dict[title][:min_step_length])) for
         data_dict in data_dicts], axis=0)
    y_max = np.max(
        [list(map(lambda x: float(x), data_dict[title][:min_step_length])) for
         data_dict in data_dicts], axis=0)

    matplot.plot(x, y_ave, format, lw=2,label=label)
    matplot.fill_between(x, y_min, y_max, alpha=0.2)
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

    limit = 500

    axs[0].set_title('ML1 Push Avg Return')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')
    plot_curve(axs[0], local_path,
               'pearl_metaworld_ml1_push_1', '-', legend='pearl_rerun', limit=200)
    # plot_curve(axs[0], leviathan_path,
    #            'pearl_metaworld_ml1_push', '-', limit=500)
    #
    # plot_curve(axs[0], local_path,
    #            'pearl_emphasized_metaworld_ml1_push', '-', limit=250, legend="pearl_emphasized")
    # plot_curve(axs[0], local_path,
    #            'pearl_emphasized_metaworld_ml1_push_1', '-', limit=250, legend="pearl_emphasized_corrected_done_signal")

    # plot_curve(axs[0], namazu_path,
    #            'multitask_oracle_metaworld_ml1_push', '-', limit=200)
    plot_curve(axs[0], leviathan_path,
               'multitask_emphasized_oracle_metaworld_ml1_push', '-', legend="multitask_oracle", limit=200)
    #
    plot_curve(axs[0], local_path,
               'curl_metaworld_ml1_push', '-', legend="curl", limit=200)
    #
    plot_curve(axs[0], local_path, 'curl_traj_emphasized_metaworld_ml1_push_2', '-', legend="curl_traj_2_step", limit=200)

    plot_curve(axs[0], local_path, 'curl_shaped_metaworld_ml1_push_2',
               '-', legend="curl_rich_traj_5_step", limit=200)


    axs[1].set_title('ML1 Reach Avg Return')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    plot_curve(axs[1], local_path,
               'pearl_metaworld_ml1_reach_1', '-', legend='pearl_rerun', limit=300)
    # plot_curve(axs[1], leviathan_path,
    #            'pearl_metaworld_ml1_reach', '-', limit=500)
    # plot_curve(axs[1], local_path,
    #            'pearl_emphasized_metaworld_ml1_reach', '-', limit=250, legend="pearl_emphasized")
    # plot_curve(axs[1], local_path,
    #            'pearl_emphasized_metaworld_ml1_reach_1', '-', limit=250, legend="pearl_emphasized_correct_done_signal")

    # plot_curve(axs[1], namazu_path,
    #            'multitask_oracle_metaworld_ml1_reach', '-', limit=250)
    plot_curve(axs[1], leviathan_path,
               'multitask_emphasized_oracle_metaworld_ml1_reach', '-', legend="multitask_oracle", limit=300)
    #
    plot_curve(axs[1], local_path,
               'curl_metaworld_ml1_reach', '-', legend='curl', limit=300)
    plot_curve(axs[1], local_path,
               'curl_traj_emphasized_metaworld_ml1_reach_2', '-', legend='curl_traj_2_step', limit=300)


    axs[2].set_title('ML1 Pick Place Avg Return')
    axs[2].set_xlabel('Total Env Steps')
    axs[2].set_ylabel('Avg Test Return')
    plot_curve(axs[2], local_path,
               'pearl_metaworld_ml1_pick_place_1', '-', legend='pearl', limit=500) # YES
    # plot_curve(axs[2], leviathan_path,
    #            'pearl_metaworld_ml1_pick_place', '-', limit=250) # Keep
    # plot_curve(axs[2], leviathan_path,
    #            'pearl_metaworld_ml1_pick_place_2', '-', limit=500)

    # plot_curve(axs[2], local_path,
    #            'pearl_emphasized_metaworld_ml1_pick_place', '-', limit=250, legend="pearl_emphasized")
    # plot_curve(axs[2], local_path,
    #            'pearl_emphasized_metaworld_ml1_pick_place_1', '-', limit=250, legend="pearl_emphasized_correct_done_signal")

    # plot_curve(axs[2], namazu_path,
    #            'multitask_oracle_metaworld_ml1_pick_place', '-', limit=250)
    # plot_curve(axs[2], leviathan_path,
    #            'multitask_emphasized_oracle_metaworld_ml1_pick_place', '-', limit=250)
    #
    plot_curve(axs[2], namazu_path,
               'curl_metaworld_ml1_pick_place', '-', legend='curl', limit=250)

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
def mlsp_plot_single():
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 2)

    exp_name = 'curl_metaworld_mlsp_ram'
    limit = 200
    axs[0].set_title('MLSP {} Avg Return'.format('MetaTest/Average/AverageReturn'))
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')

    plot_curve(axs[0], local_path,
               exp_name, '-', title='MetaTest/Average/AverageReturn', legend='',limit=limit)

    axs[1].set_title('MLSP Random Average Return Per Task')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/pick-place-v1/AverageReturn', legend='pick-place-return',limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/reach-v1/AverageReturn', legend='reach-return', limit=500)
    # plot_curve(axs[1], local_path,
    #            exp_name, '-',
    #            title='MetaTest/plate-slide-v1/AverageReturn', legend='plate-slide-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/window-open-v1/AverageReturn', legend='window-open-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/stick-pull-v1/AverageReturn', legend='stick-pull-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/push-v1/AverageReturn', legend='push-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/door-open-v1/AverageReturn', legend='door-open-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/hammer-v1/AverageReturn', legend='hammer-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/sweep-v1/AverageReturn', legend='sweep-return', limit=limit)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/sweep-into-v1/AverageReturn', legend='sweep-into-return', limit=500)
    plot_curve(axs[1], local_path,
               exp_name, '-',
               title='MetaTest/basketball-v1/AverageReturn', legend='basketball-return', limit=500)

    plt.show()

def mlsp_plot():
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    fig, axs = plt.subplots(1, 1)
    plot_order = ['MetaTest/Average/AverageReturn']

    axs.set_title('MLSP Training Average Return')
    axs.set_xlabel('Total Env Steps')
    axs.set_ylabel('Avg Test Return')
    plot_curve(axs, leviathan_path,
               'pearl_metaworld_mlsp', '-', title=plot_order[0], legend='pearl_mlsp',limit=250)

    # plot_curve(axs, local_path,
    #            'curl_metaworld_mlsp', '-', title=plot_order[0], limit=250)
    plot_curve(axs, leviathan_path,
               'curl_emphasized_metaworld_mlsp_1', '-', title=plot_order[0], legend='curl_emphasized',limit=500)
    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_2', '-', title=plot_order[0], legend='curl_harder_problems', limit=500)
    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_5', '-', title=plot_order[0], legend='curl_mlsp', limit=500)
    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_ram', '-', title=plot_order[0],
               legend='curl_mlsp_randomized', limit=500)



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


def mlsp_adapt_plot():
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    exp_name = ''
    num_plot = 1
    fig, axs = plt.subplots(1, num_plot)

    axs.set_title('MLSP Adaptation Avg Return')
    axs.set_xlabel('Adapting Epoch')
    axs.set_ylabel('Avg Test Return')

    # Original MLSP last
    # plot_curve(axs, local_path,
    #            'pearl_metaworld_mlsp_adapt_1', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1,
    #            legend='pearl_mlsp_adapt')

    # # Original MLSP last
    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_adapt_4', '-',
               title='MetaTest/Average/AverageReturn', limit=-1, legend='curl_mlsp_last_iter')

    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_adapt_new', '-',
               title='MetaTest/Average/AverageReturn', limit=-1,
               legend='curl_mlsp_best_iter')

    # # New MLSP 100th iter
    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_adapt_5', '-',
               title='MetaTest/Average/AverageReturn', limit=-1, legend='curl_harder_mlsp_best_iter')
    # # New MLSP last
    # plot_curve(axs, local_path,
    #            'curl_metaworld_mlsp_adapt_5', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1, legend='curl_harder_mlsp_last_iter')
    #
    # # Original MLSP last self training
    # plot_curve(axs, local_path,
    #            'curl_metaworld_mlsp_adapt_2nd', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1, legend='curl_mlsp_adapt')
    #
    # # New MLSP 100th iter self training
    # plot_curve(axs, local_path,
    #            'curl_metaworld_mlsp_adapt_2nd_1', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1, legend='curl_harder_mlsp_adapt')

    # corrected pearl adapt best
    # plot_curve(axs, local_path,
    #            'pearl_metaworld_mlsp_adapt_new', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1,
    #            legend='pearl_mlsp_adapt')

    # corrected pearl adapt best

    plot_curve(axs, local_path,
               'curl_metaworld_mlsp_ram_adapt_new', '-',
               title='MetaTest/Average/AverageReturn', limit=-1,
               legend='curl_mlsp_ram_adapt')
    # plot_curve(axs, local_path,
    #            'curl_metaworld_harder_mlsp_adapt_new', '-',
    #            title='MetaTest/Average/AverageReturn', limit=-1,
    #            legend='curl_harder_mlsp_best_iter')

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

def sim_policy2():
    from garage.experiment.task_sampler import SetTaskSampler
    from garage.envs import GarageEnv, normalize
    import metaworld.benchmarks as mwb
    from garage.experiment import Snapshotter
    from garage.torch import set_gpu_mode
    from garage.envs.mujoco import HalfCheetahDirEnv

    exp_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    base_agent_path = '{}/curl_origin_auto_temp_traj_cheetah_dir_1'.format(exp_path)
    snapshotter = Snapshotter()
    snapshot = snapshotter.load(base_agent_path)
    curl = snapshot['algo']
    curl_policy = curl._policy
    set_gpu_mode(True, gpu_id=0)
    curl.to()

    env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(HalfCheetahDirEnv())))
    envs = env_sampler.sample(10)
    sim_env = envs[1]()
    rander_env = sim_env.env.env
    rander_env._task['partially_observable'] = None
    rander_env.reset_model()
    rander_env.frame_skip = 300
    viewer = rander_env._get_viewer('human')

    prev_obs = sim_env.reset()
    sim_env.render()
    curl_policy.sample_from_belief()
    for i in range(200):
        a, agent_info = curl_policy.get_action(prev_obs)
        next_o, r, d, env_info = sim_env.step(a)
        print(
            'Step: {}\nObs:\n {}, \nAction: \n{}\n Reward: \n{}'.format(i, prev_obs,
                                                                        a,
                                                                        r))
        prev_obs = next_o
        sim_env.render()
        sim_env.render()
        sim_env.render()
        sim_env.render()
        sim_env.render()
        if d:
            print("Done")
        if i == 149:
            print("here")



def sim_policy():
    import pickle
    import metaworld.benchmarks as mwb
    from garage.envs import GarageEnv, normalize
    from garage.experiment.task_sampler import EnvPoolSampler
    # create multi-task environment and sample tasks
    ml_test_envs = [
        GarageEnv(normalize(mwb.ML1.get_train_tasks('push-v1')))
    ]

    env_sampler = EnvPoolSampler(ml_test_envs)
    env_sampler.grow_pool(1)
    envs = env_sampler.sample(1)
    rander_env = envs[0]._env._env.env.active_env
    rander_env._task = {'partially_observable': None}
    rander_env.reset_model()
    rander_env.frame_skip=500
    viewer = rander_env._get_viewer('human')
    env = envs[0]._env
    obs = env.reset()
    env.render()
    expert_path = {'observations': [],
                   'actions': [],
                   'rewards': [],
                   'next_observations': [],
                   'dones': []}

    while True:
        action_str = ""
        while len(action_str) == 0:
            env.render()
            env.render()
            env.render()
            env.render()
            env.render()
            action_str = input(":")
        print("action inputed: {}".format(action_str))
        action = np.array(eval(action_str))
        expert_path['observations'].append(obs)
        expert_path['actions'].append(action)
        obs, rew, done, env_infos = env.step(action)
        expert_path['rewards'].append(rew)
        expert_path['next_observations'].append(obs)
        expert_path['dones'].append(done)
        print('Obs:\n {}, \nAction: \n{}\n Reward: \n{}'.format(obs, action, rew))
        env.render()
        env.render()
        env.render()
        env.render()
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


def ml1_push_exp_plot():
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/experiment'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 3)

    limit = 200

    axs[0].set_title('Avg Return Pearl/Oracle ML1 Push ')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')
    # Observation with goal state
    # plot_curve_avg(axs[0],
    #            ['{}/pearl_metaworld_ml1_push_1'.format( local_path)], '-', legend='pearl', limit=limit)
    # plot_curve_avg(axs[0], local_path, ['pearl_auto_temp_metaworld_ml1_push',
    #                                     'pearl_auto_temp_metaworld_ml1_push_1',
    #                                     'pearl_auto_temp_metaworld_ml1_push_2'],
    #                '-', legend="pearl_auto_temp", limit=limit)

    # Only the Observation, but not goal state
    plot_curve_avg(axs[0], ['{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(local_path),
                            '{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(namazu_path),
                            '{}/pearl_origin_auto_temp_metaworld_ml1_push_1'.format(namazu_path),
                            '{}/pearl_origin_auto_temp_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="pearl_origin_auto_temp", limit=limit)

    plot_curve_avg(axs[0],
                   ['{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(local_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(namazu_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push_1'.format(namazu_path)],
                   '-', legend="multitask_oracle_origin_auto_temp", limit=limit)



    axs[1].set_title('Avg Return CURL ML1 Push Traj vs Rich Traj')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    # Observation with goal state
    # plot_curve_avg(axs[1], local_path, ['curl_traj_emphasized_metaworld_ml1_push_2'],
    #            '-', legend="curl_traj_2_step", limit=limit)
    # plot_curve_avg(axs[1], local_path,
    #                ['curl_auto_temp_traj_metaworld_ml1_push',
    #                 'curl_auto_temp_traj_metaworld_ml1_push_1',
    #                 'curl_auto_temp_traj_metaworld_ml1_push_2'],
    #                '-', legend="curl_auto_temp", limit=limit)

    # Only the Observation, but not goal state
    plot_curve_avg(axs[1],
                   ['{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(local_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_1'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="curl_origin_traj_2_auto_temp", limit=limit)

    plot_curve_avg(axs[1],
                   ['{}/curl_origin_shaped_metaworld_ml1_push'.format(local_path)],
                   '-', legend="curl_origin_rich_traj_5_step_auto_temp", limit=limit)
    plot_curve_avg(axs[1],
                   ['{}/curl_origin_shaped_metaworld_ml1_push_2'.format(local_path)],
                   '-', legend="curl_origin_rich_traj_2_step_auto_temp",limit=limit)

    # plot_curve_avg(axs[1],
    #                ['{}/curl_origin_auto_temp_traj_metaworld_ml1_push_1'.format(local_path)],
    #                '-', legend="curl_origin_traj_2_step_auto_temp_fix", limit=limit)

    # plot_curve_avg(axs[1],
    #                [
    #                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_2'.format(
    #                        local_path)],
    #                '-', legend="curl_origin_traj_1_step_auto_temp_fix",
    #                limit=limit)
    #
    # plot_curve_avg(axs[1],['{}/curl_origin_shaped_metaworld_ml1_push_1'.format(local_path)],
    #                '-', legend="curl_origin_rich_traj_1_step_auto_temp_fix", limit=limit)

    # plot_curve_avg(axs[1], [
    #     '{}/curl2_origin_auto_temp_traj_metaworld_ml1_push_2'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_vf_net",
    #                limit=limit)


    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_3'.format(local_path)],
    #                '-', legend="curl_origin_rich_traj_1_step_auto_temp_no_q_gradient",
    #                limit=limit)

    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_4'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_random_sample",
    #                limit=limit)

    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_5'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_no_kl",
    #                limit=limit)

    plot_curve_avg(axs[1], ['{}/curl_paper_ml1_push'.format(local_path),
                            '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence", limit=limit)

    plot_curve_avg(axs[1], [
        '{}/curl-push-v1_1'.format(
            local_path)],
                   '-',
                   legend="curl_no_kl_not_in_sequence",
                   limit=limit)

    plot_curve_avg(axs[1], [
        '{}/curl-push-v1_2'.format(
            local_path)],
                   '-',
                   legend="curl_no_kl_in_sequence_single_step",
                   limit=limit)

    axs[2].set_title('ML1 Comparison Avg Return')
    axs[2].set_xlabel('Total Env Steps')
    axs[2].set_ylabel('Avg Test Return')

    # Observation with goal state
    # plot_curve(axs[2], local_path,
    #            'pearl_auto_temp_metaworld_ml1_push', '-',
    #            legend="pearl_auto_temp", limit=limit)
    #
    # plot_curve(axs[2], local_path, 'curl_auto_temp_traj_metaworld_ml1_push',
    #            '-', legend="curl_traj_2_step_auto_temp", limit=limit)
    # plot_curve_avg(axs[2], local_path, ['pearl_auto_temp_metaworld_ml1_push',
    #                                     'pearl_auto_temp_metaworld_ml1_push_1',
    #                                     'pearl_auto_temp_metaworld_ml1_push_2'],
    #                '-', legend="pearl_auto_temp", limit=limit)
    # plot_curve_avg(axs[2], local_path, ['curl_auto_temp_traj_metaworld_ml1_push',
    #                                     'curl_auto_temp_traj_metaworld_ml1_push_1',
    #                                     'curl_auto_temp_traj_metaworld_ml1_push_2'],
    #                '-', legend="curl_auto_temp", limit=limit)

    # Only the Observation, but not goal state

    plot_curve_avg(axs[2], [
        '{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(local_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(namazu_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push_1'.format(namazu_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="pearl_origin_auto_temp", limit=limit)

    plot_curve_avg(axs[2],
                   ['{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(local_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(namazu_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push_1'.format(namazu_path)],
                   '-', legend="multitask_oracle_origin_auto_temp",limit=limit)

    plot_curve_avg(axs[2],
                   ['{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(local_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_1'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="curl_origin_traj_2_step_auto_temp", limit=limit)
    # ,
    #                     '{}/curl_origin_shaped_metaworld_ml1_push_1'.format(
    #                         leviathan_path),
    #                     '{}/curl_origin_shaped_metaworld_ml1_push_2'.format(
    #                         leviathan_path)
    plot_curve_avg(axs[2],
                   ['{}/curl_origin_shaped_metaworld_ml1_push'.format(local_path)],
                   '-', legend="curl_origin_rich_traj_5_step_auto_temp",limit=limit)

    plot_curve_avg(axs[2],
                   ['{}/curl_origin_shaped_metaworld_ml1_push_2'.format(
                       local_path)],
                   '-', legend="curl_origin_rich_traj_2_step_auto_temp",
                   limit=limit)

    plt.show()


def mujoco_exp_plot():
    leviathan_path = '/home/simon0xzx/research/berkely_research/garage/data/leviathan/experiment'
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/experiment'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(1, 3)

    limit = 500

    axs[0].set_title('Cheetah Vel Avg Return')
    axs[0].set_xlabel('Total Env Steps')
    axs[0].set_ylabel('Avg Test Return')

    plot_curve_avg(axs[0],
                   ['{}/pearl_origin_auto_temp_traj_cheetah_vel_1'.format(local_path)],
                   '-', legend="pearl", limit=limit)


    # Only the Observation, but not goal state
    plot_curve_avg(axs[0], ['{}/curl_origin_auto_temp_traj_cheetah_vel_1'.format(local_path)],
                   '-', legend="curl_kl_random_aug_traj_2_step", limit=limit)

    # plot_curve_avg(axs[0],
    #                [
    #                    '{}/curl_origin_auto_temp_traj_cheetah_vel_4'.format(
    #                        local_path)],
    #                '-', legend="curl_vf",
    #                limit=limit)

    # plot_curve_avg(axs[0],
    #                [
    #                    '{}/curl_origin_auto_temp_traj_cheetah_vel_5'.format(
    #                        local_path)],
    #                '-', legend="curl_no_q",
    #                limit=limit)

    # plot_curve_avg(axs[0], [
    #     '{}/curl_origin_auto_temp_traj_cheetah_vel_6'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_sample_transition",
    #                limit=limit)

    # plot_curve_avg(axs[0],
    #                ['{}/curl_origin_auto_temp_traj_cheetah_vel_8'.format(
    #                    local_path)],
    #                '-', legend="curl_2_positive_no_kl", limit=limit)

    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_3'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_no_q_gradient",
    #                limit=limit)
    #
    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_4'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_random_sample",
    #                limit=limit)
    #
    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_5'.format(
    #         local_path)],
    #                '-',
    #                legend="curl_origin_rich_traj_1_step_auto_temp_no_kl",
    #                limit=limit)
    #

    plot_curve_avg(axs[0], ['{}/curl_origin_auto_temp_traj_cheetah_vel_14'.format( local_path)],
                   '-', legend="curl_no_kl_in_sequence_aug_single_step", limit=limit)


    axs[1].set_title('Humanoid Dir Avg Return')
    axs[1].set_xlabel('Total Env Steps')
    axs[1].set_ylabel('Avg Test Return')

    # Only the Observation, but not goal state
    # plot_curve_avg(axs[1], [
    #     '{}/curl_origin_auto_temp_traj_cheetah_dir'.format(local_path),
    #     '{}/curl_origin_auto_temp_traj_cheetah_dir'.format(namazu_path),
    #     '{}/curl_origin_auto_temp_traj_cheetah_dir_1'.format(namazu_path),
    #     '{}/curl_origin_auto_temp_traj_cheetah_dir_2'.format(namazu_path)],
    #                '-', legend="curl", limit=limit)

    plot_curve_avg(axs[1],
                   ['{}/pearl_origin_auto_temp_traj_humanoid_dir_2'.format(
                       local_path)],
                   '-', legend="pearl", limit=limit)
    #
    # plot_curve_avg(axs[1],
    #                ['{}/curl_origin_auto_temp_traj_humanoid_dir_1'.format(
    #                    local_path)],
    #                '-', legend="curl", limit=limit)
    #
    # plot_curve_avg(axs[1],
    #                ['{}/curl_origin_auto_temp_traj_humanoid_dir_4'.format(
    #                    local_path)],
    #                '-', legend="curl_vf", limit=limit)
    #
    # plot_curve_avg(axs[1],
    #                ['{}/curl_origin_auto_temp_traj_humanoid_dir_5'.format(
    #                    local_path)],
    #                '-', legend="curl_no_q", limit=limit)
    #
    # plot_curve_avg(axs[1],
    #                ['{}/curl_origin_auto_temp_traj_humanoid_dir_6'.format(
    #                    local_path)],
    #                '-', legend="curl_sample_transition", limit=limit)
    # plot_curve_avg(axs[1], ['{}/curl_origin_auto_temp_traj_humanoid_dir_7'.format(
    #                    local_path)], '-', legend="curl_2_positive_no_kl", limit=limit)

    plot_curve_avg(axs[1], ['{}/curl_origin_auto_temp_traj_humanoid_dir_8'.format(
                       local_path)], '-', legend="curl_no_kl_single_step_256_batch_random_aug", limit=limit)

    plot_curve_avg(axs[1], ['{}/curl_origin_auto_temp_traj_humanoid_dir_15'.format(
                       local_path)], '-', legend="curl_no_kl_single_step_128_batch_in_sequence_aug", limit=limit)



    plt.show()

def grab_task_list(root_dir, task_name, prefix, exp_set, max_count = 5):
    pearl_exp_names = []
    for i in range(max_count):
        exp_name = '{}-{}'.format(prefix, task_name)
        if i >= 1:
            exp_name += '_{}'.format(i)
        if exp_name in exp_set:
            pearl_exp_names.append(os.path.join(root_dir, exp_name))
        else:
            break
    return pearl_exp_names

def ml1_tasks_comparison():
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(3, 5)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.25, hspace=0.30)
    limit = 250
    task_list = ['button-press-topdown-v1', 'coffee-push-v1', 'dissassemble-v1',
                 'faucet-close-v1', 'faucet-open-v1',
                 'hammer-v1', 'handle-press-v1', 'handle-pull-side-v1',
                 'lever-pull-v1', 'plate-slide-back-v1', 'plate-slide-side-v1',
                 'soccer-v1', 'stick-pull-v1', 'stick-push-v1', 'button-press-topdown-wall-v1']

    exp_set = set(os.listdir(namazu_path))

    for i in range(len(task_list)):
        task_name = task_list[i]
        print('preparing exp: {}'.format(task_name))
        plt_x = int(i / 5)
        plt_y = int(i % 5 if i % 5 >= 0 else i % 5 + 5)
        plt_axs = axs[plt_x][plt_y]
        plt_axs.set_title('Avg Return ML1 {}'.format(task_name))
        plt_axs.set_xlabel('Total Env Steps')
        plt_axs.set_ylabel('Avg Test Return')

        pearl_exp_names = grab_task_list(namazu_path, task_name, 'pearl', exp_set, max_count=5)
        plot_curve_avg(plt_axs, pearl_exp_names, '-', legend="pearl", limit=limit)

        curl_exp_names = grab_task_list(namazu_path, task_name, 'curl', exp_set, max_count=5)
        plot_curve_avg(plt_axs, curl_exp_names, '-', legend="curl", limit=limit)



    plt.show()


def ml1_push_display():
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/experiment'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    fig, axs = plt.subplots(2, 4)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.25, hspace=0.30)
    limit = 200

    axs[0][0].set_title('ML1 Push Pearl vs Oracle')
    axs[0][0].set_xlabel('Total Env Steps')
    axs[0][0].set_ylabel('Avg Test Return')
    # Only the Observation, but not goal state
    plot_curve_avg(axs[0][0], [
        '{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(local_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push'.format(namazu_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push_1'.format(namazu_path),
        '{}/pearl_origin_auto_temp_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="pearl", limit=limit)
    plot_curve_avg(axs[0][0],
                   ['{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(local_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(namazu_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push_1'.format(namazu_path)],
                   '-', legend="multitask_oracle", limit=limit)
    # ==========================================================================
    axs[0][1].set_title('ML1 Push CURL Traj vs Rich Traj')
    axs[0][1].set_xlabel('Total Env Steps')
    axs[0][1].set_ylabel('Avg Test Return')
    plot_curve_avg(axs[0][1],
                   ['{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(local_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_1'.format(namazu_path),
                    '{}/curl_origin_auto_temp_traj_metaworld_ml1_push_2'.format(namazu_path)],
                   '-', legend="curl_traj_2_step_no_q_no_kl", limit=limit)
    plot_curve_avg(axs[0][1],
                   ['{}/curl_origin_shaped_metaworld_ml1_push_2'.format(local_path)],
                   '-', legend="curl_rich_traj_2_step_no_q_no_kl", limit=limit)
    # ==========================================================================
    axs[0][2].set_title('ML1 Push CURL Rich Traj 2 step vs 5 step')
    axs[0][2].set_xlabel('Total Env Steps')
    axs[0][2].set_ylabel('Avg Test Return')
    plot_curve_avg(axs[0][2],
                   ['{}/curl_origin_shaped_metaworld_ml1_push'.format(local_path)],
                   '-', legend="curl_rich_traj_5_step_no_q_no_kl", limit=limit)
    plot_curve_avg(axs[0][2],
                   ['{}/curl_origin_shaped_metaworld_ml1_push_2'.format(local_path)],
                   '-', legend="curl_rich_traj_2_step_no_q_no_kl", limit=limit)
    # ==========================================================================
    axs[0][3].set_title('ML1 Push CURL traj 2 step vs traj 1 step')
    axs[0][3].set_xlabel('Total Env Steps')
    axs[0][3].set_ylabel('Avg Test Return')
    plot_curve_avg(axs[0][3], ['{}/curl_paper_ml1_push'.format(local_path),
                               '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence_traj_2_step", limit=limit)

    plot_curve_avg(axs[0][3], ['{}/curl-push-v1_2'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence_traj_1_step", limit=limit)
    #==========================================================================
    axs[1][0].set_title('ML1 Push CURL KL vs no KL')
    axs[1][0].set_xlabel('Total Env Steps')
    axs[1][0].set_ylabel('Avg Test Return')
    plot_curve_avg(axs[1][0],
                   ['{}/curl_origin_auto_temp_traj_metaworld_ml1_push_1'.format(local_path)],
                   '-', legend="curl_kl", limit=limit)

    plot_curve_avg(axs[1][0], ['{}/curl_paper_ml1_push'.format(local_path),
                               '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl", limit=limit)
    # ==========================================================================
    axs[1][1].set_title('ML1 Push CURL In Sequence Aug vs Random Aug')
    axs[1][1].set_xlabel('Total Env Steps')
    axs[1][1].set_ylabel('Avg Test Return')

    plot_curve_avg(axs[1][1], ['{}/curl_paper_ml1_push'.format(local_path),
                            '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence", limit=limit)

    plot_curve_avg(axs[1][1], ['{}/curl-push-v1_1'.format(local_path)],
                   '-', legend="curl_no_kl_not_in_sequence", limit=limit)

    # ==========================================================================
    axs[1][2].set_title('ML1 Push CURL rich traj Q vs no Q')
    axs[1][2].set_xlabel('Total Env Steps')
    axs[1][2].set_ylabel('Avg Test Return')

    plot_curve_avg(axs[1][2], ['{}/curl-push-v1_2'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence_traj_1_step", limit=limit)
    plot_curve_avg(axs[1][2], ['{}/curl_origin_shaped_metaworld_ml1_push_4'.format(local_path)],
                   '-', legend="curl_no_kl_no_q_in_sequence_rich_traj_1_step", limit=limit)
    # ==========================================================================
    axs[1][3].set_title('ML1 Push CURL comparison')
    axs[1][3].set_xlabel('Total Env Steps')
    axs[1][3].set_ylabel('Avg Test Return')

    plot_curve_avg(axs[1][3],
                   ['{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(local_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push'.format(namazu_path),
                    '{}/multitask_oracle_auto_temp_metaworld_ml1_push_1'.format(namazu_path)],
                   '-', legend="multitask_oracle", limit=limit)

    plot_curve_avg(axs[1][3],
                   ['{}/curl_origin_shaped_metaworld_ml1_push_2'.format(local_path)],
                   '-', legend="curl_rich_traj_2_step_no_q_no_kl", limit=limit)

    plot_curve_avg(axs[1][3], ['{}/curl-push-v1_2'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence_traj_1_step",
                   limit=limit)
    plot_curve_avg(axs[1][3], ['{}/curl_paper_ml1_push'.format(local_path),
                               '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl", limit=limit)

    plot_curve_avg(axs[1][3], ['{}/curl_paper_ml1_push'.format(local_path),
                               '{}/curl-push-v1'.format(local_path)],
                   '-', legend="curl_no_kl_in_sequence", limit=limit)
    plt.show()

def ml1_hype_param_plot():
    namazu_path = '/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results'
    local_path = '/home/simon0xzx/research/berkely_research/garage/data/local/curl_fine_tune'
    fig, axs = plt.subplots(1, 4)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.25, hspace=0.30)
    limit = 200
    task_list = ['faucet-close-v1','handle-press-v1', 'plate-slide-side-v1', 'soccer-v1']

    exp_set = set(os.listdir(local_path))
    plot_curve_avg(axs[0], ['{}/faucet-close-v1'.format(local_path)],
                   '-', legend="curl", limit=limit)
    plot_curve_avg(axs[0], ['{}/faucet-close-v1_1'.format(local_path)],
                   '-', legend="curl_emphasized", limit=limit)

    plot_curve_avg(axs[1], ['{}/handle-press-v1'.format(local_path)],
                   '-', legend="curl", limit=limit)
    plot_curve_avg(axs[1], ['{}/handle-press-v1_1'.format(local_path)],
                   '-', legend="curl_emphasized", limit=limit)

    plot_curve_avg(axs[2], ['{}/plate-slide-side-v1'.format(local_path)],
                   '-', legend="curl", limit=limit)
    plot_curve_avg(axs[2], ['{}/plate-slide-side-v1_1'.format(local_path)],
                   '-', legend="curl_emphasized", limit=limit)

    plot_curve_avg(axs[3], ['{}/soccer-v1'.format(local_path)],
                   '-', legend="curl", limit=limit)
    # plot_curve_avg(axs[3], ['{}/soccer-v1_1'.format(local_path)],
    #                '-', legend="curl_emphasized", limit=limit)

    plt.show()

if __name__ == '__main__':
    # ml1_exp_plot()
    # mlsp_plot()
    # mlsp_plot_single()
    # expert_plot()
    # mtsac_mt10_plot()
    # mlsp_adapt_plot()
    # sim_policy()
    # sim_policy2()
    # ml1_push_exp_plot()
    # ml1_push_display()
    # mujoco_exp_plot()
    ml1_hype_param_plot()
    # ml1_tasks_comparison()
