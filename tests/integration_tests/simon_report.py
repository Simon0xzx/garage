from tests.integration_tests.simon_display_helper import print_hyper_tests, read_csv_file
import matplotlib.pyplot as plt
from os import path

# Name -> (exp_root_path, labels)
def get_data_repos():
    data_dir = '/home/simon0xzx/research/berkely_research/garage/data'
    data_repo = {
    'pearl_old_ml1_result': (path.join(data_dir, '/namazu/ml1_results'), 'old_pearl_result'),
    'curl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', 'old_curl_result'),
    'large_batch': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_test2', 'curl_large_batch_size'),
    'new_label': ('/home/simon0xzx/research/berkely_research/garage/data/local/curl_new_label', 'curl_new_label_batch_16', 'curl_new_label_batch_64'),
    'namazu_new_label': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_new_label', 'curl_new_label_batch_32'),
    # full suit results
    'old_curl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/old_curl_ml1_result_suit', 'old_curl'),
    'pearl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/pearl_ml1_result_suit', 'old_pearl'),
    'curl_labeled_b16': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_labeled_b16', 'curl_labeled_b16'),
    'updated_curl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_normal', 'updated_curl'),
    'classifier_encoder':('/home/simon0xzx/research/berkely_research/garage/data/local/classifier_suit', 'classifier_encoder'),
    'curl_wasserstein_old': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein', 'curl_wasserstein_old'),
    'curl_wasserstein': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_final', 'curl_wasserstein'),
    'curl_wasserstein_2_step': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_final_2_steps', 'curl_wasserstein_2_step'),
    'curl_wasserstein_2': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_final2', 'curl_wasserstein'),
    'rl2_ppo_suit': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/rl2_ppo_suit', 'rl2_ppo'),
    # 10 envs
    'curl_wasserstein_long_10': (path.join(data_dir, 'result_suits/curl_wasserstein_long_10_suit'),'curl_wasserstein'),
    'pearl_long_10': (path.join(data_dir, 'result_suits/pearl_10_suit'), 'pearl'),

    # other methods
    'maml_trpo': ('/home/simon0xzx/research/berkely_research/garage/data/local/maml_trpo_suit_2','maml_trpo'),
    'maml_ppo': ('/home/simon0xzx/research/berkely_research/garage/data/local/maml_ppo_suit', 'maml_ppo'),
    'rl2_ppo': ('/home/simon0xzx/research/berkely_research/garage/data/local/rl2_ppo_suit', 'rl2_ppo'),
    'rl2_ppo_2': ('/home/simon0xzx/research/berkely_research/garage/data/local/rl2_ppo_suit_2', 'rl2_ppo'),
    'rl2_ppo_larger_net': ('/home/simon0xzx/research/berkely_research/garage/data/local/rl2_ppo_larger_net_suit', 'rl2_ppo_larger_net'),
    'rl2': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/rl2_suit', 'rl2'),
    'mql': ('/home/simon0xzx/research/berkely_research/meta-q-learning/log_dir', 'mql')
    }
    return data_repo

def get_metaworld_task_list(task_portion='Odd'):
    task_name_csv = '/home/simon0xzx/research/berkely_research/garage/examples/metaworld_exp_name.txt'
    csv_dict = read_csv_file(task_name_csv, type=str)
    if task_portion == 'Odd':
        task_names = [task_name for i, task_name in enumerate(csv_dict['task_name']) if i%2 ==0]
    elif task_portion == 'Even':
        task_names = [task_name for i, task_name in enumerate(csv_dict['task_name']) if i % 2 == 1]
    else:
        task_names = csv_dict['task_name']

    return task_names



def metaworld_ml1_graph(axs, task_lists, draw_repo_names, row=4, col=6, num_seeds=1,
                        limit=-1, title='MetaTest/Average/AverageReturn', x_title='TotalEnvSteps',
                        backward_smooth_window=1):
    data_repo = get_data_repos()
    for i, task in enumerate(task_lists):
        row_cnt = int(i / col)
        col_cnt = i % col if i % col >= 0 else (i % col) + col
        if col == 1 and row > 1:
            subplot_axs = axs[row_cnt]
        elif row == 1 and col > 1:
            subplot_axs = axs[col_cnt]
        elif row == 1 and col ==1:
            subplot_axs = axs
        else:
            subplot_axs = axs[row_cnt][col_cnt]
        subplot_axs.set_title('ML1 {}'.format(task))
        subplot_axs.set_xlabel(x_title)
        subplot_axs.set_ylabel(title.split('/')[-1])
        subplot_axs.ticklabel_format(useMathText=True)
        subplot_axs.axvline(x=2500000, linewidth=2, color='r')
        # subplot_axs.set_xlim(0, 2500000)
        for repo in draw_repo_names:
            exp_repo = data_repo[repo]
            print_hyper_tests(subplot_axs, exp_repo[0], task, exp_repo[1],
                              num_seeds = num_seeds, title=title, x_title=x_title, limit=limit, backward_smooth_window=backward_smooth_window)
    plt.show()


def plot_full_suits():
    sampled_task_lists = ['faucet-close-v1', 'soccer-v1', 'plate-slide-side-v1', 'stick-push-v1']
    full_suit_task_lists = ['faucet-open-v1', 'faucet-close-v1', 'lever-pull-v1', 'stick-push-v1', 'handle-pull-side-v1', 'stick-pull-v1', 'dissassemble-v1', 'coffee-push-v1', 'hammer-v1', 'plate-slide-side-v1', 'handle-press-v1', 'soccer-v1', 'plate-slide-back-v1', 'button-press-topdown-v1', 'button-press-topdown-wall-v1', 'peg-insert-side-v1', 'push-wall-v1', 'button-press-v1', 'coffee-pull-v1', 'window-close-v1', 'door-open-v1', 'drawer-open-v1', 'box-close-v1', 'door-unlock-v1', 'basketball-v1']
    row, col = 5,5
    fig, axs = plt.subplots(row, col)
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.96,
    #                     wspace=0.26, hspace=0.70)
    # row, col = 4, 6
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.96,
                        wspace=0.25, hspace=0.30)

    # metaworld_ml1_graph(axs, full_suit_task_lists, ['pearl', 'curl_wasserstein', 'curl_wasserstein_2_step'], row=row, col=col)
    # metaworld_ml1_graph(axs, full_suit_task_lists, ['old_curl', 'updated_curl', 'curl_wasserstein', 'curl_wasserstein_2_step'], row=row, col=col)
    metaworld_ml1_graph(axs, full_suit_task_lists, ['curl_wasserstein_2', 'rl2_ppo_suit', 'pearl'], num_seeds=3,  row=row, col=col, limit=100)

    plt.show()

'''
soccer-v1
push-wall-v1
coffee-pull-v1
button-press-topdown-wall-v1
faucet-close-v1
pick-out-of-hole-v1
dial-turn-v1
plate-slide-back-side-v1
window-open-v1
door-close-v1
'''
def plot_selected_suits():

    sampled_task_lists = ['soccer-v1', 'push-wall-v1', 'coffee-pull-v1',
                          'button-press-topdown-wall-v1', 'faucet-close-v1',
                          'pick-out-of-hole-v1', 'dial-turn-v1', 'plate-slide-back-side-v1',
                          'window-open-v1', 'door-close-v1']

    row, col = 2,5
    fig, axs = plt.subplots(row, col)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.96,
                        wspace=0.26, hspace=0.70)
    # row, col = 4, 6
    # plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.96,
    #                     wspace=0.25, hspace=0.30)

    # metaworld_ml1_graph(axs, full_suit_task_lists, ['pearl', 'curl_wasserstein', 'curl_wasserstein_2_step'], row=row, col=col)
    metaworld_ml1_graph(axs, sampled_task_lists, ['curl_wasserstein_long_10', 'pearl_long_10', 'rl2_ppo_2'], limit=100, num_seeds=3, row=row, col=col)

    # , 'pearl_long_10', 'pearl'
    plt.show()



def varify():
    sampled_task_lists = ['soccer-v1', 'drawer-open-v1', 'button-press-topdown-v1']
    full_suit_task_lists = ['faucet-open-v1', 'faucet-close-v1',
                            'lever-pull-v1', 'stick-push-v1',
                            'handle-pull-side-v1', 'stick-pull-v1',
                            'dissassemble-v1', 'coffee-push-v1', 'hammer-v1',
                            'plate-slide-side-v1', 'handle-press-v1',
                            'soccer-v1', 'plate-slide-back-v1',
                            'button-press-topdown-v1',
                            'button-press-topdown-wall-v1',
                            'peg-insert-side-v1', 'push-wall-v1',
                            'button-press-v1', 'coffee-pull-v1',
                            'window-close-v1', 'door-open-v1',
                            'drawer-open-v1', 'box-close-v1', 'door-unlock-v1', 'basketball-v1']

    row, col = 5, 5
    fig, axs = plt.subplots(row, col)
    metaworld_ml1_graph(axs, full_suit_task_lists, ['curl_wasserstein', 'rl2_ppo_2'], num_seeds=1, row=row, col=col, limit=100)


def process():
    stuff = "button-press-topdown-v1: 1 windows (created Fri Dec  4 00:38:34 2020) [130x71] \n\
button-press-topdown-v1_2: 1 windows (created Sun Dec  6 02:27:24 2020) [130x71]\n\
button-press-topdown-wall-v1: 1 windows (created Fri Dec  4 00:39:40 2020) [130x71]\n\
button-press-topdown-wall-v1_12: 1 windows (created Thu Nov 26 05:49:39 2020) [95x71]\n\
button-press-topdown-wall-v1_13: 1 windows (created Thu Nov 26 05:50:37 2020) [85x71]\n\
coffee-pull-v1: 1 windows (created Fri Dec  4 00:38:45 2020) [130x71]\n\
coffee-pull-v1_2: 1 windows (created Sun Dec  6 02:27:34 2020) [130x71]\n\
dial-turn-v1_11: 1 windows (created Thu Nov 26 05:51:58 2020) [85x71]\n\
dial-turn-v1_12: 1 windows (created Thu Nov 26 05:52:13 2020) [85x71]\n\
dial-turn-v1_13: 1 windows (created Thu Nov 26 05:52:18 2020) [85x71]\n\
door-open-v1: 1 windows (created Fri Dec  4 00:39:51 2020) [130x71]\n\
door-unlock-v1: 1 windows (created Fri Dec  4 00:40:00 2020) [130x71]\n\
drawer-open-v1: 1 windows (created Fri Dec  4 00:38:04 2020) [130x71]\n\
drawer-open-v1_2: 1 windows (created Sun Dec  6 02:27:02 2020) [130x71]\n\
faucet-close-v1_11: 1 windows (created Thu Nov 26 05:50:50 2020) [95x71]\n\
faucet-close-v1_12: 1 windows (created Thu Nov 26 05:50:55 2020) [95x71]\n\
faucet-close-v1_13: 1 windows (created Thu Nov 26 05:51:01 2020) [85x71]\n\
handle-press-v1: 1 windows (created Fri Dec  4 00:39:03 2020) [130x71]\n\
pearl_button-press-topdown-wall-v1_11: 1 windows (created Sun Nov 29 04:45:25 2020) [95x71]\n\
pearl_coffee-pull-v1_11: 1 windows (created Sun Nov 29 04:44:57 2020) [95x71]\n\
pearl_coffee-pull-v1_12: 1 windows (created Sun Nov 29 04:45:01 2020) [95x71]\n\
pearl_coffee-pull-v1_13: 1 windows (created Sun Nov 29 04:45:12 2020) [95x71]\n\
pearl_door-close-v1_11: 1 windows (created Sun Nov 29 04:46:23 2020) [95x71]\n\
pearl_door-close-v1_12: 1 windows (created Sun Nov 29 04:46:29 2020) [95x71]\n\
pearl_door-close-v1_13: 1 windows (created Sun Nov 29 04:46:40 2020) [95x71]\n\
pearl_push-wall-v1_11: 1 windows (created Thu Dec  3 18:07:19 2020) [85x71]\n\
pearl_push-wall-v1_12: 1 windows (created Sun Nov 29 04:43:25 2020) [85x71]\n\
pearl_push-wall-v1_13: 1 windows (created Sun Nov 29 04:44:46 2020) [95x71]\n\
pearl_soccer_11: 1 windows (created Thu Dec  3 18:01:24 2020) [85x71]\n\
pearl_soccer_12: 1 windows (created Thu Dec  3 18:02:29 2020) [85x71]\n\
pearl_soccer_13: 1 windows (created Thu Dec  3 18:06:43 2020) [85x71]\n\
pearl_window-open-v1_11: 1 windows (created Sun Nov 29 04:45:53 2020) [95x71]\n\
pearl_window-open-v1_12: 1 windows (created Sun Nov 29 04:45:58 2020) [95x71]\n\
pearl_window-open-v1_13: 1 windows (created Sun Nov 29 04:46:10 2020) [95x71]\n\
pick-out-of-hole-v1_11: 1 windows (created Thu Dec  3 18:08:49 2020) [85x71]\n\
pick-out-of-hole-v1_12: 1 windows (created Thu Dec  3 18:08:57 2020) [85x71]\n\
pick-out-of-hole-v1_13: 1 windows (created Thu Dec  3 18:09:01 2020) [85x71]\n\
plate-slide-back-side-v1_11: 1 windows (created Thu Nov 26 05:52:31 2020) [85x71]\n\
plate-slide-back-side-v1_12: 1 windows (created Thu Nov 26 05:55:12 2020) [95x71]\n\
plate-slide-back-side-v1_13: 1 windows (created Thu Nov 26 05:55:19 2020) [95x71]\n\
plate-slide-side-v1: 1 windows (created Fri Dec  4 00:39:11 2020) [130x71]\n\
push-wall-v1: 1 windows (created Fri Dec  4 00:39:30 2020) [130x71]\n\
rl2_box-close-v1: 1 windows (created Mon Dec  7 14:22:38 2020) [130x71]\n\
rl2_button-press-topdown-v1: 1 windows (created Mon Dec  7 14:23:16 2020) [130x71]\n\
rl2_button-press-topdown-wall-v1: 1 windows (created Mon Dec  7 14:24:20 2020) [130x71]\n\
rl2_button-press-v1: 1 windows (created Mon Dec  7 14:22:19 2020) [130x71]\n\
rl2_coffee-pull-v1: 1 windows (created Mon Dec  7 14:23:25 2020) [130x71]\n\
rl2_door-open-v1: 1 windows (created Mon Dec  7 14:24:30 2020) [130x71]\n\
rl2_door-unlock-v1: 1 windows (created Mon Dec  7 14:24:39 2020) [130x71]\n\
rl2_drawer-open-v1: 1 windows (created Mon Dec  7 14:23:07 2020) [130x71]\n\
rl2_hammer-v1: 1 windows (created Mon Dec  7 14:22:28 2020) [130x71]\n\
rl2_handle-press-v1: 1 windows (created Mon Dec  7 14:23:42 2020) [130x71]\n\
rl2_peg-insert-side-v1: 1 windows (created Mon Dec  7 14:22:56 2020) [130x71]\n\
rl2_plate-slide-back-v1: 1 windows (created Mon Dec  7 14:22:47 2020) [130x71]\n\
rl2_plate-slide-side-v1: 1 windows (created Mon Dec  7 14:23:52 2020) [130x71]\n\
rl2_push-wall-v1: 1 windows (created Mon Dec  7 14:24:10 2020) [130x71]\n\
rl2_soccer-v1: 1 windows (created Mon Dec  7 14:24:01 2020) [130x71]\n\
rl2_window-close-v1: 1 windows (created Mon Dec  7 14:23:33 2020) [130x71]\n\
soccer-v1: 1 windows (created Fri Dec  4 00:39:21 2020) [130x71]\n\
window-close-v1: 1 windows (created Fri Dec  4 00:38:54 2020) [130x71]\n\
window-close-v1_2: 1 windows (created Sun Dec  6 02:27:48 2020) [130x71]"

    for line in stuff.split('\n'):
        env = line.split(':')[0]
        statement = "sudo tmux kill-session -t {}".format(env)
        print(statement)

if __name__ == '__main__':
    # plot_full_suits()
    # plot_selected_suits()
    # varify()
    process()
