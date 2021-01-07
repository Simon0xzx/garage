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
    'pearl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/pearl', 'pearl'),
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
    'rl2_ppo_manual_reduce': ('/home/simon0xzx/research/berkely_research/garage/data/local/rl2_ppo_suit_2','rl2_ppo_manual_reduce'),
    'rl2_ppo_larger_net': ('/home/simon0xzx/research/berkely_research/garage/data/local/rl2_ppo_larger_net_suit', 'rl2_ppo_larger_net'),
    'rl2': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/rl2_suit', 'rl2'),
    'mql': ('/home/simon0xzx/research/berkely_research/meta-q-learning/log_dir', 'mql'),

    # local stuff
    'curl_w_env_step': ('/home/simon0xzx/research/berkely_research/garage/data/local/curl_wasserstein_env_steps', 'env_steps'),
    'curl_step_4x_reduce': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_env_4x_reduce', '4x_reduce'),
    'curl_step_2x_reduce': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_env_2x_reduce', '2x_reduce'),
    'curl_wasserstein_10x_reduce': ("/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_env_steps_400","10x_reduce_old"),
    'curl_wasserstein_2x_reduce': ("/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_wasserstein_env_steps_2000", "2x_reduce_old")

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
                        env_step_limit=-1, backward_smooth_window=1):
    data_repo = get_data_repos()
    env_reports = {}
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
        if env_step_limit != -1:
            subplot_axs.axvline(x=env_step_limit, linewidth=2, color='k')
            subplot_axs.set_xlim(0, env_step_limit)
        env_reports[task] = {}
        for repo in draw_repo_names:
            exp_repo = data_repo[repo]
            env_stats = print_hyper_tests(subplot_axs, exp_repo[0], task, exp_repo[1],
                              num_seeds = num_seeds, title=title, x_title=x_title, limit=limit, env_step_limit=env_step_limit, backward_smooth_window=backward_smooth_window)
            env_reports[task][repo] = env_stats
    return env_reports

def make_report(env_reports, valid_repos):
    env_report_list = []
    title = 'env_name, {}\n'.format(','.join(valid_repos))
    win_cnt = [0 for _ in range(len(valid_repos))]
    for task, report in env_reports.items():
        env_results = []
        winner_score, winner_idx = 0,0
        for i, repo in enumerate(valid_repos):
            seed_cnt = report[repo]['seed_cnt']
            if seed_cnt > 1:
                report_str = "{}~{}".format(report[repo]['average'], report[repo]['std'], seed_cnt)
                score = float(report[repo]['average'])
            elif seed_cnt == 0:
                report_str = '---'
                score = -999999
            else:
                report_str = str(round(float(report[repo]['average']), 4))
                score = float(report[repo]['average'])
            if score > winner_score:
                winner_score = score
                winner_idx = i
            env_results.append(report_str)
        win_cnt[winner_idx] += 1
        env_report_list.append('{},{}\n'.format(task, ','.join(env_results)))

    env_report_list.append('wins,{}\n'.format(','.join(map(lambda x: str(x), win_cnt))))
    text_file = open("/home/simon0xzx/research/berkely_research/garage/data/reports/report.csv", "w")
    text_file.write(title)
    for env_report in env_report_list:
        text_file.write(env_report)
    text_file.close()


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

    valid_repo_list = ['rl2_ppo_2', 'rl2_ppo_manual_reduce', 'pearl', 'curl_wasserstein_2', 'curl_step_2x_reduce', 'curl_step_4x_reduce', 'curl_wasserstein_10x_reduce']
    row, col = 5, 5
    fig, axs = plt.subplots(row, col)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.96,
                        wspace=0.20, hspace=0.40)
    repo_stats = metaworld_ml1_graph(axs, full_suit_task_lists, valid_repo_list, num_seeds=5, row=row, col=col, limit=100, backward_smooth_window=10, env_step_limit=1000000)

    make_report(repo_stats, valid_repo_list)
    plt.show()

def process():
    stuff = ""

    for line in stuff.split('\n'):
        env = line.split(':')[0]
        statement = "sudo tmux kill-session -t {}".format(env)
        print(statement)

if __name__ == '__main__':
#     # plot_full_suits()
#     # plot_selected_suits()
    varify()
    # process()
    # simple_draw()
