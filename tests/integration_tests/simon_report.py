from tests.integration_tests.simon_display_helper import print_hyper_tests, read_csv_file
import matplotlib.pyplot as plt

# Name -> (exp_root_path, labels)
def get_data_repos():
    data_repo = {
    'pearl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_pearl_result']),
    'curl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_curl_result']),
    'large_batch': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_test2', ['curl_large_batch_size']),
    'new_label': ('/home/simon0xzx/research/berkely_research/garage/data/local/curl_new_label', ['curl_new_label_batch_16', 'curl_new_label_batch_64']),
    'namazu_new_label': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_new_label', ['curl_new_label_batch_32']),
    # full suit results
    'old_curl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/old_curl_ml1_result_suit', ['old_curl']),
    'pearl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/pearl_ml1_result_suit', ['old_pearl']),
    'curl_labeled_b16': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_labeled_b16', ['curl_labeled_b16']),
    'updated_curl': ('/home/simon0xzx/research/berkely_research/garage/data/result_suits/curl_normal', ['updated_curl']),
    'classifier_encoder':('/home/simon0xzx/research/berkely_research/garage/data/local/classifier_suit', ['classifier_encoder'])
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



def metaworld_ml1_graph(axs, task_lists, draw_repo_names, row=4, col=6, limit=100, title='MetaTest/Average/AverageReturn', x_title='TotalEnvSteps'):

    for i, task in enumerate(task_lists):
        row_cnt = int(i / col)
        col_cnt = i % col if i % col >= 0 else (i % col) + col
        if col == 1:
            subplot_axs = axs[row_cnt]
        elif row == 1:
            subplot_axs = axs[col_cnt]
        else:
            subplot_axs = axs[row_cnt][col_cnt]
        subplot_axs.set_title('CURL ML1 {}'.format(task))
        subplot_axs.set_xlabel(x_title)
        subplot_axs.set_ylabel(title.split('/')[-1])

        for repo in draw_repo_names:
            exp_repo = data_repo[repo]
            print_hyper_tests(subplot_axs, exp_repo[0], task,
                              exp_repo[1], title=title, x_title=x_title, limit=limit)
    plt.show()


def plot_full_suits():
    sampled_task_lists = ['faucet-close-v1', 'soccer-v1', 'plate-slide-side-v1', 'stick-push-v1']
    full_suit_task_lists = ['button-press-topdown-v1', 'coffee-push-v1',
                  'dissassemble-v1', 'faucet-close-v1', 'faucet-open-v1',
                  'hammer-v1', 'handle-press-v1', 'handle-pull-side-v1',
                  'lever-pull-v1', 'plate-slide-back-v1', 'plate-slide-side-v1',
                  'soccer-v1', 'stick-pull-v1', 'stick-push-v1',
                  'button-press-topdown-wall-v1', 'peg-insert-side-v1',
                  'push-wall-v1', 'button-press-v1', 'coffee-pull-v1',
                  'window-close-v1', 'door-open-v1', 'box-close-v1',
                  'door-unlock-v1', 'drawer-open-v1']
    row, col = 2,2
    fig, axs = plt.subplots(row, col)
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.96,
                        wspace=0.25, hspace=0.30)

    # metaworld_ml1_graph(axs, full_suit_task_lists, ['old_curl', 'curl_labeled_b16', 'updated_curl', 'pearl'], row=6, col=4)

    # metaworld_ml1_graph(axs, full_suit_task_lists, ['old_curl', 'updated_curl', 'curl_labeled_b16'], row=row, col=col)
    metaworld_ml1_graph(axs, sampled_task_lists, ['curl_labeled_b16', 'classifier_encoder'], row=2, col=2)
    plt.show()



if __name__ == '__main__':
    # plot_full_suits()
    get_metaworld_task_list()
