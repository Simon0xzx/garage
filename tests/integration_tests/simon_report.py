from tests.integration_tests.simon_display_helper import print_hyper_tests
import matplotlib.pyplot as plt

# Name -> (exp_root_path, labels)
data_repo = {
'pearl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_pearl_result']),
'curl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_curl_result']),
'large_batch': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_test2', ['curl_large_batch_size']),
'new_label': ('/home/simon0xzx/research/berkely_research/garage/data/local/curl_new_label', ['curl_new_label_batch_16', 'curl_new_label_batch_64']),
'namazu_new_label': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_new_label', ['curl_new_label_batch_32']),
# full suit results
'old_curl_suit': ('/home/simon0xzx/research/berkely_research/garage/data/full_suit_results/old_curl_ml1_result_suit', ['old_curl']),
'pearl_suit': ('/home/simon0xzx/research/berkely_research/garage/data/full_suit_results/pearl_ml1_result_suit', ['old_pearl']),
'new_label_batch_16_suit': ('/home/simon0xzx/research/berkely_research/garage/data/full_suit_results/curl_index_label_meta_batch_16', ['curl_new_label_batch_16'])

}


def ml1_hype_tune_detail():
    # task_lists = ['faucet-close-v1', 'soccer-v1', 'plate-slide-side-v1', 'stick-push-v1']
    task_lists = ['button-press-topdown-v1', 'coffee-push-v1',
                 'dissassemble-v1',
                 'faucet-close-v1', 'faucet-open-v1',
                 'hammer-v1', 'handle-press-v1', 'handle-pull-side-v1',
                 'lever-pull-v1', 'plate-slide-back-v1', 'plate-slide-side-v1',
                 'soccer-v1', 'stick-pull-v1', 'stick-push-v1',
                 'button-press-topdown-wall-v1', 'peg-insert-side-v1',
                 'push-wall-v1', 'button-press-v1', 'coffee-pull-v1',
                 'window-close-v1', 'door-open-v1', 'box-close-v1',
                 'door-unlock-v1', 'drawer-open-v1']
    row, col, limit = 4, 6, 100
    fig, axs = plt.subplots(row, col)
    title = 'MetaTest/Average/AverageReturn'
    x_title = 'TotalEnvSteps'

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
        subplot_axs.set_xlabel('Total Env Steps')
        subplot_axs.set_ylabel('Total Test Return')

        # Old ML1 Results
        pearl_ml1_repo = data_repo['pearl_suit']
        print_hyper_tests(subplot_axs, pearl_ml1_repo[0], task,
                          pearl_ml1_repo[1], title=title, x_title=x_title, limit=limit)

        old_curl_ml1_repo = data_repo['old_curl_suit']
        print_hyper_tests(subplot_axs, old_curl_ml1_repo[0], task,
                          old_curl_ml1_repo[1], title=title, x_title=x_title, limit=limit)
        # batch on new labels
        new_label_batch_16_repo = data_repo['new_label_batch_16_suit']
        print_hyper_tests(subplot_axs, new_label_batch_16_repo[0], task,
                          new_label_batch_16_repo[1], title=title, x_title=x_title, limit=-1)
    plt.show()


if __name__ == '__main__':
    ml1_hype_tune_detail()
