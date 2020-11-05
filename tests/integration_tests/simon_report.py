from tests.integration_tests.simon_display_helper import print_hyper_tests
import matplotlib.pyplot as plt

# Name -> (exp_root_path, labels)
data_repo = {
'pearl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_pearl_result']),
'curl_old_ml1_result': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/ml1_results', ['old_curl_result']),
'large_batch': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_test2', ['curl_large_batch_size']),
'new_label': ('/home/simon0xzx/research/berkely_research/garage/data/local/curl_new_label', ['curl_new_label_batch_16', 'curl_new_label_batch_64']),
'namazu_new_label': ('/home/simon0xzx/research/berkely_research/garage/data/namazu/curl_new_label', ['curl_new_label_batch_32']),
}


def ml1_hype_tune_detail():
    task_lists = ['faucet-close-v1', 'soccer-v1', 'plate-slide-side-v1', 'stick-push-v1']
    row, col, limit = 4, 1, 100
    fig, axs = plt.subplots(row, col)
    title = 'MetaTest/Average/AverageReturn'
    x_title = 'TotalEnvSteps'

    for i, task in enumerate(task_lists):
        row_cnt = int(i / col)
        col_cnt = i % col if i % col >= 0 else (i % col) + col
        subplot_axs = axs[row_cnt]
        subplot_axs.set_title('CURL ML1 {}'.format(task))
        subplot_axs.set_xlabel('Total Env Steps')
        subplot_axs.set_ylabel('Total Test Return')

        # Old ML1 Results
        pearl_old_ml1_repo = data_repo['pearl_old_ml1_result']
        # print_hyper_tests(subplot_axs, pearl_old_ml1_repo[0], 'pearl-' + task,
        #                   pearl_old_ml1_repo[1], title=title, x_title=x_title, limit=limit)
        # curl_old_ml1_repo = data_repo['curl_old_ml1_result']
        # print_hyper_tests(subplot_axs, curl_old_ml1_repo[0], 'curl-' + task,
        #                   curl_old_ml1_repo[1], title=title, x_title=x_title, limit=limit)
        # Larger batch size
        large_batch_repo = data_repo['large_batch']
        print_hyper_tests(subplot_axs, large_batch_repo[0], task,
                          large_batch_repo[1], title=title, x_title=x_title, limit=limit)
        # batch on new labels
        local_new_label_repo = data_repo['new_label']
        print_hyper_tests(subplot_axs, local_new_label_repo[0], task,
                          local_new_label_repo[1], title=title, x_title=x_title, limit=-1)
        # batch on new labels on namazu
        namazu_new_label_repo = data_repo['namazu_new_label']
        print_hyper_tests(subplot_axs, namazu_new_label_repo[0], task,
                          namazu_new_label_repo[1], title=title, x_title=x_title, limit=-1)
    plt.show()


if __name__ == '__main__':
    ml1_hype_tune_detail()
