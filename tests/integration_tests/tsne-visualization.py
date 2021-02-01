import metaworld
import numpy as np
import torch
import matplotlib.pyplot as plt

from garage import StepType
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage import rollout
from garage.experiment import Snapshotter
from garage.torch import set_gpu_mode
from garage.replay_buffer import PathBuffer
from garage.experiment.task_sampler import SetTaskSampler
from garage.torch import global_device
from garage import wrap_experiment
from garage.trainer import Trainer

def augment_path(path, batch_size, in_sequence=False):
    path_len = path['actions'].shape[0]
    augmented_path = {}
    if in_sequence:
        if batch_size > path_len:
            raise Exception(
                'Embedding_batch size cannot be longer than path length {} > {}'.format(
                    batch_size, path_len))
        seq_begin = np.random.randint(0, path_len - batch_size)
        augmented_path['observations'] = path['observations'][
                                         seq_begin:seq_begin + batch_size]
        augmented_path['actions'] = path['actions'][
                                    seq_begin:seq_begin + batch_size]
        augmented_path['rewards'] = path['rewards'][
                                    seq_begin:seq_begin + batch_size]
        augmented_path['next_observations'] = path['next_observations'][
                                              seq_begin:seq_begin + batch_size]
    else:
        seq_idx = np.random.choice(path_len, batch_size)
        augmented_path['observations'] = path['observations'][seq_idx]
        augmented_path['actions'] = path['actions'][seq_idx]
        augmented_path['rewards'] = path['rewards'][seq_idx]
        augmented_path['next_observations'] = path['next_observations'][
            seq_idx]

    return augmented_path

def tsne_data_collection(model_dir, sample_env_size = 10, num_path_rollout=1, replay_buffer_size=100000, sample_from_dist = False):
    snapshotter = Snapshotter()
    data = snapshotter.load(model_dir)
    env_name = model_dir.split('/')[-1]
    path_label = '-'.join(model_dir.split('/')[-2:])
    curl = data['algo']
    contrastive_encoder = data['algo'].networks[0]

    origin_env = data['env']
    set_gpu_mode(True, gpu_id=1)
    curl.to()
    policy = data['algo'].policy

    ml1 = metaworld.ML1(env_name)
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))
    env_set = env_sampler.sample_env(sample_env_size)
    replay_buffers = PathBuffer(replay_buffer_size)
    adapt_replay_buffers = PathBuffer(replay_buffer_size)

    print("Task Exploration...")
    for task_idx, env in enumerate(env_set):
        print('building replay buffers for task: {}'.format(task_idx))
        for _ in range(1):
            origin_env.set_task(env)
            path = rollout(origin_env, policy, max_episode_length=10000, animated=False)
            terminations = np.array(
                [step_type == StepType.TERMINAL for step_type in
                 path['dones']]).reshape(-1, 1)
            p = {
                'observations': path['observations'],
                'next_observations': path['next_observations'],
                'actions': path['actions'],
                'rewards': path['rewards'].reshape(-1, 1),
                'dones': terminations
            }
            adapt_replay_buffers.add_path(p)

    print("Actual Task Rollout...")
    for task_idx, env in enumerate(env_set):
        print('building replay buffers for task: {}'.format(task_idx))
        for _ in range(10):
            print("Sampling path {}/{}".format(_+1, num_path_rollout))
            origin_env.set_task(env)

            path = adapt_replay_buffers.sample_path()
            batch = augment_path(path, 64, in_sequence=True)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            no = batch['next_observations']
            context = np.hstack((np.hstack((o, a)), r))
            context = np.hstack((context, no))  # Optional
            context_tensor = torch.as_tensor(context, device=global_device()).float()
            policy.infer_posterior(context_tensor)
            path = rollout(origin_env, policy, max_episode_length=10000, animated=False)
            terminations = np.array(
                [step_type == StepType.TERMINAL for step_type in
                 path['dones']]).reshape(-1, 1)
            p = {
                'observations': path['observations'],
                'next_observations': path['next_observations'],
                'actions': path['actions'],
                'rewards': path['rewards'].reshape(-1, 1),
                'dones': terminations
            }
            replay_buffers.add_path(p)

    encoder_sample, label = [], []

    for task_idx, env in enumerate(env_set):
        path = replay_buffers.sample_path()
        for _ in range(10):
            batch = augment_path(path, 64, in_sequence=True)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            no = batch['next_observations']
            context = np.hstack((np.hstack((o, a)), r))
            context = np.hstack((context, no)) # Optional
            context_tensor = torch.as_tensor(context, device=global_device()).float()
            policy.infer_posterior(context_tensor)
            encoder_sample.extend(policy.z)
            label.extend([task_idx] * len(policy.z))

    encoder_sample = np.array(encoder_sample)
    label = np.array(label)
    assert encoder_sample.shape[0] == label.shape[0]
    suffix = "dist" if sample_from_dist else "data"
    np.savetxt('/home/simon0xzx/research/berkely_research/garage/data/tsne_data/sample2/{}_{}_task_0.txt'.format(path_label, suffix), encoder_sample)
    np.savetxt('/home/simon0xzx/research/berkely_research/garage/data/tsne_data/sample2/{}_label_task_0.txt'.format(path_label), label)
    print('DONE')

def plot_graph():
    tcl_pearl_x_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/data_plot/tcl-pearl-window-open-v1_data_X2.txt"
    tcl_pearl_y_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/data_plot/tcl-pearl-window-open-v1_data_Y2.txt"
    tcl_pearl_label_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/sample/tcl-pearl-window-open-v1_label.txt"
    tcl_pearl_x = np.loadtxt(tcl_pearl_x_file)
    tcl_pearl_y = np.loadtxt(tcl_pearl_y_file)
    tcl_pearl_labels = np.loadtxt(tcl_pearl_label_file).tolist()

    pearl_x_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/data_plot/pearl-window-open-v1_data_X.txt"
    pearl_y_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/data_plot/pearl-window-open-v1_data_Y.txt"
    pearl_label_file = "/home/simon0xzx/research/berkely_research/garage/data/tsne_data/sample/pearl-window-open-v1_label.txt"
    pearl_x = np.loadtxt(pearl_x_file)
    pearl_y = np.loadtxt(pearl_y_file)
    pearl_labels = np.loadtxt(pearl_label_file).tolist()

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("T-SNE Encoder Space of TCL-Pearl")
    axs[0].scatter(tcl_pearl_x, tcl_pearl_y, 10, tcl_pearl_labels)
    axs[1].set_title("T-SNE Encoder Space of Pearl")
    axs[1].scatter(pearl_x, pearl_y, 10, pearl_labels)

    plt.show()

"""
This function generates the toy data for varify
"""
def generate_toy_data():
    means = np.array([[1,1,1,1]] * 200 + [[0,0,0,0]] * 200+ [[2,2,2,2]] * 200)
    vars = np.array([[1,1,1,1]] * 200 + [[0.1,0.1,0.1,0.1]] * 200 + [[0.2,0.2,0.2,0.2]] * 200)
    label = np.array([1]* 200 + [0] * 200 + [2]* 200)

    means = torch.as_tensor(means, dtype=torch.float)
    vars = torch.as_tensor(vars, dtype=torch.float)
    posteriors = [
        torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(means, vars)
    ]
    z = [d.rsample().cpu().detach().numpy() for d in posteriors]

    np.savetxt('/home/simon0xzx/research/berkely_research/tsne-pytorch/data/toy_x.txt', z)
    np.savetxt('/home/simon0xzx/research/berkely_research/tsne-pytorch/data/toy_label.txt', label)
    print('DONE')


def generate_tsne_sample():
    model_dir = '/home/simon0xzx/research/berkely_research/garage/data/tsne_data/pearl/push-v1'
    # model_dir = '/home/simon0xzx/research/berkely_research/garage/data/tsne_data/tcl-pearl/push-v1'
    tsne_data_collection(model_dir)





if __name__ == "__main__":
    generate_tsne_sample()
    # plot_graph()

