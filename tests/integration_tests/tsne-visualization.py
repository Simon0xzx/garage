import metaworld
import numpy as np
import torch

from garage import StepType
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage import rollout
from garage.experiment import Snapshotter
from garage.torch import set_gpu_mode
from garage.replay_buffer import PathBuffer
from garage.experiment.task_sampler import SetTaskSampler
from garage.torch import global_device

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
    else:
        seq_idx = np.random.choice(path_len, batch_size)
        augmented_path['observations'] = path['observations'][seq_idx]
        augmented_path['actions'] = path['actions'][seq_idx]
        augmented_path['rewards'] = path['rewards'][seq_idx]

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
    replay_buffers = {
        i: PathBuffer(replay_buffer_size)
        for i in range(sample_env_size)
    }

    for task_idx, env in enumerate(env_set):
        print('building replay buffers for task: {}'.format(task_idx))
        for _ in range(num_path_rollout):
            print("Sampling path {}/{}".format(_+1, num_path_rollout))
            origin_env.set_task(env)
            path = rollout(origin_env, policy, max_episode_length=10000, animated=False)
            terminations = np.array(
                [step_type == StepType.TERMINAL for step_type in
                 path['dones']]).reshape(-1, 1)
            p = {
                'observations': path['observations'],
                'actions': path['actions'],
                'rewards': path['rewards'].reshape(-1, 1),
                'dones': terminations
            }
            replay_buffers[task_idx].add_path(p)

    encoder_sample, label = [], []

    for task_idx, env in enumerate(env_set):
        path = replay_buffers[task_idx].sample_path()
        for _ in range(10):
            batch = augment_path(path, 64, in_sequence=True)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            context_tensor = torch.as_tensor(context, device=global_device()).float()
            task_rep = contrastive_encoder(context_tensor)

            posteriors = [
                torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                    torch.unbind(task_rep[:, :7]), torch.unbind(task_rep[:, 7:].pow(2)))
            ]
            if sample_from_dist:
                z = [d.rsample().cpu().detach().numpy() for d in posteriors]
            else:
                z = task_rep[:, :7].cpu().detach().numpy()
            encoder_sample.extend(z)
            label.extend([task_idx] * len(z))

    encoder_sample = np.array(encoder_sample)
    label = np.array(label)
    np.savetxt('/home/simon0xzx/research/berkely_research/tsne-pytorch/data/{}_data.txt'.format(path_label), encoder_sample)
    np.savetxt('/home/simon0xzx/research/berkely_research/tsne-pytorch/data/{}_label.txt'.format(path_label), label)
    print('DONE')

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

    np.savetxt( '/home/simon0xzx/research/berkely_research/tsne-pytorch/data/toy_x.txt', z)
    np.savetxt( '/home/simon0xzx/research/berkely_research/tsne-pytorch/data/toy_label.txt', label)
    print('DONE')


if __name__ == "__main__":
    tcl_pearl_dir = '/home/simon0xzx/research/berkely_research/garage/data/result_suits/tcl_pearl_new_env_no_reduce/window-open-v1_2'
    pearl_dir = '/home/simon0xzx/research/berkely_research/garage/data/result_suits/pearl_new_env_no_reduce/window-open-v1'
    tsne_data_collection(tcl_pearl_dir)
    tsne_data_collection(pearl_dir)

