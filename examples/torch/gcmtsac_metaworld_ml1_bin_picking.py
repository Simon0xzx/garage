#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on ML1.

This experiment shows how MTSAC adapts to 50 environents of the same type
but each environment has a goal variation.

https://arxiv.org/pdf/1910.10897.pdf
"""
import pickle

import click
import metaworld.benchmarks as mwb
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv, MultiEnvWrapper, normalize
from garage.envs.multi_env_wrapper import uniform_random_strategy, round_robin_strategy
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import GCMTSAC
from garage.torch.algos.gcmtsac import GCMTSACWorker
from garage.torch.policies import GoalConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--gpu_id', 'gpu_id', type=int, default=0)
@wrap_experiment(snapshot_mode='none')
def gcmtsac_metaworld_ml1_bin_picking(ctxt=None,
                                      seed=1,
                                      net_size=400,
                                      goal_dim=3,
                                      max_path_length=150,
                                      num_train_tasks=50,
                                      gpu_id=None):
    """Train MTSAC with the ML1 pick-place-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(ctxt)
    train_envs = []
    test_envs = []
    env_names = []
    for i in range(num_train_tasks):
        train_env = GarageEnv(normalize(mwb.ML1.get_train_tasks('bin-picking-v1'),
                      normalize_reward=True))
        test_env = pickle.loads(pickle.dumps(train_env))
        env_names.append('bin-picking-{}'.format(i))
        train_envs.append(train_env)
        test_envs.append(test_env)
    ml1_train_envs = MultiEnvWrapper(train_envs,
                                     sample_strategy=uniform_random_strategy,
                                     env_names=env_names)
    ml1_test_envs = MultiEnvWrapper(test_envs,
                                    sample_strategy=round_robin_strategy,
                                    env_names=env_names)
    augmented_env = GCMTSAC.augment_env_spec(ml1_train_envs, goal_dim)
    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    policy = GoalConditionedPolicy(
        latent_dim=goal_dim,
        policy=inner_policy
    )

    qf1 = ContinuousMLPQFunction(env_spec=augmented_env,
                                 hidden_sizes=[net_size, net_size, net_size],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=augmented_env,
                                 hidden_sizes=[net_size, net_size, net_size],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e5), )

    timesteps = 10000000
    batch_size = int(150 * ml1_train_envs.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    gcmtsac = GCMTSAC(policy=policy,
                    qf1=qf1,
                    qf2=qf2,
                    gradient_steps_per_itr=150,
                    max_path_length=max_path_length,
                    eval_env=ml1_test_envs,
                    goal_dim=goal_dim,
                    env_spec=augmented_env,
                    num_tasks=num_train_tasks,
                    steps_per_epoch=epoch_cycles,
                    replay_buffer=replay_buffer,
                    min_buffer_size=1500,
                    target_update_tau=5e-3,
                    discount=0.99,
                    buffer_batch_size=1280)

    set_gpu_mode(True, gpu_id=gpu_id)
    gcmtsac.to()

    runner.setup(algo=gcmtsac,
                 env=ml1_train_envs,
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=GCMTSACWorker)
    runner.train(n_epochs=epochs, batch_size=batch_size)

if __name__ == '__main__':
    gcmtsac_metaworld_ml1_bin_picking()
