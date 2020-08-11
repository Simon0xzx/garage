#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import click
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import metaworld.benchmarks as mwb
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@wrap_experiment(snapshot_mode='none')
@click.option('--gpu_id', default=0)
def sac_metaworld_ml1_lever_pull(ctxt=None, seed=1, gpu_id=0):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(mwb.ML1.get_train_tasks('shelf-place-v1')))

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=1000,
              max_path_length=150,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=1.,
              num_tasks=1)

    set_gpu_mode(True, gpu_id)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    runner.train(n_epochs=500, batch_size=1000)


s = np.random.randint(0, 1000)
sac_metaworld_ml1_lever_pull(seed=521)
