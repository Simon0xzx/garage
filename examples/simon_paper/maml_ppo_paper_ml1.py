#!/usr/bin/env python3
"""This is an example to train MAML-VPG on HalfCheetahDirEnv environment."""
# pylint: disable=no-value-for-parameter
import click
import torch
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalRunner, MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=100)
@click.option('--rollouts_per_task', default=10)
@click.option('--meta_batch_size', default=20)
@click.option('--name', default='push-v1')
@click.option('--prefix', default='maml_ppo_suit')
@wrap_experiment
def maml_ppo_paper_ml1(ctxt, seed, epochs, rollouts_per_task,
                              meta_batch_size,
                              name='push-v1',
                              prefix='maml_trpo_ml1'
                              ):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        rollouts_per_task (int): Number of rollouts per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)
    print("Running experiences on {}/{}".format(prefix, name))

    test_env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(mwb.ML1.get_test_tasks(name))))

    env = GarageEnv( normalize(mwb.ML1.get_train_tasks(name)))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(400, 400, 400),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(400, 400, 400),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    max_path_length = 200
    meta_evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)

    runner = LocalRunner(ctxt)
    algo = MAMLPPO(env=env,
                   policy=policy,
                   value_function=value_function,
                   max_path_length=max_path_length,
                   meta_batch_size=meta_batch_size,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.05,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)

    runner.setup(algo, env)
    runner.train(n_epochs=epochs,
                 batch_size=rollouts_per_task * max_path_length)

if __name__ == '__main__':
    maml_ppo_paper_ml1()
