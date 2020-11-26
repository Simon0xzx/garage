#!/usr/bin/env python3
"""Example script to run RL2 in ML1."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment.task_sampler import SetTaskSampler
from garage.experiment import LocalTFRunner, task_sampler, MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy


@click.command()
@click.option('--seed', default=1)
@click.option('--max_path_length', default=150)
@click.option('--meta_batch_size', default=40)
@click.option('--n_epochs', default=100)
@click.option('--episode_per_task', default=10)
@click.option('--name', default='push-v1')
@click.option('--prefix', default='rl2_suit')
@wrap_experiment
def rl2_ppo_paper_ml1(ctxt, seed, max_path_length, meta_batch_size,
                               n_epochs, episode_per_task, name='push-v1',
                               prefix='maml_trpo_ml1'):
    """Train PPO with ML1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_path_length (int): Maximum length of a single rollout.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            env=mwb.ML1.get_train_tasks(name)))

        env_spec = RL2Env(env=mwb.ML1.get_train_tasks(name)).spec
        policy = GaussianGRUPolicy(name=name,
                                   hidden_dim=400,
                                   env_spec=env_spec,
                                   init_std=2.0,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)
        test_env_sampler = SetTaskSampler(lambda: RL2Env(env=mwb.ML1.get_train_tasks(name)))
        meta_evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                       max_path_length=max_path_length,
                                       n_test_tasks=1,
                                       n_test_rollouts=10)

        algo = RL2(rl2_max_path_length=max_path_length,
                   meta_batch_size=meta_batch_size,
                   task_sampler=tasks,
                   meta_evaluator=meta_evaluator,
                   n_epochs_per_eval = 1,
                   env_spec=env_spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=max_path_length * episode_per_task
                   )
        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker,
                     worker_args=dict(n_paths_per_trial=episode_per_task))

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)

if __name__ == '__main__':
    rl2_ppo_paper_ml1()
