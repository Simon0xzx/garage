#!/usr/bin/env python3
"""Example script to run RL2 in ML1."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

# yapf: enable


@click.command()
@click.option('--seed', default=1)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=100)
@click.option('--episode_per_task', default=10)
@click.option('--gpu_id', default=0)
@click.option('--name', default='push-v1')
@click.option('--prefix', default='rl2_ppo_env')
@wrap_experiment
def rl2_ppo_paper_ml1(ctxt, seed, meta_batch_size,
                      n_epochs, episode_per_task,
                      gpu_id=0,
                      name='push-v1',
                      prefix='rl2_ppo_suit_2'):
    """Train RL2 PPO with ML1 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    print("Running Experiments on GPU: {}".format(gpu_id))
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

    ml1 = metaworld.ML1(name)
    task_sampler = MetaWorldTaskSampler(ml1, 'train', lambda env, _: RL2Env(env))
    env = task_sampler.sample(1)[0]()
    test_task_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                       env=MetaWorldSetTaskEnv(ml1, 'test'),
                                       wrapper=lambda env, _: RL2Env(env))
    env_spec = env.spec

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_task_sampler,
                                       n_test_tasks=10)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        envs = task_sampler.sample(meta_batch_size)
        sampler = LocalSampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      task_sampler=task_sampler,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32, max_optimization_epochs=10),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=0.02,
                      center_adv=False,
                      meta_evaluator=meta_evaluator,
                      episodes_per_trial=episode_per_task)

        trainer.setup(algo, envs)

        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task * env_spec.max_episode_length * meta_batch_size)

if __name__ == '__main__':
    rl2_ppo_paper_ml1()
