#!/usr/bin/env python3
"""PEARL MLSP example."""

import click
import metaworld.benchmarks as mwb
from garage.experiment import Snapshotter
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.experiment import MetaEvaluator
from garage.torch.algos.pearl import PEARLWorker


@click.command()
@click.option('--num_epochs', default=1000)
@click.option('--num_train_tasks', default=1)
@click.option('--num_test_tasks', default=1)
@click.option('--encoder_hidden_size', default=200)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=4000)
@click.option('--num_initial_steps', default=4000)
@click.option('--num_steps_prior', default=750)
@click.option('--num_extra_rl_steps_posterior', default=750)
@click.option('--batch_size', default=256)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@click.option('--max_path_length', default=150)
@click.option('--gpu_id', default=0)
@wrap_experiment
def pearl_metaworld_mlsp_adapt_new(ctxt=None,
                         seed=1,
                         num_epochs=1000,
                         num_train_tasks=1,
                         num_test_tasks=1,
                         latent_size=7,
                         encoder_hidden_size=200,
                         net_size=300,
                         meta_batch_size=32,
                         num_steps_per_epoch=4000,
                         num_initial_steps=4000,
                         num_tasks_sample=15,
                         num_steps_prior=750,
                         num_extra_rl_steps_posterior=750,
                         batch_size=512,
                         embedding_batch_size=128,
                         embedding_mini_batch_size=128,
                         max_path_length=150,
                         reward_scale=10.,
                         gpu_id=0,
                         use_gpu=True):
    """Train PEARL with ML10 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_path_length (int): Maximum path length.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    set_seed(seed)
    # create multi-task environment and sample tasks
    ml_test_envs = [
        GarageEnv(normalize(mwb.MLSP.from_task('button-press-wall-v1')))
    ]

    test_env_sampler = EnvPoolSampler(ml_test_envs)
    test_env_sampler.grow_pool(num_test_tasks)
    env = test_env_sampler.sample(num_train_tasks)
    runner = LocalRunner(ctxt)
    worker_args = dict(deterministic=True, accum_context=True)
    evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                    max_path_length=max_path_length,
                                    worker_class=PEARLWorker,
                                    worker_args=worker_args,
                                    n_test_tasks=num_test_tasks)


    exp_path = '/home/simon0xzx/research/berkely_research/garage/data/local/experiment'
    base_agent_path = '{}/pearl_metaworld_mlsp'.format(exp_path)
    snapshotter = Snapshotter()
    snapshot = snapshotter.load(base_agent_path)
    pearl = snapshot['algo']
    pearl.update_env(env, evaluator, 1, 1)

    set_gpu_mode(use_gpu, gpu_id=gpu_id)
    if use_gpu:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=PEARLWorker)
    expert_traj_dir = '/home/simon0xzx/research/berkely_research/garage/data/expert/metaworld_mlsp_button_press_wall'
    print(
        "==================================\nAdapting\n==================================")
    runner.adapt_policy(n_epochs=50, expert_traj_path=expert_traj_dir, batch_size=batch_size)

    print(
        "==================================\nSelf Training\n==================================")
    runner.train(n_epochs=num_epochs, batch_size=batch_size)

if __name__ == '__main__':
    pearl_metaworld_mlsp_adapt_new()
