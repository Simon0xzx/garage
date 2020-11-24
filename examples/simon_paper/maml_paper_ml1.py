#!/usr/bin/env python3
"""PEARL ML1 example."""
import click
import metaworld.benchmarks as mwb
import torch

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalRunner, MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.torch.algos import MAML
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@click.command()
@click.option('--seed', default=1)
@click.option('--num_epochs', default=100)
@click.option('--rollouts_per_task', default=10)
@click.option('--meta_batch_size', default=20)
@click.option('--net_size', default=400)
@click.option('--max_path_length', default=200)
@click.option('--num_test_tasks', default=10)
@click.option('--gpu_id', default=0)
@click.option('--name', default='push-v1')
@click.option('--prefix', default='maml_trpo_suit_2')
@wrap_experiment
def maml_trpo_paper_ml1(ctxt=None,
                             seed=1,
                             num_epochs=100,
                             meta_batch_size=20,
                             rollouts_per_task=10,
                             net_size=400,
                             max_path_length=200,
                             num_test_tasks = 10,
                             gpu_id = 0,
                             name='push-v1',
                             prefix='maml_trpo_ml1',
                             use_gpu=True):
    """Train PEARL with ML1 environments.

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
    print("Running experiences on {}/{}".format(prefix, name))
    env = GarageEnv(normalize(mwb.ML1.get_train_tasks(name),
                              expected_action_scale=10.))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(net_size, net_size, net_size),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=[net_size, net_size, net_size],
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    test_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(mwb.ML1.get_test_tasks('push-v1'))))

    meta_evaluator = MetaEvaluator(test_task_sampler=test_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=num_test_tasks)


    runner = LocalRunner(ctxt)
    algo = MAML(env=env,
                    policy=policy,
                    value_function=value_function,
                    max_path_length=max_path_length,
                    meta_batch_size=meta_batch_size,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=5,
                    meta_evaluator=meta_evaluator)

    # set_gpu_mode(use_gpu, gpu_id=gpu_id)
    # if use_gpu:
    #     algo.to()

    runner.setup(algo, env)
    runner.train(n_epochs=num_epochs,
                 batch_size=rollouts_per_task * max_path_length)

if __name__ == '__main__':
    maml_trpo_paper_ml1()
