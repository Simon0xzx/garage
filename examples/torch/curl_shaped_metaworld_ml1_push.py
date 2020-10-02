#!/usr/bin/env python3
"""PEARL ML1 example."""
import click
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import CURLShaped
from garage.torch.algos.curl_shaped import CURLShapedWorker
from garage.torch.embeddings import ContrastiveEncoder
from garage.torch.policies import CurlShapedPolicy
from garage.torch.policies import TanhGaussianContextEmphasizedPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--num_epochs', default=500)
@click.option('--seed', default=1)
@click.option('--num_train_tasks', default=50)
@click.option('--num_test_tasks', default=10)
@click.option('--encoder_hidden_size', default=300)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=4000)
@click.option('--num_initial_steps', default=4000)
@click.option('--num_steps_prior', default=750)
@click.option('--num_extra_rl_steps_posterior', default=750)
@click.option('--batch_size', default=512)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@click.option('--max_path_length', default=200)
@click.option('--context_step_len', default=2)
@click.option('--gpu_id', default=0)
@wrap_experiment
def curl_origin_shaped_metaworld_ml1_push(ctxt=None,
                             seed=1,
                             num_epochs=1000,
                             num_train_tasks=100,
                             num_test_tasks=10,
                             latent_size=10,
                             encoder_hidden_size=300,
                             net_size=300,
                             meta_batch_size=32,
                             num_steps_per_epoch=4000,
                             num_initial_steps=4000,
                             num_tasks_sample=15,
                             num_steps_prior=750,
                             num_extra_rl_steps_posterior=750,
                             batch_size=512,
                             embedding_batch_size=64,
                             embedding_mini_batch_size=64,
                             max_path_length=150,
                             reward_scale=10.,
                             context_step_len=1,
                             gpu_id = 0,
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
            context encoder.s
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
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size, encoder_hidden_size)
    # create multi-task environment and sample tasks
    env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(mwb.ML1.get_train_tasks('push-v1'))))
    env = env_sampler.sample(num_train_tasks)

    test_env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(mwb.ML1.get_test_tasks('push-v1'))))
    runner = LocalRunner(ctxt)

    # instantiate networks
    augmented_env = CURLShaped.augment_env_spec(env[0](), latent_size)
    qf_1 = ContinuousMLPQFunction(env_spec=augmented_env,
                                  hidden_sizes=[net_size, net_size, net_size])

    qf_2 = ContinuousMLPQFunction(env_spec=augmented_env,
                                  hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianContextEmphasizedPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size],
        latent_sizes=latent_size)



    curl = CURLShaped(
        env=env,
        policy_class=CurlShapedPolicy,
        encoder_class=ContrastiveEncoder,
        inner_policy=inner_policy,
        qf1=qf_1,
        qf2=qf_2,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length,
        reward_scale=reward_scale,
        replay_buffer_size=100000,
        context_step_len=context_step_len
    )

    set_gpu_mode(use_gpu, gpu_id=gpu_id)
    if use_gpu:
        curl.to()

    runner.setup(algo=curl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=CURLShapedWorker)

    runner.train(n_epochs=num_epochs, batch_size=batch_size)

if __name__ == '__main__':
    curl_origin_shaped_metaworld_ml1_push()
