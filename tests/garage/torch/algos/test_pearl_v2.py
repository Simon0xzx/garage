"""This script is a test that fails when PEARL performance is too low."""
import pickle

import pytest

from garage.envs import MetaWorldSetTaskEnv, normalize, PointEnv
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import PEARLV2
from garage.torch.algos.pearl_v2 import PEARLV2Worker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config

try:
    # pylint: disable=unused-import
    import mujoco_py  # noqa: F401
except ImportError:
    pytest.skip('To use mujoco-based features, please install garage[mujoco].',
                allow_module_level=True)
except Exception:  # pylint: disable=broad-except
    pytest.skip(
        'Skipping tests, failed to import mujoco. Do you have a '
        'valid mujoco key installed?',
        allow_module_level=True)
import metaworld  # isort:skip


@pytest.mark.mujoco
class TestPEARL:
    """Test class for PEARL."""

    # @pytest.mark.skip
    @pytest.mark.large
    def test_pearl_ml1_push(self):
        """Test PEARL with ML1 Push environment."""
        params = dict(seed=1,
                      num_epochs=1,
                      num_train_tasks=5,
                      latent_size=7,
                      encoder_hidden_sizes=[10, 10, 10],
                      net_size=30,
                      meta_batch_size=16,
                      num_steps_per_epoch=40,
                      num_initial_steps=40,
                      num_tasks_sample=15,
                      num_steps_prior=15,
                      num_extra_rl_steps_posterior=15,
                      batch_size=256,
                      embedding_batch_size=8,
                      embedding_mini_batch_size=8,
                      reward_scale=10.,
                      use_information_bottleneck=True,
                      use_next_obs_in_context=False,
                      use_gpu=True,
                      gpu=3,
                      fixed_alpha=None,
                      target_entropy=None,
                      initial_log_entropy=0.)

        net_size = params['net_size']
        set_seed(params['seed'])
        # create multi-task environment and sample tasks
        ml1 = metaworld.ML1('push-v1')
        train_env = MetaWorldSetTaskEnv(ml1, 'train')
        env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                     env=train_env,
                                     wrapper=lambda env, _: normalize(env))
        env = env_sampler.sample(params['num_train_tasks'])
        test_env = MetaWorldSetTaskEnv(ml1, 'test')
        test_env_sampler = SetTaskSampler(
            MetaWorldSetTaskEnv,
            env=test_env,
            wrapper=lambda env, _: normalize(env))

        augmented_env = PEARLV2.augment_env_spec(env[0](), params['latent_size'])
        qf1 = ContinuousMLPQFunction(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])

        qf2 = ContinuousMLPQFunction(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])

        inner_policy = TanhGaussianMLPPolicy(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])

        sampler = LocalSampler(
            agents=None,
            envs=env[0](),
            max_episode_length=env[0]().spec.max_episode_length,
            n_workers=1,
            worker_class=PEARLV2Worker)

        pearl = PEARLV2(
            env=env,
            policy_class=ContextConditionedPolicy,
            encoder_class=MLPEncoder,
            inner_policy=inner_policy,
            qf1=qf1,
            qf2=qf2,
            sampler=sampler,
            num_train_tasks=params['num_train_tasks'],
            latent_dim=params['latent_size'],
            encoder_hidden_sizes=params['encoder_hidden_sizes'],
            test_env_sampler=test_env_sampler,
            meta_batch_size=params['meta_batch_size'],
            num_steps_per_epoch=params['num_steps_per_epoch'],
            num_initial_steps=params['num_initial_steps'],
            num_tasks_sample=params['num_tasks_sample'],
            num_steps_prior=params['num_steps_prior'],
            num_extra_rl_steps_posterior=params[
                'num_extra_rl_steps_posterior'],
            batch_size=params['batch_size'],
            embedding_batch_size=params['embedding_batch_size'],
            embedding_mini_batch_size=params['embedding_mini_batch_size'],
            reward_scale=params['reward_scale'],
            fixed_alpha=params['fixed_alpha'],
            target_entropy=params['target_entropy'],
            initial_log_entropy=params['initial_log_entropy'],
        )

        set_gpu_mode(params['use_gpu'], gpu_id=params['gpu'])
        if params['use_gpu']:
            pearl.to()

        trainer = Trainer(snapshot_config)
        trainer.setup(algo=pearl, env=env[0]())

        trainer.train(n_epochs=params['num_epochs'],
                      batch_size=params['batch_size'])

    def test_pickling(self):
        """Test pickle and unpickle."""
        net_size = 10
        env_sampler = SetTaskSampler(PointEnv)
        env = env_sampler.sample(5)

        test_env_sampler = SetTaskSampler(PointEnv)

        augmented_env = PEARLV2.augment_env_spec(env[0](), 5)
        qf1 = ContinuousMLPQFunction(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])
        
        qf2 = ContinuousMLPQFunction(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])

        inner_policy = TanhGaussianMLPPolicy(
            env_spec=augmented_env,
            hidden_sizes=[net_size, net_size, net_size])

        pearl = PEARLV2(env=env,
                        inner_policy=inner_policy,
                        qf1=qf1,
                        qf2=qf2,
                        sampler=None,
                        num_train_tasks=5,
                        num_test_tasks=5,
                        latent_dim=5,
                        encoder_hidden_sizes=[10, 10],
                        test_env_sampler=test_env_sampler)

        # This line is just to improve coverage
        pearl.to()

        pickled = pickle.dumps(pearl)
        unpickled = pickle.loads(pickled)

        assert hasattr(unpickled, '_replay_buffers')
        assert hasattr(unpickled, '_context_replay_buffers')
        assert unpickled._is_resuming
