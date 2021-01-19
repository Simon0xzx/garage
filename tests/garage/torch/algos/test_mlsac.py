"""Module for testing MLSAC."""
import metaworld
import numpy as np
import pickle
import pytest
import torch
from torch.nn import functional as F

from garage.envs import GymEnv, MultiEnvWrapper, normalize, GoalAugmentedWrapper
from garage.envs.multi_env_wrapper import round_robin_strategy
from garage.experiment import deterministic, MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import global_device, set_gpu_mode
from garage.torch.algos import MLSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


# @pytest.mark.mujoco
# def test_mlsac_get_log_alpha(monkeypatch):
#     """Check that the private function _get_log_alpha functions correctly.

#     MLSAC uses disentangled alphas, meaning that

#     """
#     env_names = ['CartPole-v0', 'CartPole-v1']
#     task_envs = [GymEnv(name, max_episode_length=100) for name in env_names]
#     env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
#     print('classes: ', env._classes)
#     print('tasks: ', env._tasks)
#     deterministic.set_seed(0)
#     policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[1, 1],
#         hidden_nonlinearity=torch.nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )

#     qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)

#     qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)
#     replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

#     num_tasks = 2
#     buffer_batch_size = 2
#     mlsac = MLSAC(policy=policy,
#                   qf1=qf1,
#                   qf2=qf2,
#                   sampler=None,
#                   gradient_steps_per_itr=150,
#                   eval_env=[env],
#                   env_spec=env.spec,
#                   num_tasks=num_tasks,
#                   steps_per_epoch=5,
#                   replay_buffer=replay_buffer,
#                   min_buffer_size=1e3,
#                   target_update_tau=5e-3,
#                   discount=0.99,
#                   buffer_batch_size=buffer_batch_size)
#     monkeypatch.setattr(mlsac, '_log_alpha', torch.Tensor([1., 2.]))
#     for i, _ in enumerate(env_names):
#         obs = torch.Tensor([env.reset()[0]] * buffer_batch_size)
#         log_alpha = mlsac._get_log_alpha(dict(observation=obs))
#         assert (log_alpha == torch.Tensor([i + 1, i + 1])).all().item()
#         assert log_alpha.size() == torch.Size([mlsac._buffer_batch_size])


# @pytest.mark.mujoco
# def test_mlsac_get_log_alpha_incorrect_num_tasks(monkeypatch):
#     """Check that if the num_tasks passed does not match the number of tasks

#     in the environment, then the algorithm should raise an exception.

#     MLSAC uses disentangled alphas, meaning that

#     """
#     env_names = ['CartPole-v0', 'CartPole-v1']
#     task_envs = [GymEnv(name, max_episode_length=150) for name in env_names]
#     env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
#     deterministic.set_seed(0)
#     policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[1, 1],
#         hidden_nonlinearity=torch.nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )

#     qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)

#     qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)
#     replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

#     buffer_batch_size = 2
#     mlsac = MLSAC(policy=policy,
#                   qf1=qf1,
#                   qf2=qf2,
#                   sampler=None,
#                   gradient_steps_per_itr=150,
#                   eval_env=[env],
#                   env_spec=env.spec,
#                   num_tasks=4,
#                   steps_per_epoch=5,
#                   replay_buffer=replay_buffer,
#                   min_buffer_size=1e3,
#                   target_update_tau=5e-3,
#                   discount=0.99,
#                   buffer_batch_size=buffer_batch_size)
#     monkeypatch.setattr(mlsac, '_log_alpha', torch.Tensor([1., 2.]))
#     error_string = ('The number of tasks in the environment does '
#                     'not match self._num_tasks. Are you sure that you passed '
#                     'The correct number of tasks?')
#     obs = torch.Tensor([env.reset()[0]] * buffer_batch_size)
#     with pytest.raises(ValueError, match=error_string):
#         mlsac._get_log_alpha(dict(observation=obs))


# @pytest.mark.mujoco
# def test_mlsac_inverted_double_pendulum():
#     """Performance regression test of MTSAC on 2 InvDoublePendulum envs."""
#     env_names = ['InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v2']
#     task_envs = [GymEnv(name, max_episode_length=100) for name in env_names]
#     env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
#     test_envs = MultiEnvWrapper(task_envs,
#                                 sample_strategy=round_robin_strategy)
#     deterministic.set_seed(0)
#     trainer = Trainer(snapshot_config=snapshot_config)
#     policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[32, 32],
#         hidden_nonlinearity=torch.nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )

#     qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[32, 32],
#                                  hidden_nonlinearity=F.relu)

#     qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[32, 32],
#                                  hidden_nonlinearity=F.relu)
#     replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
#     num_tasks = 2
#     buffer_batch_size = 128
#     sampler = LocalSampler(agents=policy,
#                            envs=env,
#                            max_episode_length=env.spec.max_episode_length,
#                            worker_class=FragmentWorker)
#     mlsac = MLSAC(policy=policy,
#                   qf1=qf1,
#                   qf2=qf2,
#                   sampler=sampler,
#                   gradient_steps_per_itr=100,
#                   eval_env=[test_envs],
#                   env_spec=env.spec,
#                   num_tasks=num_tasks,
#                   steps_per_epoch=5,
#                   replay_buffer=replay_buffer,
#                   min_buffer_size=1e3,
#                   target_update_tau=5e-3,
#                   discount=0.99,
#                   buffer_batch_size=buffer_batch_size)
#     trainer.setup(mlsac, env)
#     ret = trainer.train(n_epochs=8, batch_size=128, plot=False)
#     assert ret > 0

# def test_to():
#     """Test the torch function that moves modules to GPU.

#         Test that the policy and qfunctions are moved to gpu if gpu is
#         available.

#     """
#     env_names = ['CartPole-v0', 'CartPole-v1']
#     task_envs = [GymEnv(name, max_episode_length=100) for name in env_names]
#     env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
#     deterministic.set_seed(0)
#     policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[1, 1],
#         hidden_nonlinearity=torch.nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )

#     qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)

#     qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[1, 1],
#                                  hidden_nonlinearity=F.relu)
#     replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

#     num_tasks = 2
#     buffer_batch_size = 2
#     mlsac = MLSAC(policy=policy,
#                   qf1=qf1,
#                   qf2=qf2,
#                   sampler=None,
#                   gradient_steps_per_itr=150,
#                   eval_env=[env],
#                   env_spec=env.spec,
#                   num_tasks=num_tasks,
#                   steps_per_epoch=5,
#                   replay_buffer=replay_buffer,
#                   min_buffer_size=1e3,
#                   target_update_tau=5e-3,
#                   discount=0.99,
#                   buffer_batch_size=buffer_batch_size)

#     set_gpu_mode(torch.cuda.is_available())
#     mlsac.to()
#     device = global_device()
#     for param in mlsac._qf1.parameters():
#         assert param.device == device
#     for param in mlsac._qf2.parameters():
#         assert param.device == device
#     for param in mlsac._qf2.parameters():
#         assert param.device == device
#     for param in mlsac.policy.parameters():
#         assert param.device == device
#     assert mlsac._log_alpha.device == device


# @pytest.mark.mujoco
# def test_fixed_alpha():
#     """Test if using fixed_alpha ensures that alpha is non differentiable."""
#     env_names = ['InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v2']
#     task_envs = [GymEnv(name, max_episode_length=100) for name in env_names]
#     env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
#     test_envs = MultiEnvWrapper(task_envs,
#                                 sample_strategy=round_robin_strategy)
#     deterministic.set_seed(0)
#     trainer = Trainer(snapshot_config=snapshot_config)
#     policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[32, 32],
#         hidden_nonlinearity=torch.nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )

#     qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[32, 32],
#                                  hidden_nonlinearity=F.relu)

#     qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                                  hidden_sizes=[32, 32],
#                                  hidden_nonlinearity=F.relu)
#     replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
#     num_tasks = 2
#     buffer_batch_size = 128
#     sampler = LocalSampler(agents=policy,
#                            envs=env,
#                            max_episode_length=env.spec.max_episode_length,
#                            worker_class=FragmentWorker)
#     mlsac = MLSAC(policy=policy,
#                   qf1=qf1,
#                   qf2=qf2,
#                   sampler=sampler,
#                   gradient_steps_per_itr=100,
#                   eval_env=[test_envs],
#                   env_spec=env.spec,
#                   num_tasks=num_tasks,
#                   steps_per_epoch=1,
#                   replay_buffer=replay_buffer,
#                   min_buffer_size=1e3,
#                   target_update_tau=5e-3,
#                   discount=0.99,
#                   buffer_batch_size=buffer_batch_size,
#                   fixed_alpha=np.exp(0.5))
#     if torch.cuda.is_available():
#         set_gpu_mode(True)
#     else:
#         set_gpu_mode(False)
#     mlsac.to()
#     assert torch.allclose(torch.Tensor([0.5] * num_tasks),
#                           mtlac._log_alpha.to('cpu'))
#     trainer.setup(mlsac, env)
#     trainer.train(n_epochs=1, batch_size=128, plot=False)
#     assert torch.allclose(torch.Tensor([0.5] * num_tasks),
#                           mlsac._log_alpha.to('cpu'))
#     assert not mlsac._use_automatic_entropy_tuning

@pytest.mark.mujoco
def test_mlsac_metaworld_ml1():
    """Performance regression test of MLSAC on Metaworld ML1 env."""
    ml1 = metaworld.ML1('reach-v1')
    ml1_test = metaworld.ML1('reach-v1')
    train_task_sampler = MetaWorldTaskSampler(ml1, 'train',
                                              lambda env, task: normalize(GoalAugmentedWrapper(env, task)))
    test_task_sampler = MetaWorldTaskSampler(ml1_test, 'test',
                                             lambda env, task: normalize(GoalAugmentedWrapper(env, task)))
    n_train_tasks = 50
    n_test_tasks = 10
    train_envs = train_task_sampler.sample(n_train_tasks)
    env = train_envs[0]()
    test_envs = [env_up() for env_up in test_task_sampler.sample(n_test_tasks)]
    embeddings = [pickle.loads(t._task.data)['rand_vec'].astype(np.float32) for t in train_envs]
    deterministic.set_seed(0)
    trainer = Trainer(snapshot_config=snapshot_config)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    num_tasks = 2
    buffer_batch_size = 128
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    mlsac = MLSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  sampler=sampler,
                  embeddings=embeddings,
                  gradient_steps_per_itr=100,
                  eval_env=test_envs,
                  env_spec=env.spec,
                  num_tasks=num_tasks,
                  steps_per_epoch=5,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=buffer_batch_size)
    trainer.setup(mlsac, env)
    ret = trainer.train(n_epochs=8, batch_size=128, plot=False)
    assert ret > 0