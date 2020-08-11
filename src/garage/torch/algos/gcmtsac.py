"""This modules creates a MTSAC model in PyTorch."""
import numpy as np
import torch
import copy
import akro
from dowel import tabular
import torch.nn.functional as F
from collections import deque

from garage.experiment import MetaEvaluator
from garage import log_multitask_performance, TrajectoryBatch
from garage.np import obtain_evaluation_samples
from garage.torch import dict_np_to_torch, global_device
from garage.envs import EnvSpec
from garage.sampler import DefaultWorker, RaySampler
from garage.np.algos import RLAlgorithm


class GCMTSAC(RLAlgorithm):
    """A Goal Conditioned MTSAC Model in Torch.

    This MTSAC implementation uses is the same as SAC except for a small change
    called "disentangled alphas". Alpha is the entropy coefficient that is used
    to control exploration of the agent/policy. Disentangling alphas refers to
    having a separate alpha coefficients for every task learned by the policy.
    The alphas are accessed by using a the one-hot encoding of an id that is
    assigned to each task.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Stores transitions
            that are previously collected by the sampler.
        env_spec (garage.envs.env_spec.EnvSpec): The env_spec attribute of the
            environment that the agent is being trained in. Usually accessable
            by calling env.spec.
        num_tasks (int): The number of tasks being learned.
        max_path_length (int): The max path length of the algorithm.
        eval_env (garage.envs.GarageEnv): The environment used for collecting
            evaluation trajectories.
        gradient_steps_per_itr (int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): The discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): A coefficient that controls the rate at
            which the target q_functions update over optimization iterations.
        policy_lr (float): Learning rate for policy optimizers.
        qf_lr (float): Learning rate for q_function optimizers.
        reward_scale (float): Reward multiplier. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): Optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_trajectories (int): The number of evaluation
            trajectories used for computing eval stats at the end of every
            epoch.

    """

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            replay_buffers,
            env_spec,
            num_tasks,
            max_path_length,
            eval_env,
            gradient_steps_per_itr,
            goal_dim=3,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=10.,
            optimizer=torch.optim.Adam,
            num_evaluation_trajectories=5,
    ):
        self._env = env
        self._num_tasks = num_tasks
        self._goal_dim = goal_dim
        self._eval_env = eval_env
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha

        self._qf1 = qf1
        self._qf2 = qf2
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_trajectories = num_evaluation_trajectories
        self._eval_env = eval_env

        self._min_buffer_size = min_buffer_size
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_path_length = max_path_length
        self._max_eval_path_length = None

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffers = replay_buffers

        self.sampler_cls = RaySampler

        self._reward_scale = reward_scale
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._policy_lr)
        self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
                                              lr=self._qf_lr)
        self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
                                              lr=self._qf_lr)

        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy] * self._goal_dim).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha] * self._goal_dim,
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha] * self._goal_dim).log()

        self.episode_rewards = deque(maxlen=30)
        self._epoch_mean_success_rate = []
        self._epoch_median_success_rate = []

        worker_args = dict(deterministic=True, accum_context=True)
        self._evaluator = MetaEvaluator(test_task_sampler=eval_env,
                                        max_path_length=max_path_length,
                                        worker_class=GCMTSACWorker,
                                        worker_args=worker_args,
                                        n_test_tasks=5)


    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_return = None
        for _ in runner.step_epochs():
            for _ in range(self._num_tasks):
                self._env.reset()
                task_idx = self._env.active_task_index
                if not (self.replay_buffers[task_idx].n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(runner.step_itr, batch_size,
                                                         self.policy)
                path_returns = []
                for path in runner.step_path:
                    self.replay_buffers[task_idx].add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=path['dones'].reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) is len(runner.step_path)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            # self._evaluator.evaluate(self)
            last_return = self._evaluate_policy(runner.step_itr)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            runner.step_itr += 1

        return np.mean(last_return)

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        task_idx = self._env.active_task_index
        if self.replay_buffers[task_idx].n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffers[task_idx].sample_transitions(
                self._buffer_batch_size)
            samples = dict_np_to_torch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)

        """
        obs = samples_data['observation']
        log_alpha = self._log_alpha
        goal = obs[:, -self._goal_dim:]
        ret = torch.mm(goal, log_alpha.unsqueeze(0).t()).squeeze()
        return ret

    def _temperature_objective(self, log_pi, samples_data):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            alpha_loss = (-(self._get_log_alpha(samples_data)) *
                          (log_pi.detach() + self._target_entropy)).mean()
        return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions(torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions(torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        obs = samples_data['observation']
        b, _ = obs.size()
        task_z = torch.as_tensor(self._env.active_task_goal, device=global_device()).float()
        task_z = torch.cat([z.repeat(b, 1) for z in task_z], dim=1)

        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(torch.cat([obs, task_z], dim=1), new_actions),
                                      self._qf2(torch.cat([obs, task_z], dim=1), new_actions))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        """Compute the Q-function/critic loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        new_next_actions_dist, task_z = self.policy(next_obs, self._env.active_task_goal)

        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        q1_pred = self._qf1(torch.cat([obs, task_z], dim=1), actions)
        q2_pred = self._qf2(torch.cat([obs, task_z], dim=1), actions)

        target_q_values = torch.min(
            self._target_qf1(torch.cat([next_obs, task_z], dim=1), new_next_actions),
            self._target_qf2(torch.cat([next_obs, task_z], dim=1), new_next_actions))\
                              .flatten() - (alpha * new_log_pi)
        with torch.no_grad():
            q_target = rewards * self._reward_scale + (1. - terminals) * self._discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        action_dists = self.policy(obs, self._env.active_task_goal)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic rollouts.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_trajectories
                trajectories

        """
        eval_trajs = []
        for _ in range(self._num_tasks):
            eval_trajs.append(
                obtain_evaluation_samples(
                    self.policy,
                    self._eval_env,
                    num_trajs=self._num_evaluation_trajectories))
        eval_trajs = TrajectoryBatch.concatenate(*eval_trajs)
        last_return = log_multitask_performance(epoch, eval_trajs,
                                                self._discount)
        return last_return

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim):
        """Augment environment by a size of latent dimension.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.

        Returns:
            garage.envs.EnvSpec: Augmented environment specs.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        aug_obs = akro.Box(low=-np.inf, high=np.inf,
                           shape=(obs_dim + latent_dim,),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1, high=1,
                           shape=(action_dim,),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss(torch.Tensor): loss from actor/policy network.
            qf1_loss(torch.Tensor): loss from 1st qf/critic network.
            qf2_loss(torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Training/Policy/Loss', policy_loss.item())
        tabular.record('Training/QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('Training/QF/{}'.format('Qf2Loss'), float(qf2_loss))
        for task_idx, buffer in self.replay_buffers:
            tabular.record('ReplayBuffer/buffer_{}_size'.format(task_idx),
                           buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha] * self._goal_dim)\
                .log().to(device)
        else:
            self._log_alpha = torch.Tensor( [self._initial_log_entropy] * self._goal_dim)\
                .to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)



class GCMTSACWorker(DefaultWorker):
    """A worker class used in sampling for GCMTSAC.

    It stores context and resample belief in the policy every step.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic(bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        accum_context(bool): If true, update context of the agent.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 deterministic=False,
                 accum_context=False):
        self._deterministic = deterministic
        self._accum_context = accum_context
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_path_length`.

        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            if self._deterministic:
                a = agent_info['mean']
            next_o, r, d, env_info = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            if not d:
                self._prev_obs = next_o
                return False
        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.agent.infer_posterior(np.array([self.env.active_task_goal]))
        self.start_rollout()
        while not self.step_rollout():
            pass
        self._agent_infos['context'] = [self.env] * self._max_path_length
        return self.collect_rollout()
