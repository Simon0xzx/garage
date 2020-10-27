"""CURL and CURLWorker in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

import copy
import pickle
import akro
from dowel import logger
import numpy as np
import torch
import torch.nn.functional as F
from dowel import tabular

from garage import InOutSpec, TimeStep
from garage.envs import EnvSpec
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch import global_device
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import CurlPolicy


class CURL(MetaRLAlgorithm):
    """A PEARL model based on https://arxiv.org/abs/1903.08254.

    PEARL, which stands for Probablistic Embeddings for Actor-Critic
    Reinforcement Learning, is an off-policy meta-RL algorithm. It is built
    on top of SAC using two Q-functions and a value function with an addition
    of an inference network that estimates the posterior :math:`q(z \| c)`.
    The policy is conditioned on the latent variable Z in order to adpat its
    behavior to specific tasks.

    Args:
        env (list[GarageEnv]): Batch of sampled environment updates(EnvUpdate),
            which, when invoked on environments, will configure them with new
            tasks.
        policy_class (garage.torch.policies.Policy): Context-conditioned policy
            class.
        encoder_class (garage.torch.embeddings.ContextEncoder): Encoder class
            for the encoder in context-conditioned policy.
        inner_policy (garage.torch.policies.Policy): Policy.
        qf (torch.nn.Module): Q-function.
        vf (torch.nn.Module): Value function.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        encoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of
            the context encoder.
        test_env_sampler (garage.experiment.SetTaskSampler): Sampler for test
            tasks.
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        context_lr (float): Inference network learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the
            soft target update.
        kl_lambda (float): KL lambda value.
        optimizer_class (callable): Type of optimizer for training networks.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_steps_posterior (int): Number of transitions to obtain per task
            with z ~ posterior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_path_length (int): Maximum path length.
        discount (float): RL discount factor.
        replay_buffer_size (int): Maximum samples in replay buffer.
        reward_scale (int): Reward scale.
        update_post_train (int): How often to resample context when obtaining
            data during training (in trajectories).

    """

    # pylint: disable=too-many-statements
    def __init__(self,
                 env,
                 inner_policy,
                 qf1,
                 qf2,
                 num_train_tasks,
                 num_test_tasks,
                 latent_dim,
                 encoder_hidden_sizes,
                 test_env_sampler,
                 policy_class=CurlPolicy,
                 encoder_class=MLPEncoder,
                 policy_lr=3E-4,
                 qf_lr=3E-4,
                 context_lr=3E-4,
                 policy_mean_reg_coeff=1E-3,
                 policy_std_reg_coeff=1E-3,
                 policy_pre_activation_coeff=0.,
                 soft_target_tau=0.005,
                 kl_lambda=.1,
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 optimizer_class=torch.optim.Adam,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,
                 use_kl_loss=False,
                 use_q_loss=True,
                 meta_batch_size=64,
                 num_steps_per_epoch=1000,
                 num_initial_steps=100,
                 num_tasks_sample=100,
                 num_steps_prior=100,
                 num_steps_posterior=0,
                 num_extra_rl_steps_posterior=100,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=1000,
                 encoder_path_sample_len=64,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 reward_scale=1,
                 embedding_batch_in_sequence=False,
                 num_pos_contrastive = 2,
                 num_neg_contrastive = 0,
                 contrastive_mean_only = False,
                 new_contrastive_formula = False,
                 new_weight_update = False,
                 encoder_common_net=True,
                 single_alpha = False,
                 update_post_train=1):

        self._env = env
        self._qf1 = qf1
        self._qf2 = qf2
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)

        # Contrastive Encoder setting
        self._embedding_batch_in_sequence = embedding_batch_in_sequence
        self._num_pos_contrastive = num_pos_contrastive # TODO
        self._num_neg_contrastive = num_neg_contrastive # TODO

        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context
        self._use_kl_loss = use_kl_loss
        self._use_q_loss = use_q_loss
        self._encoder_path_sample_len = encoder_path_sample_len

        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._update_post_train = update_post_train
        self._task_idx = None
        self._is_resuming = False

        # Architecture choice
        self._contrastive_mean_only = contrastive_mean_only
        self._new_contrastive_formula = new_contrastive_formula
        self._new_weight_update = new_weight_update
        self._encoder_common_net = encoder_common_net
        self._single_alpha = single_alpha

        worker_args = dict(deterministic=True, accum_context=True)
        self._evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                        max_path_length=max_path_length,
                                        worker_class=CURLWorker,
                                        worker_args=worker_args,
                                        n_test_tasks=num_test_tasks)

        env_spec = env[0]()
        encoder_spec = self.get_env_spec(env_spec, latent_dim, 'encoder', use_information_bottleneck)
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        if self._use_next_obs_in_context:
            encoder_in_dim += int(np.prod(env[0]().observation_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        self._context_encoder = encoder_class(input_dim=encoder_in_dim,
                                        output_dim=encoder_out_dim,
                                        common_network= self._encoder_common_net,
                                        hidden_sizes=encoder_hidden_sizes)
        if self._contrastive_mean_only:
            encoder_out_dim = self._latent_dim
        self._contrastive_weight = torch.rand(encoder_out_dim, encoder_out_dim, device=global_device(), requires_grad=True)

        # Automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._initial_log_entropy = initial_log_entropy
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    env_spec.action_space.shape).item()
            if self._single_alpha:
                self._log_alpha = torch.Tensor([self._initial_log_entropy]).requires_grad_()
            else:
                self._log_alpha = torch.Tensor([self._initial_log_entropy] * self._num_train_tasks).requires_grad_()
            self._alpha_optimizer = optimizer_class([self._log_alpha], lr=policy_lr)
        else:
            if self._single_alpha:
                self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
            else:
                self._log_alpha = torch.Tensor([self._fixed_alpha] * self._num_train_tasks).log()

        self._context_lr = context_lr
        self._policy = policy_class(
            latent_dim=latent_dim,
            context_encoder=self._context_encoder,
            policy=inner_policy,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context)

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._policy_optimizer = optimizer_class(
            self._policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        if self._encoder_common_net:
            self.context_optimizer = optimizer_class(
                self._context_encoder.networks[0].parameters(),
                lr=context_lr,
            )
        if self._new_weight_update:
            self.contrastive_weight_optimizer = optimizer_class(
                [self._contrastive_weight],
                lr=context_lr,
            )
        self.query_optimizer = optimizer_class(
            self._context_encoder.networks[1].parameters(),
            lr=context_lr,
        )


    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }
        self._is_resuming = True

    def update_env(self, env, evaluator, num_train_tasks, num_test_tasks):
        print("Updating environments")
        self._env = env
        self._evaluator = evaluator
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }
        self._task_idx = 0
        print("Updated with new environment setup")

        self._policy_optimizer = torch.optim.Adam(
            self._policy.networks[1].parameters(),
            lr=3E-4,
        )
        self.qf1_optimizer = torch.optim.Adam(
            self._qf1.parameters(),
            lr=3E-4,
        )
        self.qf2_optimizer = torch.optim.Adam(
            self._qf2.parameters(),
            lr=3E-4,
        )
        if self._encoder_common_net:
            self.context_optimizer = torch.optim.Adam(
                self._context_encoder.networks[0].parameters(),
                lr=3E-4,
            )
        self.query_optimizer = torch.optim.Adam(
            self._context_encoder.get_query_net().parameters(),
            lr=3E-4,
        )
        print('Reset optimizer state')

    def get_encoder_info(self):
        mean = self._policy.z_means.detach().cpu().numpy()
        mean_norm = np.linalg.norm(mean)
        var = self._policy.z_vars.detach().cpu().numpy()
        var_norm = np.linalg.norm(var)
        return mean_norm, var_norm

    def fill_expert_traj(self, expert_traj_dir):
        print("Filling Expert trajectory to replay buffer")
        from os import listdir
        from os.path import isfile, join
        expert_traj_paths = [join(expert_traj_dir, f) for f in
                             listdir(expert_traj_dir) if
                             isfile(join(expert_traj_dir, f))]
        expert_trajs = []
        for exp_path in expert_traj_paths:
            with open(exp_path, 'rb') as handle:
                data = pickle.load(handle)
                expert_trajs.append(data)

        for path in expert_trajs:
            p = {
                'observations': path['observations'],
                'actions': path['actions'],
                'rewards': path['rewards'].reshape(-1, 1),
                'next_observations': path['next_observations'],
                'dones': path['dones'].reshape(-1, 1)
            }
            self._replay_buffers[self._task_idx].add_path(p)
            self._context_replay_buffers[self._task_idx].add_path(p)

    def adapt_expert_traj(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

                Args:
                    runner (LocalRunner): LocalRunner is passed to give algorithm
                        the access to runner.step_epochs(), which provides services
                        such as snapshotting and sampler control.

                """
        for _ in runner.step_epochs():
            logger.log('Adapting Policy {}...'.format(runner.step_itr))
            self._train_once()
            runner.step_itr += 1
            logger.log('Evaluating...')
            # evaluate
            self._policy.reset_belief()
            self._evaluator.evaluate(self)

    def train(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_samples(runner, epoch,
                                         self._num_initial_steps, np.inf)
                    self._is_resuming = False

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self._obtain_samples(runner, epoch, self._num_steps_prior,
                                         np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self._obtain_samples(runner, epoch,
                                         self._num_steps_posterior,
                                         self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self._obtain_samples(runner,
                                         epoch,
                                         self._num_extra_rl_steps_posterior,
                                         self._update_post_train,
                                         add_to_enc_buffer=False)

            logger.log('Training...')
            # sample train tasks and optimize networks
            self._train_once()
            runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self._policy.reset_belief()
            self._evaluator.evaluate(self)

    def _train_once(self):
        """Perform one iteration of training."""
        policy_loss_list = []
        qf_loss_list = []
        contrastive_loss_list = []
        alpha_loss_list = []
        alpha_list = []
        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            policy_loss, qf_loss, contrastive_loss, alpha_loss, alpha = self._optimize_policy(indices)
            policy_loss_list.append(policy_loss)
            qf_loss_list.append(qf_loss)
            contrastive_loss_list.append(contrastive_loss)
            alpha_loss_list.append(alpha_loss)

        with tabular.prefix('MetaTrain/Average/'):
            tabular.record('PolicyLoss', np.average(np.array(policy_loss_list)))
            tabular.record('QfLoss', np.average(np.array(qf_loss_list)))
            tabular.record('ContrastiveLoss', np.average(np.array(contrastive_loss_list)))
            tabular.record('AlphaLoss', np.average(np.array(alpha_loss_list)))
            tabular.record('AlphaLoss', np.average(np.array(alpha_loss_list)))
            tabular.record('Alpha', np.average(np.array(alpha_list)))

    def augment_path(self, path, batch_size, in_sequence = False):
        path_len = path['actions'].shape[0]
        augmented_path = {}
        if in_sequence:
            if self._encoder_path_sample_len > path_len:
                raise Exception('Embedding_batch size cannot be longer than path length {} > {}'.format(batch_size, path_len))
            seq_begin = np.random.randint(0, path_len - self._encoder_path_sample_len, batch_size)
            augmented_path['observations'] = np.vstack([path['observations'][i:i + self._encoder_path_sample_len] for i in seq_begin])
            augmented_path['actions'] = np.vstack([path['actions'][i:i + self._encoder_path_sample_len] for i in seq_begin])
            augmented_path['rewards'] = np.vstack([path['rewards'][i:i + self._encoder_path_sample_len] for i in seq_begin])
            augmented_path['next_observations'] = np.vstack([path['next_observations'][i:i + self._encoder_path_sample_len] for i in seq_begin])
        else:
            seq_idx = np.random.choice(path_len, batch_size * self._encoder_path_sample_len)
            augmented_path['observations'] = path['observations'][seq_idx]
            augmented_path['actions'] = path['actions'][seq_idx]
            augmented_path['rewards'] = path['rewards'][seq_idx]
            augmented_path['next_observations'] = path['next_observations'][seq_idx]

        return augmented_path


    def augment_path_old(self, path, batch_size, in_sequence = False):
        path_len = path['actions'].shape[0]
        augmented_path = {}
        if in_sequence:
            if batch_size > path_len:
                raise Exception(
                    'Embedding_batch size cannot be longer than path length {} > {}'.format(batch_size, path_len))
            seq_begin = np.random.randint(0, path_len - batch_size)
            augmented_path['observations'] = path['observations'][seq_begin:seq_begin + batch_size]
            augmented_path['actions'] = path['actions'][seq_begin:seq_begin + batch_size]
            augmented_path['rewards'] = path['rewards'][seq_begin:seq_begin + batch_size]
            augmented_path['next_observations'] = path['next_observations'][seq_begin:seq_begin + batch_size]
        else:
            seq_idx = np.random.choice(path_len, batch_size)
            augmented_path['observations'] = path['observations'][seq_idx]
            augmented_path['actions'] = path['actions'][seq_idx]
            augmented_path['rewards'] = path['rewards'][seq_idx]
            augmented_path['next_observations'] = path['next_observations'][seq_idx]

        return augmented_path

    def _compute_contrastive_loss_new(self, indices):
        # Optimize CURL encoder
        context_augs = self._sample_contrastive_pairs(indices, num_aug=2)
        aug1 = torch.as_tensor(context_augs[0], device=global_device())
        aug2 = torch.as_tensor(context_augs[1], device=global_device())
        # path_batches = self.sample_path_batch(indices)

        # similar_contrastive
        query = self._context_encoder(aug1, query=True)
        key = self._context_encoder(aug2, query=False)
        t,b,d = query.size()
        query = query.view(t * b, d)
        key = key.view(t * b, d)
        if self._contrastive_mean_only:
            assert self._contrastive_weight.size()[0] == self._latent_dim
            query = query[:, :self._latent_dim]
            key = key[:, :self._latent_dim]
        loss_fun = torch.nn.CrossEntropyLoss()
        left_product = torch.matmul(query, self._contrastive_weight.to(global_device()))
        logits = torch.matmul(left_product, key.T)
        logits = logits - torch.max(logits, axis=1)[0]
        labels = torch.arange(logits.shape[0]).to(global_device())
        loss = loss_fun(logits, labels)
        return loss

    def _compute_contrastive_loss(self, indices):
        # Optimize CURL encoder
        context_augs = self._sample_contrastive_pairs(indices, num_aug=2)
        aug1 = torch.as_tensor(context_augs[0], device=global_device())
        aug2 = torch.as_tensor(context_augs[1], device=global_device())
        # path_batches = self.sample_path_batch(indices)

        # similar_contrastive
        query = self._context_encoder(aug1, query=True)
        key = self._context_encoder(aug2, query=False)
        if self._contrastive_mean_only:
            assert self._contrastive_weight.size()[0] == self._latent_dim
            query = query[:, :self._latent_dim]
            key = key[:, :self._latent_dim]
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = None
        for i in range(len(indices)):
            left_product = torch.matmul(query[i], self._contrastive_weight.to(global_device()))
            logits = torch.matmul(left_product, key[i].T)
            logits = logits - torch.max(logits, axis=1)[0]
            labels = torch.arange(logits.shape[0]).to(global_device())
            if not loss:
                loss = loss_fun(logits, labels)
            else:
                loss += loss_fun(logits, labels)
        return loss

    def _optimize_policy(self, indices):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.

        """
        num_tasks = len(indices)
        if self._new_contrastive_formula:
            contrastive_loss = self._compute_contrastive_loss_new(indices)
        else:
            contrastive_loss = self._compute_contrastive_loss(indices)
        context = self._sample_context(indices)

        # clear context and reset belief of policy
        self._policy.reset_belief(num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self._sample_data(indices)

        # flatten out the task dimension
        t, b, _ = obs.size()
        batch_obs = obs.view(t * b, -1)
        batch_action = actions.view(t * b, -1)
        batch_next_obs = next_obs.view(t * b, -1)

        policy_outputs, task_z = self._policy(next_obs, context)
        new_next_actions, policy_mean, policy_log_std, log_pi, pre_tanh = policy_outputs

        # ===== Critic Objective =====
        with torch.no_grad():
            alpha = self._get_log_alpha(indices).exp()
        q1_pred = self._qf1(torch.cat([batch_obs, batch_action], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([batch_obs, batch_action], dim=1), task_z)
        target_q_values = torch.min(self._target_qf1(torch.cat([batch_next_obs, new_next_actions], dim=1), task_z),
            self._target_qf2(torch.cat([batch_next_obs, new_next_actions], dim=1), task_z)).flatten() - (alpha * log_pi.flatten())

        rewards_flat = rewards.view(self._batch_size * num_tasks, -1).flatten()
        rewards_flat = rewards_flat * self._reward_scale
        terms_flat = terms.view(self._batch_size * num_tasks, -1).flatten()

        with torch.no_grad():
            q_target = rewards_flat + ((1. - terms_flat) * self._discount) * target_q_values

        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)
        qf_loss = qf1_loss + qf2_loss

        # Optimize Q network and context encoder
        if self._encoder_common_net:
            self.context_optimizer.zero_grad()
        self.query_optimizer.zero_grad()
        if self._new_weight_update:
            self.contrastive_weight_optimizer.zero_grad()

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        if not self._use_q_loss:
            if self._encoder_common_net:
                self.context_optimizer.zero_grad()
            self.query_optimizer.zero_grad()
            if self._new_weight_update:
                self.contrastive_weight_optimizer.zero_grad()

        if self._use_kl_loss and self._use_information_bottleneck:
            # KL constraint on z if probabilistic
            kl_div = self._policy.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward()

        contrastive_loss.backward()
        self.query_optimizer.step()
        if self._encoder_common_net:
            self.context_optimizer.step()

        if self._new_weight_update:
            self.contrastive_weight_optimizer.step()

        query_net = self._context_encoder.get_query_net()
        key_net = self._context_encoder.get_key_net()
        with torch.no_grad():
            if not self._new_weight_update:
                self._contrastive_weight -= self._context_lr * self._contrastive_weight.grad
                self._contrastive_weight.grad.zero_()
            # update key net with 0.05 of query net
            for target_param, param in zip(key_net.parameters(), query_net.parameters()):
                target_param.data.copy_(self._soft_target_tau * param.data +
                                        target_param.data * (1 - self._soft_target_tau))

        # ===== Actor Objective =====
        policy_outputs, task_z = self._policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi, pre_tanh = policy_outputs
        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([batch_obs, new_actions], dim=1), task_z.detach())
        q2 = self._qf2(torch.cat([batch_obs, new_actions], dim=1), task_z.detach())
        min_q = torch.min(q1, q2)
        # optimize policy
        policy_loss = ((alpha * log_pi) - min_q.flatten()).mean()
        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * ((pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = (mean_reg_loss + std_reg_loss + pre_activation_reg_loss)
        policy_loss += policy_reg_loss

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ===== Temperature Objective =====
        alpha_loss_cpu = np.array([0])
        if self._use_automatic_entropy_tuning:
            alpha = (self._get_log_alpha(indices)).exp()
            alpha_loss = (-alpha * (log_pi.detach() + self._target_entropy)).mean()
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            alpha_avg_cpu = np.average(alpha.detach().cpu().numpy())
            alpha_loss_cpu = alpha_loss.detach().cpu().numpy()

        # ===== Update Target Network =====
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._soft_target_tau) + param.data * self._soft_target_tau)

        qf_loss_cpu = qf_loss.detach().cpu().numpy()
        policy_loss_cpu = policy_loss.detach().cpu().numpy()
        contrastive_loss_cpu = contrastive_loss.detach().cpu().numpy()
        return policy_loss_cpu, qf_loss_cpu, contrastive_loss_cpu, alpha_loss_cpu, alpha_avg_cpu


    def _obtain_samples(self,
                        runner,
                        itr,
                        num_samples,
                        update_posterior_rate,
                        add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            runner (LocalRunner): LocalRunner.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often (in trajectories) to infer
                posterior of policy.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder
                buffer.

        """
        self._policy.reset_belief()
        total_samples = 0

        if update_posterior_rate != np.inf:
            num_samples_per_batch = (update_posterior_rate * self.max_path_length)
        else:
            num_samples_per_batch = num_samples

        while total_samples < num_samples:
            paths = runner.obtain_samples(itr, num_samples_per_batch,
                                          self._policy,
                                          self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])

            for path in paths:
                p = {
                    'observations': path['observations'],
                    'actions': path['actions'],
                    'rewards': path['rewards'].reshape(-1, 1),
                    'next_observations': path['next_observations'],
                    'dones': path['dones'].reshape(-1, 1)
                }
                self._replay_buffers[self._task_idx].add_path(p)

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            if update_posterior_rate != np.inf:
                context = self._sample_context(self._task_idx)
                self._policy.infer_posterior(context)

    def _sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, O^*)` where X
                is the number of tasks. N is batch size.
            torch.Tensor: Actions, with shape :math:`(X, N, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, 1)`.
            torch.Tensor: Next obervations, with shape :math:`(X, N, O^*)`.
            torch.Tensor: Dones, with shape :math:`(X, N, 1)`.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                d = batch['dones'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                d = np.vstack((d, batch['dones'][np.newaxis]))

        o = torch.as_tensor(o, device=global_device()).float()
        a = torch.as_tensor(a, device=global_device()).float()
        r = torch.as_tensor(r, device=global_device()).float()
        no = torch.as_tensor(no, device=global_device()).float()
        d = torch.as_tensor(d, device=global_device()).float()

        return o, a, r, no, d

    def _sample_contrastive_pairs(self, indices, num_aug=2):
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        path_augs = []
        for j in range(num_aug):
            initialized = False
            for idx in indices:
                path = self._context_replay_buffers[idx].sample_path()
                batch_aug = self.augment_path(path, self._embedding_batch_size, in_sequence=self._embedding_batch_in_sequence) # conduct random path augmentations
                o = batch_aug['observations']
                a = batch_aug['actions']
                r = batch_aug['rewards']
                context = np.hstack((np.hstack((o, a)), r))
                if self._use_next_obs_in_context:
                    context = np.hstack((context, batch_aug['next_observations']))

                if not initialized:
                    final_context = context[np.newaxis]
                    initialized = True
                else:
                    final_context = np.vstack((final_context, context[np.newaxis]))

            final_context = torch.as_tensor(final_context,
                                            device=global_device()).float()
            if len(indices) == 1:
                final_context = final_context.unsqueeze(0)

            path_augs.append(final_context)

        return path_augs

    def sample_path_batch(self, indices):
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        path_batch = {}
        for idx in indices:
            path_batch[idx] = self._context_replay_buffers[idx].sample_path()
        return path_batch

    def _sample_context(self, indices):
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Context data, with shape :math:`(X, N, C)`. X is the
                number of tasks. N is batch size. C is the combined size of
                observation, action, reward, and next observation if next
                observation is used in context. Otherwise, C is the combined
                size of observation, action, and reward.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            path = self._context_replay_buffers[idx].sample_path()
            batch = self.augment_path(path, self._embedding_batch_size, in_sequence=self._embedding_batch_in_sequence)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            if self._use_next_obs_in_context:
                context = np.hstack((context, batch['next_observations']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                new_context = context[np.newaxis]
                if final_context.shape[1] != new_context.shape[1]:
                    min_length = min(final_context.shape[1],
                                     new_context.shape[1])
                    new_context = new_context[:, :min_length, :]
                    final_context = np.vstack(
                        (final_context[:, :min_length, :], new_context))
                final_context = np.vstack((final_context, new_context))

        final_context = torch.as_tensor(final_context,
                                        device=global_device()).float()
        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    @property
    def policy(self):
        """Return all the policy within the model.

        Returns:
            garage.torch.policies.Policy: Policy within the model.

        """
        return self._policy

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self._policy.networks + [self._policy, self._qf1, self._qf2,
                                        self._target_qf1, self._target_qf2, self._contrastive_weight]

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            garage.Policy: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """
        return self._policy

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (garage.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (garage.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            garage.Policy: A policy adapted to the task represented by the
                exploration_trajectories.

        """
        total_steps = sum(exploration_trajectories.lengths)
        o = exploration_trajectories.observations
        a = exploration_trajectories.actions
        r = exploration_trajectories.rewards.reshape(total_steps, 1)
        no = exploration_trajectories.next_observations
        ctxt = np.hstack((o, a, r))
        if self._use_next_obs_in_context:
            ctxt = np.hstack((ctxt, no))
        ctxt = ctxt.reshape(1, total_steps, -1)
        context = torch.as_tensor(ctxt, device=global_device()).float()
        self._policy.infer_posterior(context)

        return self._policy

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)
        if self._use_automatic_entropy_tuning:
            self._log_alpha = self._log_alpha.to(device).requires_grad_()
        else:
            self._log_alpha = self._log_alpha.to(device)

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
        aug_obs = akro.Box(low=-1,
                           high=1,
                           shape=(obs_dim + latent_dim, ),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1,
                           high=1,
                           shape=(action_dim, ),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module, use_information_bottleneck=False):
        """Get environment specs of encoder with latent dimension.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specs.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.

        Returns:
            garage.envs.InOutSpec: Module environment specs with latent
                dimension.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = obs_dim + action_dim + 1
            out_dim = latent_dim
            if use_information_bottleneck:
                out_dim = out_dim * 2

        elif module == 'vf':
            in_dim = obs_dim
            out_dim = latent_dim
        in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim, ),
                             dtype=np.float32)
        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)

        return spec

    def _get_log_alpha(self, indices):
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
        if self._single_alpha:
            return self._log_alpha
        else:
            log_alpha = self._log_alpha
            one_hots = np.zeros((len(indices) * self._batch_size, self._num_train_tasks), dtype=np.float32)
            for i in range(len(indices)):
                one_hots[self._batch_size * i: self._batch_size * (i + 1), indices[i]] = 1
            one_hots = torch.as_tensor(one_hots, device=global_device())
            ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
            return ret

class CURLWorker(DefaultWorker):
    """A worker class used in sampling for CURL.

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
            if self._accum_context:
                s = TimeStep(env_spec=self.env,
                             observation=self._prev_obs,
                             next_observation=next_o,
                             action=a,
                             reward=float(r),
                             terminal=d,
                             env_info=env_info,
                             agent_info=agent_info)
                self.agent.update_context(s)
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
        self.agent.sample_from_belief()
        self.start_rollout()
        while not self.step_rollout():
            pass
        self._agent_infos['context'] = [self.agent.z.detach().cpu().numpy()
                                        ] * self._path_length
        return self.collect_rollout()
