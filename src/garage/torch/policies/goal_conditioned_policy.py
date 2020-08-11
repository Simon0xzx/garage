"""A policy used in training meta reinforcement learning algorithms.

It is used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement
Learning). The paper on PEARL can be found at https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from garage.torch import global_device, product_of_gaussians


# pylint: disable=attribute-defined-outside-init
# pylint does not recognize attributes initialized as buffers in constructor
class GoalConditionedPolicy(nn.Module):
    def __init__(self, latent_dim, policy):
        super().__init__()
        self._latent_dim = latent_dim
        self._policy = policy

    def update_context(self, timestep):

        o = torch.as_tensor(timestep.observation[None, None, ...],
                            device=global_device()).float()
        a = torch.as_tensor(timestep.action[None, None, ...],
                            device=global_device()).float()
        r = torch.as_tensor(np.array([timestep.reward])[None, None, ...],
                            device=global_device()).float()
        no = torch.as_tensor(timestep.next_observation[None, None, ...],
                             device=global_device()).float()

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        """
            Context in this case is the goal vector of the active env
        """
        self.z = torch.as_tensor(context, device=global_device()).float()

    # pylint: disable=arguments-differ
    def forward(self, obs, context):
        self.infer_posterior(context)
        task_z = self.z

        # task, batch
        b, _ = obs.size()
        obs = obs.view(b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in task_z], dim=1)

        # run policy, get log probs and new actions
        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        dist = self._policy(obs_z)[0]

        return dist, task_z

    def reset_belief(self, env, num_tasks=1):
        r"""Reset :math:`q(z \| c)` to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        """

        self.z = torch.as_tensor(env._task['info'][None], device=global_device()).float()


    def reset(self):
        pass

    def get_action(self, obs):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values, with shape :math:`(1, O)`.
                O is the size of the flattened observation space.

        Returns:
            torch.Tensor: Output action value, with shape :math:`(1, A)`.
                A is the size of the flattened action space.
            dict:
                * np.ndarray[float]: Mean of the distribution.
                * np.ndarray[float]: Standard deviation of logarithmic values
                    of the distribution.

        """
        z = self.z
        obs = torch.as_tensor(obs[None], device=global_device()).float()
        obs_in = torch.cat([obs, z], dim=1)
        action, info = self._policy.get_action(obs_in)
        action = np.squeeze(action, axis=0)
        info['mean'] = np.squeeze(info['mean'], axis=0)
        return action, info

    def compute_kl_div(self):
        r"""Compute :math:`KL(q(z|c) \| p(z))`.

        Returns:
            float: :math:`KL(q(z|c) \| p(z))`.

        """
        prior = torch.distributions.Normal(
            torch.zeros(self._latent_dim).to(global_device()),
            torch.ones(self._latent_dim).to(global_device()))
        posteriors = [
            torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
                torch.unbind(self.z_means), torch.unbind(self.z_vars))
        ]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior)
            for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._policy]

    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: Context values, with shape :math:`(X, N, C)`.
                X is the number of tasks. N is batch size. C is the combined
                size of observation, action, reward, and next observation if
                next observation is used in context. Otherwise, C is the
                combined size of observation, action, and reward.

        """
        return self._context
