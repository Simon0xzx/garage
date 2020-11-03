"""Variant of the HalfCheetahEnv with different target directions."""
import numpy as np

from garage.envs.mujoco.ant_env_meta_base import AntEnvMetaBase  # noqa: E501


class AntDirEnv(AntEnvMetaBase):
    """Ant environment with target direction, as described in [1].

    The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

    Args:
        task (dict or None):
            direction (float): Target direction, either -1 or 1.

    """

    def __init__(self, task=None):
        super().__init__(task or {'direction': 1.})

    def step(self, action):
        """Take one step in the environment.

        Equivalent to step in Ant, but with different rewards.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step. Always False for this environment.
                * infos (dict):
                    * reward_forward (float): Reward for moving, ignoring the
                        control cost.
                    * reward_ctrl (float): The reward for acting i.e. the
                        control cost (always negative).
                    * task_dir (float): Target direction. 1.0 for forwards,
                        -1.0 for backwards.

        """
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._task['direction']), np.sin(self._task['direction']))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        ob = self._get_obs()
        state = self.state_vector()
        termination = not (np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0)
        done = False
        info = dict(reward_forward=forward_reward,
                    reward_ctrl=-ctrl_cost,
                    reward_contact=-contact_cost,
                    reward_survive=survive_reward,
                    torso_velocity=torso_velocity,
                    task_dir=self._task['direction'],
                    termination = termination)
        return ob, reward, done, info

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "direction", mapping to -1
                or 1.

        """
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'direction': d} for d in directions]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "direction", mapping to -1 or 1).

        """
        self._task = task