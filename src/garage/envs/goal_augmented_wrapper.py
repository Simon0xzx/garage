"""Wrapper for appending one-hot task encodings to individual task envs.

See `~TaskOnehotWrapper.wrap_env_list` for the main way of using this module.

"""
import akro
import numpy as np
import pickle

from garage import EnvSpec, EnvStep, Wrapper


class GoalAugmentedWrapper(Wrapper):
    """Append a goal representation to an environment.

    See GoalAugmentedWrapper.wrap_env_list for the recommended way of creating
    this class.

    Args:
        env (Environment): The environment to wrap.
        task (NamedTuple)

    """

    def __init__(self, env, task):
        super().__init__(env)
        task_data = pickle.loads(task.data)
        goal_space = task_data['env_cls']()._random_reset_space
        self._task_embedding = task_data['rand_vec']
        env_lb = self._env.observation_space.low
        env_ub = self._env.observation_space.high

        self._observation_space = akro.Box(
            np.concatenate([env_lb, goal_space.low]),
            np.concatenate([env_ub, goal_space.high]))
        self._spec = EnvSpec(
            action_space=self.action_space,
            observation_space=self.observation_space,
            max_episode_length=self._env.spec.max_episode_length)

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """Return the environment specification.

        Returns:
            EnvSpec: The envionrment specification.

        """
        return self._spec

    def reset(self):
        """Sample new task and call reset on new task env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        first_obs, episode_info = self._env.reset()
        first_obs = self._obs_with_task_embedding(first_obs)

        return first_obs, episode_info

    def step(self, action):
        """Environment step for the active task env.

        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.

        Returns:
            EnvStep: The environment step resulting from the action.

        """
        es = self._env.step(action)
        obs = es.observation
        emb_obs = self._obs_with_task_embedding(obs)

        env_info = es.env_info

        env_info['task_embedding'] = self._task_embedding

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=es.reward,
                       observation=emb_obs,
                       env_info=env_info,
                       step_type=es.step_type)

    def _obs_with_task_embedding(self, obs):
        """Concatenate observation and task embedding. 

        Args:
            obs (numpy.ndarray): observation

        Returns:
            numpy.ndarray: observation + task embedding. 

        """
        return np.concatenate([obs, self._task_embedding])

    @classmethod
    def wrap_env_list(cls, envs):
        """Wrap a list of environments, giving each environment a one-hot.

        This is the primary way of constructing instances of this class.
        It's mostly useful when training multi-task algorithms using a
        multi-task aware sampler.

        For example:
        '''
        .. code-block:: python

            envs = get_mt10_envs()
            wrapped = TaskOnehotWrapper.wrap_env_list(envs)
            sampler = trainer.make_sampler(LocalSampler, env=wrapped)
        '''

        Args:
            envs (list[Environment]): List of environments to wrap. Note
            that the
                order these environments are passed in determines the value of
                their one-hot encoding. It is essential that this list is
                always in the same order, or the resulting encodings will be
                inconsistent.

        Returns:
            list[TaskOnehotWrapper]: The wrapped environments.

        """
        # n_total_tasks = len(envs)
        # wrapped = []
        # for i, env in enumerate(envs):
        #     wrapped.append(cls(env, task_index=i, n_total_tasks=n_total_tasks))
        # return wrapped
        raise NotImplementedError

    @classmethod
    def wrap_env_cons_list(cls, env_cons):
        """Wrap a list of environment constructors, giving each a one-hot.

        This function is useful if you want to avoid constructing any
        environments in the main experiment process, and are using a multi-task
        aware remote sampler (i.e. `~RaySampler`).

        For example:
        '''
        .. code-block:: python

            env_constructors = get_mt10_env_cons()
            wrapped = TaskOnehotWrapper.wrap_env_cons_list(env_constructors)
            env_updates = [NewEnvUpdate(wrapped_con)
                           for wrapped_con in wrapped]
            sampler = trainer.make_sampler(RaySampler, env=env_updates)
        '''


        Args:
            env_cons (list[Callable[Environment]]): List of environment
            constructor
                to wrap. Note that the order these constructors are passed in
                determines the value of their one-hot encoding. It is essential
                that this list is always in the same order, or the resulting
                encodings will be inconsistent.

        Returns:
            list[Callable[TaskOnehotWrapper]]: The wrapped environments.

        """
        # n_total_tasks = len(env_cons)
        # wrapped = []
        # for i, con in enumerate(env_cons):
        #     # Manually capture this value of i by introducing a new scope.
        #     wrapped.append(lambda i=i, con=con: cls(
        #         con(), task_index=i, n_total_tasks=n_total_tasks))
        # return wrapped
        raise NotImplementedError
