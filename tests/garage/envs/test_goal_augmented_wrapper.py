"""Tests for garage.envs.GoalAugmentedWrapper"""

import metaworld
import numpy as np
import pickle
import random

from garage.envs import PointEnv, GoalAugmentedWrapper, GymEnv, TaskNameWrapper


class TestSingleWrappedEnv:

    def setup_method(self):
        ml1 = metaworld.ML1('reach-v1')
        self.task = random.choice(ml1.train_tasks)
        env = ml1.train_classes['reach-v1']()
        env = GymEnv(env, max_episode_length=env.max_path_length)
        self.env = TaskNameWrapper(env, task_name=self.task.env_name)
        self.env.set_task(self.task)

        obs, _ = self.env.reset()

        self.base_len = len(obs)
        self.goal_dim = 6
        self.task_embedding = pickle.loads(self.task.data)['rand_vec']
        self.wrapped = GoalAugmentedWrapper(self.env, self.task)

    def test_produces_correct_goal(self):
        obs, _ = self.wrapped.reset()
        assert len(obs) == self.base_len + self.goal_dim
        assert (obs[-self.goal_dim:] == self.task_embedding).all()

    def test_spec_obs_space(self):
        obs, _ = self.wrapped.reset()
        assert self.wrapped.observation_space.contains(obs)
        assert self.wrapped.spec.observation_space.contains(obs)
        assert (self.wrapped.spec.observation_space ==
                self.wrapped.observation_space)

    def test_visualization(self):
        assert self.env.render_modes == self.wrapped.render_modes
        mode = self.env.render_modes[0]
        assert self.env.render(mode) == self.wrapped.render(mode)
