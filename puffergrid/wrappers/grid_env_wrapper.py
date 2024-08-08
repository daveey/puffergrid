from pdb import set_trace as T
from types import SimpleNamespace
import numpy as np

import pettingzoo
import gymnasium as gym

from env.griddly.builder import action
from env.mettagrid import render
import pufferlib
from pufferlib.environment import PufferEnv

class PufferGridEnv(PufferEnv):
    def __init__(self, c_env) -> None:
        super().__init__()
        self._c_env = c_env

        (o, tm, tr, re) = self._c_env.get_buffers()
        self._buffers = SimpleNamespace(
            observations=o,
            terminals=tm,
            truncations=tr,
            rewards=re
        )

    @property
    def observation_space(self):
        return self._c_env.observation_space

    @property
    def action_space(self):
        return self._c_env.action_space

    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        self._c_env.set_buffers(
            self._buffers.observations,
            self._buffers.terminals,
            self._buffers.terminals,
            self._buffers.rewards)

        self._c_env.reset()
        return self._buffers.observations, {}

    def step(self, actions):
        self._c_env.step(actions)

        infos = {}
        # if self.current_timestep >= self._max_timesteps:
        #     infos = {
        #         "episode_return": self._episode_rewards.mean(),
        #         "episode_length": self.current_timestep,
        #         "episode_stats": self._c_env.stats()
        #     }
        return (self._buffers.observations,
                self._buffers.rewards,
                self._buffers.terminals,
                self._buffers.truncations,
                infos)

    @property
    def current_timestep(self):
        return self._c_env.current_timestep()

    @property
    def unwrapped(self):
        return self

    @property
    def player_count(self):
        return self._num_agents

    @property
    def grid_features(self):
        return self._c_env.grid_features()

    @property
    def global_features(self):
        return self._c_env.global_features()
