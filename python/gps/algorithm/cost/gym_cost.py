""" This file defines a cost sum of arbitrary other costs. """
import copy

from gps.algorithm.cost.config import COST_SUM
from gps.algorithm.cost.cost import Cost
import numpy as np


class gym_cost(Cost):
    """ A wrapper cost function that adds other cost functions. """

    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        self._config = config
        Cost.__init__(self, config)
        self._build_env()

    def _build_env(self):
        if self._config['env_name'] in ['gym_cheetah', 'gym_ant', 'gym_hopper',
                                        'gym_swimmer', 'gym_walker2d']:
            from mbbl.env.gym_env import walker
            self._env = walker.env(self._config['env_name'], 1234,
                                   misc_info={})
        elif self._config['env_name'] in ['gym_reacher']:
            from mbbl.env.gym_env import reacher
            self._env = reacher.env(self._config['env_name'], 1234,
                                    misc_info={})

        elif self._config['env_name'] in ['gym_pendulum']:
            from mbbl.env.gym_env import pendulum
            self._env = pendulum.env(self._config['env_name'], 1234,
                                     misc_info={})
        elif self._config['env_name'] in ['gym_invertedPendulum']:
            from mbbl.env.gym_env import invertedPendulum
            self._env = invertedPendulum.env(self._config['env_name'], 1234,
                                             misc_info={})
        elif self._config['env_name'] in ['gym_acrobot']:
            from mbbl.env.gym_env import acrobot
            self._env = acrobot.env(self._config['env_name'], 1234,
                                    misc_info={})
        elif self._config['env_name'] in ['gym_mountain']:
            from mbbl.env.gym_env import mountain_car
            self._env = mountain_car.env(self._config['env_name'], 1234,
                                         misc_info={})
        elif self._config['env_name'] in ['gym_cartpole']:
            from mbbl.env.gym_env import cartpole
            self._env = cartpole.env(self._config['env_name'], 1234,
                                     misc_info={})
        elif self._config['env_name'] in ['gym_petsCheetah', 'gym_petsPusher', 'gym_petsReacher']:
            from mbbl.env.gym_env import pets
            self._env = pets.env(self._config['env_name'], 1234,
                                 misc_info={})
        elif self._config['env_name'] in ['gym_cheetahO01', 'gym_cheetahO001',
                                          'gym_cheetahA01', 'gym_cheetahA003']:
            from mbbl.env.gym_env import noise_gym_cheetah
            self._env = noise_gym_cheetah.env(self._config['env_name'], 1234,
                                              misc_info={})
        elif self._config['env_name'] in ['gym_pendulumO01', 'gym_pendulumO001']:
            from mbbl.env.gym_env import noise_gym_pendulum
            self._env = noise_gym_pendulum.env(self._config['env_name'], 1234,
                                               misc_info={})
        elif self._config['env_name'] in ['gym_cartpoleO01', 'gym_cartpoleO001']:
            from mbbl.env.gym_env import noise_gym_cartpole
            self._env = noise_gym_cartpole.env(self._config['env_name'], 1234,
                                               misc_info={})
        elif self._config['env_name'] in ['gym_fwalker2d', 'gym_fhopper', 'gym_fant']:
            from mbbl.env.gym_env import fixed_walker
            self._env = fixed_walker.env(self._config['env_name'], 1234,
                                         misc_info={})
        elif self._config['env_name'] in ['gym_fswimmer']:
            from mbbl.env.gym_env import fixed_swimmer
            self._env = fixed_swimmer.env(self._config['env_name'], 1234,
                                          misc_info={})
        elif self._config['env_name'] in ['gym_nostopslimhumanoid']:
            from mbbl.env.gym_env import humanoid
            self._env = humanoid.env(self._config['env_name'], 1234,
                                     misc_info={})
        else:
            raise NotImplementedError

    def eval(self, sample):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """
        T, Du, Dx = sample.T, sample.dU, sample.dX  # the dimentions
        # l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)

        data_dict = {'start_state': sample.get_X(), 'action': sample.get_U(),
                     'next_state': sample.get_X()}
        l_reward = np.zeros(T)

        # the constant term l
        for i_pos in range(T):
            l_reward[i_pos] = -self._env.reward({key: data_dict[key][i_pos]
                                                 for key in data_dict})
        lx = -self._env.reward_derivative(data_dict, 'state')
        lu = -self._env.reward_derivative(data_dict, 'action')
        lxx = -self._env.reward_derivative(data_dict, 'state-state')
        luu = -self._env.reward_derivative(data_dict, 'action-action')
        lux = -self._env.reward_derivative(data_dict, 'action-state')

        return l_reward, lx, lu, lxx, luu, lux
