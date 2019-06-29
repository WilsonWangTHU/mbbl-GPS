""" This file defines utility classes and functions for agents. """
import numpy as np


def get_x0(env_name):
    if env_name in ['gym_walker2d', 'gym_hopper', 'gym_swimmer',
                    'gym_cheetah', 'gym_ant']:
        from mbbl.env.gym_env.walker import env
    elif env_name in ['gym_nostopslimhumanoid']:
        from mbbl.env.gym_env.humanoid import env
    elif env_name in ['gym_reacher']:
        from mbbl.env.gym_env.reacher import env

    elif env_name in ['gym_acrobot']:
        from mbbl.env.gym_env.acrobot import env
    elif env_name in ['gym_cartpole']:
        from mbbl.env.gym_env.cartpole import env
    elif env_name in ['gym_cartpoleO01', 'gym_cartpoleO001']:
        from mbbl.env.gym_env.noise_gym_cartpole import env
    elif env_name in ['gym_mountain']:
        from mbbl.env.gym_env.mountain_car import env
    elif env_name in ['gym_pendulum']:
        from mbbl.env.gym_env.pendulum import env
    elif env_name in ['gym_pendulumO01', 'gym_pendulumO001']:
        from mbbl.env.gym_env.noise_gym_pendulum import env
    elif env_name in ['gym_invertedPendulum']:
        from mbbl.env.gym_env.invertedPendulum import env
    elif env_name in ['gym_petsReacher', 'gym_petsCheetah', 'gym_petsPusher']:
        from mbbl.env.gym_env.pets import env

    elif env_name in ['gym_fwalker2d', 'gym_fhopper', 'gym_fant']:
        from mbbl.env.gym_env.fixed_walker import env

    elif env_name in ['gym_fswimmer']:
        from mbbl.env.gym_env.fixed_swimmer import env

    elif env_name in ['gym_cheetahO01', 'gym_cheetahO001',
                      'gym_cheetahA01', 'gym_cheetahA003']:
        from mbbl.env.gym_env.noise_gym_cheetah import env
    else:
        raise NotImplementedError

    gym_env = env(env_name, 1234, misc_info={})

    # get the x0
    if env_name in ['gym_cheetah', 'gym_walker2d', 'gym_hopper',
                    'gym_cheetahO01', 'gym_cheetahO001',
                    'gym_cheetahA01', 'gym_cheetahA003']:
        x0 = np.concatenate([gym_env._env.env.init_qpos[1:],
                             gym_env._env.env.init_qvel])
    elif env_name in ['gym_nostopslimhumanoid']:
        x0 = np.concatenate([gym_env._env.env.init_qpos[2:],
                             gym_env._env.env.init_qvel])
    elif env_name in ['gym_swimmer', 'gym_ant']:
        x0 = np.concatenate([gym_env._env.env.init_qpos[2:],
                             gym_env._env.env.init_qvel])

    elif env_name in ['gym_acrobot']:
        # np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
        x0 = np.array([1, 0, 1, 0, 0, 0])
    elif env_name in ['gym_cartpole', 'gym_cartpoleO01', 'gym_cartpoleO001']:
        x0 = np.zeros(4)
    elif env_name in ['gym_mountain']:
        x0 = np.array([-0.5, 0])
    elif env_name in ['gym_pendulum', 'gym_pendulumO01', 'gym_pendulumO001']:
        #  return np.array([np.cos(theta), np.sin(theta), thetadot])
        x0 = np.array([1, 0, 0])

    elif env_name in ['gym_fswimmer']:
        x0 = np.zeros(9)
    elif env_name in ['gym_fant']:
        x0 = np.concatenate([gym_env._env.env.init_qpos[2:],
                             gym_env._env.env.init_qvel])
    elif env_name in ['gym_fwalker2d', 'gym_fhopper']:
        x0 = np.concatenate([gym_env._env.env.init_qpos[1:],
                             gym_env._env.env.init_qvel])
    elif env_name in ['gym_invertedPendulum']:
        x0 = np.concatenate([gym_env._env.env.init_qpos,
                             gym_env._env.env.init_qvel])

    elif env_name in ['gym_petsReacher']:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        qpos[-3:] = 0.0
        qvel[-3:] = 0.0
        raw_obs = np.concatenate([qpos, qvel[:-3]])
        EE_pos = np.reshape(gym_env._env.get_EE_pos(raw_obs[None]), [-1])

        x0 = np.concatenate([raw_obs, EE_pos])
    elif env_name in ['gym_petsCheetah']:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        x0 = np.concatenate([np.array([0]), qpos[1:], qvel])
    elif env_name in ['gym_petsPusher']:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        qpos[-4: -2] = [-0.25, 0.15]
        qpos[-2:] = [0.0]
        qvel[-4:] = 0
        qvel = np.copy(gym_env._env.init_qvel)
        other_obs = gym_env._env._get_obs()
        x0 = np.concatenate([qpos[:7], qvel[:7], other_obs[14:]])
    else:
        assert env_name == 'gym_reacher'
        """ @brief: note that the initialization of reacher is as following
            np.cos(theta),  the theta -> 0 / 0
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")

            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            qpos[-2:] = self.goal

            the mean of self.get_body_com("fingertip")
        """
        x0 = np.concatenate([
            [1, 1, 0, 0],
            [0, 0],  # gym_env.env.init_qpos[2:],
            gym_env._env.env.init_qvel[:2],
            gym_env._env.env.get_body_com("fingertip") -
            np.array([0, 0, 0])  # gym_env.env.get_body_com("target")
        ])

    return x0
