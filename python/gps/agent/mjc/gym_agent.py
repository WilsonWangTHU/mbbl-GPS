""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import NOISE

from gps.sample.gym_sample import Gym_sample
from mbbl.env.gym_env import walker
from mbbl.env.gym_env import reacher
from mbbl.env.gym_env import pendulum
from mbbl.env.gym_env import invertedPendulum
from mbbl.env.gym_env import acrobot
from mbbl.env.gym_env import mountain_car
from mbbl.env.gym_env import cartpole
from mbbl.env.gym_env import pets
from mbbl.env.gym_env import noise_gym_cheetah
from mbbl.env.gym_env import noise_gym_pendulum
from mbbl.env.gym_env import noise_gym_cartpole
from mbbl.env.gym_env import fixed_swimmer
from mbbl.env.gym_env import fixed_walker
from mbbl.env.gym_env import humanoid


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()
        self._setup_world(hyperparams['env_name'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        pass

    def _setup_world(self, env_name):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        if env_name in ['gym_reacher']:
            self._env = \
                reacher.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_cheetah', 'gym_walker2d',
                          'gym_hopper', 'gym_swimmer', 'gym_ant']:
            self._env = \
                walker.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_nostopslimhumanoid']:
            self._env = \
                humanoid.env(env_name, 1234, misc_info={'reset_type': 'gym'})

        elif env_name in ['gym_pendulum']:
            self._env = \
                pendulum.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_invertedPendulum']:
            self._env = invertedPendulum.env(env_name, 1234,
                                             misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_acrobot']:
            self._env = \
                acrobot.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_mountain']:
            self._env = mountain_car.env(env_name, 1234,
                                         misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_cartpole']:
            self._env = \
                cartpole.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_petsCheetah', 'gym_petsReacher', 'gym_petsPusher']:
            self._env = \
                pets.env(env_name, 1234, misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_cheetahO01', 'gym_cheetahO001',
                          'gym_cheetahA01', 'gym_cheetahA003']:
            self._env = noise_gym_cheetah.env(env_name, 1234,
                                              misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_pendulumO01', 'gym_pendulumO001']:
            self._env = noise_gym_pendulum.env(env_name, 1234,
                                               misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_cartpoleO01', 'gym_cartpoleO001']:
            self._env = noise_gym_cartpole.env(env_name, 1234,
                                               misc_info={'reset_type': 'gym'})
        elif env_name in ['gym_fwalker2d', 'gym_fant', 'gym_fhopper']:
            self._env = fixed_walker.env(
                env_name, 1234,
                misc_info={'reset_type': 'gym', 'no_termination': True}
            )
        elif env_name in ['gym_fswimmer']:
            self._env = fixed_swimmer.env(
                env_name, 1234,
                misc_info={'reset_type': 'gym', 'no_termination': True}
            )
        else:
            raise NotImplementedError

        self.x0 = [np.array(self._hyperparams['x0'])
                   for _ in range(self._hyperparams['conditions'])]

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        assert condition < self._hyperparams['conditions']

        # Create new sample, populate first time step (reset the env)
        new_sample = self._init_sample(condition, feature_fn=None)
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if (t + 1) < self.T:
                new_ob, reward, done, _ = \
                    self._env.step(np.array(mj_U).flatten())
                new_sample.set('observation', new_ob, t=t + 1)
        new_sample.set('action', U)
        new_sample.set(NOISE, noise)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics
        self._world[condition].set_model(self._model[condition])
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        self._world[condition].set_data(data)
        self._world[condition].kinematics()

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        assert feature_fn is None
        sample = Gym_sample(self)

        obs = self._env.reset()
        sample.set('observation', obs, t=0)

        return sample
