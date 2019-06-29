""" This file defines the sample class. """
import numpy as np

from gps.proto.gps_pb2 import NOISE
from gps.sample.sample import Sample


class Gym_sample(Sample):
    """ @brief:
            For the gym agent, we only store the self._data['observation'] and
            self._data['action']

            X and obs are self._data['observation']

            Does the proto serve as an important component?
    """

    def set(self, sensor_name, sensor_data, t=None):
        """ Set trajectory data for a particular sensor. """
        assert sensor_name in ['observation', 'action', NOISE]
        if t is None:
            self._data[sensor_name] = sensor_data
            self._X.fill(np.nan)  # Invalidate existing X.
            self._obs.fill(np.nan)  # Invalidate existing obs.
            self._meta.fill(np.nan)  # Invalidate existing meta data.

        else:
            if sensor_name not in self._data:
                # init the sensor data
                self._data[sensor_name] = \
                    np.empty((self.T,) + sensor_data.shape)
                self._data[sensor_name].fill(np.nan)
            self._data[sensor_name][t, :] = sensor_data
            self._X[t, :].fill(np.nan)
            self._obs[t, :].fill(np.nan)

    def get(self, sensor_name, t=None):
        """ Get trajectory data for a particular sensor. """
        assert sensor_name in ['observation', 'action', NOISE]
        return (self._data[sensor_name] if t is None
                else self._data[sensor_name][t, :])

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. """
        X = self._data['observation'] if t is None \
            else self._data['observation'][t, :]
        assert not (np.any(np.isnan(X)))
        return X

    def get_U(self, t=None):
        """ Get the action. """
        return self._data['action'] if t is None else self._data['action'][t, :]

    def get_obs(self, t=None):
        return self.get_X(t)
