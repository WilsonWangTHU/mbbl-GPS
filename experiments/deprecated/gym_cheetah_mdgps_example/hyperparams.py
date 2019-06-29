""" @brief: The hyperparams for the halfcheetah
"""
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.gym_agent import AgentMuJoCo
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.gym_cost import gym_cost
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython

from gps.algorithm.policy_opt.tf_model_example import tf_network
# from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.config import generate_experiment_info
from gps.agent.gym_env_util import get_x0

""" @brief:
    hyperparams to consider:
        Number of samples
"""

env_name = 'gym_cheetah'
num_samples = 5

SENSOR_DIMS = {
    "observation": 17,
    'action': 6,
}

# PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
PR2_GAINS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/' + env_name + '_mdgps_example/'

common = {
    'experiment_name': env_name + '_experiments' + '_' +
    datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': get_x0(env_name),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([-0.08, -0.08, 0])], [np.array([-0.08, 0.08, 0])],
                        [np.array([0.08, 0.08, 0])], [np.array([0.08, -0.08, 0])]],
    'sensor_dims': SENSOR_DIMS,
    'state_include': ['observation'],
    'obs_include': ['observation'],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),

    # the real config files
    'T': 1000,
    'env_name': env_name
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 200,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS['action']),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['cost'] = {
    'type': gym_cost,
    'env_name': env_name
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': ['observation'],
        'sensor_dims': SENSOR_DIMS,
    },
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 3000,
    'network_model': tf_network
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'gui_on': False,
    'iterations': algorithm['iterations'],
    'num_samples': num_samples,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
