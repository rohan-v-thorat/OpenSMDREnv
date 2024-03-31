""""
Authors of this code:
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek

Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

"""

import gymnasium as gym
from scipy.linalg import expm
from gym import spaces
from gym.utils import seeding
import numpy as np
data = sio.loadmat('5dof_MKC_matrix.mat')
K = data('K5')
C = data('C5')
class DynamicEnv(gym.Env):
    def __init__(self, time_step= 0.01, system_parameter= {'M':data('M5'),\
                    'K':data('K5'),\
                    'C':data('C5')}, \
                 space_bounds = {'action_space_lowerbound':-10*np.ones(5), 'action_space_upperbound':10*np.ones(5),\
                                  'observation_space_lowerbound':-10*np.ones(11), 'observation_space_upperbound': 10*np.ones(11)},\
                reward_weights = {'displacement_weights':np.ones(5), 'velocity_weights':np.ones(5), 'acceleration_weights':np.ones(5), 'control_force_weights':np.ones(5)}):
        self.observation_space_lowerbound = space_bounds["observation_space_lowerbound"] 
        self.observation_space_upperbound = space_bounds["observation_space_upperbound"]
        self.action_space_lowerbound = space_bounds["action_space_lowerbound"]
        self.action_space_upperbound = space_bounds["action_space_upperbound"]
        self.action_space = spaces.Box(low=self.action_space_lowerbound, high=self.action_space_upperbound, dtype=np.float32)
        self.observation_space = spaces.Box(self.observation_space_lowerbound, self.observation_space_upperbound, dtype=np.float32)
        M = system_parameter["M"]
        K = system_parameter["K"]
        C = system_parameter["C"]
        dt = time_step
        self.reward_weights = reward_weights
        self.observation_size = len(self.observation_space_lowerbound)
