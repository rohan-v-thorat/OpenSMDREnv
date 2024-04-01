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
                    
        # continuous system 
        Ac = np.concatenate((np.concatenate((np.zeros((5,5)),np.eye(5)),axis=1),\
                        np.concatenate((-np.linalg.inv(M)@K, -np.linalg.inv(M)@C),axis=1)),axis=0)
        Bc = np.concatenate(([np.zeros((5,5)),np.linalg.inv(M)]),axis=0)
        Gc = np.concatenate([np.zeros((5,1)),-np.ones((5,1))])

        # discrete system
        Ad = expm(Ac*dt)

        try:
            Bd = np.linalg.inv(Ac)@(Ad - np.eye(10))@Bc  
        except:
            Bd = Bc*dt
        
        try:
            Gd = np.linalg.inv(Ac)@(Ad - np.eye(10))@Gc
        except:
            Gd = Gc*dt

        Cd = Ac[5:,:]
        Dd = Bc[5:,:]

        self.Ad = Ad
        self.Bd = Bd
        self.Gd = Gd
        self.Cd = Cd
        self.Dd = Dd

        self.seed()   
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, env_state, ground_acceleration):
        action = np.clip(action, self.action_space.low, self.action_space.high) # might be of no use, can be removed
        
        Ad = self.Ad
        Bd = self.Bd
        Gd = self.Gd
        Cd = self.Cd
        Dd = self.Dd

        w1 = self.reward_weights["displacement_weights"]
        w2 = self.reward_weights["velocity_weights"]
        w3 = self.reward_weights["acceleration_weights"]
        w4 = self.reward_weights["control_force_weights"]

        # process equation / discrete state-space model
        env_state = Ad@np.transpose(env_state) + 0*Bd@action + 0*Gd@ground_acceleration 

        # acceleration calculated based on the state of the environment
        env_acceleration = Cd@np.transpose(env_state) + 0*Dd@action
        
        # assignment of the displacement, velocity and acceleration values
        x = env_state[0:5]
        xdot = env_state[5:10]
        xddot = env_acceleration

        # calculation of the reward
        reward =  -(w1@abs(x) + w2@abs(xdot) + w3@abs(xddot) + w4@abs(action)) 

        return reward, env_state, env_acceleration

    def render(self):
        pass
    
    def reset(self):
        self.agent_state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(self.observation_size,))
        self.env_state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(self.observation_size,))
        self.steps_beyonDdone = None
        return np.array(self.agent_state), np.array(self.env_state)
