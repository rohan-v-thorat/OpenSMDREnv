""""
Authors of this code:
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek

"""

import gymnasium as gym
from scipy.linalg import expm
from gym import spaces
from gym.utils import seeding
import numpy as np

class DynamicEnv(gym.Env):
    def __init__(self, time_step= 0.01, system_parameter= {'M':np.array([1]), 'K':np.array([100]), 'C':np.array([0.4])}, \
                 space_bounds = {'action_space_lowerbound':-np.array([10.]), 'action_space_upperbound':np.array([10.]), 'observation_space_lowerbound':-np.array([10.,10.,10]), 'observation_space_upperbound': np.array([10.,10.,10])},\
                reward_weights = {'displacement_weights':np.array([1.]), 'velocity_weights':np.array([1.]), 'acceleration_weights':np.array([1]), 'control_force_weights':np.array([1.])}):
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
        A_c = np.array([[0.,1],[-1/M@K,-1/M@C]]) # shall we write A_c as Ac ?
        B_c = np.array([[0.],[1/M[0]]])
        G_c = np.array([[0.],[-1]])

        # discrete system
        A_d = expm(A_c*dt)

        try:
            B_d = np.linalg.inv(A_c)@(A_d - np.eye(2))@B_c  
        except:
            B_d = B_c*dt
        
        try:
            G_d = np.linalg.inv(A_c)@(A_d - np.eye(2))@G_c
        except:
            G_d = G_c*dt
            
        C_d = A_c[1:2,:]
        D_d = B_c[1:2,:]

        self.A_d = A_d
        self.B_d = B_d
        self.G_d = G_d
        self.C_d = C_d
        self.D_d = D_d

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, env_state, ground_acceleration):
        action = np.clip(action, self.action_space.low, self.action_space.high) # Just done for precaution

        A_d = self.A_d
        B_d = self.B_d
        G_d = self.G_d
        C_d = self.C_d
        D_d = self.D_d

        w1 = self.reward_weights["displacement_weights"]
        w2 = self.reward_weights["velocity_weights"]
        w3 = self.reward_weights["acceleration_weights"]
        w4 = self.reward_weights["control_force_weights"]

        # process equation / discrete state-space model
        env_state = A_d@np.transpose(env_state) + B_d@action + G_d@ground_acceleration 

        # acceleration calculated based on the state of the environment
        env_acceleration = C_d@np.transpose(env_state) + D_d@action
        
        # assignment of the displacement, velocity and acceleration values
        x = env_state[0:1]
        xdot = env_state[1:2]
        xddot = env_acceleration

        # calculation of the reward
        reward =  -(w1@abs(x) + w2@abs(xdot) + w3@abs(xddot) + w4@abs(action))  

        return reward, env_state, env_acceleration

    def render(self):
        pass

    def reset(self):
        self.agent_state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(self.observation_size,))
        self.steps_beyond_done = None
        return np.array(self.state)
