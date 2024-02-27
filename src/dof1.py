import gymnasium as gym
from scipy.linalg import expm
from operator import matmul
from gym import spaces
from gym.utils import seeding
import numpy as np

class DynamicEnv(gym.Env):
    def __init__(self, time_step= 0.01, system_parameter= {'M':1, 'K':100, 'C':0.4}, \
                 space_bounds = {'action_space_lowerbound':-np.array([10.]), 'action_space_upperbound':np.array([10.]), 'observation_space_lowerbound':-np.array([10.,10.,10]), 'observation_space_upperbound': np.array([10.,10.,10])},\
                reward_weights = {'displacement_weights':np.array([1.]), 'velocity_weights':np.array([0.]), 'acceleration_weights':np.array([1/100])}):
        self.observation_space_lowerbound = space_bounds["observation_space_lowerbound"] 
        self.observation_space_upperbound = space_bounds["observation_space_upperbound"]
        self.action_space_lowerbound = space_bounds["action_space_lowerbound"]
        self.action_space_upperbound = space_bounds["action_space_upperbound"]
        self.action_space = spaces.Box(low=self.action_space_lowerbound, high=self.action_space_upperbound, dtype=np.float32)
        self.observation_space = spaces.Box(self.observation_space_lowerbound, self.observation_space_upperbound, dtype=np.float32)
        self.M = system_parameter["M"]
        self.K = system_parameter["K"]
        self.C = system_parameter["C"]
        self.dt = time_step
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, env_state, ground_acceleration):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        M = self.M
        K = self.K
        C = self.C
        dt = self.dt

        #action = np.array([0])
        state = self.state
        state = np.concatenate([[env_state],[state]],axis=1)[0]
 
        input = np.concatenate([[state[0:2]], [action]], axis=1)

        # Dynamic model
        A_c = np.array([[0.,1.],[-K/M,-C/M]])
        B_c = np.array([[0.],[1/1.0]])
        G_c = np.array([[0.],[1]])

        A_d = expm(A_c*dt)
        B_d = matmul(matmul(np.linalg.inv(A_c),(A_d - np.eye(2) )),B_c)  
        G_d = matmul(matmul(np.linalg.inv(A_c),(A_d - np.eye(2) )),G_c) 

        C_d = A_c[1,:]
        D_d = B_c[1,:]

        x_pre = matmul(A_d,np.transpose(input[:,:2][0])) + matmul(B_d,action) + matmul(G_d,np.array([ground_acceleration]))
        y_pre = matmul(C_d,np.transpose(x_pre) ) + D_d*action[0]

        
        reward =  -(abs(x_pre[0])  + abs(y_pre[0])/100) #+ abs(action[0])/100
        env_state = np.transpose(x_pre)
        self.state = y_pre # plus ground acceleration plus action ( read data file)
        return self.state, reward, env_state, y_pre

    def render(self):
        pass

    def reset(self):
        self.state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state)