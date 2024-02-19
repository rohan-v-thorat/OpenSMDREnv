import gymnasium as gym
from scipy.linalg import expm
from operator import matmul
from gym import spaces
from gym.utils import seeding
import numpy as np

class DynamicEnv(gym.Env):
    def __init__(self,action_space_lowerbound=-np.array([10.]),action_space_upperbound=np.array([10.]),observation_space_lowerbound=-np.array([10.,10.,10]),observation_space_upperbound = np.array([10.,10.,10])):
        self.observation_space_lowerbound = observation_space_lowerbound
        self.observation_space_upperbound = observation_space_upperbound
        self.action_space_lowerbound = action_space_lowerbound
        self.action_space_upperbound = action_space_upperbound
        self.action_space = spaces.Box(low=action_space_lowerbound, high=action_space_upperbound, dtype=np.float32)
        self.observation_space = spaces.Box(-observation_space_lowerbound, observation_space_upperbound, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, X_):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        m = 1
        k = 100
        c = 0.4
        dt = 0.01
        
        #action = np.array([0])
        state = self.state
        state = np.concatenate([[X_],[state]],axis=1)[0]
 
        input = np.concatenate([[state[0:2]], [action]], axis=1)

        # Dynamic model
        A_c = np.array([[0.,1.],[-k/m,-c/m]])
        B_c = np.array([[0.],[1/1.0]])
        A_d = expm(A_c*dt)
        try:
            B_d = matmul(matmul(np.linalg.inv(A_c),(A_d - np.eye(2) )),B_c)    # why can't we use non-linear equation
        except:
            B_d = B_c*dt

        C_d = np.array([-10**2,-2*10*0.02])
        D_d = np.array([1/1.0])
        a = matmul(B_d,action)
        b = matmul(A_d,np.transpose(input[:,:2]))
        c = np.transpose(input[:,:2])
        d = input[:,:2]
        ground_acc = state[3]
        ground_force = np.array([1*ground_acc])
        x_pre = matmul(A_d,np.transpose(input[:,:2][0])) + matmul(B_d,action + ground_force)
        y_pre = matmul(C_d,np.transpose(x_pre) ) + D_d*action[0]

        
        reward =  -(abs(x_pre[0])  + abs(y_pre[0])/100) #+ abs(action[0])/100
        X_ = np.transpose(x_pre)
        self.state = y_pre # plus ground acceleration plus action ( read data file)
        return self.state, reward, X_, y_pre

    def render(self):
        pass

    def reset(self):
        self.state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state)