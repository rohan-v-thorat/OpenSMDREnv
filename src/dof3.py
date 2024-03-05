import gymnasium as gym
from scipy.linalg import expm
from operator import matmul
from gym import spaces
from gym.utils import seeding
import numpy as np

class DynamicEnv(gym.Env):
    def __init__(self, time_step= 0.01, system_parameter= {'M':np.diag([1,1,1]),\
                    'K':np.array([[100, -100, 0],[-100,200,-100],[0,-100,200]]),\
                    'C':np.array([[0.4, -0.4, 0],[-0.4,0.8,-0.4],[0,-0.4,0.8]])}, \
                 space_bounds = {'action_space_lowerbound':-10*np.ones(3), 'action_space_upperbound':10*np.ones(3),\
                                  'observation_space_lowerbound':-10*np.ones(7), 'observation_space_upperbound': 10*np.ones(7)},\
                reward_weights = {'displacement_weights':np.ones(3), 'velocity_weights':np.ones(3), 'acceleration_weights':np.ones(3)}):
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
        self.reward_weights = reward_weights
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, env_state, ground_acceleration):
        action = np.clip(action, self.action_space.low, self.action_space.high) # might be of no use, can be removed
        M = self.M
        K = self.K
        C = self.C
        dt = self.dt
        w1 = self.reward_weights["displacement_weights"]
        w2 = self.reward_weights["velocity_weights"]
        w3 = self.reward_weights["acceleration_weights"]

        # continuous system 
        A_c = np.array([[0.,1.],[-K/M,-C/M]])
        B_c = np.array([[0.],[1/1.0]])
        G_c = np.array([[0.],[1]])

        # discrete system
        A_d = expm(A_c*dt)
        B_d = matmul(matmul(np.linalg.inv(A_c),(A_d - np.eye(2) )),B_c)  
        G_d = matmul(matmul(np.linalg.inv(A_c),(A_d - np.eye(2) )),G_c) 
        C_d = A_c[1,:]
        D_d = B_c[1,:]

        # process equation / discrete state-space model
        env_state = matmul(A_d,np.transpose(env_state)) + matmul(B_d,action) + matmul(G_d,np.array([ground_acceleration]))

        # acceleration calculated based on the state of the environment
        env_acceleration = matmul(C_d,np.transpose(env_state) ) + D_d*action[0]
        
        # assignment of the displacement, velocity and acceleration values
        x = env_state[0:1]
        xdot = env_state[1:2]
        xddot = env_acceleration

        # calculation of the reward
        reward =  -((matmul(w1,abs(x)))+(matmul(w2,abs(xdot)))+(matmul(w3,abs(xddot))))

        env_state = np.transpose(env_state)
        return reward, env_state, env_acceleration

    def render(self):
        pass

    def reset(self):
        self.agent_state = self.np_random.uniform(low=self.observation_space_lowerbound/2, high=self.observation_space_upperbound/2, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state)