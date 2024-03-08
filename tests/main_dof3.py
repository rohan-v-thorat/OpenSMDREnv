import argparse
import datetime
import gym
from gym import spaces
import sys
sys.path.insert(0,'../src')
from dof3 import DynamicEnv
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import scipy.io as sio
from scipy.io import savemat

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetahBulletEnv-v0",
                    help='name of the environment to run')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Read data
data = sio.loadmat('data_0.mat')
data = data['T']

############# user defined parameters ############
# system parameter are taken from this article: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9399(1989)115:8(1609)
time_step = 0.01
system_parameter = {'M':np.diag([1002.4,1002.4,1002.4]),\
                    'K':1e6*np.array([[2.80, -1.68, 0.38],[-1.68, 3.09, -1.66],[0.38, -1.66, 1.36]]),\
                    'C':np.array([[391.12, -58.53, 63.01],[-58.53, 466.83, -0.27],[63.01, -0.27, 446.97]])}   
space_bounds = {'action_space_lowerbound':-np.ones(3), 'action_space_upperbound':np.ones(3), 'observation_space_lowerbound':-10*np.ones(7), 'observation_space_upperbound': 10*np.ones(7)}
reward_weights = {'displacement_weights':np.ones(3), 'velocity_weights':np.ones(3), 'acceleration_weights':np.ones(3), 'control_force_weights':np.ones(3)}

# shall we remove observation_space_bound ? 
##################################################

# Environment
#env = gym.make(args.env_name)
env = DynamicEnv(time_step, system_parameter, space_bounds, reward_weights ) #
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

writer = SummaryWriter('runs/{}_SAC_V_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    #agent_state, env_state = env.reset()
    max_ep_steps = 1000
    start_point = 0
    agent_state = np.array([0.,0.,0.,data[start_point,0],0.,0.,0.])
    # env_state = np.zeros(2*3)
    env_state = np.array([0.,0.1,0,0,0,0 ])
    for j in range(start_point + 1, start_point+1+max_ep_steps):
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(agent_state)  # Sample action from policy

                  
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):  # Number of updates per step in environment
                # Update parameters of all the networks
                value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1
        ground_acceleration = np.array([data[start_point,0]])
        reward, env_state, env_acceleration = env.step(action, env_state, ground_acceleration) # Step
        next_agent_state = np.concatenate([env_acceleration,[data[j,0]],action], axis=0)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_ep_steps else float(not done)

        memory.push(agent_state, action, reward, next_agent_state, mask) # Append transition to memory

        agent_state = next_agent_state
    if total_numsteps > args.num_steps:
        break
    
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    y_store = []
    if i_episode % 2 == 0 and args.eval == True:
        avg_reward = 0.
        episodes = 1
        for _ in range(episodes):
            #state = env.reset()
            episode_reward = 0
            done = False
            agent_state = np.array([0.,0.,0.,data[start_point,0],0.,0.,0.])
            env_state = np.zeros(2*3)
            for j in range(start_point + 1, start_point+1+max_ep_steps):
                action = agent.select_action(agent_state, eval=True)

                ground_acceleration = np.array([data[start_point,0]])
                reward, env_state, env_acceleration = env.step(action, env_state, ground_acceleration) # Step
                next_agent_state = np.concatenate([env_acceleration,[data[j,0]],action], axis=0)
                episode_reward += reward
                if len(y_store) == 0:
                    y_store = env_acceleration
                else:
                    y_store = np.vstack((y_store, env_acceleration))

                agent_state = next_agent_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        data_store = {'response':y_store}
        savemat('SAC_response.mat',data_store)
env.close()

