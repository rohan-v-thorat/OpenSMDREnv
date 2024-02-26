import argparse
import datetime
import gym
import sys
sys.path.insert(0,'../src')
from dof1 import DynamicEnv
import numpy as np
import itertools
import torch
# import pybullet  # 'import pybullet_envs' has been removed
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
# Environment
#env = gym.make(args.env_name)
env = DynamicEnv()
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
    #state = env.reset()
    max_ep_steps = 1000
    start_point = 0
    state = np.array([0.,data[start_point,0],0.])
    env.state = state
    for j in range(start_point + 1, start_point+1+max_ep_steps):
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)  # Sample action from policy
        if j==1:
           system_state = np.array([0.,0.])
                  
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):  # Number of updates per step in environment
                # Update parameters of all the networks
                value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1
        ground_acceleration = data[start_point,0]
        next_state, reward, system_state, y_pre= env.step(action,system_state, ground_acceleration) # Step
        next_state = np.concatenate([y_pre,[data[j,0]],action], axis=0)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_ep_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        env.state = next_state
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
            state = np.array([0.,data[start_point,0],0.])
            env.state = state
            episode_reward = 0
            done = False
            for j in range(start_point + 1, start_point+1+max_ep_steps):
                action = agent.select_action(state, eval=True)
                if j==1:
                   system_state = np.array([0.,0.])

                next_state, reward, done,system_state , y_pre= env.step(action,system_state) # Step
                next_state = np.concatenate([y_pre,[data[j,0]],action], axis=0)
                episode_reward += reward
                if y_store == []:
                    y_store = y_pre
                else:
                    y_store = np.vstack((y_store, y_pre))

                state = next_state
                env.state = state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        data_store = {'response':y_store}
        savemat('SAC_response.mat',data_store)
env.close()

