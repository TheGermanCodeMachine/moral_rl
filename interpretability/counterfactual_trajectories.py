import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
import matplotlib.pyplot as plt
from envs.gym_wrapper import *
from moral.preference_giver import *
from utils.evaluate_ppo import evaluate_ppo
import argparse
import sys
from create_data_interp import *
import tensorflow as tf
from linear_regression import *
import helper
from copy import *
from statistics import mean

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
    env_id= 'randomized_v3'
    env_steps= 8e6
    batchsize_ppo= 12
    n_queries= 50
    preference_noise= 0
    n_workers= 1
    lr_ppo= 3e-4
    entropy_reg= 0.25
    gamma= 0.999
    epsilon= 0.1
    ppo_epochs= 5
    max_steps = 50

def accumulate_reward_over_traj(reward_function, trajectory):
    # there is probably some more efficient vectorised way to do this
    sum_reward = 0
    for t in range(len(trajectory['states'])):
        state = trajectory['states'][t]
        reward = reward_function.g(state)
        sum_reward += reward
    return sum_reward


if __name__ == '__main__':

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers, seed=157)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    ppo.load_state_dict(torch.load('saved_models/ppo_airl_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))

    discriminator = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models/discriminator_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))
    # utop_0 = discriminator.estimate_utopia(ppo, config)
    # print(f'Reward Normalization 0: {utop_0}')
    discriminator.set_eval()

    # create one trajectory with ppo
    org_traj = {'states': [], 'actions': []}
    for t in tqdm(range(config.max_steps-1)):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = vec_env.step(actions)
        org_traj['states'].append(states_tensor)
        # Note: Actions currently append as arrays and not integers!
        org_traj['actions'].append(actions)

        if done[0]:
            next_states = vec_env.reset()
            break

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    org_reward = accumulate_reward_over_traj(discriminator, org_traj)

    # make perturbations of the original trajectory

    # make a list of copies of the original trajectory
    # for each copy, change one action; for each action, change it to the best action that is not the same as the original action
    counterfactual_trajs = []
    counterfactual_rewards = []
    for step in range(1, len(org_traj['actions'])):
        counterfactual_traj = {'states': [], 'actions': []}

        # create a new environment to make the counterfactual trajectory in

        vec_env_counter = VecEnv(config.env_id, config.n_workers, seed=157)
        states = vec_env_counter.reset()
        states_tensor = torch.tensor(states).float().to(device)
        
        counterfactual_traj['states'].append(states_tensor)
        # retrace the steps of original trajectory
        for t in range(step):
            next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][t])
            if done[0]:
                next_states = vec_env_counter.reset()
                break

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)
            counterfactual_traj['states'].append(states_tensor)
            counterfactual_traj['actions'].append(org_traj['actions'][t])


        states_tensor = torch.tensor(org_traj['states'][step]).float().to(device)
        counterfact_action, log_probs = ppo.pick_another_action(states_tensor, org_traj['actions'][step])

        # take the best action that is not the same as the original action
        # remove the action which is the same as the orignal action
        counterfactual_traj['actions'].append(counterfact_action)

        next_states, reward, done, info = vec_env_counter.step(counterfact_action)
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
        

        # finish the counterfactual trajectory
        for t in range(step+1, config.max_steps):
            counterfactual_traj['states'].append(states_tensor)
            actions, log_probs = ppo.act(states_tensor)
            counterfactual_traj['actions'].append(actions)
            next_states, reward, done, info = vec_env_counter.step(actions)
            if done[0]:
                next_states = vec_env_counter.reset()
                break
        counterfactual_trajs.append(counterfactual_traj)
        counterfactual_rewards.append(accumulate_reward_over_traj(discriminator, counterfactual_traj))

    # return the index of the counterfactual reward with the largest absolute difference from the original reward
    max_diff = max(counterfactual_rewards, key=lambda x: abs(x-org_reward))
    # get the index of the max_diff
    max_diff_index = counterfactual_rewards.index(max_diff)
    # get the counterfactual trajectory with the largest absolute difference from the original reward
    max_diff_traj = counterfactual_trajs[max_diff_index]