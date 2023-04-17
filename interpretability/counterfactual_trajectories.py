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
from utils.visualize_trajectory import visualize_two_trajectories
from utils.util_functions import *
import random 
import time

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
    env_id= 'randomized_v2'
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

# tests whether the current state is in the set of states that have been visited in the orignial trajectory after timestep step
def test_rejoined_org_traj(org_traj, state, step):
    (x,y) = extract_player_position(state)
    # ensure that the point of rejoining is not too far away in the future. This would otherwise make for unnatural rejoins. I consider 5 steps to be a reasonable limit
    # TODO Experiment with different values for this limit
    for t in range(step, min(len(org_traj['states']), step+6)):
        # test whether position is the same
        (x_org, y_org) = extract_player_position(org_traj['states'][t])
        if x == x_org and y == y_org:
            return t
    return False

def retrace_original(cf_traj, org_traj, start, end=None):
    for t in range(step):
        cf_traj['states'].append(states_tensor)
        counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
        counterfactual_traj['actions'].append(org_traj['actions'][t])

        next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][t])
        if done[0]:
            next_states = vec_env_counter.reset()
            break

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

def deviation_metric(org_traj, cf_traj):
    # TODO: Implement this
    return 0

def quality_criterion(org_reward, org_traj, cf_traj, cf_reward):
    # alpha determines how much to prioritise similarity of trajectories or difference in outputs
    alpha = 1
    # difference in reward
    diff_reward = abs(org_reward - cf_reward)
    # difference in trajectories
    diff_traj = deviation_metric(org_traj, cf_traj)
    return diff_reward + alpha*diff_traj


if __name__ == '__main__':

    # make a random number based on the time
    random.seed(time.time())
    seed_env = random.randint(0, 100)
    torch.manual_seed(seed_env)
    np.random.seed(seed_env)
    
    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
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
    ppo.load_state_dict(torch.load('saved_models/ppo_airl_v2_[0,1].pt', map_location=torch.device('cpu')))

    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models/discriminator_v2_[0,1].pt', map_location=torch.device('cpu')))
    # utop_0 = discriminator.estimate_utopia(ppo, config)
    # print(f'Reward Normalization 0: {utop_0}')
    # discriminator.set_eval()

    # create one trajectory with ppo
    org_traj = {'states': [], 'actions': [], 'rewards': []}
    for t in tqdm(range(config.max_steps-1)):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = vec_env.step(actions)
        org_traj['states'].append(states_tensor)
        # Note: Actions currently append as arrays and not integers!
        org_traj['actions'].append(actions)
        org_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())

        if done[0]:
            next_states = vec_env.reset()
            break

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # sum up the reward of the original trajectory
    org_reward = torch.mean(torch.tensor(org_traj['rewards']))


    # Now we make counterfactual trajectories by changing one action at a time and see how the reward changes

    # we make copies of the original trajectory and each changes one action at a timestep step
    # for each copy, change one action; for each action, change it to the best action that is not the same as the original action
    counterfactual_trajs = []
    counterfactual_rewards = []
    counterfactual_deviations = []

    # this loop is over all timesteps in the original and each loop creates one counterfactual with the action at that timestep changed
    for step in range(1, len(org_traj['actions'])):
        counterfactual_traj = {'states': [], 'actions': [], 'rewards': []}
        counterfactual_deviation = 0

        # create a new environment to make the counterfactual trajectory in; this has the same seed as the original so the board is the same
        vec_env_counter = VecEnv(config.env_id, config.n_workers, seed=seed_env)
        states = vec_env_counter.reset()
        states_tensor = torch.tensor(states).float().to(device)
        
        # retrace the steps of original trajectory until the point of divergence
        for t in range(step):
            counterfactual_traj['states'].append(states_tensor)
            counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
            counterfactual_traj['actions'].append(org_traj['actions'][t])

            next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][t])
            if done[0]:
                next_states = vec_env_counter.reset()
                break

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        # TODO: Make sure (through debugging) that the same state is not being saved twice here
        states_tensor = torch.tensor(org_traj['states'][step]).float().to(device)
        counterfactual_traj['states'].append(states_tensor)
        reward = discriminator.g(states_tensor)[0][0].item()
        counterfactual_traj['rewards'].append(reward)

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
            counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
            actions, log_probs = ppo.act(states_tensor)
            counterfactual_traj['actions'].append(actions)
            counterfactual_deviation += 1
            next_states, reward, done, info = vec_env_counter.step(actions)
            if done[0]:
                next_states = vec_env_counter.reset()
                break

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

            # test if this next step will rejoin the orinigal trajectory
            rejoin_step = test_rejoined_org_traj(org_traj, states_tensor, t)
            # TODO I might have to add a test whether there has been enough difference between the two trajectories yet
            if rejoin_step and rejoin_step < len(org_traj['states'])-1:
                # finish the trajectory following the original actions
                for i in range(rejoin_step, len(org_traj['states'])):
                    counterfactual_traj['states'].append(states_tensor)
                    counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
                    counterfactual_traj['actions'].append(org_traj['actions'][i])

                    next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][i])
                    if done[0]:
                        next_states = vec_env_counter.reset()
                        break

                    states = next_states.copy()
                    states_tensor = torch.tensor(states).float().to(device)
                                                          
                next_states = vec_env_counter.reset()
                break

        counterfactual_trajs.append(counterfactual_traj)
        counterfactual_rewards.append(torch.mean(torch.tensor(counterfactual_traj['rewards'])))
        counterfactual_deviations.append(counterfactual_deviation)

    # alpha determines how much to prioritise similarity of trajectories or difference in outputs
    alpha = 0.1
    # return the index of the counterfactual reward with the largest absolute difference from the original reward
    # max_diff = max(range(1, len(counterfactual_rewards)), key=lambda x: quality_criterion(org_reward, org_traj, counterfactual_trajs[x], counterfactual_rewards[x]))
    max_diff = max(counterfactual_rewards, key=lambda x: abs(x-org_reward) - alpha * counterfactual_deviations[counterfactual_rewards.index(x)])

    # get the index of the max_diff
    max_diff_index = counterfactual_rewards.index(max_diff)
    print("original reward: ", org_reward, "counterfactual reward: ", counterfactual_rewards[max_diff_index])
    # get the counterfactual trajectory with the largest absolute difference from the original reward
    max_diff_traj = counterfactual_trajs[max_diff_index]

    # print out the players path over the trajectory
    # for i in range(len(counterfactual_trajs[1]['states'])):
        # print player position
        # print(extract_player_position(org_traj['states'][i]), extract_player_position(counterfactual_trajs[1]['states'][i]))

    visualize_two_trajectories(org_traj, max_diff_traj)