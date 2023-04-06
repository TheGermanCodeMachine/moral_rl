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
from copy import deepcopy
from statistics import mean

print(sys.path)

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
        

def iterate_through_settings(pairs):
        rewards = []
        for state, next_state in pairs:
            airl_state = torch.tensor(state).to(device).float()
            airl_next_state = torch.tensor(next_state).to(device).float()
            airl_rewards = discriminator_0.forward(airl_state, airl_next_state, config.gamma).squeeze(1)
            rewards.append(airl_rewards.detach().numpy()[0])
        return rewards

if __name__ == '__main__':

    # Fetch ratio args for automatic preferences

    parser = argparse.ArgumentParser(description='Number of negative and positive examples')
    parser.add_argument('-n', nargs=1, type=int)
    parser.add_argument('-t', nargs=1, type=int)
    args = parser.parse_args()

    num_neg, num_pos = args.n[0], args.n[0]
    test_type = args.t[0]
    if test_type < 2 or test_type > 5:
         test_type = 2

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
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

    vases = counterfactual_num_surrounding_types(config, ppo, num_examples=num_neg, test_type=test_type)

    org, close, far = counterfactuals_close_far_removal(config, ppo, num_examples=num_neg, test_type=test_type)

    # Expert 0
    discriminator_0 = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator_0.load_state_dict(torch.load('saved_models/discriminator_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))
    ppo_0 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo_0.load_state_dict(torch.load('saved_models/ppo_airl_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))
    utop_0 = discriminator_0.estimate_utopia(ppo_0, config)
    print(f'Reward Normalization 0: {utop_0}')
    discriminator_0.set_eval()

    # calculate outputs
    
    vases_rewards0 = iterate_through_settings(vases[0])
    vases_rewards1 = iterate_through_settings(vases[1])
    vases_rewards2 = iterate_through_settings(vases[2])
    vases_rewards3 = iterate_through_settings(vases[3])
    vases_rewards4 = iterate_through_settings(vases[4])

    rewards_org = iterate_through_settings(org)
    rewards_close = iterate_through_settings(close)
    rewards_far = iterate_through_settings(far)

    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    test_all_changes(config, ppo, discriminator_0)
    

    print("avg reward 0:", mean(vases_rewards0))
    print("avg reward 1:", mean(vases_rewards1))
    print("avg reward 2:", mean(vases_rewards2))
    print("avg reward 3:", mean(vases_rewards3))
    print("avg reward 4:", mean(vases_rewards4))

    print("avg reward org:", mean(rewards_org))
    print("avg reward close:", mean(rewards_close))
    print("avg reward far:", mean(rewards_far))