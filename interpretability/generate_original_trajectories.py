import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
from envs.gym_wrapper import *
from moral.preference_giver import *
import sys
import tensorflow as tf
from copy import *
from helpers.util_functions import *
import random 
import time
import pickle
import evaluation.reward_features as erf

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
    max_steps = 75
    num_runs = 50

def generate_original_trajectory(ppo, discriminator, vec_env, states_tensor):
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

    return org_traj

if __name__ == '__main__':

    # make a random number based on the time
    random.seed(3)
    seed_env = random.randint(0, 100000)
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
    ppo.load_state_dict(torch.load('saved_models\ppo_airl_v2_[1,10]_new.pt', map_location=torch.device('cpu')))

    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models/discriminator_v2_[1,10]_new.pt', map_location=torch.device('cpu')))

    original_trajectories_and_seeds = []

    for runs in range(config.num_runs):
        print("run: ", runs)
        # reset the environment
        seed_env = random.randint(0, 100000)
        vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
        states = vec_env.reset()
        states_tensor = torch.tensor(states).float().to(device)
        
        # generate the original trajectory
        org_traj = generate_original_trajectory(ppo, discriminator, vec_env, states_tensor)

        original_trajectories_and_seeds.append((org_traj, seed_env))

    # save the original trajectories
    with open('demonstrations/original_trajectories_new_maxsteps75_airl_50_new.pkl', 'wb') as f:
        pickle.dump(original_trajectories_and_seeds, f)