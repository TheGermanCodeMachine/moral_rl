import pickle
import torch
import numpy as np
import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from moral.ppo import PPO
import random
from envs.gym_wrapper import *
from utils.evaluate_ppo import evaluate_ppo

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
    base_path = '.\datasets\\100mcts\\'
    measure_statistics = True
    num_runs = 100
    criteria = ['validity', 'diversity', 'proximity', 'critical_state', 'realisticness', 'sparsity']
    # criteria = ['baseline']
    # criteria = ['validity']
    cf_method = 'mcts' # 'mcts' or 'deviation'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_env = random.randint(0, 100000)

vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
states = vec_env.reset()
states_tensor = torch.tensor(states).float().to(device)

# Fetch Shapes
n_actions = vec_env.action_space.n
obs_shape = vec_env.observation_space.shape
state_shape = obs_shape[:-1]
in_channels = obs_shape[-1]

# load pickle file
ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
ppo.load_state_dict(torch.load('saved_models\ppo_v2_[1, 10].pt', map_location=torch.device('cpu')))

obj_means, obj_std, scal_mean, scal_std = evaluate_ppo(ppo, config, n_eval=1000)