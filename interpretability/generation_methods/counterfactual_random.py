import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import random
import torch
from envs.gym_wrapper import *
from helpers.util_functions import *
from torch.distributions import Categorical

def generate_counterfactual_random(org_traj, ppo, discriminator, seed_env, config):
    long_enough = False
    # while not long_enough:
    start = random.randint(0, len(org_traj['states'])-2)

    vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env_cf.reset()
    states_tensor = torch.tensor(states).float().to(device)
    
    full_traj = {'states': [], 'actions': [], 'rewards': []}
    full_traj, states_tensor = retrace_original(0, start, full_traj, org_traj, vec_env_cf, states_tensor, discriminator)
    
    action = -1
    done = [False]
    cf_traj = {'states': [], 'actions': [], 'rewards': []}
    while not done[0] and action != 9:
        cf_traj['states'].append(states_tensor)
        cf_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())

        action_distribution = ppo.action_distribution_torch(states_tensor)
        # remove all actions below the threshold

        if len(cf_traj['states']) > 2:
            action_distribution = torch.cat((action_distribution, torch.tensor([0.2]).view(1,1)), 1)
        m = Categorical(action_distribution)
        action = m.sample()
        action = action.item()
        cf_traj['actions'].append(action)
        state, reward, done, info = vec_env_cf.step([action])
        states_tensor = torch.tensor(state).float().to(device)
        

    part_org_traj = partial_trajectory(org_traj, start, start + len(cf_traj['states']) - 1)
        # if len(cf_traj['states']) > 1 and len(part_org_traj['states']) > 1:
        #     long_enough = True
        # else:
        #     random.seed(time.time())
        #     torch.seed(time.time())
            
    if len(cf_traj['states']) <2:
        a=0

    return part_org_traj, cf_traj, start