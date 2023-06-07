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
from utils.visualize_trajectory import visualize_two_trajectories, visualize_two_part_trajectories
from utils.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, get_all_combinations_of_qc
from quality_metrics.diversity_measures import diversity_all
from quality_metrics.validity_measures import validity_all as validity
from quality_metrics.critical_state_measures import critical_state_all as critical_state
from quality_metrics.distance_measures import distance_all as distance_all
from copy import deepcopy
import pickle
import evaluation.extract_reward_features as erf
from quality_metrics.distance_measures import my_distance as distance

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
    max_steps = 75
    num_runs = 100
    criteria = ['validity', 'diversity', 'proximity', 'critical_state']
    # criteria = ['validity']

def parse_attributes(arg_str):
    attr_map = {
        'p': 'proximity',
        'v': 'validity',
        'c': 'critical_state',
        'd': 'diversity',
    }
    return [attr_map[c] for c in arg_str if c in attr_map]

# tests whether the current state is in the set of states that have been visited in the orignial trajectory after timestep step
def test_rejoined_org_traj(org_traj, state, step, start):
    if step > start+1:
        (x,y) = extract_player_position(state)
        # ensure that the point of rejoining is not too far away in the future. This would otherwise make for unnatural rejoins. I consider 5 steps to be a reasonable limit
        # TODO Experiment with different values for this limit
        s = max(start+1, step-1)
        e = min(len(org_traj['states']), step+2)
        for t in range(s, e):
            # test whether position is the same
            (x_org, y_org) = extract_player_position(org_traj['states'][t])
            if x == x_org and y == y_org:
                return t
    return False

def generate_original_trajectory(ppo, discriminator, vec_env, states_tensor):
     # create one trajectory with ppo
    org_traj = {'states': [], 'actions': [], 'rewards': []}
    citizen_sum = 100
    for t in tqdm(range(config.max_steps-1)):
        if t==6:
            a=0
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = vec_env.step(actions)
        org_traj['states'].append(states_tensor)
        # Note: Actions currently append as arrays and not integers!
        org_traj['actions'].append(actions)
        org_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
        sum = torch.sum(org_traj['states'][-1][0][2], dim = (0,1)).item()
        if torch.sum(org_traj['states'][-1][0][2], dim = (0,1)).item() > citizen_sum:
            a=0
        citizen_sum = torch.sum(org_traj['states'][-1][0][2], dim = (0,1)).item()

        if done[0]:
            next_states = vec_env.reset()
            break

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    return org_traj

def retrace_original(start, divergence, counterfactual_traj, org_traj, vec_env_counter, states_tensor, discriminator):
    # retrace the steps of original trajectory until the point of divergence
    for i in range(start, divergence):
        counterfactual_traj['states'].append(states_tensor)
        counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
        counterfactual_traj['actions'].append(org_traj['actions'][i])

        next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][i])
        if done[0]:
            next_states = vec_env_counter.reset()
            break

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
    return counterfactual_traj, states_tensor

def generate_counterfactuals(org_traj, ppo, discriminator, seed_env):
    # Now we make counterfactual trajectories by changing one action at a time and see how the reward changes

    # we make copies of the original trajectory and each changes one action at a timestep step
    # for each copy, change one action; for each action, change it to the best action that is not the same as the original action
    counterfactual_trajs = []
    counterfactual_rewards = []
    counterfactual_deviations = []
    # the timestep where the counterfactual diverges from the original trajectory and rejoins again
    starts = []
    end_orgs = []
    end_cfs = []

    # create a new environment to make the counterfactual trajectory in; this has the same seed as the original so the board is the same
    vec_env_counter = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env_counter.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # this loop is over all timesteps in the original and each loop creates one counterfactual with the action at that timestep changed
    for step in range(0, len(org_traj['actions'])-1):
        if step == 9:
            a = 0
        counterfactual_traj = {'states': [], 'actions': [], 'rewards': []}
        counterfactual_deviation = 0
        
        # follow the steps of the original trajectory until the point of divergence (called step here)
        counterfactual_traj, states_tensor = retrace_original(0, step, counterfactual_traj, org_traj, vec_env_counter, states_tensor, discriminator)

        # TODO: Make sure (through debugging) that the same state is not being saved twice here
        # states_tensor = torch.tensor(org_traj['states'][step]).float().to(device)
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
        for t in range(step+1, config.max_steps-1):
            # print("cf: ", t, " ", extract_player_position(states_tensor))
            # print("org: ", t, " ", extract_player_position(org_traj['states'][t]))
            counterfactual_traj['states'].append(states_tensor)
            counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
            actions, log_probs = ppo.act(states_tensor)
            counterfactual_traj['actions'].append(actions)
            counterfactual_deviation += 1

            # test if this next step will rejoin the original trajectory
            rejoin_step = test_rejoined_org_traj(org_traj, states_tensor, t, step)
            # TODO I might have to add a test whether there has been enough difference between the two trajectories yet
            if rejoin_step and rejoin_step < len(org_traj['states'])-1:
                end_part_cf = t

                # check if there is a difference in the reward between the counterfactual and the original
                # if there is no difference, then the counterfactual is not informative 
                # note: it can still be different due to taking different grabbing actions, but gets the same reward because discrminator.g only considers the state

                # follow the steps of the original trajectory until the length of the original trajectory
                counterfactual_traj, states_tensor = retrace_original(rejoin_step, len(org_traj['states']), counterfactual_traj, org_traj, vec_env_counter, states_tensor, discriminator)
                break

            next_states, reward, done, info = vec_env_counter.step(actions)
            if done[0]:
                next_states = vec_env_counter.reset()
                break
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        if not rejoin_step:
            rejoin_step = len(org_traj['states']) - 1
            end_part_cf = len(counterfactual_traj['states']) - 1
            
        # if the rewards are the same, then the counterfactual is not informative and we don't include it
        if np.mean(counterfactual_traj['rewards'][step:end_part_cf+1]) - np.mean(org_traj['rewards'][step:rejoin_step+1]) != 0:
            counterfactual_trajs.append(counterfactual_traj)
            counterfactual_rewards.append(torch.mean(torch.tensor(counterfactual_traj['rewards'])))
            counterfactual_deviations.append(counterfactual_deviation)
            starts.append(step)
            end_orgs.append(rejoin_step)
            end_cfs.append(end_part_cf)

            ## This code tests whether the phenomenon of "standing on the citizen makes the citizens disappear" is present in the counterfactual trajectory
            # part_traj = {'states' : counterfactual_traj['states'][step:end_part_cf+1],
            #         'actions': counterfactual_traj['actions'][step:end_part_cf+1],
            #         'rewards': counterfactual_traj['rewards'][step:end_part_cf+1]}
            
            # if erf.citizens_saved(part_traj) < 0:
            #     print('start', torch.sum(part_traj['states'][0][0][2], dim=(0,1)))
            #     print('end', torch.sum(part_traj['states'][-1][0][2], dim=(0,1)))
            #     a=0
        vec_env_counter = VecEnv(config.env_id, config.n_workers, seed=seed_env)
        states = vec_env_counter.reset()
        states_tensor = torch.tensor(states).float().to(device)


    return counterfactual_trajs, counterfactual_rewards, counterfactual_deviations, starts, end_cfs, end_orgs



if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Parse attributes string')
    parser.add_argument('attributes', type=str, help='Attribute string to be parsed')
    if len(sys.argv) >= 2:
        args = parser.parse_args()
        parsed_attrs = parse_attributes(args.attributes)
        config.criteria = parsed_attrs

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
    ppo.load_state_dict(torch.load('saved_models/ppo_airl_v2_[0,1].pt', map_location=torch.device('cpu')))

    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models/discriminator_v2_[0,1].pt', map_location=torch.device('cpu')))

    all_org_trajs = []
    all_cf_trajs = []
    all_starts = []
    all_end_orgs = []
    all_end_cfs = []
    all_part_orgs = []
    all_part_cfs = []
    all_full_orgs = []
    random_baseline_cfs = []
    random_baseline_orgs = []

    org_traj_seed = pickle.load(open('original_trajectories_and_seeds.pkl', 'rb'))
    run = 0
    for org_traj, seed_env in org_traj_seed:
        print('run', run)
        run += 1
        # generate the counterfactual trajectories
        counterfactual_trajs, counterfactual_rewards, counterfactual_deviations, starts, end_cfs, end_orgs = generate_counterfactuals(org_traj, ppo, discriminator, seed_env)

        # calculate the quality criteria for each counterfactual trajectory
        qc_values = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs, config.criteria)
        # get the qc_values[0] with the highest qc_values[1]
        max_qc_index = qc_values[0][0]
        # get the counterfactual trajectory with the largest absolute difference from the original reward
        best_counterfactual_trajectory = counterfactual_trajs[max_qc_index]

        # visualize only the part of the trajectory that is different
        # visualize_two_part_trajectories(org_traj, best_counterfactual_trajectory, starts[max_qc_index], end_cfs[max_qc_index],  end_orgs[max_qc_index])

        # add the original trajectory and the counterfactual trajectory to the list of all trajectories
        all_org_trajs.append(org_traj)
        all_cf_trajs.append(best_counterfactual_trajectory)
        all_starts.append(starts[max_qc_index])
        all_end_orgs.append(end_orgs[max_qc_index])
        all_end_cfs.append(end_cfs[max_qc_index])

        part_org = {'states' : org_traj['states'][starts[max_qc_index]:end_orgs[max_qc_index]+1],
                    'actions': org_traj['actions'][starts[max_qc_index]:end_orgs[max_qc_index]+1],
                    'rewards': org_traj['rewards'][starts[max_qc_index]:end_orgs[max_qc_index]+1]}
        part_rewards = sum(part_org['rewards'])
        all_part_orgs.append((part_org, part_rewards))

        part_cf = {'states' : best_counterfactual_trajectory['states'][starts[max_qc_index]:end_cfs[max_qc_index]+1],
                    'actions': best_counterfactual_trajectory['actions'][starts[max_qc_index]:end_cfs[max_qc_index]+1],
                    'rewards': best_counterfactual_trajectory['rewards'][starts[max_qc_index]:end_cfs[max_qc_index]+1]}
        part_rewards_cf = sum(part_cf['rewards'])
        all_part_cfs.append((part_cf, part_rewards_cf))


        full_rewards = sum(org_traj['rewards'])
        all_full_orgs.append((org_traj, starts[max_qc_index], end_cfs[max_qc_index]+1 - starts[max_qc_index], full_rewards))

        # pick a random counterfactual as the baseline
        random_cf_index = random.randint(0, len(counterfactual_trajs)-1)
        part_random_org = {'states' : org_traj['states'][starts[random_cf_index]:end_orgs[random_cf_index]+1],
                    'actions': org_traj['actions'][starts[random_cf_index]:end_orgs[random_cf_index]+1],
                    'rewards': org_traj['rewards'][starts[random_cf_index]:end_orgs[random_cf_index]+1]}
        random_org_reward = sum(part_random_org['rewards'])
        random_baseline_orgs.append((part_random_org, random_org_reward))

        part_random_cf = {'states' : counterfactual_trajs[random_cf_index]['states'][starts[random_cf_index]:end_cfs[random_cf_index]+1],
                    'actions': counterfactual_trajs[random_cf_index]['actions'][starts[random_cf_index]:end_cfs[random_cf_index]+1],
                    'rewards': counterfactual_trajs[random_cf_index]['rewards'][starts[random_cf_index]:end_cfs[random_cf_index]+1]}
        random_cf_reward = sum(part_random_cf['rewards'])
        random_baseline_cfs.append((part_random_cf, random_cf_reward))



    base_path = '.\evaluation\datasets\\100_ablations\\baseline'
    # save the trajectories
    # with open(base_path + '\org_trajectories.pkl', 'wb') as f:
    #     pickle.dump(all_part_orgs, f)
    # with open(base_path + '\cf_trajectories.pkl', 'wb') as f:
    #     pickle.dump(all_part_cfs, f)
    # with open(base_path + '\\full_trajectories.pkl', 'wb') as f:
    #     pickle.dump(all_full_orgs, f)
    with open(base_path + '\\cf_random_baselines.pkl', 'wb') as f:
        pickle.dump(random_baseline_cfs, f)
    with open(base_path + '\org_random_baselines.pkl', 'wb') as f:
        pickle.dump(random_baseline_orgs, f)