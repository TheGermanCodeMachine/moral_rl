import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
from envs.gym_wrapper import *
from moral.preference_giver import *
import argparse
import sys
import tensorflow as tf
from copy import *
from helpers.visualize_trajectory import visualize_two_part_trajectories, visualize_two_part_trajectories_part
from helpers.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, evaluate_qcs_for_cte, compare_cte_methods
from quality_metrics.distance_measures import distance_all as distance_all
import pickle
from helpers.parsing import sort_args, parse_attributes
from generation_methods.counterfactual_mcts import *
from generation_methods.counterfactual_step import *
from generation_methods.counterfactual_random import *
from quality_metrics.critical_state_measures import critical_state_all as critical_state
from normalising_qc import normalising_qcs

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
    base_path = '.\datasets\\100mcts\\'
    measure_statistics = True
    num_runs = 10
    criteria = ['validity', 'diversity', 'proximity', 'critical_state', 'realisticness', 'sparsity']
    # criteria = ['baseline']
    # criteria = ['validity']
    cf_method = 'mcts' # 'mcts' or 'step'


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Parse attributes string')
    parser.add_argument('attributes', type=str, help='Attribute string to be parsed')
    folder_string = ''
    if len(sys.argv) >= 2:
        args = parser.parse_args()
        parsed_attrs = parse_attributes(args.attributes)
        folder_string = sort_args(args.attributes)
        config.criteria = parsed_attrs

    # determine whether this is a baseline run or not
    baseline = 'baseline' in config.criteria

    print('Criteria: ', config.criteria, baseline)
    
    # make a random number based on the time
    random.seed(4)
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
    ppo.load_state_dict(torch.load('saved_models/ppo_airl_v2_[1,10].pt', map_location=torch.device('cpu')))
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models/discriminator_v2_[1,10].pt', map_location=torch.device('cpu')))

    all_org_trajs, all_cf_trajs, all_starts, all_end_orgs, all_end_cfs, all_part_orgs, all_part_cfs, random_baseline_cfs, random_baseline_orgs = [], [], [], [], [], [], [], [], []
    lengths_org, lengths_cf, start_points, quality_criteria, effiencies, qc_statistics = [], [], [], [], [], []

    # load the original trajectories
    org_traj_seed = pickle.load(open('demonstrations/original_trajectories_new_maxsteps75_airl_1000.pkl', 'rb'))

    normalising_qcs(ppo, discriminator, org_traj_seed, config)

    # stop after the number of runs config.num_runs or iterate through all original trajectories
    run = 0
    for org_traj, seed_env in org_traj_seed:
        print(run)
        if run >= config.num_runs: break
        run += 1

        time_start = time.time()

        # generate the counterfactual trajectories
        if config.cf_method == 'mcts':
            # Method 1: MCTS            
            mcts_org, mcts_cf, mcts_start = generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env, all_org_trajs, all_cf_trajs, all_starts)
            efficiency = time.time() - time_start
            # visualize_two_part_trajectories_part(mcts_org, mcts_cf)


            mcts_rewards = sum(mcts_org['rewards'])
            all_part_orgs.append((mcts_org, mcts_rewards))
            mcts_rewards_cf = sum(mcts_cf['rewards'])
            all_part_cfs.append((mcts_cf, mcts_rewards_cf))

            # Method 2: 1-step deviation
            counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs = generate_counterfactuals(org_traj, ppo, discriminator, seed_env, config)
            if not baseline:
                # use the quality criteria to determine the best counterfactual trajectory
                sort_index, qc_stats = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, config.criteria)
                qc_statistics.append(qc_stats)
            else:
                # use a random baseline to determine the best counterfactual trajectory
                sort_index = random.randint(0, len(counterfactual_trajs)-1)

            chosen_counterfactual_trajectory = counterfactual_trajs[sort_index]
            chosen_start = starts[sort_index]
            chosen_end_cf = end_cfs[sort_index]
            chosen_end_org = end_orgs[sort_index]

            efficiency = time.time() - time_start

            step_org = partial_trajectory(org_traj, chosen_start, chosen_end_org)
            step_rewards = sum(step_org['rewards'])
            # all_part_orgs.append((step_org, step_rewards))

            step_cf = partial_trajectory(chosen_counterfactual_trajectory, chosen_start, chosen_end_cf)
            step_rewards_cf = sum(step_cf['rewards'])
            # all_part_cfs.append((step_cf, step_rewards_cf))

            # Method 3: Random
            random_org, random_cf, random_start = generate_counterfactual_random(org_traj, ppo, discriminator, seed_env, config)

            # Compare the methods
            compare_cte_methods(mcts_org, mcts_cf, mcts_start, all_org_trajs, all_cf_trajs, all_starts, config.criteria, ppo, step_org, step_cf, chosen_start, random_org, random_cf, random_start)
            # compare_cte_methods(mcts_org, mcts_cf, mcts_start, all_org_trajs, all_cf_trajs, all_starts, config.criteria, ppo)

        # uncomment below if the trajectories should be visualized:
        # visualize_two_part_trajectories(org_traj, chosen_counterfactual_trajectory, chosen_start, chosen_end_cf,  chosen_end_org)



        if config.measure_statistics:
            # record stastics
            lengths_org.append(len(mcts_org['states']))
            lengths_cf.append(len(mcts_cf['states']))
            start_points.append(mcts_start)
            chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar = evaluate_qcs_for_cte(mcts_org, mcts_cf, mcts_start, ppo, all_org_trajs, all_cf_trajs, all_starts)
            quality_criteria.append((chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar))
            effiencies.append(efficiency)

        # add the original trajectory and the counterfactual trajectory to the list of all trajectories
        all_org_trajs.append(mcts_org)
        all_cf_trajs.append(mcts_cf)
        all_starts.append(mcts_start)


    print('avg length org: ', np.mean(lengths_org))
    print('avg length cf: ', np.mean(lengths_cf))
    print('avg start point: ', np.mean(start_points))
    print('avg quality criteria: ', np.mean(quality_criteria, axis=0))
    print('avg generation time: ', np.mean(effiencies))

    if not baseline:
        path_folder = config.base_path + folder_string + str(config.num_runs)
    else:
        path_folder = config.base_path + '\\baseline' + str(config.num_runs)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    #save the trajectories
    with open(path_folder + '\org_trajectories.pkl', 'wb') as f:
        pickle.dump(all_part_orgs, f)
    with open(path_folder + '\cf_trajectories.pkl', 'wb') as f:
        pickle.dump(all_part_cfs, f)

    if config.measure_statistics:
        # saving statistics
        # check if folder statistics exists
        if not os.path.exists(path_folder + '\statistics'):
            os.makedirs(path_folder + '\statistics')

        with open(path_folder + '\statistics\lengths_org.pkl', 'wb') as f:
            pickle.dump(lengths_org, f)
        with open(path_folder + '\statistics\lengths_cf.pkl', 'wb') as f:
            pickle.dump(lengths_cf, f)
        with open(path_folder + '\statistics\start_points.pkl', 'wb') as f:
            pickle.dump(start_points, f)
        with open(path_folder + '\statistics\quality_criteria.pkl', 'wb') as f:
            pickle.dump(quality_criteria, f)
        with open(path_folder + '\statistics\effiencies.pkl', 'wb') as f:
            pickle.dump(effiencies, f)
        with open(path_folder + '\statistics\qc_statistics.pkl', 'wb') as f:
            pickle.dump(qc_statistics, f)