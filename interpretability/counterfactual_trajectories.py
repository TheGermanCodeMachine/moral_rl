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
from helpers.visualize_trajectory import visualize_two_part_trajectories
from helpers.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, evaluate_qcs_for_cte
from quality_metrics.distance_measures import distance_all as distance_all
import pickle
from helpers.parsing import sort_args, parse_attributes
from counterfactual_mcts import *
from quality_metrics.critical_state_measures import critical_state_all as critical_state


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
    base_path = '.\datasets\\1000\\'
    measure_statistics = True
    num_runs = 1000
    criteria = ['validity', 'diversity', 'proximity', 'critical_state', 'realisticness', 'sparsity']
    # criteria = ['baseline']
    # criteria = ['validity']
    cf_method = 'mcts' # 'mcts' or 'deviation'

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

# this tests that the states of the original and counterfactual are actually different after the deviation has been taken
def test_same_next_state(states_tensor, next_state, org_action, cf_action):
    # option 1 for difference: any of them takes a step (this guarantees different states since they can't take the same step)
    if org_action in [0,1,2,3] or cf_action in [0,1,2,3]:
        return False
    # option 2 for difference: a citizen has been grabbed in the original
    if count_citizens(states_tensor) != count_citizens(next_state):
        return False
    # option 3: a citizen will be grabbed in the counterfactual
    pp = extract_player_position(states_tensor)
    if cf_action == 5:
        if states_tensor[0][2][pp[0]][pp[1]+1] == 1:
            return False
    elif cf_action == 6:
        if states_tensor[0][2][pp[0]][pp[1]-1] == 1:
            return False
    elif cf_action == 7:
        if states_tensor[0][2][pp[0]+1][pp[1]] == 1:
            return False
    elif cf_action == 8:
        if states_tensor[0][2][pp[0]-1][pp[1]] == 1:
            return False
    return True
    
def generate_counterfactual_random(org_traj, ppo, discriminator, seed_env):
    start = random.randint(0, len(org_traj['states'])-1)

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

        action_distribution = torch.cat((action_distribution, torch.tensor([0.2]).view(1,1)), 1)
        m = Categorical(action_distribution)
        action = m.sample()
        action = action.item()
        cf_traj['actions'].append(action)
        state, reward, done, info = vec_env_cf.step([action])
        states_tensor = torch.tensor(state).float().to(device)
        

    part_org_traj = partial_trajectory(org_traj, start, start + len(cf_traj['states']) - 1)
    return part_org_traj, cf_traj, start



def generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env):
    
    vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env_cf.reset()
    states_tensor = torch.tensor(states).float().to(device)

    critical_states = critical_state(ppo, org_traj['states'])
    # get the index of the 5 states with the highest critical state
    critical_states = [(i,j) for i,j in zip(critical_states, range(len(critical_states)))]
    critical_states.sort(key=lambda x: x[0], reverse=True)
    critical_states = critical_states[:1]

    part_orgs, part_cfs, q_values = [], [], []
    for (i, starting_position) in critical_states:
        part_org, part_cf, q_value = run_mcts_from(vec_env_cf, org_traj, starting_position, ppo, discriminator, seed_env)
        part_orgs.append(part_org)
        part_cfs.append(part_cf)
        q_values.append(q_value)

    # select the best counterfactual trajectory
    sort_index = np.argmax(q_values)
    chosen_part_org = part_orgs[sort_index]
    chosen_part_cf = part_cfs[sort_index]
    chosen_start = critical_states[sort_index][1]
    return chosen_part_org, chosen_part_cf, chosen_start

def generate_counterfactuals(org_traj, ppo, discriminator, seed_env):
    # Now we make counterfactual trajectories by changing one action at a time and see how the reward changes

    # we make copies of the original trajectory and each changes one action at a timestep step
    # for each copy, change one action; for each action, change it to the best action that is not the same as the original action
    counterfactual_trajs, counterfactual_rewards = [], []
    # the timestep where the counterfactual diverges from the original trajectory and rejoins again
    starts, end_orgs, end_cfs = [], [], []

    # create a new environment to make the counterfactual trajectory in; this has the same seed as the original so the board is the same
    vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env_cf.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # this loop is over all timesteps in the original and each loop creates one counterfactual with the action at that timestep changed
    for step in range(0, len(org_traj['actions'])-1):
        counterfactual_traj = {'states': [], 'actions': [], 'rewards': []}
        counterfactual_deviation = 0

        # follow the steps of the original trajectory until the point of divergence (called step here)
        counterfactual_traj, states_tensor = retrace_original(0, step, counterfactual_traj, org_traj, vec_env_cf, states_tensor, discriminator)

        # now we are at the point of divergence
        counterfactual_traj['states'].append(states_tensor)
        reward = discriminator.g(states_tensor)[0][0].item()
        counterfactual_traj['rewards'].append(reward)
        same_next_state = True
        while same_next_state:
            counterfact_action, log_probs = ppo.pick_another_action(states_tensor, org_traj['actions'][step])
            same_next_state = test_same_next_state(states_tensor, org_traj['states'][step+1], org_traj['actions'][step], counterfact_action)
        # take the best action that is not the same as the original action
        # remove the action which is the same as the orignal action
        counterfactual_traj['actions'].append(counterfact_action)

        next_states, reward, done, info = vec_env_cf.step(counterfact_action)
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

        # continue the counterfactual trajectory with the policy until the end of the trajectory or until it rejoins the original trajectory
        for t in range(step+1, config.max_steps-1):
            counterfactual_traj['states'].append(states_tensor)
            counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
            actions, log_probs = ppo.act(states_tensor)
            counterfactual_traj['actions'].append(actions)

            # test if this next step will rejoin the original trajectory
            rejoin_step = test_rejoined_org_traj(org_traj, states_tensor, t, step)
            if rejoin_step and rejoin_step < len(org_traj['states']):
                end_part_cf = t
                # follow the steps of the original trajectory until the length of the original trajectory
                counterfactual_traj, states_tensor = retrace_original(rejoin_step, len(org_traj['states']), counterfactual_traj, org_traj, vec_env_cf, states_tensor, discriminator)
                break

            next_states, reward, done, info = vec_env_cf.step(actions)
            if done[0]:
                next_states = vec_env_cf.reset()
                break
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        if not rejoin_step:
            rejoin_step = len(org_traj['states']) - 1
            end_part_cf = len(counterfactual_traj['states']) - 1
            
        # if the rewards are the same, then the counterfactual is not informative and we don't include it
        # NOTE: By including 'test_same_next_state' this if-statement should always be true, because the next state should always be different
        if np.mean(counterfactual_traj['rewards'][step:end_part_cf+1]) - np.mean(org_traj['rewards'][step:rejoin_step+1]) != 0:
            counterfactual_trajs.append(counterfactual_traj)
            counterfactual_rewards.append(torch.mean(torch.tensor(counterfactual_traj['rewards'])))
            starts.append(step)
            end_orgs.append(rejoin_step)
            end_cfs.append(end_part_cf)

        vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
        states = vec_env_cf.reset()
        states_tensor = torch.tensor(states).float().to(device)

    return counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs



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

    # stop after the number of runs config.num_runs or iterate through all original trajectories
    run = 0
    for org_traj, seed_env in org_traj_seed:
        print(run)
        if run >= config.num_runs: break
        run += 1

        time_start = time.time()

        # generate the counterfactual trajectories
        if config.cf_method == 'mcts':
            random_org, random_cf, random_start = generate_counterfactual_random(org_traj, ppo, discriminator, seed_env)
            part_org, part_cf, chosen_start = generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env)
            
            efficiency = time.time() - time_start

            part_rewards = sum(part_org['rewards'])
            all_part_orgs.append((part_org, part_rewards))
            part_rewards_cf = sum(part_cf['rewards'])
            all_part_cfs.append((part_cf, part_rewards_cf))
        
            counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs = generate_counterfactuals(org_traj, ppo, discriminator, seed_env)
            if not baseline:
                # use the quality criteria to determine the best counterfactual trajectory
                sort_index, qc_stats = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, config.criteria, part_org, part_cf, chosen_start, random_org, random_cf, random_start)
                qc_statistics.append(qc_stats)
            else:
                # use a random baseline to determine the best counterfactual trajectory
                sort_index = random.randint(0, len(counterfactual_trajs)-1)

            chosen_counterfactual_trajectory = counterfactual_trajs[sort_index]
            chosen_start = starts[sort_index]
            chosen_end_cf = end_cfs[sort_index]
            chosen_end_org = end_orgs[sort_index]

            efficiency = time.time() - time_start

            part_org = partial_trajectory(org_traj, chosen_start, chosen_end_org)
            part_rewards = sum(part_org['rewards'])
            all_part_orgs.append((part_org, part_rewards))

            part_cf = partial_trajectory(chosen_counterfactual_trajectory, chosen_start, chosen_end_cf)
            part_rewards_cf = sum(part_cf['rewards'])
            all_part_cfs.append((part_cf, part_rewards_cf))

        # uncomment below if the trajectories should be visualized:
        # visualize_two_part_trajectories(org_traj, chosen_counterfactual_trajectory, chosen_start, chosen_end_cf,  chosen_end_org)



        if config.measure_statistics:
            # record stastics
            lengths_org.append(len(part_org['states']))
            lengths_cf.append(len(part_cf['states']))
            start_points.append(chosen_start)
            chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar = evaluate_qcs_for_cte(part_org, part_cf, chosen_start, ppo, all_org_trajs, all_cf_trajs, all_starts)
            quality_criteria.append((chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar))
            effiencies.append(efficiency)

        # add the original trajectory and the counterfactual trajectory to the list of all trajectories
        all_org_trajs.append(part_org)
        all_cf_trajs.append(part_cf)
        all_starts.append(chosen_start)


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