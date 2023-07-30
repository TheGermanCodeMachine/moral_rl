import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))

import numpy as np
import torch
import random
from helpers.util_functions import extract_player_position, normalise_values, partial_trajectory
from quality_metrics.distance_measures import my_distance, distance_subtrajectories, state_action_diff

# global variable to store the rotated trajectories (so they don't have to be recalculated every time)
ROTATED_TRAJ = None

# inputs:
# - whole trajectories
# - rewards for the trajectories
# - start and end of the part of the trajectory that make the CTE
# - other quality metrics assigned to the CTE?
# all trajectories have the form {'states': [s1, s2, s3, ...], 'actions': [a1, a2, a3, ...], 'rewards': [r1, r2, r3, ...]}
def diversity(traj1, traj2, start, end_cf, end_org, prev_org_traj, prev_cf_traj, prev_starts, prev_ends_cf, prev_ends_org, all_rotated_trajs):
    iterate_prev = range(len(prev_starts))

    # calculate the diversity of the length of the trajectories and start and end times
    length_div = length_diversity(end_cf - start, [prev_ends_cf[x] - prev_starts[x] for x in iterate_prev])
    start_time_div = timestep_diversity(start, prev_starts)
    # end_time_div = timestep_diversity(end_cf, prev_ends_cf)
    
    # calculate the diversity of start and end states

    all_prev_first_states = [i['states'][0] for i in all_rotated_trajs]
    all_prev_first_actions = [i['actions'][0] for i in all_rotated_trajs]
    start_state_div = state_diversity(traj1["states"][start], traj1["actions"][start], all_prev_first_states, all_prev_first_actions)

    org_traj = partial_trajectory(traj1, start, end_org)
    cf_traj = partial_trajectory(traj2, start, end_cf)
    up_down_div = up_down_diversity(org_traj, cf_traj, prev_org_traj, prev_cf_traj)
    # prev_last_states = [i['states'][-1] for i in prev_org_traj]
    # prev_last_actions = [i['actions'][-1] for i in prev_org_traj]
    # end_state_div = state_diversity(traj1['states'][end_org], traj1['actions'][end_org], prev_last_states, prev_last_actions)
    # prev_last_cf_states = [i['states'][-1] for i in prev_cf_traj]
    # prev_last_cf_actions = [i['actions'][-1] for i in prev_cf_traj]
    # endcf_state_div = state_diversity(traj2['states'][end_cf], traj2['actions'][end_cf], prev_last_cf_states, prev_last_cf_actions)


    # This makes the code super inefficient because it causes thousands of calls to the state_action_diff function
    # part_traj1 = {'states': traj1['states'][start:end_org+1], 'actions': traj1['actions'][start:end_org+1], 'rewards': traj1['rewards'][start:end_org+1]}
    # part_traj2 = {'states': traj2['states'][start:end_cf+1], 'actions': traj2['actions'][start:end_cf+1], 'rewards': traj2['rewards'][start:end_cf+1]}s
    # trajectory_div_org = part_traj_diversity(part_traj1, all_rotated_trajs)
    # trajectory_div_cf = part_traj_diversity(part_traj2, all_rotated_trajs)

    # print("length_div: ", length_div, "start_time_div: ", start_time_div, "start_state_div: ", start_state_div)
    return length_div, start_time_div, up_down_div # + trajectory_div_org + trajectory_div_cf

def diversity_all(org_traj, cf_trajs, starts, end_cfs, end_orgs, prev_org_trajs, prev_cf_trajs, prev_starts, prev_ends_cf, prev_ends_org):
    if len(prev_starts) == 0:
        return [0 for x in range(len(end_cfs))]
    # pick 20 random values from a range
    if len(prev_starts) > 10:
        iterate_prev = random.sample(list(range(len(prev_starts))), 2)
    else:
        iterate_prev = range(len(prev_starts))
    # take only the part of the previous trajectories between the start and end of the CTE
    prev_org_trajs = [partial_trajectory(prev_org_trajs[x], prev_starts[x], prev_ends_org[x]) for x in iterate_prev]
    prev_cf_trajs = [partial_trajectory(prev_cf_trajs[x], prev_starts[x], prev_ends_cf[x]) for x in iterate_prev]

    # make a list of all the previous original and counterfactual trajectories and their rotations
    all_prev_trajs = prev_org_trajs.copy()
    all_prev_trajs.extend(prev_cf_trajs)
    all_rotated_trajs = [t for x in all_prev_trajs for t in rotated_trajectories(x)]
    length_divs = []
    start_time_divs = []
    up_down_divs = []
    for x in range(len(end_cfs)):
        length_div, start_time_div, up_down_div = diversity(org_traj, cf_trajs[x], starts[x], end_cfs[x], end_orgs[x], prev_org_trajs, prev_cf_trajs, prev_starts, prev_ends_cf, prev_ends_org, all_rotated_trajs)
        length_divs.append(length_div)
        start_time_divs.append(start_time_div)
        up_down_divs.append(up_down_div)
    # length_divs = normalise_values(length_divs)
    # start_time_divs = normalise_values(start_time_divs)
    # start_state_divs = normalise_values(start_state_divs)
    # log_length_divs = [np.log(i) if i != 0 else 0 for i in length_divs]
    # log_start_time_divs = [np.log(i) if i != 0 else 0 for i in start_time_divs]
    log_length_divs = length_divs
    log_start_time_divs = start_time_divs

    sum = [log_length_divs[i] + log_start_time_divs[i] + up_down_divs[i] for i in range(len(length_divs))]
    return sum

def diversity_single(org_traj, cf_traj, start, end_cf, end_org, prev_org_trajs, prev_cf_trajs, prev_starts, prev_ends_cf, prev_ends_org):
    # BUG: This is not normalised in the same way as the diversity_all function. THus it leads to different results.
    # TODO: Fix this
    if len(prev_starts) == 0:
        return 0
    iterate_prev = range(len(prev_starts))
    # take only the part of the previous trajectories between the start and end of the CTE
    prev_org_traj = [partial_trajectory(prev_org_trajs[x], prev_starts[x], prev_ends_org[x]) for x in iterate_prev]
    prev_cf_traj = [partial_trajectory(prev_cf_trajs[x], prev_starts[x], prev_ends_cf[x]) for x in iterate_prev]
    all_prev_traj = prev_org_traj.copy()
    all_prev_traj.extend(prev_cf_traj)
    all_rotated_trajs = [t for x in all_prev_traj for t in rotated_trajectories(x)]
    
    length_div, start_time_div, start_state_div = diversity(org_traj, cf_traj, start, end_cf, end_org, prev_org_traj, prev_cf_traj, prev_starts, prev_ends_cf, prev_ends_org, all_rotated_trajs)
    return length_div + start_time_div + start_state_div
    
    

def up_down_diversity(org_traj, cf_traj, prev_org_trajs, prev_cf_trajs):
    # calculate the average rewards
    org_rewards = np.mean(org_traj['rewards'])
    cf_rewards = np.mean(cf_traj['rewards'])
    diff = org_rewards - cf_rewards

    # calculate the average rewards of the previous trajectories
    prev_org_rewards = [np.mean(x['rewards']) for x in prev_org_trajs]
    prev_cf_rewards = [np.mean(x['rewards']) for x in prev_cf_trajs]
    # calculate the difference between org and cf for the previous trajectories
    prev_diff = [prev_org_rewards[x] - prev_cf_rewards[x] for x in range(len(prev_org_rewards))]
    # count how many prev_diffs are positive and negative (downwards and upwards counterfactuals)
    down = len([x for x in prev_diff if x >= 0])
    up = len([x for x in prev_diff if x < 0])
    # if the current cf is a downwards cf
    if diff >= 0:
        return up / (down + up)
    else:
        return down / (down + up)

def length_diversity(dev, prev_dev):
    # pick out the 3 values in prev_dev that are closest to dev
    # return the average of the 3 values
    # if there are less than 3 values in prev_dev, return the average of all values
    perc = 0.25

    if len(prev_dev) == 0:
        return 0
    else:
        #TODO: This is a very prude implementation. A better version would take a distribution of the possible lengths and sample dev so it covers a new space in the distribution compared to prev_dev
        # if you have 3 or less values take those, otherwise take 25% of the values
        num = max(round(len(prev_dev)*0.25), min(len(prev_dev), 3))
        # pick the num closest values to dev
        diff = [abs(x - dev) for x in prev_dev]
        diff.sort()
        diff = diff[:num]
        return np.mean(diff)
        

def timestep_diversity(point, points):
    # pick out the 3 values in prev_dev that are closest to dev
    # return the average of the 3 values
    # if there are less than 3 values in prev_dev, return the average of all values
    perc = 0.25

    if len(points) == 0:
        return 0
    else:
        #TODO: This is a very prude implementation. A better version would take a distribution of the possible lengths and sample dev so it covers a new space in the distribution compared to prev_dev
        # if you have 3 or less values take those, otherwise take 25% of the values
        num = max(round(len(points)*0.25), min(len(points), 3))
        # pick the num closest values to dev
        diff = [abs(x - point) for x in points]
        diff.sort()
        diff = diff[:num]
        return np.mean(diff)
    

# this method can calculate the diversity of one state-action pair compared to a set of state-action pairs
# this is mostly used to calculate how similar the start state-action pair of a CTE is to the start state-action pairs of previous CTEs
def state_diversity(state, action, prev_states, prev_actions):
    if len(prev_states) == 0:
        return 0
    else:
        #TODO: This implementation just picks the distance to the closest state-action pair (nearest neighbor). Anothe option would be to take the average distance to the k nearest neighbors.
        diff = []
        for x in range(len(prev_states)):
            sad = state_action_diff(state, action, prev_states[x], prev_actions[x])
            # check if the fire-extinguisher is in the usual position (7,7) in the rotated previous state. If not, apply a penalty to the similarity
            if state[0][4][7][7] != prev_states[x][0][4][7][7]: sad += 1
        diff = [state_action_diff(state, action, prev_states[x], prev_actions[x]) for x in range(len(prev_states))]
        return min(diff)
    
# this method helps to rotate the actions of a trajectory
# num_rotations: 0->0°, 1->90°, 2->180°, 3->270°
def map_action(action, num_rotations):
    interact_actions = [5,7,6,8]
    walk_action = [0,2,1,3] 
    if action == 4: return 4
    if action in interact_actions: 
        index = (interact_actions.index(action) + num_rotations) % 4
        return np.array(interact_actions[index], dtype=np.int64)
    if action in walk_action:
        index = (walk_action.index(action) + num_rotations) % 4
        return np.array(walk_action[index], dtype=np.int64)
    return np.array(-1, dtype=np.int64)

# this method takes a (partial) trajectory and returns a list of all possible rotations of that trajectory (0°, 90°, 180°, 270°)
def rotated_trajectories(traj):
    # tmp_traj = torch.Tensor(traj['states'])
    tmp2_traj = torch.stack(traj['states'])
    # torch.rot90(tmp_traj, k=1, dims=[-2, -1])
    torch.rot90(tmp2_traj, k=2, dims=[-2, -1])
    rotated_states1 = torch.rot90(torch.stack(traj['states']), k=1, dims=[-2, -1])
    rotated_states2 = torch.rot90(torch.stack(traj['states']), k=2, dims=[-2, -1])
    rotated_states3 = torch.rot90(torch.stack(traj['states']), k=3, dims=[-2, -1])
    # convert the first dimension of rotated_states to a list

    rotated_states1 = rotated_states1.tolist()
    rotated_states2 = rotated_states2.tolist()
    rotated_states3 = rotated_states3.tolist()

    rotated_states1 = [torch.tensor(x) for x in rotated_states1]
    rotated_states2 = [torch.tensor(x) for x in rotated_states2]
    rotated_states3 = [torch.tensor(x) for x in rotated_states3]

    t1 = {'states': rotated_states1, 'actions': [map_action(x, 1) for x in traj['actions']], 'rewards': traj['rewards']}
    t2 = {'states': rotated_states2 , 'actions': [map_action(x, 2) for x in traj['actions']], 'rewards': traj['rewards']}
    t3 = {'states': rotated_states3, 'actions': [map_action(x, 3) for x in traj['actions']], 'rewards': traj['rewards']}
    return traj, t1, t2, t3

# this method calculates how similar one trajectory is to a set of other trajectories
def part_traj_diversity(traj, rotated_trajs):
    if len(rotated_trajs) == 0:
        return 0
    else:
        for t in rotated_trajs:
            if len(t['states']) == 0:
                a=0
        dist = [distance_subtrajectories(traj, x) for x in rotated_trajs]
        return min(dist)

        
