import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))

import numpy as np
import torch
from utils.util_functions import state_action_diff, distance_subtrajectories, extract_player_position, normalise_values
from quality_metrics.distance_measures import my_distance

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

    prev_first_states = [i['states'][0] for i in prev_org_traj]
    prev_first_actions = [i['actions'][0] for i in prev_org_traj]
    start_state_div = state_diversity(traj1["states"][start], traj1["actions"][start], prev_first_states, prev_first_actions)
    # prev_last_states = [i['states'][-1] for i in prev_org_traj]
    # prev_last_actions = [i['actions'][-1] for i in prev_org_traj]
    # end_state_div = state_diversity(traj1['states'][end_org], traj1['actions'][end_org], prev_last_states, prev_last_actions)
    # prev_last_cf_states = [i['states'][-1] for i in prev_cf_traj]
    # prev_last_cf_actions = [i['actions'][-1] for i in prev_cf_traj]
    # endcf_state_div = state_diversity(traj2['states'][end_cf], traj2['actions'][end_cf], prev_last_cf_states, prev_last_cf_actions)


    # This makes the code super inefficient because it causes thousands of calls to the state_action_diff function
    # part_traj1 = {'states': traj1['states'][start:end_org+1], 'actions': traj1['actions'][start:end_org+1], 'rewards': traj1['rewards'][start:end_org+1]}
    # part_traj2 = {'states': traj2['states'][start:end_cf+1], 'actions': traj2['actions'][start:end_cf+1], 'rewards': traj2['rewards'][start:end_cf+1]}
    # trajectory_div_org = part_traj_diversity(part_traj1, all_rotated_trajs)
    # trajectory_div_cf = part_traj_diversity(part_traj2, all_rotated_trajs)

    # print("length_div: ", length_div, "start_time_div: ", start_time_div, "start_state_div: ", start_state_div)
    return length_div, start_time_div, start_state_div # + trajectory_div_org + trajectory_div_cf

def diversity_all(org_traj, cf_trajs, starts, end_cfs, end_orgs, prev_traj_orgs, prev_traj_cfs, prev_starts, prev_ends_cf, prev_ends_org):
    if len(prev_starts) == 0:
        return [0 for x in range(len(end_cfs))]
    iterate_prev = range(len(prev_starts))
    # take only the part of the previous trajectories between the start and end of the CTE
    prev_org_traj = [{'states': prev_traj_orgs[x]['states'][prev_starts[x]:prev_ends_org[x]+1], 'actions': prev_traj_orgs[x]['actions'][prev_starts[x]:prev_ends_org[x]+1], 'rewards': prev_traj_orgs[x]['rewards'][prev_starts[x]:prev_ends_org[x]+1]} for x in iterate_prev]
    prev_cf_traj = [{'states': prev_traj_cfs[x]['states'][prev_starts[x]:prev_ends_cf[x]+1], 'actions': prev_traj_cfs[x]['actions'][prev_starts[x]:prev_ends_cf[x]+1], 'rewards': prev_traj_cfs[x]['rewards'][prev_starts[x]:prev_ends_cf[x]+1]} for x in iterate_prev]

    # make a list of all the previous original and counterfactual trajectories and their rotations
    all_prev_traj = prev_org_traj.copy()
    all_prev_traj.extend(prev_cf_traj)
    all_rotated_trajs = [t for x in all_prev_traj for t in rotated_trajectories(x)]
    length_divs = []
    start_time_divs = []
    start_state_divs = []
    for x in range(len(end_cfs)):
        length_div, start_time_div, start_state_div = diversity(org_traj, cf_trajs[x], starts[x], end_cfs[x], end_orgs[x], prev_org_traj, prev_cf_traj, prev_starts, prev_ends_cf, prev_ends_org, all_rotated_trajs)
        length_divs.append(length_div)
        start_time_divs.append(start_time_div)
        start_state_divs.append(start_state_div)
    length_divs = normalise_values(length_divs)
    start_time_divs = normalise_values(start_time_divs)
    start_state_divs = normalise_values(start_state_divs)
    return length_divs + start_time_divs + start_state_divs

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
        diff = [state_action_diff(state, action, prev_states[x], prev_actions[x]) for x in range(len(prev_states))]
        return min(diff)
    
# this method helps to rotate the actions of a trajectory
# num_rotations: 0->0°, 1->90°, 2->180°, 3->270°
def map_action(action, num_rotations):
    grab_actions = [5,7,6,8]
    walk_action = [0,2,1,3] 
    if action == 4: return 4
    if action in grab_actions: 
        index = (grab_actions.index(action) + num_rotations) % 4
        return np.array(grab_actions[index], dtype=np.int64)
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

        
