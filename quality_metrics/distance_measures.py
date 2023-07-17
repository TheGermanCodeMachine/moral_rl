# in this file there are multiple measures for the distance/similarity between two trajectories
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))

from skimage.metrics import hausdorff_distance
import numpy as np
import torch
import time
import keyboard
from helpers.util_functions import partial_trajectory, extract_player_position
from helpers.visualize_trajectory import visualize_two_part_trajectories

# (Unused) Calculates how many states are different between two trajectories
def deviation_count(traj1, traj2, start_part, end_part_cf, end_part_org):
    return len(end_part_cf - start_part)

def my_distance(traj1, traj2, start_part, end_part_cf, end_part_org):
    """
    My distance is a measure of the distance between two trajectories
    :param traj1: trajectory 1
    :param traj2: trajectory 2
    :param start_part: where the two trajectories start deviating
    :param end_part_cf: where the deviation of the counterfactual trajectory ends
    :param end_part_org: timestep in the original trajectory where the counterfactual rejoins
    :return: distance
    """
        
    # select subtrajectory between start_part and end_part_cf
    traj1_sub = partial_trajectory(traj1, start_part, end_part_org)
    traj2_sub = partial_trajectory(traj2, start_part, end_part_cf)
    return distance_subtrajectories(traj1_sub, traj2_sub)

def distance_all(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs):
    dist = []
    for i in range(len(counterfactual_trajs)):
        distance = my_distance(org_traj, counterfactual_trajs[i], starts[i], end_cfs[i], end_orgs[i])
        
        dist.append(distance)
        # visualize_two_part_trajectories(org_traj, counterfactual_trajs[i], starts[i], end_cfs[i], end_orgs[i])
    return dist

def distance_single(org_traj, counterfactual_traj, start, end_cf, end_org):
    return my_distance(org_traj, counterfactual_traj, start, end_cf, end_org)

# Modified Hausdorff distance between two trajectories (see: Dubuisson, M. P., & Jain, A. K. (1994, October). A modified Hausdorff distance for object matching. In Proceedings of 12th international conference on pattern recognition (Vol. 1, pp. 566-568). IEEE.)
def distance_subtrajectories(traj1, traj2):
    # calucalte the distance between every state-action pair in traj1 and traj2
    dist_table = np.zeros((len(traj1['states']), len(traj2['states'])))
    act_diffs, cit_divs, pos_divs = np.zeros((len(traj1['states']), len(traj2['states']))), np.zeros((len(traj1['states']), len(traj2['states']))), np.zeros((len(traj1['states']), len(traj2['states'])))
    for i in range(len(traj1['states'])):
        for j in range(len(traj2['states'])):
            dist_table[i,j], act_diffs[i,j], cit_divs[i,j], pos_divs[i,j] = state_action_diff(traj1['states'][i], traj1['actions'][i], traj2['states'][j], traj2['actions'][j])

    dist_A_B = np.mean(np.min(dist_table, axis=1))
    dist_B_A = np.mean(np.min(dist_table, axis=0))
    deviation = 0.05*len((traj1['states'])) + 0.05*len((traj2['states']))
    # if dist_A_B > dist_B_A:
    #     indices = np.argmin(dist_table, axis=1)
    #     indices0 = np.argmin(dist_table, axis=0)
    #     # get the act_diffs at the indices
    #     act_div = np.mean([act_diffs[i, indices[i]] for i in range(act_diffs.shape[0])])
    #     cit_div = np.mean([cit_divs[i, indices[i]] for i in range(act_diffs.shape[0])])
    #     pos_div = np.mean([pos_divs[i, indices[i]] for i in range(act_diffs.shape[0])])
    # else:
    #     indices = np.argmin(dist_table, axis=0)
    #     act_div = np.mean([act_diffs[indices[i], i] for i in range(act_diffs.shape[1])])
    #     cit_div = np.mean([cit_divs[indices[i], i] for i in range(act_diffs.shape[1])])
    #     pos_div = np.mean([pos_divs[indices[i], i] for i in range(act_diffs.shape[1])])
    # print(round(max(dist_A_B, dist_B_A)+deviation,1), round(act_div,1), round(cit_div,1), round(pos_div,1), round(deviation,1), round(len((traj1['states']))))
    # keyboard.read_event()
    sum = max(dist_A_B, dist_B_A) + deviation
    # take log of sum
    return np.log(sum)

# this method calculates how different two state-action pairs in the randomized_v2 environment are
# WARNING: this method works sensibly only for the randomized_v2 environment; other enivronments will need a different implementation
def state_action_diff(s1, a1, s2, a2):
    dist = 0
    # add 1 if the action is different
    if a1 != a2: dist += 0.5
    if a1 != a2: act_div = 0.5 
    else: act_div = 0
    # add the edit distance between the two matrices of unsaved citizens
    dist += np.sum(np.abs(s1[0][2] - s2[0][2]).detach().numpy())
    cit_div = np.sum(np.abs(s1[0][2] - s2[0][2]).detach().numpy())

    # manhattan distance between the player positions
    pos_1 = extract_player_position(s1)
    pos_2 = extract_player_position(s2)
    dist += (abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1]))*1.5
    pos_div = abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])*1.5
    return dist, act_div, cit_div, pos_div