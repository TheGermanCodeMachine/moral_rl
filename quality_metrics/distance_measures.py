# in this file there are multiple measures for the distance/similarity between two trajectories
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))

from skimage.metrics import hausdorff_distance
import numpy as np
import torch
from utils.util_functions import distance_subtrajectories

# the modified hausdorff distance
# see: Dubuisson, M. P., & Jain, A. K. (1994, October). A modified Hausdorff distance for object matching. In Proceedings of 12th international conference on pattern recognition (Vol. 1, pp. 566-568). IEEE.
def mdh(traj1, traj2, start_part, end_part_cf, end_part_org):
    """
    Mean Displacement of Head (MDH) is a measure of the distance between two trajectories
    :param traj1: trajectory 1
    :param traj2: trajectory 2
    :param start_part: where the two trajectories start deviating
    :param end_part_cf: where the deviation of the counterfactual trajectory ends
    :param end_part_org: timestep in the original trajectory where the counterfactual rejoins
    :return: MDH
    """
    
    # select the parts of the trajectories that make the explanation (the deviations) & translate this list of tensorss to an ndarray
    traj1_array = torch.stack(traj1['states'][start_part:end_part_org+1]).squeeze().numpy()
    traj2_array = torch.stack(traj2['states'][start_part:end_part_cf+1]).squeeze().numpy()
    
    distance = hausdorff_distance(traj1_array, traj2_array, method='modified')
    
    return distance

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
    traj1_sub = {'states': traj1['states'][start_part:end_part_org+1], 'actions': traj1['actions'][start_part:end_part_org+1]}
    traj2_sub = {'states': traj2['states'][start_part:end_part_cf+1], 'actions': traj2['actions'][start_part:end_part_cf+1]}
    return distance_subtrajectories(traj1_sub, traj2_sub)

def distance_all(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs):
    dist = []
    for i in range(len(counterfactual_trajs)):
        dist.append(my_distance(org_traj, counterfactual_trajs[i], starts[i], end_cfs[i], end_orgs[i]))
    return dist

def distance_single(org_traj, counterfactual_traj, start, end_cf, end_org):
    return my_distance(org_traj, counterfactual_traj, start, end_cf, end_org)