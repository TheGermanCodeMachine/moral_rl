import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from copy import deepcopy
import numpy as np
import time
from helpers.util_functions import *
from quality_metrics.validity_measures import validity_all as validity
from quality_metrics.validity_measures import validity_single
from quality_metrics.distance_measures import distance_all as distance
from quality_metrics.distance_measures import distance_single
from quality_metrics.diversity_measures import diversity_all as diversity
from quality_metrics.diversity_measures import diversity_single
from quality_metrics.critical_state_measures import critical_state_all as critical_state
from quality_metrics.critical_state_measures import critical_state_single

weight = {'validity': 1, 'proximity': 1, 'critical_state': 0.5, 'diversity': 0.5}

def evaluate_qcs_for_cte(org_traj, counterfactual_traj, start, end_org, end_cf, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs):
    best_val = validity_single(org_traj, counterfactual_traj, start, end_cf, end_org)
    best_prox = distance_single(org_traj, counterfactual_traj, start, end_cf, end_org)
    best_crit = critical_state_single(ppo, org_traj['states'][start])
    best_div = diversity_single(org_traj, counterfactual_traj, start, end_cf, end_org, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
    return best_val, best_prox, best_crit, best_div

def measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs, criteria_to_use):

    validity_qc, proximity_qc, critical_state_qc, diversity_qc = None, None, None, None

    qc_values = [(x, 0) for x in range(len(counterfactual_rewards))]
    if 'validity' in criteria_to_use:
        validity_qc = validity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs)
        validity_qc = normalise_values(validity_qc)
        # multiply by weight
        validity_qc = [val * weight['validity'] for val in validity_qc]
        # add to qc_values
        qc_values = [(x, qc_values[x][1] + validity_qc[x]) for x in range(len(counterfactual_rewards))]
    if 'proximity' in criteria_to_use:
        proximity_qc = distance(org_traj, counterfactual_trajs, starts, end_cfs,end_orgs)
        proximity_qc = normalise_values(proximity_qc)
        proximity_qc = [val * weight['proximity'] for val in proximity_qc]
        qc_values = [(x, qc_values[x][1] - proximity_qc[x]) for x in range(len(counterfactual_rewards))]
    if 'critical_state' in criteria_to_use:
        critical_state_qc = critical_state(ppo, org_traj['states'])
        critical_state_qc = normalise_values(critical_state_qc)
        critical_state_qc = [val * weight['critical_state'] for val in critical_state_qc]
        qc_values = [(x, qc_values[x][1] + critical_state_qc[x]) for x in range(len(counterfactual_rewards))]
    # sort the qc_values in descending order
    qc_values.sort(key=lambda x: x[1], reverse=True)

    if 'diversity' in criteria_to_use:
        # Now we only take the 25% of counterfactuals that perform best on the validity, proximity and critical state metrics and compute diversity for them to save computational time
        # pick the  25% best ones
        best_qc_values = qc_values[:int(len(qc_values)/4)]
        if all([x[1] for x in qc_values] == 0):
            best_qc_values = qc_values[:int(len(qc_values))]
        #get the indices of the best ones
        best_cfs_indices = [x[0] for x in best_qc_values]
        # get the best counterfactuals
        best_counterfactual_trajs = [counterfactual_trajs[x] for x in best_cfs_indices]
        best_starts = [starts[x] for x in best_cfs_indices]
        best_end_cfs = [end_cfs[x] for x in best_cfs_indices]
        best_end_orgs = [end_orgs[x] for x in best_cfs_indices]

        diversity_qc = diversity(org_traj, best_counterfactual_trajs, best_starts, best_end_cfs, best_end_orgs, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
        diversity_qc = normalise_values(diversity_qc)
        diversity_qc = [val * weight['diversity'] for val in diversity_qc]
        # add diversity_qc values to qc_values
        best_qc_values = [(x[0], x[1] + diversity_qc[i]) for i,x in enumerate(best_qc_values)]
        best_qc_values.sort(key=lambda x: x[1], reverse=True)
        qc_values = best_qc_values

        # # THIS CODE TEST WHETHER LEAVING OUT THE OPTIONS THAT SCORE WORSE ON THE QC METRICS IGNORES SOME CFs THAT WOULD HAVE OTHERWISE BEEN GOOD
        # qc_values_all = deepcopy(qc_values)
        # # times = time.time()
        # # diversity_qc_all = diversity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
        # # print("time for diversity_all: ", time.time() - times)
        # # qc_values_all = [(x[0], diversity_qc_all[x[0]]) for i,x in enumerate(qc_values_all)]
        # # qc_values_all.sort(key=lambda x: x[1], reverse=True)
        # # best_qc_values.sort(key=lambda x: x[1], reverse=True)    
        # # print out indices of the top 5 counterfactuals
        # # print("top 5 counterfactuals: ", [x[1] for x in best_qc_values[:5]])
        # # print("top 5 counterfactuals all: ", [x[1] for x in qc_values_all[:5]])
        # # print("diverstiy_qc_all", sum(diversity_qc_all)/len(diversity_qc_all), "validity_qc_all", sum(validity_qc)/len(validity_qc), "proximity_qc_all", sum(proximity_qc)/len(proximity_qc), "critical_state_qc_all", sum(critical_state_qc)/len(critical_state_qc))
        # # print("diverstiy_qc", max(diversity_qc), "validity_qc", max(validity_qc), "proximity_qc", max(proximity_qc), "critical_state_qc", max(critical_state_qc)

    max_index = qc_values[0][0]
    return max_index