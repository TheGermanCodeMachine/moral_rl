import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from copy import deepcopy
import numpy as np
import time
from utils.util_functions import *
from quality_metrics.validity_measures import validity_all as validity
from quality_metrics.distance_measures import distance_all as distance
from quality_metrics.diversity_measures import diversity_all as diversity
from quality_metrics.critical_state_measures import critical_state_all as critical_state

def print_prox_val(validity_qc, proximity_qc):
    for i in range(len(validity_qc)):
        print("i: ", i, "Validity: ", validity_qc[i], " Proximity: ", proximity_qc[i], "sum: ", validity_qc[i] - proximity_qc[i])
    # average validity and proximity
    print("Average validity: ", np.mean(validity_qc), " Average proximity: ", np.mean(proximity_qc))
    # variance of validity and proximity
    print("Variance validity: ", np.var(validity_qc), " Variance proximity: ", np.var(proximity_qc))
    # correlation between validity and proximity
    print("Correlation validity and proximity: ", np.corrcoef(validity_qc, proximity_qc))


#NOTE: currently i am only using proximity and validity
def measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs, criteria_to_use):
    # alpha determines how much to prioritise similarity of trajectories or difference in outputs
    weight = {'validity': 1, 'proximity': 1, 'critical_state': 0.5, 'diversity': 0.5}

    qc_values = [(x, 0) for x in range(len(counterfactual_rewards))]
    if 'validity' in criteria_to_use:
        validity_qc = validity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs)
        validity_qc = normalise_values(validity_qc)
        validity_qc = [val * weight['validity'] for val in validity_qc]
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
    # print_prox_val(validity_qc, proximity_qc)
    # Now we only take the 25% of counterfactuals that perform best on the validity, proximity and critical state metrics and compute diversity for them to save computational time
    qc_values.sort(key=lambda x: x[1], reverse=True)

    if 'diversity' in criteria_to_use:
        # Now we only take the 25% of counterfactuals that perform best on the validity, proximity and critical state metrics and compute diversity for them to save computational time
        # pick the  25% best ones
        best_qc_values = qc_values[:int(len(qc_values)/4)]
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

        return best_qc_values   

    return qc_values

def get_all_combinations_of_qc(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs):
    # alpha determines how much to prioritise similarity of trajectories or difference in outputs
    alpha = 0.1
    beta = 1

    validity_qc = validity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs)
    proximity_qc = distance(org_traj, counterfactual_trajs, starts, end_cfs,end_orgs)
    critical_state_qc = critical_state(ppo, org_traj['states'])
    # Now we only take the 25% of counterfactuals that perform best on the validity, proximity and critical state metrics and compute diversity for them to save computational time
    only_validity = [(x, validity_qc[x]) for x in range(len(counterfactual_rewards))]
    best_only_validity = only_validity.sort(key=lambda x: x[1], reverse=True)
    only_proximity = [(x, proximity_qc[x]) for x in range(len(counterfactual_rewards))]
    best_only_proximity = only_proximity.sort(key=lambda x: x[1], reverse=True)
    proximity_validity = [(x, validity_qc[x] - proximity_qc[x]) for x in range(len(counterfactual_rewards))]
    best_proximity_validity = proximity_validity.sort(key=lambda x: x[1], reverse=True)
    proximity_validity_critical = [(x, validity_qc[x] - proximity_qc[x] + critical_state_qc[x]) for x in range(len(counterfactual_rewards))]
    best_proximity_validity_critical = proximity_validity_critical.sort(key=lambda x: x[1], reverse=True)


    # pick the  25% best ones
    best_qc_values = best_proximity_validity[:int(len(best_proximity_validity)/4)]
    #get the indices of the best ones
    best_cfs_indices = [x[0] for x in best_qc_values]
    # get the best counterfactuals
    best_counterfactual_trajs = [counterfactual_trajs[x] for x in best_cfs_indices]
    best_starts = [starts[x] for x in best_cfs_indices]
    best_end_cfs = [end_cfs[x] for x in best_cfs_indices]
    best_end_orgs = [end_orgs[x] for x in best_cfs_indices]

    diversity_qc = diversity(org_traj, best_counterfactual_trajs, best_starts, best_end_cfs, best_end_orgs, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
    # add diversity_qc values to qc_values
    best_qc_values = [(x[0], x[1] + diversity_qc[i]) for i,x in enumerate(best_qc_values)]

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
    # # print("diverstiy_qc", max(diversity_qc), "validity_qc", max(validity_qc), "proximity_qc", max(proximity_qc), "critical_state_qc", max(critical_state_qc))

    proximity_validity_diversity = [(x, validity_qc[x] - proximity_qc[x] + diversity_qc[x]) for x in range(len(counterfactual_rewards))]
    best_proximity_validity_diversity = proximity_validity_diversity.sort(key=lambda x: x[1], reverse=True)[0][0]
    proximity_validity_diversity_critical = [(x, validity_qc[x] - proximity_qc[x] + diversity_qc[x] + critical_state_qc[x]) for x in range(len(counterfactual_rewards))]
    best_proximity_validity_diversity_critical = proximity_validity_diversity_critical.sort(key=lambda x: x[1], reverse=True)[0][0]

    return best_only_validity, best_only_proximity, best_proximity_validity, best_proximity_validity_critical, best_proximity_validity_diversity, best_proximity_validity_diversity_critical