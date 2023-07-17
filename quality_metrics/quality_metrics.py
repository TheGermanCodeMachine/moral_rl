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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

weight = {'validity': 1, 'proximity': 1, 'critical_state': 0.5, 'diversity': 0.5}

def evaluate_qcs_for_cte(org_traj, counterfactual_traj, start, end_org, end_cf, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs):
    best_val = validity_single(org_traj, counterfactual_traj, start, end_cf, end_org)
    best_prox = distance_single(org_traj, counterfactual_traj, start, end_cf, end_org)
    best_crit = critical_state_single(ppo, org_traj['states'][start])
    best_div = diversity_single(org_traj, counterfactual_traj, start, end_cf, end_org, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
    return best_val, best_prox, best_crit, best_div

def measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs, criteria_to_use):

    validity_qc, proximity_qc, critical_state_qc, diversity_qc = None, None, None, None

    # fig, ax = plt.subplots()
    # xx = range(len(starts))
    qc_values = [(x, 0) for x in range(len(counterfactual_rewards))]
    if 'validity' in criteria_to_use:
        validity_qc = validity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs)
        validity_qc = normalise_values_01(validity_qc)
        # ax.plot(xx, validity_qc, 'ro', label='validity')
        # multiply by weight
        validity_qc = [val * weight['validity'] for val in validity_qc]
        # add to qc_values
        qc_values = [(x, qc_values[x][1] + validity_qc[x]) for x in range(len(counterfactual_rewards))]
    if 'proximity' in criteria_to_use:
        proximity_qc = distance(org_traj, counterfactual_trajs, starts, end_cfs,end_orgs)
        proximity_qc = normalise_values_01(proximity_qc)
        # ax.plot(xx, [-i for i in proximity_qc], 'go', label='proximity')
        proximity_qc = [val * weight['proximity'] for val in proximity_qc]
        qc_values = [(x, qc_values[x][1] - proximity_qc[x]) for x in range(len(counterfactual_rewards))]
    if 'critical_state' in criteria_to_use:
        critical_state_qc = critical_state(ppo, [counterfactual_traj['states'][starts[i]] for i, counterfactual_traj in enumerate(counterfactual_trajs)])
        critical_state_qc = normalise_values_01(critical_state_qc)
        # ax.plot(xx, critical_state_qc, 'bo', label='critical_state')
        critical_state_qc = [val * weight['critical_state'] for val in critical_state_qc]
        qc_values = [(x, qc_values[x][1] + critical_state_qc[x]) for x in range(len(counterfactual_rewards))]
    # IN THIS VERSION OF DIVERSITY WE COMPUTE DIVERSITY FOR ALL COUNTERFACTUALS, WHICH TAKES MORE TIME
    if 'diversity' in criteria_to_use:
        diversity_qc = diversity(org_traj, counterfactual_trajs, starts, end_cfs, end_orgs, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
        diversity_qc = normalise_values_01(diversity_qc)
        # ax.plot(xx, diversity_qc, 'yo', label='diversity')
        diversity_qc = [val * weight['diversity'] for val in diversity_qc]
        qc_values = [(x, qc_values[x][1] + diversity_qc[x]) for x in range(len(counterfactual_rewards))]

    # ax.plot(xx, [i[1] for i in qc_values], 'yo', label='qc')
    # plt.legend()
    # plt.show()
    ## IN THIS CODE WE ONLY USE THE 25% BEST COUNTERFACTUALS TO COMPUTE THE DIVERSITY FOR TO SAVE COMPUTE TIME
    # qc_values.sort(key=lambda x: x[1], reverse=True)
    # full_qc_values = deepcopy(qc_values)
    # best_index = qc_values[0][0]

    # if 'diversity' in criteria_to_use:
    #     # Now we only take the 25% of counterfactuals that perform best on the validity, proximity and critical state metrics and compute diversity for them to save computational time
    #     # pick the  25% best ones
    #     best_qc_values = qc_values[:int(len(qc_values)/4)]
    #     if all([x[1]==0 for x in qc_values]):
    #         best_qc_values = qc_values[:int(len(qc_values))]
    #     #get the indices of the best ones
    #     best_cfs_indices = [x[0] for x in best_qc_values]
    #     # get the best counterfactuals
    #     best_counterfactual_trajs = [counterfactual_trajs[x] for x in best_cfs_indices]
    #     best_starts = [starts[x] for x in best_cfs_indices]
    #     best_end_cfs = [end_cfs[x] for x in best_cfs_indices]
    #     best_end_orgs = [end_orgs[x] for x in best_cfs_indices]

    #     diversity_qc = diversity(org_traj, best_counterfactual_trajs, best_starts, best_end_cfs, best_end_orgs, all_org_trajs, all_cf_trajs, all_starts, all_end_cfs, all_end_orgs)
    #     diversity_qc = normalise_values_01(diversity_qc)
    #     diversity_qc = [val * weight['diversity'] for val in diversity_qc]
    #     # add diversity_qc values to qc_values
    #     best_qc_values = [(x[0], x[1] + diversity_qc[i]) for i,x in enumerate(best_qc_values)]
    #     best_qc_values.sort(key=lambda x: x[1], reverse=True)
    #     qc_values = best_qc_values


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


    qc_val_spear, qc_val_pear = spearmanr([i[1] for i in qc_values], validity_qc)[0], pearsonr([i[1] for i in qc_values], validity_qc)[0]
    print('qc-validity', qc_val_spear, qc_val_pear)
    qc_prox_spear, qc_prox_pear = -spearmanr([i[1] for i in qc_values], proximity_qc)[0], -pearsonr([i[1] for i in qc_values], proximity_qc)[0]
    print('qc-proximity', qc_prox_spear, qc_prox_pear)
    qc_crit_spear, qc_crit_pear = spearmanr([i[1] for i in qc_values], critical_state_qc)[0], pearsonr([i[1] for i in qc_values], critical_state_qc)[0]
    print('qc-critical_state', qc_crit_spear, qc_crit_pear)
    qc_div_spear, qc_div_pear = spearmanr([i[1] for i in qc_values], diversity_qc)[0], pearsonr([i[1] for i in qc_values], diversity_qc)[0]
    print('qc-diversity', qc_div_spear, qc_div_pear)
    val_prox_spear, val_prox_pear = -spearmanr(validity_qc, proximity_qc)[0], -pearsonr(validity_qc, proximity_qc)[0]
    print('validity-proximity', val_prox_spear, val_prox_pear)
    val_crit_spear, val_crit_pear = spearmanr(validity_qc, critical_state_qc)[0], pearsonr(validity_qc, critical_state_qc)[0]
    print('validity-critical_state', val_crit_spear, val_crit_pear)
    val_div_spear, val_div_pear = spearmanr(validity_qc, diversity_qc)[0], pearsonr(validity_qc, diversity_qc)[0]
    print('validity-diversity', val_div_spear, val_div_pear)
    prox_crit_spear, prox_crit_pear = -spearmanr(proximity_qc, critical_state_qc)[0], -pearsonr(proximity_qc, critical_state_qc)[0]
    print('proximity-critical_state', prox_crit_spear, prox_crit_pear)
    prox_div_spear, prox_div_pear = -spearmanr(proximity_qc, diversity_qc)[0], -pearsonr(proximity_qc, diversity_qc)[0]
    print('proximity-diversity', prox_div_spear, prox_div_pear)
    crit_div_spear, crit_div_pear = spearmanr(critical_state_qc, diversity_qc)[0], pearsonr(critical_state_qc, diversity_qc)[0]
    print('critical_state-diversity', crit_div_spear, crit_div_pear)
    qc_values.sort(key=lambda x: x[1], reverse=True)
    best_index = qc_values[0][0]        
    chosen_val = validity_qc[best_index]
    chosen_prox =  -proximity_qc[best_index]+1
    chosen_crit = 2*critical_state_qc[best_index]
    chosen_div = 2*diversity_qc[best_index]
    print('VALUES: validity:', validity_qc[best_index], 'proximity:', -proximity_qc[best_index]+1, 'critical_state:', 2*critical_state_qc[best_index], 'diversity:', 2*diversity_qc[best_index], 'qc:', qc_values[0][1])

    validity_qc = [(i,j) for i,j in enumerate(validity_qc)]
    validity_qc.sort(key=lambda x: x[1], reverse=True)
    proximity_qc = [(i,-j+1) for i,j in enumerate(proximity_qc)]
    proximity_qc.sort(key=lambda x: x[1], reverse=True)
    critical_state_qc = [(i,2*j) for i,j in enumerate(critical_state_qc)]
    critical_state_qc.sort(key=lambda x: x[1], reverse=True)
    diversity_qc = [(i,2*j) for i,j in enumerate(diversity_qc)]
    diversity_qc.sort(key=lambda x: x[1], reverse=True)

    # find the position of the best_index in each of the sorted lists
    pos_val = np.where(np.array([i[0] for i in validity_qc])==best_index)[0][0]
    pos_prox = np.where(np.array([i[0] for i in proximity_qc])==best_index)[0][0]
    pos_crit = np.where(np.array([i[0] for i in critical_state_qc])==best_index)[0][0]
    pos_div = np.where(np.array([i[0] for i in diversity_qc])==best_index)[0][0]
    print('POSITIONS: validity:', pos_val / len(validity_qc), 'proximity:', pos_prox /len(proximity_qc), 'critical_state:', pos_crit/len(critical_state_qc), 'diversity:', pos_div/len(diversity_qc))

    fix, ax = plt.subplots()
    ax.plot(range(len(validity_qc)), [i[1] for i in validity_qc], 'ro', label='validity')
    ax.plot(range(len(proximity_qc)), [i[1] for i in proximity_qc], 'go', label='proximity')
    ax.plot(range(len(critical_state_qc)), [i[1] for i in critical_state_qc], 'bo', label='critical_state')
    ax.plot(range(len(diversity_qc)), [i[1] for i in diversity_qc], 'co', label='diversity')

    
    ax.plot(range(len(qc_values)), [i[1] for i in qc_values], 'yo', label='qc')
    plt.legend()
    plt.show()

    spear_correlations = {'qc-validity': qc_val_spear, 'qc-proximity': qc_prox_spear, 'qc-critical_state': qc_crit_spear, 'qc-diversity': qc_div_spear, 'validity-proximity': val_prox_spear, 'validity-critical_state': val_crit_spear, 'validity-diversity': val_div_spear, 'proximity-critical_state': prox_crit_spear, 'proximity-diversity': prox_div_spear, 'critical_state-diversity': crit_div_spear}
    pear_correlations = {'qc-validity': qc_val_pear, 'qc-proximity': qc_prox_pear, 'qc-critical_state': qc_crit_pear, 'qc-diversity': qc_div_pear, 'validity-proximity': val_prox_pear, 'validity-critical_state': val_crit_pear, 'validity-diversity': val_div_pear, 'proximity-critical_state': prox_crit_pear, 'proximity-diversity': prox_div_pear, 'critical_state-diversity': crit_div_pear}
    perc_positions = {'validity': pos_val / len(validity_qc), 'proximity': pos_prox /len(proximity_qc), 'critical_state': pos_crit/len(critical_state_qc), 'diversity': pos_div/len(diversity_qc)}
    chosen_values = {'validity': chosen_val, 'proximity': chosen_prox, 'critical_state': chosen_crit, 'diversity': chosen_div, 'qc': qc_values[0][1]}
    statistics = {'spear_correlations': spear_correlations, 'pear_correlations': pear_correlations, 'perc_positions': perc_positions, 'chosen_values': chosen_values}

    max_index = qc_values[0][0]
    return max_index, statistics