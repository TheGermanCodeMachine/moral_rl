import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from quality_metrics.distance_measures import distance_subtrajectories
from quality_metrics.diversity_measures import diversity_single
from quality_metrics.validity_measures import validity_single_partial
from quality_metrics.critical_state_measures import critical_state_single
from quality_metrics.realisticness_measures import realisticness_single_partial
from quality_metrics.sparsity_measure import sparsitiy_single_partial
from interpretability.generation_methods.counterfactual_random import generate_counterfactual_random
from interpretability.generation_methods.counterfactual_mcts import generate_counterfactual_mcts
from interpretability.generation_methods.counterfactual_step import generate_counterfactual_step
from quality_metrics.quality_metrics import measure_quality
import numpy as np
import pickle
from helpers.util_functions import partial_trajectory


def normalising_qcs(ppo, discriminator, org_traj_seed, config):
    # load original trajectories
    proxs, vals, divs, crits, spars, reals = [], [], [], [], [], []
    prev_org_trajs, prev_cf_trajs, prev_starts = [], [], []
    num = 0
    for org_traj, seed_env in org_traj_seed:
        print(num)
        random_org, random_cf, random_start = generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env, prev_org_trajs, prev_cf_trajs, prev_starts, config)
        proxs.append(distance_subtrajectories(random_org, random_cf))
        vals.append(validity_single_partial(random_org, random_cf))
        divs.append(diversity_single(random_org, random_cf, random_start, prev_org_trajs, prev_cf_trajs, prev_starts))
        crits.append(critical_state_single(ppo, random_org['states'][0]))
        spars.append(sparsitiy_single_partial(random_org, random_cf))
        reals.append(realisticness_single_partial(random_org, random_cf))
        prev_org_trajs.append(random_org)
        prev_cf_trajs.append(random_cf)
        prev_starts.append(random_start)
        num += 1
        if num == 10:
            break
    
    num = 0
    prev_org_trajs, prev_cf_trajs, prev_starts = [], [], []
    for org_traj, seed_env in org_traj_seed:
        if num < 10:
            num += 1
            continue
        if num == 20:
            break
        print(num)
        num+=1
        counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs = generate_counterfactual_step(org_traj, ppo, discriminator, seed_env, config)
        sort_index, qc_stats = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, prev_org_trajs, prev_cf_trajs, prev_starts, config.criteria)
        chosen_counterfactual_trajectory = counterfactual_trajs[sort_index]
        chosen_start = starts[sort_index]
        chosen_end_cf = end_cfs[sort_index]
        chosen_end_org = end_orgs[sort_index]
        step_org = partial_trajectory(org_traj, chosen_start, chosen_end_org)
        step_cf = partial_trajectory(chosen_counterfactual_trajectory, chosen_start, chosen_end_cf)
        prev_org_trajs.append(step_org)
        prev_cf_trajs.append(step_cf)
        prev_starts.append(chosen_start)

        proxs.append(distance_subtrajectories(step_org, step_cf))
        vals.append(validity_single_partial(step_org, step_cf))
        divs.append(diversity_single(step_org, step_cf, random_start, prev_org_trajs, prev_cf_trajs, prev_starts))
        crits.append(critical_state_single(ppo, step_org['states'][0]))
        spars.append(sparsitiy_single_partial(step_org, step_cf))
        reals.append(realisticness_single_partial(step_org, step_cf))

    normalisation = {'validity': [min(vals), max(vals)], 'diversity': [min(divs), max(divs)], 'proximity': [min(proxs), max(proxs)], 'critical_state': [min(crits), max(crits)], 'realisticness': [min(reals), max(reals)], 'sparsity': [min(spars), max(spars)]}
    print(normalisation)
    # write into pickle
    with open('interpretability\\normalisation_values_new.pkl', 'wb') as f:
        pickle.dump(normalisation, f)