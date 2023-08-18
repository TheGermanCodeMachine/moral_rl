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
from generation_methods.counterfactual_random import generate_counterfactual_random

normalisation = {'validity': 0, 'diversity': 0, 'proximity': 0, 'critical_state': 0, 'realisticness': 0, 'sparsity': 0}

def normalising_qcs(ppo, discriminator, org_traj_seed, config):
    # load original trajectories
    proxs, vals, divs, crits, spars, reals = [], [], [], [], [], []
    prev_org_trajs, prev_cf_trajs, prev_starts = [], [], []
    num = 0
    for org_traj, seed_env in org_traj_seed:
        random_org, random_cf, random_start = generate_counterfactual_random(org_traj, ppo, discriminator, seed_env, config)
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
        if num == 20:
            break

    return normalisation