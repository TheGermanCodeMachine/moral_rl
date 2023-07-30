
# This criteria measures how long the trajectories are. The shorter, the more sparse. It returns the negative sum of the length of the original and counterfactual trajectories.

def sparsity_all(starts, end_cfs, end_orgs):
    spars = [-(end_cfs[i] + end_orgs[i] - 2*starts[i]) for i in range(len(starts))]
    return spars

def sparsitiy_single_partial(part_org, part_cf):
    return - len(part_org['rewards']) - len(part_cf['rewards'])