import numpy as np

def validity_all(org, cfs, starts, end_cfs, end_orgs):
    vals = []
    for r in range(len(starts)):
        org_reward = np.mean(org['rewards'][starts[r]:end_orgs[r]+1])
        cf_reward = np.mean(cfs[r]['rewards'][starts[r]:end_cfs[r]+1])
        vals.append(abs(org_reward-cf_reward))
    return vals

def validity_single(org, cf, start, end_cf, end_org):
    org_reward = np.mean(org['rewards'][start:end_org+1])
    cf_reward = np.mean(cf['rewards'][start:end_cf+1])
    return abs(org_reward - cf_reward)

def validity_single_partial(part_org, part_cf):
    org_reward = np.mean(part_org['rewards'])
    cf_reward = np.mean(part_cf['rewards'])
    return abs(org_reward - cf_reward)