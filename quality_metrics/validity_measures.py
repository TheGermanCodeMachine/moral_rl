import numpy as np

def abs_diff(r1, r2):
    return abs(r1-r2)

def threshold_diff(r1, r2, threshold):
    return abs(r1-r2) > threshold

def validity_all(org, cfs, starts, end_cfs, end_orgs):
    vals = []
    for r in range(len(starts)):
        org_reward = np.mean(org['rewards'][starts[r]:end_orgs[r]+1])
        cf_reward = np.mean(cfs[r]['rewards'][starts[r]:end_cfs[r]+1])
        vals.append(abs_diff(org_reward, cf_reward))
    return vals