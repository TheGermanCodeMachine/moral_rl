from helpers.util_functions import partial_trajectory


# measures how realistic a counterfactual trajectory is by comparing the average reward per step to that of the original
# returns the difference between average rewards normalised by the average reward of the original trajectory
def realisticness_all(org_traj, cfs, starts, end_cfs, end_orgs):
    real = []
    for i in range(len(cfs)):
        org_part = partial_trajectory(org_traj, starts[i], end_orgs[i])
        avg_org_reward = sum(org_part['rewards'])/len(org_part['rewards'])
        cf_part = partial_trajectory(cfs[i], starts[i], end_cfs[i])
        avg_reward = sum(cf_part['rewards'])/len(cf_part['rewards'])

        diff = (avg_reward - avg_org_reward)/avg_org_reward
        real.append(diff)
    return real