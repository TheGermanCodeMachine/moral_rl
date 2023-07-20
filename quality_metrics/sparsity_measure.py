


def sparsity_all(starts, end_cfs, end_orgs):
    spars = [-(end_cfs[i] + end_orgs[i] - 2*starts[i]) for i in range(len(starts))]
    return spars