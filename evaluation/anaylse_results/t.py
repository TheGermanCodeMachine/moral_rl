import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent.parent
sys.path.append(str(adjacent_folder))
from copy import deepcopy
import pickle as pkl
import numpy as np
import torch
import os
from quality_metrics.diversity_measures import diversity_single

weight = {'validity': 1, 'proximity': 1, 'critical_state': 0.5, 'diversity': 0.5, 'realisticness': 0.2, 'sparsity': 0.5}
with open('interpretability\\normalisation_values_new.pkl', 'rb') as f:
    normalisation = pkl.load(f)


with open('datasets\\100random\\baseline\cf_trajectories.pkl', 'rb') as f:
    cf_trajs = pkl.load(f)
with open('datasets\\100random\\baseline\org_trajectories.pkl', 'rb') as f:
    org_trajs = pkl.load(f)
with open('datasets\\100random\\baseline\statistics\start_points.pkl', 'rb') as f:
    starts = pkl.load(f)

old_orgs = []
old_cfs = []
old_starts = []
divs_random = []
for i in range(len(cf_trajs)):
    div = diversity_single(org_trajs[i][0], cf_trajs[i][0], starts[i], old_orgs, old_cfs, old_starts)
    divs_random.append(div)
    old_orgs.append(deepcopy(org_trajs[i][0]))
    old_cfs.append(deepcopy(cf_trajs[i][0]))
    old_starts.append(starts[i])

with open('datasets\\1000\\1000\cf_trajectories.pkl', 'rb') as f:
    cf_trajs = pkl.load(f)[:100]
with open('datasets\\1000\\1000\org_trajectories.pkl', 'rb') as f:
    org_trajs = pkl.load(f)[:100]
with open('datasets\\1000\\1000\statistics\start_points.pkl', 'rb') as f:
    starts = pkl.load(f)[:100]


old_orgs = []
old_cfs = []
old_starts = []
divs_step = []
for i in range(len(cf_trajs)):
    div = diversity_single(org_trajs[i][0], cf_trajs[i][0], starts[i], old_orgs, old_cfs, old_starts)
    divs_step.append(div)
    old_orgs.append(deepcopy(org_trajs[i][0]))
    old_cfs.append(deepcopy(cf_trajs[i][0]))
    old_starts.append(starts[i])


# read line by line through a text file
saved_lines = []
with open('interpretability\logs\qc_comparison.txt', 'r') as f:
    lines = f.readlines()
    i=0
    for line in lines:
        if 'diversity' in line:
            split = line.split(' ')
            new_line = split[0] + ' ' + split[1] + ' ' + str(round(divs_step[i],2)) + ' ' + str(round(divs_random[i],2)) + '\n'
            i+=1
            saved_lines.append(new_line)
        else:
            saved_lines.append(line)
        
with open('interpretability\logs\qc_comparison2.txt', 'w') as f:
    for line in saved_lines:
        f.write(line)