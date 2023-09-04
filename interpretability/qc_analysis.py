import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))

import numpy as np
from quality_metrics.quality_metrics import weight
import pickle
from helpers.util_functions import normalise_value

with open('interpretability\\normalisation_values.pkl', 'rb') as f:
    normalisation = pickle.load(f)

mcts_prox, step_prox, random_prox = [], [], []
mcts_val, step_val, random_val = [], [], []
mcts_div, step_div, random_div = [], [], []
mcts_crit, step_crit, random_crit = [], [], []
mcts_real, step_real, random_real = [], [], []
mcts_spar, step_spar, random_spar = [], [], []
mcts_qc, step_qc, random_qc = [], [], []

with open('interpretability\logs\qc_comparison.txt', 'r') as f:
    for line in f:
        parts = line.split(' ')
        if 'validity' in line:
            mcts_val.append(normalise_value(float(parts[1]), normalisation, 'validity') * weight['validity'])
            if len(parts) > 2:
                step_val.append(normalise_value(float(parts[2]), normalisation, 'validity') * weight['validity'])
                random_val.append(normalise_value(float(parts[3]), normalisation, 'validity') * weight['validity'])
        elif 'diversity' in line:
            mcts_div.append(normalise_value(float(parts[1]), normalisation, 'diversity') * weight['diversity'])
            if len(parts) > 2:
                step_div.append(normalise_value(float(parts[2]), normalisation, 'diversity') * weight['diversity'])
                random_div.append(normalise_value(float(parts[3]), normalisation, 'diversity') * weight['diversity'])
        elif 'proximity' in line:
            mcts_prox.append(normalise_value(- float(parts[1]), normalisation, 'proximity') * weight['proximity'])
            if len(parts) > 2:
                step_prox.append(normalise_value(-float(parts[2]), normalisation, 'proximity') * weight['proximity'])
                random_prox.append(normalise_value(-float(parts[3]), normalisation, 'proximity') * weight['proximity'])
        elif 'critical' in line:
            mcts_crit.append(normalise_value(float(parts[1]), normalisation, 'critical_state') * weight['critical_state'])
            if len(parts) > 2:
                step_crit.append(normalise_value(float(parts[2]), normalisation, 'critical_state') * weight['critical_state'])
                random_crit.append(normalise_value(float(parts[3]), normalisation, 'critical_state') * weight['critical_state'])
        elif 'realistic' in line:
            mcts_real.append(normalise_value(float(parts[1]), normalisation, 'realisticness') * weight['realisticness'])
            if len(parts) > 2:
                step_real.append(normalise_value(float(parts[2]), normalisation, 'realisticness') * weight['realisticness'])
                random_real.append(normalise_value(float(parts[3]), normalisation, 'realisticness') * weight['realisticness'])
        elif 'sparsity' in line:
            mcts_spar.append(normalise_value(float(parts[1]), normalisation, 'sparsity') * weight['sparsity'])
            if len(parts) > 2:
                step_spar.append(normalise_value(float(parts[2]), normalisation, 'sparsity') * weight['sparsity'])
                random_spar.append(normalise_value(float(parts[3]), normalisation, 'sparsity') * weight['sparsity'])
        elif 'qc' in line:
            mcts_qc.append(float(parts[1]))
            if len(parts) > 2:
                step_qc.append(float(parts[2]))
                random_qc.append(float(parts[3]))

    
    # print the averages
    if len(step_prox) > 0:
        print('Proximity', round(np.mean(mcts_prox),2), round(np.mean(step_prox),2), round(np.mean(random_prox),2))
        print('Validity', round(np.mean(mcts_val),2), round(np.mean(step_val),2), round(np.mean(random_val),2))
        print('Diversity', round(np.mean(mcts_div),2), round(np.mean(step_div),2), round(np.mean(random_div),2))
        print('Critical', round(np.mean(mcts_crit),2), round(np.mean(step_crit),2), round(np.mean(random_crit),2))
        print('Realistic', round(np.mean(mcts_real),2), round(np.mean(step_real),2), round(np.mean(random_real),2))
        print('Sparsity', round(np.mean(mcts_spar),2), round(np.mean(step_spar),2), round(np.mean(random_spar),2))

        qc_mcts = np.mean(mcts_prox) + np.mean(mcts_val) + np.mean(mcts_div) + np.mean(mcts_crit) + np.mean(mcts_real) + np.mean(mcts_spar)
        qc_step = np.mean(step_prox) + np.mean(step_val) + np.mean(step_div) + np.mean(step_crit) + np.mean(step_real) + np.mean(step_spar)
        qc_random = np.mean(random_prox) + np.mean(random_val) + np.mean(random_div) + np.mean(random_crit) + np.mean(random_real) + np.mean(random_spar)
        print('QC', round(np.mean(qc_mcts),2), round(np.mean(qc_step),2), round(np.mean(qc_random),2))

    else:
        # print averages where step and random are precomputed
        print('Proximity', round(np.mean(mcts_prox),2), -0.74, -0.71)
        print('Validity', round(np.mean(mcts_val),2), 0.68, 0.2)
        print('Diversity', round(np.mean(mcts_div),2), 1.3, 1.44)
        print('Critical', round(np.mean(mcts_crit),2), -0.07, -0.15)
        print('Realistic', round(np.mean(mcts_real),2), 0.81, 0.16)
        print('Sparsity', round(np.mean(mcts_real),2), -0.24, -0.19)

        print('QC', round(np.mean(mcts_qc),2), 1.74, 0.69)
