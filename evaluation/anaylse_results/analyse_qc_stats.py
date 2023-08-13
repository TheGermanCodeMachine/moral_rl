import pickle
import numpy as np

path = "datasets\\1000\\1000\statistics\qc_statistics.pkl"
with(open(path, 'rb')) as f:
    qc_stats = pickle.load(f)

qc_val_spears, qc_prox_spears, qc_crit_spears, qc_div_spears, qc_real_spears, qc_spar_spears, val_prox_spears, val_crit_spears, val_div_spears, prox_crit_spears, prox_div_spears, crit_div_spears = [], [], [], [], [], [], [], [], [], [], [], []
qc_val_pears, qc_prox_pears, qc_crit_pears, qc_div_pears, qc_real_pears, qc_spar_pears, val_prox_pears, val_crit_pears, val_div_pears, prox_crit_pears, prox_div_pears, crit_div_pears = [], [], [], [], [], [], [], [], [], [], [], []
pos_val, pos_prox, pos_crit, pos_div, pos_real, pos_spar = [], [], [], [], [], []
chosen_val, chosen_prox, chosen_crit, chosen_div, chosen_real, chosen_spar = [], [], [], [], [], []

for i in range(len(qc_stats)):
    # load all the correlations saved as this dictionary
    # {'qc-validity': qc_val_spear, 'qc-proximity': qc_prox_spear, 'qc-critical_state': qc_crit_spear, 'qc-diversity': qc_div_spear, 'validity-proximity': val_prox_spear, 'validity-critical_state': val_crit_spear, 'validity-diversity': val_div_spear, 'proximity-critical_state': prox_crit_spear, 'proximity-diversity': prox_div_spear, 'critical_state-diversity': crit_div_spear}
    qc_val_spears.append(qc_stats[i]['spear_correlations']['qc-validity'])
    qc_prox_spears.append(qc_stats[i]['spear_correlations']['qc-proximity'])
    qc_crit_spears.append(qc_stats[i]['spear_correlations']['qc-critical_state'])
    qc_div_spears.append(qc_stats[i]['spear_correlations']['qc-diversity'])
    qc_real_spears.append(qc_stats[i]['spear_correlations']['qc-realisticness'])
    qc_spar_spears.append(qc_stats[i]['spear_correlations']['qc-sparsity'])
    val_prox_spears.append(qc_stats[i]['spear_correlations']['validity-proximity'])
    val_crit_spears.append(qc_stats[i]['spear_correlations']['validity-critical_state'])
    val_div_spears.append(qc_stats[i]['spear_correlations']['validity-diversity'])
    prox_crit_spears.append(qc_stats[i]['spear_correlations']['proximity-critical_state'])
    prox_div_spears.append(qc_stats[i]['spear_correlations']['proximity-diversity'])
    crit_div_spears.append(qc_stats[i]['spear_correlations']['critical_state-diversity'])


    qc_val_pears.append(qc_stats[i]['pear_correlations']['qc-validity'])
    qc_prox_pears.append(qc_stats[i]['pear_correlations']['qc-proximity'])
    qc_crit_pears.append(qc_stats[i]['pear_correlations']['qc-critical_state'])
    qc_div_pears.append(qc_stats[i]['pear_correlations']['qc-diversity'])
    qc_real_pears.append(qc_stats[i]['pear_correlations']['qc-realisticness'])
    qc_spar_pears.append(qc_stats[i]['pear_correlations']['qc-sparsity'])
    val_prox_pears.append(qc_stats[i]['pear_correlations']['validity-proximity'])
    val_crit_pears.append(qc_stats[i]['pear_correlations']['validity-critical_state'])
    val_div_pears.append(qc_stats[i]['pear_correlations']['validity-diversity'])
    prox_crit_pears.append(qc_stats[i]['pear_correlations']['proximity-critical_state'])
    prox_div_pears.append(qc_stats[i]['pear_correlations']['proximity-diversity'])
    crit_div_pears.append(qc_stats[i]['pear_correlations']['critical_state-diversity'])

    pos_val.append(qc_stats[i]['perc_positions']['validity'])
    pos_prox.append(qc_stats[i]['perc_positions']['proximity'])
    pos_crit.append(qc_stats[i]['perc_positions']['critical_state'])
    pos_div.append(qc_stats[i]['perc_positions']['diversity'])
    pos_real.append(qc_stats[i]['perc_positions']['realisticness'])
    pos_spar.append(qc_stats[i]['perc_positions']['sparsity'])


    chosen_val.append(qc_stats[i]['chosen_values']['validity'])
    chosen_prox.append(qc_stats[i]['chosen_values']['proximity'])
    chosen_crit.append(qc_stats[i]['chosen_values']['critical_state'])
    chosen_div.append(qc_stats[i]['chosen_values']['diversity'])
    chosen_real.append(qc_stats[i]['chosen_values']['realisticness'])
    chosen_spar.append(qc_stats[i]['chosen_values']['sparsity'])

# remove the first values of each correlation with diversity
qc_div_spears = qc_div_spears[1:]
val_div_spears = val_div_spears[1:]
prox_div_spears = prox_div_spears[1:]
crit_div_spears = crit_div_spears[1:]
qc_div_pears = qc_div_pears[1:]
val_div_pears = val_div_pears[1:]
prox_div_pears = prox_div_pears[1:]
crit_div_pears = crit_div_pears[1:] 

print("qc-validity", round(np.mean(qc_val_spears),2), round(np.mean(qc_val_pears),2))
print("qc-proximity", round(np.mean(qc_prox_spears),2), round(np.mean(qc_prox_pears),2))
print("qc-critical_state", round(np.mean(qc_crit_spears),2), round(np.mean(qc_crit_pears),2))
print("qc-diversity", round(np.mean(qc_div_spears),2), round(np.mean(qc_div_pears),2))
print("qc-realisticness", round(np.mean(qc_real_spears),2), round(np.mean(qc_real_pears),2))
print("qc-sparsity", round(np.mean(qc_spar_spears),2), round(np.mean(qc_spar_pears),2))
print('')

print("validity-proximity", round(np.mean(val_prox_spears),2), round(np.mean(val_prox_pears),2))
print("validity-critical_state", round(np.mean(val_crit_spears),2), round(np.mean(val_crit_pears),2))
print("validity-diversity", round(np.mean(val_div_spears),2), round(np.mean(val_div_pears),2))
print("proximity-critical_state", round(np.mean(prox_crit_spears),2), round(np.mean(prox_crit_pears),2))
print("proximity-diversity", round(np.mean(prox_div_spears),2), round(np.mean(prox_div_pears),2))
print("critical_state-diversity", round(np.mean(crit_div_spears),2), round(np.mean(crit_div_pears),2))
print('')

print("pos_val", round(np.mean(pos_val),2))
print("pos_prox", round(np.mean(pos_prox),2))
print("pos_crit", round(np.mean(pos_crit),2))
print("pos_div", round(np.mean(pos_div),2))
print("pos_real", round(np.mean(pos_real),2))
print("pos_spar", round(np.mean(pos_spar),2))
print('')

print("chosen_val", round(np.mean(chosen_val),2))
print("chosen_prox", round(np.mean(chosen_prox),2))
print("chosen_crit", round(np.mean(chosen_crit),2))
print("chosen_div", round(np.mean(chosen_div),2))
print("chosen_real", round(np.mean(chosen_real),2))
print("chosen_spar", round(np.mean(chosen_spar),2))