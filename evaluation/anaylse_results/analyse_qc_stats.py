import pickle
import numpy as np

path = "datasets\\100_ablations_3\\100\statistics\qc_statistics.pkl"
with(open(path, 'rb')) as f:
    qc_stats = pickle.load(f)

qc_val_spears, qc_prox_spears, qc_crit_spears, qc_div_spears, val_prox_spears, val_crit_spears, val_div_spears, prox_crit_spears, prox_div_spears, crit_div_spears = [], [], [], [], [], [], [], [], [], []
qc_val_pears, qc_prox_pears, qc_crit_pears, qc_div_pears, val_prox_pears, val_crit_pears, val_div_pears, prox_crit_pears, prox_div_pears, crit_div_pears = [], [], [], [], [], [], [], [], [], []
pos_val, pos_prox, pos_crit, pos_div = [], [], [], []
chosen_val, chosen_prox, chosen_crit, chosen_div = [], [], [], []

for i in range(len(qc_stats)):
    # load all the correlations saved as this dictionary
    # {'qc-validity': qc_val_spear, 'qc-proximity': qc_prox_spear, 'qc-critical_state': qc_crit_spear, 'qc-diversity': qc_div_spear, 'validity-proximity': val_prox_spear, 'validity-critical_state': val_crit_spear, 'validity-diversity': val_div_spear, 'proximity-critical_state': prox_crit_spear, 'proximity-diversity': prox_div_spear, 'critical_state-diversity': crit_div_spear}
    qc_val_spears.append(qc_stats[i]['spear_correlations']['qc-validity'])
    qc_prox_spears.append(qc_stats[i]['spear_correlations']['qc-proximity'])
    qc_crit_spears.append(qc_stats[i]['spear_correlations']['qc-critical_state'])
    qc_div_spears.append(qc_stats[i]['spear_correlations']['qc-diversity'])
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

    chosen_val.append(qc_stats[i]['chosen_values']['validity'])
    chosen_prox.append(qc_stats[i]['chosen_values']['proximity'])
    chosen_crit.append(qc_stats[i]['chosen_values']['critical_state'])
    chosen_div.append(qc_stats[i]['chosen_values']['diversity'])

# remove the first values of each correlation with diversity
qc_div_spears = qc_div_spears[1:]
val_div_spears = val_div_spears[1:]
prox_div_spears = prox_div_spears[1:]
crit_div_spears = crit_div_spears[1:]
qc_div_pears = qc_div_pears[1:]
val_div_pears = val_div_pears[1:]
prox_div_pears = prox_div_pears[1:]
crit_div_pears = crit_div_pears[1:] 

print("qc-validity", np.mean(qc_val_spears), np.mean(qc_val_pears))
print("qc-proximity", np.mean(qc_prox_spears), np.mean(qc_prox_pears))
print("qc-critical_state", np.mean(qc_crit_spears), np.mean(qc_crit_pears))
print("qc-diversity", np.mean(qc_div_spears), np.mean(qc_div_pears))
print("validity-proximity", np.mean(val_prox_spears), np.mean(val_prox_pears))
print("validity-critical_state", np.mean(val_crit_spears), np.mean(val_crit_pears))
print("validity-diversity", np.mean(val_div_spears), np.mean(val_div_pears))
print("proximity-critical_state", np.mean(prox_crit_spears), np.mean(prox_crit_pears))
print("proximity-diversity", np.mean(prox_div_spears), np.mean(prox_div_pears))
print("critical_state-diversity", np.mean(crit_div_spears), np.mean(crit_div_pears))

print("pos_val", np.mean(pos_val))
print("pos_prox", np.mean(pos_prox))
print("pos_crit", np.mean(pos_crit))
print("pos_div", np.mean(pos_div))

print("chosen_val", np.mean(chosen_val))
print("chosen_prox", np.mean(chosen_prox))
print("chosen_crit", np.mean(chosen_crit))
print("chosen_div", np.mean(chosen_div))