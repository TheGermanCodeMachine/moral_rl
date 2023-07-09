# This file is for analysing the resuults of the experiment. It assumes the results and statistics are already computed.

import os
import pickle
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

class config:
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'could_have_saved', 'final_number_of_unsaved_citizens', 'moved_towards_closest_citizen', 'bias']
    print_stuff = False

def letters_to_words(string):
    # "pvcd100" -> "proximity, validity, critical state, diversity"
    string = string.replace(".pkl", "")
    # remove anything that is not p, v, c or d
    if string == "baseline":
        print("no_quality_criteria")
        return "no_quality_criteria"
    string = ''.join([i for i in string if i in ['p', 'v', 'c', 'd']])

    criteria = []
    if "p" in string: criteria.append("proximity")
    if "v" in string: criteria.append("validity")
    if "c" in string: criteria.append("critical state")
    if "d" in string: criteria.append("diversity")
    print(criteria)
    return criteria


def print_results_from_file(data):
    # to_save = {'train_losses': train_lossess, 'test_losses': test_losses, 'test_mean_errors': test_mean_errors, 
    #         'train_mean_errors': train_mean_errors, 'pearson_correlations': pearson_correlations, 'spearman_correlations': spearman_correlations, 
    #         'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses}
    print("train losses: ", np.mean(data['train_losses']))
    print("test losses: ", np.mean(data['test_losses']))
    print("test mean errors: ", np.mean(data['test_mean_errors']))
    print("train mean errors: ", np.mean(data['train_mean_errors']))
    print("pearson correlations: ",  np.mean(data['pearson_correlations']))
    print("spearman correlations: ",  np.mean(data['spearman_correlations']))
    print("\n")

    
def get_weights(data):
    if config.print_stuff:
        print("weights: ", data['weights'])
        print('citizens_saved', data['weights'][0].item())
        print('unsaved_citizens', data['weights'][1].item())
        print('distance_to_citizen', data['weights'][2].item())
        print('standing_on_extinguisher', data['weights'][3].item())
        print('length', data['weights'][4].item())
        print('could_have_saved', data['weights'][5].item())
        print('final_number_of_unsaved_citizens', data['weights'][6].item())
        print('moved_towards_closest_citizen', data['weights'][7].item())
        print('bias', data['weights'][8].item())
    output = {'citizens_saved': data['weights'][0].item(), 'unsaved_citizens': data['weights'][1].item(), 'distance_to_citizen': data['weights'][2].item(), 'standing_on_extinguisher': data['weights'][3].item(), 'length': data['weights'][4].item(), 'could_have_saved': data['weights'][5].item(), 'final_number_of_unsaved_citizens': data['weights'][6].item(), 'moved_towards_closest_citizen': data['weights'][7].item(), 'bias': data['weights'][8].item()}
    return output

def load_and_get_rid_of_nans(data, string):
    data = np.array(data[string])
    data = data[~np.isnan(data)]
    return data

def get_results(data):
    train_losses = np.mean(data['train_losses'])
    test_losses = np.mean(data['test_losses'])
    test_mean_errors = np.mean(data['test_mean_errors'])
    train_mean_errors = np.mean(data['train_mean_errors'])
    # remove nan values
    pearson_correlations = np.mean(load_and_get_rid_of_nans(data, 'pearson_correlations'))
    spearman_correlations = np.mean(load_and_get_rid_of_nans(data, 'spearman_correlations'))
    train_rmse = np.mean(data['train_rmses'])
    test_rmse = np.mean(data['test_rmses'])
    r2 = np.mean(data['r2s'])
    output = {'train_losses': train_losses, 'test_losses': test_losses, 'test_mean_errors': test_mean_errors, 'train_mean_errors': train_mean_errors, 'pearson_correlations': pearson_correlations, 'spearman_correlations': spearman_correlations, 'train_rmse': train_rmse, 'test_rmse': test_rmse, 'r2': r2}
    return output

def get_results_ood(data):
    test_loss_oods = np.mean(data['test_losses'])
    test_mean_error_oods = np.mean(data['test_mean_errors'])
    test_rmse_oods = np.mean(data['test_rmses'])
    pearson_correlation_oods = np.mean(load_and_get_rid_of_nans(data, 'pearson_correlations'))
    spearman_correlation_oods = np.mean(load_and_get_rid_of_nans(data, 'spearman_correlations'))
    pred_label_pairs_oods = data['pred_label_pairs']
    r2_oods = np.mean(data['r2s'])
    output = {'test_losses': test_loss_oods, 'test_mean_errors': test_mean_error_oods, 'test_rmse': test_rmse_oods, 'pearson_correlations': pearson_correlation_oods, 'spearman_correlations': spearman_correlation_oods, 'pred_label_pairs': pred_label_pairs_oods, 'r2': r2_oods}
    return output

def get_statistics(data):
    efficiencies = np.mean(pickle.load(open(statistics_path + "\effiencies.pkl", 'rb')))
    lengths_cf = np.mean(pickle.load(open(statistics_path + "\\lengths_cf.pkl", 'rb')))
    lengths_org = np.mean(pickle.load(open(statistics_path + "\\lengths_org.pkl", 'rb')))
    quality_criteria_all = pickle.load(open(statistics_path + "\\quality_criteria.pkl", 'rb'))
    quality_criteria = np.mean(pickle.load(open(statistics_path + "\\quality_criteria.pkl", 'rb')), axis=0)
    start_points = np.mean(pickle.load(open(statistics_path + "\\start_points.pkl", 'rb')))
    org_feature_stats = pickle.load(open(statistics_path + "\\org_feature_stats.pkl", 'rb'))
    cf_feature_stats = pickle.load(open(statistics_path + "\\cf_feature_stats.pkl", 'rb'))
    if config.print_stuff:
        print("efficiencies: ", efficiencies)
        print("lengths_cf: ", lengths_cf)
        print("lengths_org: ", lengths_org)
        print("quality_criteria: ", quality_criteria)
        print("start_points: ", start_points)
        print("validity", quality_criteria[0])
        print("proximity", quality_criteria[1])
        print("critical state", quality_criteria[2])
        print("diversity", quality_criteria[3])
        print(lengths_org, lengths_cf)
        print(round(efficiencies, 2))
        print(start_points)
        print(round(quality_criteria[1], 2))
        print(round(quality_criteria[0], 2))
        print(round(quality_criteria[2], 2))
        print(round(quality_criteria[3], 2))
    output = {'efficiencies': efficiencies, 'lengths_cf': lengths_cf, 'lengths_org': lengths_org, 'quality_criteria': quality_criteria, 'start_points': start_points, 'org_feature_stats': org_feature_stats, 'cf_feature_stats': cf_feature_stats}
    return output

base_path = "evaluation\datasets\\100_ablations_3"

model_type = 'LM' # 'LM' or 'NN2'

options = ['c', 'd', 'p', 'pv', 'pvc', 'pvcd', 'pvd', 'v']
options_full = ['critical state', 'diversity', 'proximity', 'proximity-validity', 'proximity-validity-critical_state', 'proximity-validity-critical state-diversity', 'proximity-validity-diversity', 'validity']

results, results_ood, weights, statistics, feature_states = [], [], [], [], []

# iterate through all the folders
for folder in os.listdir(base_path):
    if folder == 'description.txt' or folder == "baseline1":
        continue   
    # go into the results folder
    results_path = os.path.join(base_path, folder, "results_normaliserewards" + model_type)
    statistics_path = os.path.join(base_path, folder, "statistics")
    # get the name of the folder
    folder_name = os.path.basename(folder)
    # get the name of used quality criteria
    name_of_criteria = letters_to_words(folder_name)
    # iterate through all the files in the results folder
    
    # get the specific file
    

    # extract the results
    results_data = pickle.load(open(results_path + "\\contrastive_learning_counterfactual.pkl", 'rb'))
    results_data_ood = pickle.load(open(results_path + "\\contrastive_learning_counterfactual_ood.pkl", 'rb'))
    # weights.append(get_weights(results_data))
    results.append(get_results(results_data))
    results_ood.append(get_results_ood(results_data_ood))
    statistics.append(get_statistics(statistics_path))

# print the results

# print the results as a latex table
# the top head should include all the options
# the left side should include test_mean_error, spearman correlation and pearson correlation

print('----------------results measures ----------------')
# print each item in options seperated by an &
print('& no qc & critical & diversity & proximity & proximity & proximity & proximity & proximity & validity \\\\')
print('& & state & & & validity & validity & validity & validity & \\\\')
print('& & & & & & critical & critical state & diversity & \\\\')
print('& & & & & & state & diversity & & \\\\')
# print each item in results seperated by an &
print('\hline')
print('\\textbf{Test measures:}\\\\')
print("mean error \downarrow & " + " & ".join([str(round(result['test_mean_errors'], 2)) for result in results]) + "\\\\")
print("rmse \downarrow & " + " & ".join([str(round(result['test_rmse'], 2)) for result in results]) + "\\\\")
print("r2 \\uparrow & " + " & ".join([str(round(result['r2'], 2)) for result in results]) + "\\\\")
print("spearman \\uparrow & " + " & ".join([str(round(result['spearman_correlations'], 2)) for result in results]) + "\\\\")
print("pearson \\uparrow & " + " & ".join([str(round(result['pearson_correlations'], 2)) for result in results]) + "\\\\")
print("\hline")
print('\\textbf{Test measures ood:}\\\\')
print("mean error \downarrow & " + " & ".join([str(round(result['test_mean_errors'], 2)) for result in results_ood]) + "\\\\")
print("rmse \downarrow & " + " & ".join([str(round(result['test_rmse'], 2)) for result in results_ood]) + "\\\\")
print("r2 \\uparrow & " + " & ".join([str(round(result['r2'], 2)) for result in results_ood]) + "\\\\")
print("spearman \\uparrow & " + " & ".join([str(round(result['spearman_correlations'], 2)) for result in results_ood]) + "\\\\")
print("pearson \\uparrow & " + " & ".join([str(round(result['pearson_correlations'], 2)) for result in results_ood]) + "\\\\")
print("\hline")
print("\\textbf{Statistics:}\\\\")
print("efficiencies \downarrow & " + " & ".join([str(round(statistic['efficiencies'], 2)) for statistic in statistics]) + "\\\\")
print("lengths cf & " + " & ".join([str(round(statistic['lengths_cf'], 2)) for statistic in statistics]) + "\\\\")
print("lengths org & " + " & ".join([str(round(statistic['lengths_org'], 2)) for statistic in statistics]) + "\\\\")
print("start points & " + " & ".join([str(round(statistic['start_points'], 2)) for statistic in statistics]) + "\\\\")
print("\hline")
print("\\textbf{Quality criteria:}\\\\")
print("validity \\uparrow & " + " & ".join([str(round(statistic['quality_criteria'][0], 2)) for statistic in statistics]) + "\\\\")
print("proximity \downarrow & " + " & ".join([str(round(statistic['quality_criteria'][1], 2)) for statistic in statistics]) + "\\\\")
print("critical state \\uparrow & " + " & ".join([str(round(statistic['quality_criteria'][2], 2)) for statistic in statistics]) + "\\\\")
print("diversity \\uparrow & " + " & ".join([str(round(statistic['quality_criteria'][3], 2)) for statistic in statistics]) + "\\\\")
print("\hline")
# print("Learned weights:\\\\")
# print("citizens saved & " + " & ".join([str(round(weight['citizens_saved'], 2)) for weight in weights]) + "\\\\")
# print("unsaved citizens & " + " & ".join([str(round(weight['unsaved_citizens'], 2)) for weight in weights]) + "\\\\")
# print("distance to citizen & " + " & ".join([str(round(weight['distance_to_citizen'], 2)) for weight in weights]) + "\\\\")
# print("standing on extinguisher & " + " & ".join([str(round(weight['standing_on_extinguisher'], 2)) for weight in weights]) + "\\\\")
# print("length & " + " & ".join([str(round(weight['length'], 2)) for weight in weights]) + "\\\\")
# print("could have saved & " + " & ".join([str(round(weight['could_have_saved'], 2)) for weight in weights]) + "\\\\")
# print("final number of unsaved citizens & " + " & ".join([str(round(weight['final_number_of_unsaved_citizens'], 2)) for weight in weights]) + "\\\\")
# print("moved towards closest citizen & " + " & ".join([str(round(weight['moved_towards_closest_citizen'], 2)) for weight in weights]) + "\\\\")
# print("bias & " + " & ".join([str(round(weight['bias'], 2)) for weight in weights]) + "\\\\")
# print('\hline')
print("\\textbf{Feature values:}\\\\")
print("citizens saved & " + " & ".join([str(round(statistic['cf_feature_stats'][0], 2)) for statistic in statistics]) + "\\\\")
print("unsaved citizens & " + " & ".join([str(round(statistic['cf_feature_stats'][1], 2)) for statistic in statistics]) + "\\\\")
print("distance to citizen & " + " & ".join([str(round(statistic['cf_feature_stats'][2], 2)) for statistic in statistics]) + "\\\\")
print("standing on extinguisher & " + " & ".join([str(round(statistic['cf_feature_stats'][3], 2)) for statistic in statistics]) + "\\\\")
print("length & " + " & ".join([str(round(statistic['cf_feature_stats'][4], 2)) for statistic in statistics]) + "\\\\")
print("could have saved & " + " & ".join([str(round(statistic['cf_feature_stats'][5], 2)) for statistic in statistics]) + "\\\\")
print("final number of unsaved citizens & " + " & ".join([str(round(statistic['cf_feature_stats'][6], 2)) for statistic in statistics]) + "\\\\")
print("moved towards closest citizen & " + " & ".join([str(round(statistic['cf_feature_stats'][7], 2)) for statistic in statistics]) + "\\\\")
print("reward & " + " & ".join([str(round(statistic['cf_feature_stats'][8], 2)) for statistic in statistics]) + "\\\\")

# print('\n')
# print('----------------statistics ----------------')
# print(" & " + " & ".join(options_full) + "\\\\")
# print('& critical state & diversity & proximity & proximity & proximity & proximity & proximity & validity \\\\')
# print('& & & & validity & validity & validity & validity & \\\\')
# print('& & & & & critical state & critical state & diversity & \\\\')
# print('& & & & & & diversity & & \\\\')
# # print each item in results seperated by an &
# print("efficiencies & " + " & ".join([str(round(statistic['efficiencies'], 2)) for statistic in statistics]) + "\\\\")
# print("lengths cf & " + " & ".join([str(round(statistic['lengths_cf'], 2)) for statistic in statistics]) + "\\\\")
# print("lengths org & " + " & ".join([str(round(statistic['lengths_org'], 2)) for statistic in statistics]) + "\\\\")
# print("start_points & " + " & ".join([str(round(statistic['start_points'], 2)) for statistic in statistics]) + "\\\\")

# print('\n')
# print('----------------quality criterias ----------------')
# print(" & " + " & ".join(options_full) + "\\\\")
# print('& critical state & diversity & proximity & proximity & proximity & proximity & proximity & validity \\\\')
# print('& & & & validity & validity & validity & validity & \\\\')
# print('& & & & & critical state & critical state & diversity & \\\\')
# print('& & & & & & diversity & & \\\\')
# # print each item in results seperated by an &
# print("validity & " + " & ".join([str(round(statistic['quality_criteria'][0], 2)) for statistic in statistics]) + "\\\\")
# print("proximity & " + " & ".join([str(round(statistic['quality_criteria'][1], 2)) for statistic in statistics]) + "\\\\")
# print("critical state & " + " & ".join([str(round(statistic['quality_criteria'][2], 2)) for statistic in statistics]) + "\\\\")
# print("diversity & " + " & ".join([str(round(statistic['quality_criteria'][3], 2)) for statistic in statistics]) + "\\\\")


# print('\n')
# print('----------------weights ----------------')
# print(" & " + " & ".join(options_full) + "\\\\")
# print('& critical state & diversity & proximity & proximity & proximity & proximity & proximity & validity \\\\')
# print('& & & & validity & validity & validity & validity & \\\\')
# print('& & & & & critical state & critical state & diversity & \\\\')
# print('& & & & & & diversity & & \\\\')
# # print each item in results seperated by an &
# print("citizens saved & " + " & ".join([str(round(weight['citizens_saved'], 2)) for weight in weights]) + "\\\\")
# print("unsaved citizens & " + " & ".join([str(round(weight['unsaved_citizens'], 2)) for weight in weights]) + "\\\\")
# print("distance to citizen & " + " & ".join([str(round(weight['distance_to_citizen'], 2)) for weight in weights]) + "\\\\")
# print("standing on extinguisher & " + " & ".join([str(round(weight['standing_on_extinguisher'], 2)) for weight in weights]) + "\\\\")
# print("length & " + " & ".join([str(round(weight['length'], 2)) for weight in weights]) + "\\\\")
# print("could have saved & " + " & ".join([str(round(weight['could_have_saved'], 2)) for weight in weights]) + "\\\\")
# print("final number of unsaved citizens & " + " & ".join([str(round(weight['final_number_of_unsaved_citizens'], 2)) for weight in weights]) + "\\\\")
# print("moved towards closest citizen & " + " & ".join([str(round(weight['moved_towards_closest_citizen'], 2)) for weight in weights]) + "\\\\")
# print("bias & " + " & ".join([str(round(weight['bias'], 2)) for weight in weights]) + "\\\\")

# print('\n')
# print('----------------feature values ----------------')
# print(" & " + " & ".join(options_full) + "\\\\")
# print('& critical state & diversity & proximity & proximity & proximity & proximity & proximity & validity \\\\')
# print('& & & & validity & validity & validity & validity & \\\\')
# print('& & & & & critical state & critical state & diversity & \\\\')
# print('& & & & & & diversity & & \\\\')
# # print each item in results seperated by an &
# print("citizens saved & " + " & ".join([str(round(statistic['cf_feature_stats'][0], 2)) for statistic in statistics]) + "\\\\")
# print("unsaved citizens & " + " & ".join([str(round(statistic['cf_feature_stats'][1], 2)) for statistic in statistics]) + "\\\\")
# print("distance to citizen & " + " & ".join([str(round(statistic['cf_feature_stats'][2], 2)) for statistic in statistics]) + "\\\\")
# print("standing on extinguisher & " + " & ".join([str(round(statistic['cf_feature_stats'][3], 2)) for statistic in statistics]) + "\\\\")
# print("length & " + " & ".join([str(round(statistic['cf_feature_stats'][4], 2)) for statistic in statistics]) + "\\\\")
# print("could have saved & " + " & ".join([str(round(statistic['cf_feature_stats'][5], 2)) for statistic in statistics]) + "\\\\")
# print("final number of unsaved citizens & " + " & ".join([str(round(statistic['cf_feature_stats'][6], 2)) for statistic in statistics]) + "\\\\")
# print("moved towards closest citizen & " + " & ".join([str(round(statistic['cf_feature_stats'][7], 2)) for statistic in statistics]) + "\\\\")
# print("reward & " + " & ".join([str(round(statistic['cf_feature_stats'][8], 2)) for statistic in statistics]) + "\\\\")


print('\n')
print('----------------correlations ----------------')
# calculate correlations between factors and results
# correlation between test_mean_error and length_cf
print('mean_error, length_cf', pearsonr([result['test_mean_errors'] for result in results], [statistic['lengths_cf'] for statistic in statistics])[0])
# correlation between test_mean_error and length_org
print('mean_error, length_org', pearsonr([result['test_mean_errors'] for result in results], [statistic['lengths_org'] for statistic in statistics])[0])
# correlation between test_mean_error and start_points
print('mean_error, start_points', pearsonr([result['test_mean_errors'] for result in results], [statistic['start_points'] for statistic in statistics])[0])

print('pearsonr, length_cf', pearsonr([result['pearson_correlations'] for result in results], [statistic['lengths_cf'] for statistic in statistics])[0])
print('pearsonr, length_org', pearsonr([result['pearson_correlations'] for result in results], [statistic['lengths_org'] for statistic in statistics])[0])
print('pearsonr, start_points', pearsonr([result['pearson_correlations'] for result in results], [statistic['start_points'] for statistic in statistics])[0])

print('spearmanr, length_cf', pearsonr([result['spearman_correlations'] for result in results], [statistic['lengths_cf'] for statistic in statistics])[0])
print('spearmanr, length_org', pearsonr([result['spearman_correlations'] for result in results], [statistic['lengths_org'] for statistic in statistics])[0])
print('spearmanr, start_points', pearsonr([result['spearman_correlations'] for result in results], [statistic['start_points'] for statistic in statistics])[0])

print('mean_error, spearman', pearsonr([result['test_mean_errors'] for result in results], [result['spearman_correlations'] for result in results])[0])
print('mean_error, pearson', pearsonr([result['test_mean_errors'] for result in results], [result['pearson_correlations'] for result in results])[0])
print('spearman, pearson', pearsonr([result['spearman_correlations'] for result in results], [result['pearson_correlations'] for result in results])[0])


# print('----------------avg values----------------\n')
# print('mean_error', np.mean([result['test_mean_errors'] for result in results]))
# print('rmse', np.mean([result['test_rmse'] for result in results]))
# print('r2', np.mean([result['r2'] for result in results]))
# print('pearsonr', np.mean([result['pearson_correlations'] for result in results]))
# print('spearmanr', np.mean([result['spearman_correlations'] for result in results]))
# print('mean_error_ood', np.mean([result['test_mean_errors'] for result in results_ood]))
# print('rmse_ood', np.mean([result['test_rmse'] for result in results_ood]))
# print('r2_ood', np.mean([result['r2'] for result in results_ood]))
# print('pearsonr_ood', np.mean([result['pearson_correlations'] for result in results_ood]))
# print('spearmanr_ood', np.mean([result['spearman_correlations'] for result in results_ood]))




print('----------------graphics----------------\n')
p = pickle.load(open('evaluation\datasets\\100_ablations_3\p100\\results_normaliserewards' +  model_type + '\contrastive_learning_counterfactual.pkl', 'rb'))
p_me = p['test_mean_errors']
p_r2 = p['r2s']
p_spear = load_and_get_rid_of_nans(p, 'spearman_correlations')
v = pickle.load(open('evaluation\datasets\\100_ablations_3\\v100\\results_normaliserewards' +  model_type + '\contrastive_learning_counterfactual.pkl', 'rb'))
v_me = v['test_mean_errors']
v_r2 = v['r2s']
v_spear = load_and_get_rid_of_nans(v, 'spearman_correlations')
pvcd = pickle.load(open('evaluation\datasets\\100_ablations_3\\pvcd100\\results_normaliserewards' +  model_type + '\contrastive_learning_counterfactual.pkl', 'rb'))
pvcd_me = pvcd['test_mean_errors']
pvcd_r2 = pvcd['r2s']
pvcd_spear = load_and_get_rid_of_nans(pvcd, 'spearman_correlations')
noqc = pickle.load(open('evaluation\datasets\\100_ablations_3\\baseline\\results_normaliserewards' +  model_type + '\contrastive_learning_counterfactual.pkl', 'rb'))
noqc_me = noqc['test_mean_errors']
noqc_r2 = noqc['r2s']
noqc_spear = load_and_get_rid_of_nans(noqc, 'spearman_correlations')

barWidth = 0.3

bars_avg_me = [np.mean(p_me), np.mean(v_me), np.mean(pvcd_me), np.mean(noqc_me)]
bars_median_me = [np.median(p_me), np.median(v_me), np.median(pvcd_me), np.median(noqc_me)]
yerr_lower_normal_me = [np.mean(p_me) - np.percentile(p_me, 25), np.mean(v_me) - np.percentile(v_me, 25), np.mean(pvcd_me) - np.percentile(pvcd_me, 25), np.mean(noqc_me) - np.percentile(noqc_me, 25)]
yerr_upper_normal_me = [np.percentile(p_me, 75) - np.mean(p_me), np.percentile(v_me, 75) - np.mean(v_me), np.percentile(pvcd_me, 75) - np.mean(pvcd_me), np.percentile(noqc_me, 75) - np.mean(noqc_me)]

bars_avg_r2 = [np.mean(p_r2), np.mean(v_r2), np.mean(pvcd_r2), np.mean(noqc_r2)]
bars_median_r2 = [np.median(p_r2), np.median(v_r2), np.median(pvcd_r2), np.median(noqc_r2)]
yerr_lower_normal_r2 = [np.mean(p_r2) - np.percentile(p_r2, 25), np.mean(v_r2) - np.percentile(v_r2, 25), np.mean(pvcd_r2) - np.percentile(pvcd_r2, 25), np.mean(noqc_r2) - np.percentile(noqc_r2, 25)]
yerr_upper_normal_r2 = [np.percentile(p_r2, 75) - np.mean(p_r2), np.percentile(v_r2, 75) - np.mean(v_r2), np.percentile(pvcd_r2, 75) - np.mean(pvcd_r2), np.percentile(noqc_r2, 75) - np.mean(noqc_r2)]

bars_avg_spear = [np.mean(p_spear), np.mean(v_spear), np.mean(pvcd_spear), np.mean(noqc_spear)]
bars_median_spear = [np.median(p_spear), np.median(v_spear), np.median(pvcd_spear), np.median(noqc_spear)]
yerr_lower_normal_spear = [np.mean(p_spear) - np.percentile(p_spear, 25), np.mean(v_spear) - np.percentile(v_spear, 25), np.mean(pvcd_spear) - np.percentile(pvcd_spear, 25), np.mean(noqc_spear) - np.percentile(noqc_spear, 25)]
yerr_upper_normal_spear = [np.percentile(p_spear, 75) - np.mean(p_spear), np.percentile(v_spear, 75) - np.mean(v_spear), np.percentile(pvcd_spear, 75) - np.mean(pvcd_spear), np.percentile(noqc_spear, 75) - np.mean(noqc_spear)]

x = np.arange(len(bars_avg_me))
ax1 = plt.subplot(1,1,1)
w = 0.3

plt.xticks(x+w, ['proximity', 'validity', 'all criteria', 'no criteria'])
bar_me = ax1.bar(x, bars_avg_me, width=w, color='r', align='center', yerr=[yerr_lower_normal_me,yerr_upper_normal_me], capsize=4, zorder=1)
point_me = ax1.scatter(x, bars_median_me, s=25, color='black', zorder=2)
# make the y-axis go from -2 to 5
ax1.set_ylim([-0.2,4])

# ax2 = ax1.twinx()
ax3 = ax1.twinx()
bar_r2 = ax3.bar(x+w, bars_avg_r2, width=w, color='g', align='center', yerr=[yerr_lower_normal_r2,yerr_upper_normal_r2], capsize=4, zorder=1)
point_r2 = ax3.scatter(x+w, bars_median_r2, s=25, color='black', zorder=2)
# ax2.set_ylim([-2,4])
# ax1.set_ylim([-2,4])

bar_spear = ax3.bar(x+2*w, bars_avg_spear, width=w, color='b', align='center', yerr=[yerr_lower_normal_spear,yerr_upper_normal_spear], capsize=4, zorder=1)
point_spear = ax3.scatter(x+2*w, bars_median_spear, s=25, color='black', zorder=2)
ax3.set_ylim([-0.05,1])

plt.legend([bar_me, bar_r2, bar_spear], ['mean error', 'r2', 'spearman'], loc='upper left')
plt.show()