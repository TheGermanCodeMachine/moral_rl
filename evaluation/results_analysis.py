import os
import pickle
import numpy as np

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

def get_results(data):
    train_losses = np.mean(data['train_losses'])
    test_losses = np.mean(data['test_losses'])
    test_mean_errors = np.mean(data['test_mean_errors'])
    train_mean_errors = np.mean(data['train_mean_errors'])
    pearson_correlations = np.mean(data['pearson_correlations'])
    spearman_correlations = np.mean(data['spearman_correlations'])
    output = {'train_losses': train_losses, 'test_losses': test_losses, 'test_mean_errors': test_mean_errors, 'train_mean_errors': train_mean_errors, 'pearson_correlations': pearson_correlations, 'spearman_correlations': spearman_correlations}
    return output

def get_statistics(data):
    efficiencies = np.mean(pickle.load(open(statistics_path + "\effiencies.pkl", 'rb')))
    lengths_cf = np.mean(pickle.load(open(statistics_path + "\\lengths_cf.pkl", 'rb')))
    lengths_org = np.mean(pickle.load(open(statistics_path + "\\lengths_org.pkl", 'rb')))
    quality_criteria_all = pickle.load(open(statistics_path + "\\quality_criteria.pkl", 'rb'))
    quality_criteria = np.mean(pickle.load(open(statistics_path + "\\quality_criteria.pkl", 'rb')), axis=0)
    start_points = np.mean(pickle.load(open(statistics_path + "\\start_points.pkl", 'rb')))
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
    output = {'efficiencies': efficiencies, 'lengths_cf': lengths_cf, 'lengths_org': lengths_org, 'quality_criteria': quality_criteria, 'start_points': start_points}
    return output

base_path = "evaluation\datasets\\100_ablations_3"

options = ['c', 'd', 'p', 'pv', 'pvc', 'pvcd', 'pcd', 'v']


# these vectors collect the results and have the form: axis 0 are the different measures and axis 1 are the different combinations of quality criteria
# results = {'p': [], 'v': [], 'c': [], 'd': [], 'pv': [], 'pvc': [], 'pcd': [], 'pvcd': []}
# weights = {'p': [], 'v': [], 'c': [], 'd': [], 'pv': [], 'pvc': [], 'pcd': [], 'pvcd': []}
# statistics = {'p': [], 'v': [], 'c': [], 'd': [], 'pv': [], 'pvc': [], 'pcd': [], 'pvcd': []}
results = []
weights = []
statistics = []

# iterate through all the folders
for folder in os.listdir(base_path):
    if folder == 'description.txt':
        continue   
    # go into the results folder
    results_path = os.path.join(base_path, folder, "results")
    statistics_path = os.path.join(base_path, folder, "statistics")
    # get the name of the folder
    folder_name = os.path.basename(folder)
    # get the name of used quality criteria
    name_of_criteria = letters_to_words(folder_name)
    if 'no_quality_criteria' in name_of_criteria:
        continue
    # iterate through all the files in the results folder
    
    # get the specific file
    

    # extract the results
    results_data = pickle.load(open(results_path + "\\contrastive_learning_counterfactual.pkl", 'rb'))
    weights.append(get_weights(results_data))
    results.append(get_results(results_data))
    statistics.append(get_statistics(statistics_path))

# print the results
print('hi')
