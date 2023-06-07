import os
import pickle
import numpy as np

def letters_to_words(string):
    # "pvcd100" -> "proximity, validity, critical state, diversity"
    string = string.replace(".pkl", "")
    # remove anything that is not p, v, c or d
    if string == "baseline":
        print("no_quality_criteria")
        return
    string = ''.join([i for i in string if i in ['p', 'v', 'c', 'd']])

    criteria = []
    if "p" in string: criteria.append("proximity")
    if "v" in string: criteria.append("validity")
    if "c" in string: criteria.append("critical state")
    if "d" in string: criteria.append("diversity")
    print(criteria)


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
    print("weights: ", data['weights'])
    print("\n")
    


base_path = "evaluation\datasets\\100_ablations"

all_results = []

# iterate through all the folders
for folder in os.listdir(base_path):
    # go into the results folder
    results_path = os.path.join(base_path, folder, "results")
    # get the name of the folder
    folder_name = os.path.basename(folder)
    # get the name of used quality criteria
    letters_to_words(folder_name)
    # iterate through all the files in the results folder
    for file in os.listdir(results_path):
        print(file)
        data = pickle.load(open(results_path + "\\" + file, 'rb'))
        print_results_from_file(data)
