import os
import pickle

def iterate_through_folder(folder_path):
    # are there subfolders?
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    # if there are subfolders, iterate through them
    if subfolders:
        all_folder_base_paths = []
        for subfolder in subfolders:
            # subfolder shouldn't be called results
            if not 'results' in subfolder and not subfolder.endswith('statistics'):
                subsubfolders = iterate_through_folder(subfolder)
                all_folder_base_paths.extend(subsubfolders)
            else:
                # in this case we have reached the relevant folder and don't want to go deeper
                return [folder_path]
        # remove paths ending with 'baseline'
        # all_folder_base_paths = [path for path in all_folder_base_paths if not path.endswith('baseline')]
        return all_folder_base_paths
    else:
        return [folder_path]
    
def save_results(to_save, path, baseline=0, type='results', data_mixture=(0,1), con=True):
    #  if the results folder does not exist, create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if type=='hyper_params':
        path += "hyperparameters.pkl"
    elif type=='results':
        # make the file name
        if con: path += "contrastive_" 
        else: path += "single_"
        if baseline==2: path += "_random_baseline.pkl"
        elif baseline==0: path += "_counterfactual.pkl"
        else: path += "_no_quality_baseline.pkl"
    elif type=='model':
        path += "model.pkl"
    elif type=='results_ood':
        if con: path += "contrastive_" 
        else: path += "single_"
        if baseline==2: path += "_random_baseline_ood.pkl"
        elif baseline==0: path += "_counterfactual_ood.pkl"
        else: path += "_no_quality_baseline_ood.pkl"
    # save the results
    write(to_save, path)

def write(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)