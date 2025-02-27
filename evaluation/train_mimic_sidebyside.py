import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import torch
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from evaluate_mimic import evaluate_mimic
import sys
import random
from helpers.folder_util_functions import iterate_through_folder, save_results, read, write
from copy import deepcopy
from utils_evaluation import *
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from stepwise_linear_model import train_step_wise_linear_model

class hyperparameters:
    learning_rate = 1e-1
    regularisation = 1e-2
    l1_lambda = 1e-1
    epochs_non_contrastive = 10000
    epochs_contrastive = 6000
    number_of_seeds = 30

class LM_params:
    learning_rate = 0.1
    regularisation = 0.001

class NN_params:
    # learning_rate = 0.01
    # regularisation = 0.01
    # num_layers = 4
    # hidden_layer_sizes = [8,4]
    learning_rate = 0.01
    regularisation = 0.001
    num_layers = 4
    hidden_layer_sizes = [60, 30]
    epochs = 1000
    
class config:    
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'could_have_saved', 'final_number_of_unsaved_citizens', 'moved_towards_closest_citizen', 'bias']
    model_type = 'linear' # model_type = 'NN' or 'linear' or 'stepwise'
    data_folds = 5
    results_path = "\\results_sidebysideLM\\" # Foldername to save res  ults to
    print_plot = False
    print_examples = False
    print_weights = False
    save_results = True
    print_worst_examples = False
    print_best_examples = False
    save_model = True

# randomises the order of trajectories, while keeping the pairs of original and counterfactual trajectories together
# also makes them into tensors
def shuffle_together(org_trajs, cf_trajs):
    org_cf_trajs = list(zip(org_trajs, cf_trajs))
    random.shuffle(org_cf_trajs)
    org_trajs, cf_trajs = zip(*org_cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    return org_trajs, cf_trajs

def train_test_split(org_trajs, cf_trajs, num_features, train_ratio=0.8):
    # randomise the order of the trajectories
    org_trajs = np.random.permutation(org_trajs)
    cf_trajs = np.random.permutation(cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    n_train = int(train_ratio * len(org_trajs))

    train = torch.cat((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=0)
    train_labels = torch.cat((org_trajs[:n_train,-1], cf_trajs[:n_train,-1]), dim=0)
    test = torch.cat((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=0)
    test_labels = torch.cat((org_trajs[n_train:,-1], cf_trajs[n_train:,-1]), dim=0)

    return train, train_labels, test, test_labels

def train_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio=0.8):
    n_train = int(train_ratio * len(org_trajs))

    org_trajs, cf_trajs = shuffle_together(org_trajs, cf_trajs)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,-1] - cf_trajs[n_train:,-1]

    return train, train_labels, test, test_labels

def train_test_split_contrastive_sidebyside(org_trajs, cf_trajs, num_features, n_train):
    org_trajs, cf_trajs = shuffle_together(org_trajs, cf_trajs)


    # train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    # combine the orginal and counterfactual trajectory features into one feature vector
    train = torch.stack((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=1).view(n_train, num_features*2)
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    test = torch.stack((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=1).view(len(org_trajs) - n_train, num_features*2)
    # test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,-1] - cf_trajs[n_train:,-1]

    return train, train_labels, test, test_labels 

def train_validation_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio=0.6, validation_ratio=0.2):
    n_train = int(train_ratio * len(org_trajs))
    n_validation = int(validation_ratio * len(org_trajs))

    org_trajs, cf_trajs = shuffle_together(org_trajs, cf_trajs)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    validation = org_trajs[n_train:n_train+n_validation,:num_features] - cf_trajs[n_train:n_train+n_validation,:num_features]
    validation_labels = org_trajs[n_train:n_train+n_validation,-1] - cf_trajs[n_train:n_train+n_validation,-1]
    test = org_trajs[n_train+n_validation:,:num_features] - cf_trajs[n_train+n_validation:,:num_features]
    test_labels = org_trajs[n_train+n_validation:,-1] - cf_trajs[n_train+n_validation:,-1]

    return train, train_labels, validation, validation_labels, test, test_labels


def train_model(train_set, train_labels, test_set, test_labels, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=hyperparameters.learning_rate, regularisation = hyperparameters.regularisation, num_layers = None, hidden_layer_sizes = None, base_path=None, l2=None, stop_epoch = 0, task_weight=None):

    # Initialise the model (either NN or LM)
    if config.model_type=='NN':
        # make a list of the layer dimensions: num_features, hidden_layer_sizes, 1

        # Initialise the neural network with the number of layers and the hidden_sizes
        layers = [num_features] + hidden_layer_sizes + [1]
        model = torch.nn.Sequential()
        model.add_module('linear' + str(0), torch.nn.Linear(layers[0], layers[1]))
        for i in range(2, num_layers):
            model.add_module('relu' + str(i-2), torch.nn.ReLU())
            model.add_module('linear' + str(i-1), torch.nn.Linear(layers[i-1], layers[i]))

    elif config.model_type=='linear':
        model = torch.nn.Linear(num_features, 1)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularisation)

    train_losses, test_losses, stop_train_losses, stop_test_losses = [], [], [], []

    for t in range(epochs):
        # if we want to stop the training at a certain epoch, save the model and the losses, but continue training until the end
        if t == stop_epoch-1:
            stop_model = deepcopy(model)
            stop_train_losses = deepcopy(train_losses)
            stop_test_losses = deepcopy(test_losses)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(train_set).squeeze()

        # Compute and print loss.
        loss = loss_fn(y_pred, train_labels)
        train_losses.append(loss.item())
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # record the loss at this epoch for the test set
        with torch.no_grad():
            model.eval()
            y_pred_test = model(test_set).squeeze()
            loss = loss_fn(y_pred_test, test_labels)
            test_losses.append(loss.item())
            model.train()
    
    model.eval()
    if stop_epoch != 0:
        stop_model.eval()
        return stop_model, train_losses, test_losses, stop_train_losses, stop_test_losses
    return model, train_losses, test_losses



def learning_repeats(path_org, path_cf, base_path, contrastive=True, baseline=0, n_train=800, task_weight=None):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1
    results_path = base_path + config.results_path + str(n_train) + "\\"

    test_lossess, train_lossess, all_train_losses, test_mean_errors, test_rmses, test_r2s, train_mean_errors, train_rmses, pearson_correlations, spearman_correlations, weights, all_test_losses, epochss, learning_rates, regularisations = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    test_loss_oods, test_mean_error_oods, test_rmse_oods, r2_oods, pearson_correlation_oods, spearman_correlation_oods, pred_label_pairs_oods = [], [], [], [], [], [], []

    # load ood test data
    # go up one folder from path
    path = base_path.split('\\')
    path = '\\'.join(path[:-1])

    num_layers = None
    hidden_sizes = None

    # load the data
    if os.path.isfile(results_path + 'data_split.pkl'):
        with open(results_path + 'data_split.pkl', 'rb') as f:
            train_set, train_labels, test_set, test_labels = pickle.load(f)
    else:
        train_set, train_labels, test_set, test_labels = train_test_split_contrastive_sidebyside(org_features, cf_features, num_features, n_train=n_train)
        # save the split of the data
        with open(results_path + 'data_split.pkl', 'wb') as f:
            pickle.dump([train_set, train_labels, test_set, test_labels], f)

    with open('datasets\\1000_ablations\only_one\combined_test_set.pkl', 'rb') as f:
        test_set_ood, test_labels_ood = pickle.load(f)

    # epochs, learning_rate, regularisation, num_layers, hidden_sizes = 221, 0.3, 0.1, 4, [12,6]
    if config.model_type == 'NN':
        learning_rate, regularisation, num_layers, hidden_sizes = NN_params.learning_rate, NN_params.regularisation, NN_params.num_layers, NN_params.hidden_layer_sizes
        learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization_architecture(train_set, train_labels, task_weight=task_weight)
        print(learning_rate, regularisation, num_layers, hidden_sizes)
    elif config.model_type == 'stepwise':
        train_step_wise_linear_model(train_set, train_labels)
        a = 1/0
    else:
        learning_rate, regularisation = LM_params.learning_rate, LM_params.regularisation
    # epochs = tune_epochs(train_set, train_labels, learning_rate, regularisation)
    epochs = 100

    for repeat in range(hyperparameters.number_of_seeds):
        # train the model (works for both, the linear and NN model)
        model, full_train_losses, full_test_losses, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features*2, epochs=hyperparameters.epochs_contrastive, stop_epoch=epochs, learning_rate=learning_rate, regularisation=regularisation, num_layers=num_layers, hidden_layer_sizes=hidden_sizes, task_weight=task_weight)

        # here we test on the left out test set
        test_loss, test_mean_error, test_rmse, r2, pearson_correlation, spearman_correlation, pred_label_pairs = evaluate_mimic(model, test_set, test_labels, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        # here we test on a seperate test set from a different distribution
        test_loss_ood, test_mean_error_ood, test_rmse_ood, r2_ood, pearson_correlation_ood, spearman_correlation_ood, pred_label_pairs_ood = evaluate_mimic(model, test_set_ood, test_labels_ood, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)

        test_lossess.append(test_loss)
        test_mean_errors.append(test_mean_error)
        test_rmses.append(test_rmse)
        test_r2s.append(r2)
        pearson_correlations.append(pearson_correlation)
        spearman_correlations.append(spearman_correlation)
        if config.model_type == 'linear':
            weights.append(get_weights(model, config.print_weights))
        train_lossess.append(train_losses[-1])
        all_train_losses.append(full_train_losses)
        all_test_losses.append(full_test_losses)
        train_mean_errors.append(torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())
        train_rmses.append(torch.sqrt(torch.mean((model(train_set).squeeze() - train_labels)**2)).item())

        test_loss_oods.append(test_loss_ood)
        test_mean_error_oods.append(test_mean_error_ood)
        test_rmse_oods.append(test_rmse_ood)
        r2_oods.append(r2_ood)
        pearson_correlation_oods.append(pearson_correlation_ood)
        spearman_correlation_oods.append(spearman_correlation_ood)
        pred_label_pairs_oods.append(pred_label_pairs_ood)
        print(test_mean_error, test_mean_error_ood)

        if config.save_model:
            path = results_path + "saved_models\\"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path + 'model' + str(repeat) + '.pt')


    # print('final train_loss', np.mean(train_losses))
    print('test_loss', np.mean(test_lossess))
    print('test mean error', np.mean(test_mean_errors))
    print('test rmse', np.mean(test_rmses))
    print('test r2', np.mean(test_r2s))
    print('pearson correlation', np.mean(pearson_correlations))
    print('spearman correlation', np.mean(spearman_correlations))
    print('average reward', torch.mean(train_labels))
    print('average prediction', torch.mean(model(train_set).squeeze()))
    if config.model_type == 'linear':
        weights = np.mean(weights, axis=0)
        print('weights', [v + ": " + str(weights[k]) for k,v in enumerate(config.features) if k < num_features])
    all_train_losses = np.mean(all_train_losses, axis=0)
    all_test_losses = np.mean(all_test_losses, axis=0)
    show_loss_plot(all_train_losses, all_test_losses, show=config.print_plot, save_path=results_path, epochs=epochs)
    # print('predicition label pairs', pred_label_pairs)

    print('test_loss_ood', np.mean(test_loss_oods))
    print('test mean error ood', np.mean(test_mean_error_oods))
    print('test rmse ood', np.mean(test_rmse_oods))
    print('test r2 ood', np.mean(r2_oods))
    print('pearson correlation ood', np.mean(pearson_correlation_oods))
    print('spearman correlation ood', np.mean(spearman_correlation_oods))

    if config.save_results:
        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess, 'test_mean_errors': test_mean_errors, 
                   'train_mean_errors': train_mean_errors, 'train_rmses': train_rmses, 'test_rmses': test_rmses, 'pearson_correlations': pearson_correlations, 'spearman_correlations': spearman_correlations, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses, 'pred_label_pairs': pred_label_pairs, 'r2s': test_r2s}

        save_results(to_save, results_path, baseline, type='results', data_mixture=(0,0), con=True)

        hyper_params = {'epochs': epochs, 'learning_rate': learning_rate, 'l2_regularisation': regularisation}
        save_results(hyper_params, results_path, baseline, type='hyper_params', data_mixture=(0,0))

        to_save = {'test_losses': test_loss_oods, 'test_mean_errors': test_mean_error_oods, 'test_rmses': test_rmse_oods, 'pearson_correlations': pearson_correlation_oods, 'spearman_correlations': spearman_correlation_oods, 'pred_label_pairs': pred_label_pairs_oods, 'r2s': r2_oods}
        save_results(to_save, results_path, baseline, type='results_ood', data_mixture=(0,0), con=True)

# split the data into k folds to run cross validation on
def split_for_cross_validation(train_set, train_labels, k=5):
    # split data into k folds
    train_set_folds = []
    train_labels_folds = []
    fold_size = int(len(train_set) / k)
    for i in range(k):
        train_set_folds.append(train_set[i*fold_size:(i+1)*fold_size])
        train_labels_folds.append(train_labels[i*fold_size:(i+1)*fold_size])
    return train_set_folds, train_labels_folds

# from the k-folds of the training data, return the training and validation sets for the kth fold
def cross_validate(train_set_folds, train_labels_folds, k):
    train_set_f = torch.cat(train_set_folds[:k] + train_set_folds[k+1:])
    train_labels_f = torch.cat(train_labels_folds[:k] + train_labels_folds[k+1:])
    validation_set_f = train_set_folds[k]
    validation_labels_f = train_labels_folds[k]
    return train_set_f, train_labels_f, validation_set_f, validation_labels_f

def tune_epochs(train_set, train_labels, lr, l2, num_layers=None, hidden_layer_sizes=None, task_weight=None):
    data_folds = config.data_folds
    train_set_folds, train_labels_folds = split_for_cross_validation(train_set, train_labels, k=data_folds)
    test_lossess = []
    num_features = num_features = len(train_set[0])

    for k in range(data_folds):
        train_set_f, train_labels_f, validation_set_f, validation_labels_f = cross_validate(train_set_folds, train_labels_folds, k)
        model, train_losses, test_losses = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lr, regularisation=l2, num_layers=num_layers, hidden_layer_sizes=hidden_layer_sizes, task_weight=task_weight)
        test_lossess.append(test_losses)
    
    avg_test_losses = np.mean(test_lossess, axis=0)
    # get minimum value and index
    min_test_loss = np.amin(avg_test_losses)
    min_index = np.argmin(avg_test_losses)
    print(min_test_loss, min_index)
    return min_index

def hyper_param_optimization_architecture(train_set, train_labels, task_weight=None):
    # we use 5-fold cross validation to find the best hyper parameters
    data_folds = config.data_folds
    train_set_folds, train_labels_folds = split_for_cross_validation(train_set, train_labels, k=data_folds)

    num_features = len(train_set[0])
    best_loss = 100000000
    best_epoch = 0
    best_lr = 0
    best_l2 = 0
    best_num_layers = 0
    best_hidden_layer_sizes = []

    # different NN architectures with (number of layers, hidden layer sizes)
    if config.model_type == 'NN':
        architectures = [(2, [[]]), (3, [[16], [8]]), (4, [[8,4], [8,8], [12,6], [12,12], [16,8], [16,16], [20,20]])]
        architectures = [(3, [[20], [40], [60], [92]]), (4, [[20,10], [30,15], [40,20], [60,30]]), (5, [[20,20,10], [30,30,10], [40,40,20], [50,50,50]])]
    else:
        architectures = [(None, [None])]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    # learning_rates = [0.01]
    l2_lambdas = [0.0001, 0.001, 0.01, 0.1]

    # loop over network architectures (not relevant for the LM)
    for num_layers, hidden_layer_sizes in architectures:
        for hidden_layer_size in hidden_layer_sizes:
            print(hidden_layer_size)
            # loop over hyper parameters
            for lrs in learning_rates:
                print('learning rate', lrs)
                for l2 in l2_lambdas:
                    # iterate over the k folds for cross validation
                    test_lossess = []
                    for k in range(data_folds):
                        train_set_f, train_labels_f, validation_set_f, validation_labels_f = cross_validate(train_set_folds, train_labels_folds, k)
                        model, train_losses, test_losses = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lrs, regularisation=l2, num_layers=num_layers, hidden_layer_sizes=hidden_layer_size, task_weight=task_weight)
                        test_lossess.append(test_losses)

                    # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
                    avg_test_losses = np.mean(test_lossess, axis=0)
                    # get minimum value and index
                    min_test_loss = np.amin(avg_test_losses)
                    min_index = np.argmin(avg_test_losses)
                    print(min_test_loss, min_index)

                    if min_test_loss < best_loss:
                        best_loss = min_test_loss
                        best_lr = lrs
                        best_l2 = l2
                        best_epoch = min_index
                        best_num_layers = num_layers
                        best_hidden_layer_sizes = hidden_layer_size
            # plt.legend()
            # plt.show()
    if config.model_type == 'NN':
        print(best_loss, best_epoch, best_lr, best_l2, best_hidden_layer_sizes, best_num_layers)
    else:
        print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2, best_num_layers, best_hidden_layer_sizes

def hyper_param_optimization(train_set, train_labels, task_weight=None):
    # we use 5-fold cross validation to find the best hyper parameters
    data_folds = config.data_folds
    train_set_folds, train_labels_folds = split_for_cross_validation(train_set, train_labels, k=data_folds)

    num_features = len(train_set[0])
    best_loss = 100000000
    best_epoch = 0
    best_lr = 0
    best_l2 = 0

    learning_rates = [0.001, 0.01, 0.1]
    # learning_rates = [0.01]
    l2_lambdas = [0.001, 0.01, 0.1]

    # return 32, 0.01, 0.01, 4, [8,4]

    # loop over hyper parameters
    for lrs in learning_rates:
        print('learning rate', lrs)
        for l2 in l2_lambdas:
            # iterate over the k folds for cross validation
            test_lossess = []
            for k in range(data_folds):
                train_set_f, train_labels_f, validation_set_f, validation_labels_f = cross_validate(train_set_folds, train_labels_folds, k)
                model, train_losses, test_losses = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lrs, regularisation=l2, num_layers=NN_params.num_layers, hidden_layer_sizes=NN_params.hidden_layer_sizes, task_weight=task_weight)
                test_lossess.append(test_losses)

            # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
            avg_test_losses = np.mean(test_lossess, axis=0)
            # get minimum value and index
            min_test_loss = np.amin(avg_test_losses)
            min_index = np.argmin(avg_test_losses)
            print(min_test_loss, min_index)

            if min_test_loss < best_loss:
                best_loss = min_test_loss
                best_lr = lrs
                best_l2 = l2
                best_epoch = min_index
            # plt.legend()
            # plt.show()
    if config.model_type == 'NN':
        print(best_loss, best_epoch, best_lr, best_l2)
    else:
        print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2

if __name__ == '__main__':
    folder_path = 'datasets\\1000_ablations\weights\\set5\\100'
    print(folder_path, config.model_type)

    # if there is an argument in the console
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    path_org_cte = folder_path + '\org_features_new.pkl'
    path_cf_cte = folder_path + '\cf_features_new.pkl'

    task_weight = (1,3)

    # training_numbers = [5,10,20,50, 100, 200, 500, 800]
    # for n_train in training_numbers:
        # print(n_train)
        # learning_repeats(path_org_cte, path_cf_cte, folder_path, contrastive=True, baseline=0, n_train=n_train)

    learning_repeats(path_org_cte, path_cf_cte, folder_path, multi_task=True, baseline=0, n_train=80, task_weight=task_weight)