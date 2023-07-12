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

class hyperparameters:
    learning_rate = 1e-1
    regularisation = 1e-2
    l1_lambda = 1e-1
    epochs_non_contrastive = 10000
    epochs_contrastive = 1000
    number_of_seeds = 10
    
    
class config:    
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'could_have_saved', 'final_number_of_unsaved_citizens', 'moved_towards_closest_citizen', 'bias']
    model_type = 'linear' # model_type = 'NN' or 'linear'
    data_folds = 4
    results_path = "\\results_mixedLM\\" # Foldername to save results to
    print_plot = False
    print_examples = False
    print_weights = False
    save_results = True
    print_worst_examples = False
    print_best_examples = False
    save_model = False

# randomises the order of trajectories, while keeping the pairs of original and counterfactual trajectories together
# also makes them into tensors
def shuffle_together(org_trajs, cf_trajs):
    org_cf_trajs = list(zip(org_trajs, cf_trajs))
    random.shuffle(org_cf_trajs)
    org_trajs, cf_trajs = zip(*org_cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    return org_trajs, cf_trajs

def train_test_split_single(org_trajs, cf_trajs, num_features, train_ratio=0.8):
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

# splits the data into train and test sets, where the train set contains both contrastive and single trajectories according to the percentages specified in config
def train_test_split_mixed(org_trajs, cf_trajs, num_features, train_ratio=0.8, data_mixture=(1,0)):
    n_train = int(train_ratio * len(org_trajs))

    train_con, train_labels_con, test_con, test_labels_con = train_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio)
    train_sin, train_labels_sin, test_sin, test_labels_sin = train_test_split_single(org_trajs, cf_trajs, num_features, train_ratio)

    n_single = int(data_mixture[1] * len(train_sin))
    n_contrastive = int(data_mixture[0] * len(train_con))
    train = np.concatenate((train_sin[:n_single], train_con[:n_contrastive]))
    labels = np.concatenate((train_labels_sin[:n_single], train_labels_con[:n_contrastive]))
    train_set, train_labels = shuffle_together(train, labels)

    return train_set, train_labels, test_con, test_labels_con, test_sin, test_labels_sin


def train_model(train_set, train_labels, test_set_con, test_labels_con, test_set_sin=None, test_labels_sin=None, num_features=8, epochs = hyperparameters.epochs_contrastive, learning_rate=hyperparameters.learning_rate, regularisation = hyperparameters.regularisation, num_layers = None, hidden_layer_sizes = None, base_path=None, l2=None, stop_epoch = 0):

    # Initialise the model (either NN or LM)
    if config.model_type=='NN':
        # make a list of the layer dimensions: num_features, hidden_layer_sizes, 1
        layer_dims = [num_features] + hidden_layer_sizes + [1]

        # Initialise the neural network with the number of layers and the hidden_sizes
        model = torch.nn.Sequential()
        if num_layers:
            model.add_module('linear' + str(0), torch.nn.Linear(layer_dims[0], layer_dims[1]))
            for i in range(1, num_layers-1):
                model.add_module('relu' + str(i), torch.nn.ReLU())
                model.add_module('linear' + str(i), torch.nn.Linear(layer_dims[i], layer_dims[i+1]))

    elif config.model_type=='linear':
        model = torch.nn.Linear(num_features, 1)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularisation)

    train_losses, test_losses_con, test_losses_sin, stop_train_losses, stop_test_losses_con, stop_test_losses_sin = [], [], [], [], [], []

    for t in range(epochs):
        # if we want to stop the training at a certain epoch, save the model and the losses, but continue training until the end
        if t == stop_epoch-1:
            stop_model = deepcopy(model)
            stop_train_losses = deepcopy(train_losses)
            stop_test_losses_con = deepcopy(test_losses_con)
            stop_test_losses_sin = deepcopy(test_losses_sin)

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
            y_pred_test = model(test_set_con).squeeze()
            loss = loss_fn(y_pred_test, test_labels_con)
            test_losses_con.append(loss.item())

            if test_set_sin != None:
                y_pred_test = model(test_set_sin).squeeze()
                loss = loss_fn(y_pred_test, test_labels_sin)
                test_losses_sin.append(loss.item())
            model.train()
    
    model.eval()
    if stop_epoch != 0:
        stop_model.eval()
        if test_set_sin != None:
            return stop_model, train_losses, test_losses_con, test_losses_sin, stop_train_losses, stop_test_losses_con, stop_test_losses_sin
        else:
            return stop_model, train_losses, test_losses_con, stop_train_losses, stop_test_losses_con
    if test_set_sin != None:
        return model, train_losses, test_losses_con, test_losses_sin, None, None, None
    else:
        return model, train_losses, test_losses_con

def learning_repeats(path_org, path_cf, base_path, baseline=0, data_mixture = (1,0)):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1

    results_path = base_path + config.results_path + str(data_mixture[0]) + ',' + str(data_mixture[1]) + "\\"

    num_layers, hidden_sizes = None, None

    test_lossess_con, train_lossess, all_train_losses, test_mean_errors_con, test_rmses_con, test_r2s_con, train_mean_errors, train_rmses, pearson_correlations_con, spearman_correlations_con, weights, all_test_losses_con, epochss, learning_rates, regularisations = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    test_lossess_sin, test_mean_errors_sin, test_rmses_sin, test_r2s_sin, pearson_correlations_sin, spearman_correlations_sin, all_test_losses_sin = [], [], [], [], [], [], []
    test_loss_oods_con, test_mean_error_oods_con, test_rmse_oods_con, r2_oods_con, pearson_correlation_oods_con, spearman_correlation_oods_con, pred_label_pairs_oods_con = [], [], [], [], [], [], []
    test_loss_oods_sin, test_mean_error_oods_sin, test_rmse_oods_sin, r2_oods_sin, pearson_correlation_oods_sin, spearman_correlation_oods_sin, pred_label_pairs_oods_sin = [], [], [], [], [], [], []

    # load ood test data
    # go up one folder from path
    path = base_path.split('\\')
    path = '\\'.join(path[:-1])
    # load the data
    org_features_ood = read(path + '\\baseline1\org_features_norm3.pkl')
    cf_features_ood = read(path + '\\baseline1\cf_features_norm3.pkl')
    test_set_ood_con, test_labels_ood_con, _ , _ = train_test_split_contrastive(org_features_ood, cf_features_ood, num_features, train_ratio=1)
    test_set_ood_sin, test_labels_ood_sin, _ , _ = train_test_split_single(org_features_ood, cf_features_ood, num_features, train_ratio=1)

    train_set, train_labels, test_set_con, test_labels_con, test_set_sin, test_labels_sin = train_test_split_mixed(org_features, cf_features, num_features, train_ratio=0.8, data_mixture=data_mixture)
    epochs, learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization(train_set, train_labels)

    for repeat in range(hyperparameters.number_of_seeds):
        # train the model (works for both, the linear and NN model)
        model, full_train_losses, full_test_losses_con, full_test_losses_sin, train_losses, test_losses_con, test_losses_sin = train_model(train_set, train_labels, test_set_con, test_labels_con, test_set_sin, test_labels_sin, num_features, epochs=hyperparameters.epochs_contrastive, stop_epoch=epochs, learning_rate=learning_rate, regularisation=regularisation, num_layers=num_layers, hidden_layer_sizes=hidden_sizes)

        # here we test on the left out test set
        test_loss_con, test_mean_error_con, test_rmse_con, r2_con, pearson_correlation_con, spearman_correlation_con, pred_label_pairs_con = evaluate_mimic(model, test_set_con, test_labels_con, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        test_loss_sin, test_mean_error_sin, test_rmse_sin, r2_sin, pearson_correlation_sin, spearman_correlation_sin, pred_label_pairs_sin = evaluate_mimic(model, test_set_sin, test_labels_sin, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        # here we test on a seperate test set from a different distribution
        test_loss_ood_con, test_mean_error_ood_con, test_rmse_ood_con, r2_ood_con, pearson_correlation_ood_con, spearman_correlation_ood_con, pred_label_pairs_ood_con = evaluate_mimic(model, test_set_ood_con, test_labels_ood_con, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        test_loss_ood_sin, test_mean_error_ood_sin, test_rmse_ood_sin, r2_ood_sin, pearson_correlation_ood_sin, spearman_correlation_ood_sin, pred_label_pairs_ood_sin = evaluate_mimic(model, test_set_ood_sin, test_labels_ood_sin, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)

        test_lossess_con.append(test_loss_con)
        test_lossess_sin.append(test_loss_sin)
        test_mean_errors_con.append(test_mean_error_con)
        test_mean_errors_sin.append(test_mean_error_sin)
        test_rmses_con.append(test_rmse_con)
        test_rmses_sin.append(test_rmse_sin)
        test_r2s_con.append(r2_con)
        test_r2s_sin.append(r2_sin)
        pearson_correlations_con.append(pearson_correlation_con)
        pearson_correlations_sin.append(pearson_correlation_sin)
        spearman_correlations_con.append(spearman_correlation_con)
        spearman_correlations_sin.append(spearman_correlation_sin)
        if config.model_type == 'linear':
            weights.append(get_weights(model, config.print_weights))
        train_lossess.append(train_losses[-1])
        all_train_losses.append(full_train_losses)
        all_test_losses_con.append(full_test_losses_con)
        all_test_losses_sin.append(full_test_losses_sin)
        train_mean_errors.append(torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())
        train_rmses.append(torch.sqrt(torch.mean((model(train_set).squeeze() - train_labels)**2)).item())

        test_loss_oods_con.append(test_loss_ood_con)
        test_loss_oods_sin.append(test_loss_ood_sin)
        test_mean_error_oods_con.append(test_mean_error_ood_con)
        test_mean_error_oods_sin.append(test_mean_error_ood_sin)
        test_rmse_oods_con.append(test_rmse_ood_con)
        test_rmse_oods_sin.append(test_rmse_ood_sin)
        r2_oods_con.append(r2_ood_con)
        r2_oods_sin.append(r2_ood_sin)
        pearson_correlation_oods_con.append(pearson_correlation_ood_con)
        pearson_correlation_oods_sin.append(pearson_correlation_ood_sin)
        spearman_correlation_oods_con.append(spearman_correlation_ood_con)
        spearman_correlation_oods_sin.append(spearman_correlation_ood_sin)
        pred_label_pairs_oods_con.append(pred_label_pairs_ood_con)
        pred_label_pairs_oods_sin.append(pred_label_pairs_ood_sin)
        print(test_mean_error_con, test_mean_error_ood_con)
        print(test_mean_error_sin, test_mean_error_ood_sin)

        if config.save_model:
            path = base_path + "\\results_normaliserewards3\\saved_models\\"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path + 'model' + str(repeat) + '.pt')


    # print('final train_loss', np.mean(train_losses))
    print('test_loss', np.mean(test_lossess_con), np.mean(test_lossess_sin))
    print('test mean error', np.mean(test_mean_errors_con), np.mean(test_mean_errors_sin))
    print('test rmse', np.mean(test_rmses_con), np.mean(test_rmses_sin))
    print('test r2', np.mean(test_r2s_con), np.mean(test_r2s_sin))
    print('pearson correlation', np.mean(pearson_correlations_con), np.mean(pearson_correlations_sin))
    print('spearman correlation', np.mean(spearman_correlations_con), np.mean(spearman_correlations_sin))
    average_reward = torch.mean(train_labels)
    print('average reward', average_reward)
    average_prediction = torch.mean(model(train_set).squeeze())
    print('average prediction', average_prediction)
    if config.model_type == 'linear':
        weights = np.mean(weights, axis=0)
        print('weights', [v + ": " + str(weights[k]) for k,v in enumerate(config.features) if k < num_features])
    all_train_losses = np.mean(all_train_losses, axis=0)
    all_test_losses_con = np.mean(all_test_losses_con, axis=0)
    all_test_losses_sin = np.mean(all_test_losses_sin, axis=0)
    show_loss_plot(all_train_losses, all_test_losses_con, show=config.print_plot, save_path=results_path, epochs=epochs, third_graph=all_test_losses_sin)
    print('predicition label pairs', pred_label_pairs_con)
    print('predicition label pairs', pred_label_pairs_sin)

    print('test_loss_ood', np.mean(test_loss_oods_con), np.mean(test_loss_oods_sin))
    print('test mean error ood', np.mean(test_mean_error_oods_con), np.mean(test_mean_error_oods_sin))
    print('test rmse ood', np.mean(test_rmse_oods_con), np.mean(test_rmse_oods_sin))
    print('test r2 ood', np.mean(r2_oods_con), np.mean(r2_oods_sin))
    print('pearson correlation ood', np.mean(pearson_correlation_oods_con), np.mean(pearson_correlation_oods_sin))
    print('spearman correlation ood', np.mean(spearman_correlation_oods_con), np.mean(spearman_correlation_oods_sin))

    if config.save_results:
        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess_con, 'test_mean_errors': test_mean_errors_con, 
                   'train_mean_errors': train_mean_errors, 'train_rmses': train_rmses, 'test_rmses': test_rmses_con, 'pearson_correlations': pearson_correlations_con, 'spearman_correlations': spearman_correlations_con, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses_con, 'average_reward': average_reward, 'pred_label_pairs': pred_label_pairs_con, 'average_prediction': average_prediction, 'r2s': test_r2s_con}
        save_results(to_save, results_path, baseline, type='results', data_mixture=data_mixture, con=True)

        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess_sin, 'test_mean_errors': test_mean_errors_sin, 
                   'train_mean_errors': train_mean_errors, 'train_rmses': train_rmses, 'test_rmses': test_rmses_sin, 'pearson_correlations': pearson_correlations_sin, 'spearman_correlations': spearman_correlations_sin, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses_sin, 'average_reward': average_reward, 'pred_label_pairs': pred_label_pairs_sin, 'average_prediction': average_prediction, 'r2s': test_r2s_sin}
        save_results(to_save, results_path, baseline, type='results', data_mixture=data_mixture, con=False)

        hyper_params = {'epochs': epochss, 'learning_rate': learning_rates, 'l2_regularisation': regularisations}
        save_results(hyper_params, results_path, baseline, type='hyper_params', data_mixture=data_mixture)

        to_save = {'test_losses': test_loss_oods_con, 'test_mean_errors': test_mean_error_oods_con, 'test_rmses': test_rmse_oods_con, 'pearson_correlations': pearson_correlation_oods_con, 'spearman_correlations': spearman_correlation_oods_con, 'pred_label_pairs': pred_label_pairs_oods_con, 'r2s': r2_oods_con}
        save_results(to_save, results_path, baseline, type='results_ood', data_mixture=data_mixture, con=True)

        to_save = {'test_losses': test_loss_oods_sin, 'test_mean_errors': test_mean_error_oods_sin, 'test_rmses': test_rmse_oods_sin, 'pearson_correlations': pearson_correlation_oods_sin, 'spearman_correlations': spearman_correlation_oods_sin, 'pred_label_pairs': pred_label_pairs_oods_sin, 'r2s': r2_oods_sin}
        save_results(to_save, results_path, baseline, type='results_ood', data_mixture=data_mixture, con=False)

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

def hyper_param_optimization(train_set, train_labels):
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
        architectures = [(4, [[4,2], [4,4], [6,3], [6,6], [8,4], [8,8], [10,10]]), (5, [[8,6,4], [10,10,5], [8,8,8]])]
        # architectures = [(4, [[4,4]])]
    else:
        architectures = [(None, [None])]
    learning_rates = [0.001, 0.01, 0.1, 0.3]
    l2_lambdas = [0, 0.01, 0.1, 1]

    # loop over network architectures (not relevant for the LM)
    for num_layers, hidden_layer_sizes in architectures:
        for hidden_layer_size in hidden_layer_sizes:
            best_loss_tmp = 10000000000000000
            # loop over hyper parameters
            for lrs in learning_rates:
                print('learning rate', lrs)
                for l2 in l2_lambdas:
                    # iterate over the k folds for cross validation
                    test_lossess = []
                    for k in range(data_folds):
                        train_set_f, train_labels_f, validation_set_f, validation_labels_f = cross_validate(train_set_folds, train_labels_folds, k)
                        model, train_losses, test_losses = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features=num_features, epochs=hyperparameters.epochs_contrastive, learning_rate=lrs, regularisation=l2, num_layers=num_layers, hidden_layer_sizes=hidden_layer_size)
                        test_lossess.append(test_losses)

                    # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
                    avg_test_losses = np.mean(test_lossess, axis=0)
                    # get minimum value and index
                    min_test_loss = np.amin(avg_test_losses)
                    min_index = np.argmin(avg_test_losses)

                    if min_test_loss < best_loss:
                        best_loss = min_test_loss
                        best_lr = lrs
                        best_l2 = l2
                        best_epoch = min_index
                        best_num_layers = num_layers
                        best_hidden_layer_sizes = hidden_layer_size
                    if min_test_loss < best_loss_tmp:
                        best_loss_tmp = min_test_loss
            print(num_layers, hidden_layer_size, best_loss_tmp)
            # plt.legend()
            # plt.show()
    if config.model_type == 'NN':
        print(best_loss, best_epoch, best_lr, best_l2, best_hidden_layer_sizes, best_num_layers)
    else:
        print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2, best_num_layers, best_hidden_layer_sizes

def experiment_over_all_folders():
    folder_path = 'datasets\\100_ablations_3'

    # if there is an argument in the console
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    all_folder_base_paths = iterate_through_folder(folder_path)
    all_folder_base_paths.reverse()

    for base_path in all_folder_base_paths:
        print(base_path)
        path_org_cte = base_path + '\org_features_norm3.pkl'
        path_cf_cte = base_path + '\cf_features_norm3.pkl'

        # see if base_path contains 'baseline'
        if 'baseline' in base_path:
            print('# BASELINE Contrastive Learning')
            learning_repeats(path_org_cte, path_cf_cte, base_path, contrastive=True, baseline=1)
            continue
        else:
            print('# COUNTERFACTUAL Contrastive Learning')
            learning_repeats(path_org_cte, path_cf_cte, base_path, contrastive=True, baseline=0)
        print('-------------------------')
        print('\n')

def experiment_for_single_folder():
    folder_path = 'datasets\\100_ablations_3\\pvcd100'

    # if there is an argument in the console
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    path_org_cte = folder_path + '\org_features_norm3.pkl'
    path_cf_cte = folder_path + '\cf_features_norm3.pkl'

    data_mixtures = [(1,0), (1,0.25), (1,0.5), (1,0.75), (1,1), (0.75,1), (0.5,1), (0.25,1), (0,1)]
    for data_mixture in data_mixtures:
        print('# MIXTURE: ', data_mixture)
        learning_repeats(path_org_cte, path_cf_cte, folder_path, baseline=0, data_mixture=data_mixture)

if __name__ == '__main__':
    experiment_for_single_folder()