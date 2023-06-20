import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from evaluate_mimic import evaluate_mimic
import sys
import random
from utils.util_functions import iterate_through_folder, save_results

class hyperparameters:
    learning_rate = 1e-1
    regularisation = 1e-2
    l1_lambda = 1e-1
    epochs_non_contrastive = 10000
    epochs_contrastive = 10000
    number_of_seeds = 5
    
    
class config:    
    print_plot = True
    print_examples = False
    print_weights = False
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'could_have_saved', 'final_number_of_unsaved_citizens', 'moved_towards_closest_citizen', 'bias']
    save_results = True
    print_worst_examples = False
    print_best_examples = False

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
    # TODO: this needs to be rewritten
    n_train = int(train_ratio * len(org_trajs))

    # randomise org_trajs and cf_trajs together
    org_cf_trajs = list(zip(org_trajs, cf_trajs))
    random.shuffle(org_cf_trajs)
    org_trajs, cf_trajs = zip(*org_cf_trajs)

    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]

    test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,-1] - cf_trajs[n_train:,-1]

    return train, train_labels, test, test_labels

def train_validation_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio=0.6, validation_ratio=0.2):
    test_ratio = 1 - train_ratio - validation_ratio
    n_train = int(train_ratio * len(org_trajs))
    n_validation = int(validation_ratio * len(org_trajs))
    n_test = int(test_ratio * len(org_trajs))

    # randomise org_trajs and cf_trajs together
    org_cf_trajs = list(zip(org_trajs, cf_trajs))
    random.shuffle(org_cf_trajs)
    org_trajs, cf_trajs = zip(*org_cf_trajs)

    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]

    validation = org_trajs[n_train:n_train+n_validation,:num_features] - cf_trajs[n_train:n_train+n_validation,:num_features]
    validation_labels = org_trajs[n_train:n_train+n_validation,-1] - cf_trajs[n_train:n_train+n_validation,-1]

    test = org_trajs[n_train+n_validation:,:num_features] - cf_trajs[n_train+n_validation:,:num_features]
    test_labels = org_trajs[n_train+n_validation:,-1] - cf_trajs[n_train+n_validation:,-1]

    return train, train_labels, validation, validation_labels, test, test_labels


# def contrastive_loss(pred, )

def show_loss_plot(train_losses, test_losses, show=True, lr=None, l2=None, basepath=None):
    if not config.print_plot:
        return

    # plot the training and test losses on a log scale
    #show labels
    plt.semilogy(test_losses, label='test' + ' lr={}'.format(lr))
    plt.semilogy(train_losses, label='train' + ' lr={}'.format(lr), linestyle='--')
    if show:
        if l2:
            plt.title('l2', l2)
        # plt.show()
        path = base_path + "\\results\\"
        #  if the results folder does not exist, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path + 'loss_plot.png')
        plt.close()

def print_example_predictions(model, test_set, test_labels):
    if not config.print_examples:
        return
    # print the first 5 predictions
    with torch.no_grad():
        y_pred_test = model(test_set).squeeze()
        for i in range(5):
            print('features', test_set[i], 'prediction', y_pred_test[i].item(), 'label', test_labels[i].item())

def get_weights(model):
    weights = [model.weight[0][i].item() for i in range(len(model.weight[0]))]
    weights.append(model.bias[0].item())
    # weights = [model.weight[0][0].item(), model.weight[0][1].item(), model.weight[0][2].item(), model.weight[0][3].item(), model.weight[0][4].item(), model.weight[0][5].item(), model.weight[0][6].item(), model.weight[0][7].item(), model.bias[0].item()]
    if config.print_weights:
        print('citizens_saved', model.weight[0][0].item())
        print('unsaved_citizens', model.weight[0][1].item())
        print('distance_to_citizen', model.weight[0][2].item())
        print('standing_on_extinguisher', model.weight[0][3].item())
        print('length', model.weight[0][4].item())
        print('could_have_saved', model.weight[0][5].item())
        print('final_number_of_unsaved_citizens', model.weight[0][6].item())
        print('moved_towards_closest_citizen', model.weight[0][7].item())
        print('bias', model.bias[0].item())
    return weights

def train_model(train_set, train_labels, test_set, test_labels, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=hyperparameters.learning_rate, regularisation = hyperparameters.regularisation):
    # a linear model with the features as input
    model = torch.nn.Linear(num_features, 1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularisation)
    train_losses = []
    test_losses = []
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(train_set).squeeze()

        # Compute and print loss.
        # mse_looss = loss_fn(y_pred, train_labels)
        loss = loss_fn(y_pred, train_labels)
        
        # add l1 regularisation
        # l1_reg = torch.tensor(0.)
        # for param in model.parameters():
        #     l1_reg += torch.sum(torch.abs(param))
        # loss = mse_loss + regularisation * l1_reg

        # train_losses.append(mse_loss.item())
        train_losses.append(loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        with torch.no_grad():
            y_pred_test = model(test_set).squeeze()
            loss = loss_fn(y_pred_test, test_labels)
            test_losses.append(loss.item())
    
    model.eval()
    return model, train_losses, test_losses

def learning_repeats(path_org, path_cf, base_path, contrastive=True, baseline=0):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1

    test_lossess = []
    train_lossess = []
    all_train_losses = []
    test_mean_errors = []
    train_mean_errors = []
    pearson_correlations = []
    spearman_correlations = []
    weights = []
    all_test_losses = []

    if contrastive:
    #     num_features=4
        train_set, train_labels, validation_set, validation_labels = train_test_split_contrastive(org_features, cf_features, num_features, train_ratio=0.6)
    #     # only use first 4 features
    #     train_set = train_set[:, :4]
    #     validation_set = validation_set[:, :4]
        epochs, learning_rate, regularisation = hyper_param_optimization(train_set, train_labels, validation_set, validation_labels)

    # only for the purpose of testing. Remove later
    # epochs = 10000
    for repeat in range(hyperparameters.number_of_seeds):
        if contrastive:
            # num_features = 4
            train_set, train_labels, test_set, test_labels = train_test_split_contrastive(org_features, cf_features, num_features, train_ratio=0.6)
            # only use first 4 features
            # train_set = train_set[:, :4]
            # test_set = test_set[:, :4]
            # num_features = len(train_set[0])
        else:
            train_set, train_labels, test_set, test_labels = train_test_split(org_features, cf_features, num_features, train_ratio=0.6)

        model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features, epochs=epochs)
        # model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features, epochs=epochs, learning_rate=learning_rate, regularisation=regularisation)

        test_loss, test_mean_error, pearson_correlation, spearman_correlation = evaluate_mimic(model, test_set, test_labels, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)

        test_lossess.append(test_loss)
        test_mean_errors.append(test_mean_error)
        pearson_correlations.append(pearson_correlation)
        spearman_correlations.append(spearman_correlation)
        weights.append(get_weights(model))
        train_lossess.append(train_losses[-1])
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        train_mean_errors.append(torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    print('final train_loss', np.mean(train_losses))
    print('final train mean error', np.mean(train_mean_errors))
    print('final test_loss', np.mean(test_lossess))
    print('final test mean error', np.mean(test_mean_errors))
    print('final pearson correlation', np.mean(pearson_correlations))
    print('final spearman correlation', np.mean(spearman_correlations))
    weights = np.mean(weights, axis=0)
    print('final weights', [v + ": " + str(weights[k]) for k,v in enumerate(config.features) if k < num_features])
    all_train_losses = np.mean(all_train_losses, axis=0)
    all_test_losses = np.mean(all_test_losses, axis=0)
    show_loss_plot(all_train_losses, all_test_losses, base_path)

    if config.save_results:
        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess, 'test_mean_errors': test_mean_errors, 
                   'train_mean_errors': train_mean_errors, 'pearson_correlations': pearson_correlations, 'spearman_correlations': spearman_correlations, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses}
        
        save_results(to_save, base_path, contrastive, baseline)

        hyper_params = {'epochs': epochs, 'learning_rate': learning_rate, 'l2_regularisation': regularisation}
        save_results(hyper_params, base_path, contrastive, baseline, hyper_params=True)


def hyper_param_optimization(train_set, train_labels, validation_set, validation_labels):
    # epochs = [1000, 2000, 3000, 5000]
    # learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    # l1_lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1]

    learning_rates = [0.001, 0.01, 0.03, 0.1, 0.3]
    l2_lambdas = [0, 0.1, 0.3, 1, 3, 10, 100]
    # learning_rates = [0.1]
    # l2_lambdas = [0]


    num_features = len(train_set[0])

    best_loss = 100000000
    best_epoch = 0
    best_lr = 0
    best_l2 = 0

    all_test_losses = []
    for lrs in learning_rates:
        for l2 in l2_lambdas:
            model, train_losses, test_losses = train_model(train_set, train_labels, validation_set, validation_labels, num_features, 10000, lrs, regularisation=0)
            # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
            all_test_losses.append(test_losses[-1])
            if test_losses[-1] < best_loss:
                best_loss = test_losses[-1]
                best_lr = lrs
                best_l2 = l2
                best_epoch = test_losses.index(best_loss)
    print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2

if __name__ == '__main__':
    folder_path = 'evaluation\datasets\\100_ablations_3'

    # if there is an argument in the console
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    all_folder_base_paths = iterate_through_folder(folder_path)
    all_folder_base_paths.reverse()

    for base_path in all_folder_base_paths:
        print(base_path)
        path_org_cte = base_path + '\org_features.pkl'
        path_cf_cte = base_path + '\cf_features.pkl'
        path_org_baseline = base_path + '\org_features_baseline.pkl'
        path_cf_baseline = base_path + '\cf_features_baseline.pkl'

        # contrastive learning
        print('# COUNTERFACTUAL Contrastive Learning')
        learning_repeats(path_org_cte, path_cf_cte, base_path, contrastive=True, baseline=0)
        print('-------------------------')
    #     print('# BASELINE Contrastive Learning')
    #     learning_repeats(path_org_baseline, path_cf_baseline, base_path, contrastive=True, baseline=2)
    #     print('-------------------------')
        # print('# BASELINE Non-Contrastive Learning')
        # learning_repeats(path_org_baseline, path_cf_baseline, base_path, contrastive=False, baseline=2)
        # print('-------------------------')
        # print('# COUNTERFACTUAL Non-Contrastive Learning')
        # learning_repeats(path_org_cte, path_cf_cte, base_path, contrastive=False, baseline=0)

        # print('\n')

    # remove all letters from the folder path starting from the last backslash (going up one folder)
    # base_path_root = folder_path + '\\baseline'
    # path_org_no_quality = base_path_root + '\org_features.pkl'
    # path_cf_no_quality = base_path_root + '\cf_features.pkl'

    # print('-------------------------')
    # print('# NO QUALITY Contrastive Learning')
    # learning_repeats(path_org_no_quality, path_cf_no_quality, base_path_root, contrastive=True, baseline=1)
    # print('-------------------------')
    # print('# NO QUALITY Non-Contrastive Learning')
    # learning_repeats(path_org_no_quality, path_cf_no_quality, base_path_root, contrastive=False, baseline=1)