import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from evaluate_mimic import evaluate_mimic
import sys
import random

class hyperparameters:
    learning_rate = 1e-1
    epochs_non_contrastive = 10000
    epochs_contrastive = 3000
    number_of_seeds = 50
    
class config:    
    print_plot = True
    print_examples = True
    print_weights = True
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'bias']

def train_test_split(org_trajs, cf_trajs, num_features, train_ratio=0.8):
    # randomise the order of the trajectories
    org_trajs = np.random.permutation(org_trajs)
    cf_trajs = np.random.permutation(cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    n_train = int(train_ratio * len(org_trajs))

    train = torch.cat((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=0)
    train_labels = torch.cat((org_trajs[:n_train,num_features], cf_trajs[:n_train,num_features]), dim=0)
    test = torch.cat((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=0)
    test_labels = torch.cat((org_trajs[n_train:,num_features], cf_trajs[n_train:,num_features]), dim=0)

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
    train_labels = org_trajs[:n_train,num_features] - cf_trajs[:n_train,num_features]

    test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,num_features] - cf_trajs[n_train:,num_features]

    return train, train_labels, test, test_labels

# def contrastive_loss(pred, )

def show_loss_plot(train_losses, test_losses):
    if not config.print_plot:
        return

    # plot the training and test losses on a log scale
    #show labels
    plt.semilogy(test_losses, label='test')
    plt.semilogy(train_losses, label='train')
    plt.legend()
    plt.show()

def print_example_predictions(model, test_set, test_labels):
    if not config.print_examples:
        return
    # print the first 5 predictions
    with torch.no_grad():
        y_pred_test = model(test_set).squeeze()
        for i in range(5):
            print('features', test_set[i], 'prediction', y_pred_test[i].item(), 'label', test_labels[i].item())

def get_weights(model, print=False):
    if not config.print_weights:
        return
    weights = [model.weight[0][0].item(), model.weight[0][1].item(), model.weight[0][2].item(), model.weight[0][3].item(), model.weight[0][4].item(), model.bias[0].item()]
    if print:
        print('citizens_saved', model.weight[0][0].item())
        print('unsaved_citizens', model.weight[0][1].item())
        print('distance_to_citizen', model.weight[0][2].item())
        print('standing_on_extinguisher', model.weight[0][3].item())
        print('length', model.weight[0][4].item())
        print('bias', model.bias[0].item())
    return weights

def train_model(train_set, train_labels, test_set, test_labels, num_features, epochs):
    # a linear model with the features as input
    model = torch.nn.Linear(num_features, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    train_losses = []
    test_losses = []
    for t in range(epochs):
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

        with torch.no_grad():
            y_pred_test = model(test_set).squeeze()
            loss = loss_fn(y_pred_test, test_labels)
            test_losses.append(loss.item())
    
    model.eval()
    return model, train_losses, test_losses


def non_contrastive_learning(path_org, path_cf):
    #load the datasets
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1
    
    # put the datasets into the approriate format
    train_set, train_labels, test_set, test_labels = train_test_split(org_features, cf_features, num_features, train_ratio=0.6)

    model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features, hyperparameters.epochs_non_contrastive)
    
    print('final train_loss', train_losses[-1])
    print('final train mean error', torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    # test the model
    evaluate_mimic(model, test_set, test_labels)
    get_weights(model)
    print_example_predictions(model, test_set, test_labels)
    show_loss_plot(train_losses, test_losses)

def learning_repeats(path_org, path_cf, contrastive=True):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1

    test_losses = []
    train_lossess = []
    all_train_losses = []
    test_mean_errors = []
    train_mean_errors = []
    pearson_correlations = []
    spearman_correlations = []
    weights = []
    all_test_losses = []

    for repeat in range(hyperparameters.number_of_seeds):
        if contrastive:
            train_set, train_labels, test_set, test_labels = train_test_split_contrastive(org_features, cf_features, num_features, train_ratio=0.6)
        else:
            train_set, train_labels, test_set, test_labels = train_test_split(org_features, cf_features, num_features, train_ratio=0.6)

        model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features, hyperparameters.epochs_contrastive)

        test_loss, test_mean_error, pearson_correlation, spearman_correlation = evaluate_mimic(model, test_set, test_labels)

        test_losses.append(test_loss)
        test_mean_errors.append(test_mean_error)
        pearson_correlations.append(pearson_correlation)
        spearman_correlations.append(spearman_correlation)
        weights.append(get_weights(model, print=False))
        train_lossess.append(train_losses[-1])
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        train_mean_errors.append(torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    print('final train_loss', np.mean(train_losses))
    print('final train mean error', np.mean(train_mean_errors))
    print('final test_loss', np.mean(test_losses))
    print('final test mean error', np.mean(test_mean_errors))
    print('final pearson correlation', np.mean(pearson_correlations))
    print('final spearman correlation', np.mean(spearman_correlations))
    weights = np.mean(weights, axis=0)
    print('final weights', [v + ": " + str(weights[k]) for k,v in enumerate(config.features)])
    train_lossess = np.mean(train_lossess, axis=0)
    test_losses = np.mean(test_losses, axis=0)
    all_train_losses = np.mean(all_train_losses, axis=0)
    all_test_losses = np.mean(all_test_losses, axis=0)
    show_loss_plot(all_train_losses, all_test_losses)


def contrastive_learning(path_org, path_cf):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1

    # put the datasets into the approriate format
    train_set, train_labels, test_set, test_labels = train_test_split_contrastive(org_features, cf_features, num_features, train_ratio=0.6)

    model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features, hyperparameters.epochs_contrastive)
    
    print('final train_loss', train_losses[-1])
    print('final train mean error', torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    # test the model
    evaluate_mimic(model, test_set, test_labels)
    get_weights(model)
    print_example_predictions(model, test_set, test_labels)
    show_loss_plot(train_losses, test_losses)

    


if __name__ == '__main__':
    # read console arguments
    # arg[1]: Learn in a contrastive way or not
    # c or contrastive: use contrastive learning between original and counterfactual
    # nc or non-contrastive: learn in a non-contrastive way
    # arg[2]: Test the counterfactual trajectory generator or a baseline
    # b or baseline: test the baseline model
    # cf or counterfactual: test counterfactuals

    # test if one of the args is a number and extract the number
    

    
    base_path = 'evaluation\datasets\\100_ablations\pv100'
    path_org_cte = base_path + '\org_features.pkl'
    path_cf_cte = base_path + '\cf_features.pkl'
    path_org_baseline = base_path + '\org_features_baseline.pkl'
    path_cf_baseline = base_path + '\cf_features_baseline.pkl'

    # contrastive learning
    print('# BASELINE Contrastive Learning')
    learning_repeats(path_org_baseline, path_cf_baseline, contrastive=True)
    print('-------------------------')
    print('# COUNTERFACTUAL Contrastive Learning')
    learning_repeats(path_org_cte, path_cf_cte, contrastive=True)
    print('-------------------------')
    print('# BASELINE Non-Contrastive Learning')
    learning_repeats(path_org_baseline, path_cf_baseline, contrastive=False)
    print('-------------------------')
    print('# COUNTERFACTUAL Non-Contrastive Learning')
    learning_repeats(path_org_cte, path_cf_cte, contrastive=False)
