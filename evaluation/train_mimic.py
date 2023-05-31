import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from evaluate_mimic import evaluate_mimic, print_weights
import sys
import random

class hyperparameters:
    learning_rate = 1e-1
    epochs = 10

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

    # plot the training and test losses on a log scale
    #show labels
    plt.semilogy(test_losses, label='test')
    plt.semilogy(train_losses, label='train')
    plt.legend()
    plt.show()

def train_model(train_set, train_labels, test_set, test_labels, num_features):
    # a linear model with the features as input
    model = torch.nn.Linear(num_features, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    train_losses = []
    test_losses = []
    for t in range(hyperparameters.epochs):
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

        # with torch.no_grad():
        #     y_pred_test = model(test_set).squeeze()
        #     loss = loss_fn(y_pred_test, test_labels)
        #     test_losses.append(loss.item())
    
    model.eval()
    return model, train_losses, test_losses


def non_contrastive_learning(path_org, path_cf):
    #load the datasets
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1
    
    # put the datasets into the approriate format
    train_set, train_labels, test_set, test_labels = train_test_split(org_features, cf_features, num_features, train_ratio=0.6)
    train_set = torch.tensor(train_set, dtype=torch.float)
    test_set = torch.tensor(test_set, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features)
    
    print('final train_loss', train_losses[-1])
    print('final train mean error', torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    # test the model
    evaluate_mimic(model, test_set, test_labels)
    print_weights(model)
    show_loss_plot(train_losses, test_losses)


def contrastive_learning(path_org, path_cf):
    org_features = pickle.load(open(path_org, 'rb'))
    cf_features = pickle.load(open(path_cf, 'rb'))
    num_features = len(org_features[0])-1

    # put the datasets into the approriate format
    train_set, train_labels, test_set, test_labels = train_test_split_contrastive(org_features, cf_features, num_features, train_ratio=0.6)

    model, train_losses, test_losses = train_model(train_set, train_labels, test_set, test_labels, num_features)
    
    print('final train_loss', train_losses[-1])
    print('final train mean error', torch.mean(torch.abs(model(train_set).squeeze() - train_labels)).item())

    # test the model
    evaluate_mimic(model, test_set, test_labels)
    print_weights(model)
    show_loss_plot(train_losses, test_losses)


if __name__ == '__main__':
    # read console arguments
    # arg[1]: Learn in a contrastive way or not
    # c or contrastive: use contrastive learning between original and counterfactual
    # nc or non-contrastive: learn in a non-contrastive way
    # arg[2]: Test the counterfactual trajectory generator or a baseline
    # b or baseline: test the baseline model
    # cf or counterfactual: test counterfactuals

    args = sys.argv

    contrastive = True
    counterfactuals = True
    if 'nc' in args or 'non-contrastive' in args:
        contrastive = False
    
    base_path = 'evaluation\datasets\pv10'
    path_org_cte = base_path + '\org_features.pkl'
    path_cf_cte = base_path + '\cf_features.pkl'
    path_org_baseline = base_path + '\org_features_baseline.pkl'
    path_cf_baseline = base_path + '\cf_features_baseline.pkl'

    if contrastive:
        print('# BASELINE Contrastive Learning')
        contrastive_learning(path_org_baseline, path_cf_baseline)
        print('-------------------------')
        print('# COUNTERFACTUAL Contrastive Learning')
        contrastive_learning(path_org_cte, path_cf_cte)
    else:
        # print('# BASELINE Non-Contrastive Learning')
        # non_contrastive_learning(path_org_baseline, path_cf_baseline)
        print('-------------------------')
        print('# COUNTERFACTUAL Non-Contrastive Learning')
        non_contrastive_learning(path_org_cte, path_cf_cte)

    print('done')