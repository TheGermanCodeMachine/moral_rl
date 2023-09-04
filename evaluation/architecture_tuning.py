from train_mimic_sidebyside import *
import random
import pickle
import numpy
import torch

# Load data from all 3 types

path_org_mcts = 'datasets\\1000mcts\\1000\org_features.pkl'
path_cf_mcts = 'datasets\\1000mcts\\1000\cf_features.pkl'
org_features_mcts = pickle.load(open(path_org_mcts, 'rb'))
cf_features_mcts = pickle.load(open(path_cf_mcts, 'rb'))
num_features = len(org_features_mcts[0])-1
train_set_mcts, train_labels_mcts, test_set, test_labels = train_test_split_contrastive_sidebyside(org_features_mcts, cf_features_mcts, num_features, n_train=267)

path_org_step = 'datasets\\1000step\\1000\org_features.pkl'
path_cf_step = 'datasets\\1000step\\1000\cf_features.pkl'
org_features_step = pickle.load(open(path_org_step, 'rb'))
cf_features_step = pickle.load(open(path_cf_step, 'rb'))
train_set_step, train_labels_step, test_set, test_labels = train_test_split_contrastive_sidebyside(org_features_step, cf_features_step, num_features, n_train=267)

path_org_random = 'datasets\\1000random\\1000\org_features.pkl'
path_cf_random = 'datasets\\1000random\\1000\cf_features.pkl'
org_features_random = pickle.load(open(path_org_random, 'rb'))
cf_features_random = pickle.load(open(path_cf_random, 'rb'))
train_set_random, train_labels_random, test_set, test_labels = train_test_split_contrastive_sidebyside(org_features_random, cf_features_random, num_features, n_train=266)

# append training data together
train_set = torch.cat((train_set_mcts, train_set_step, train_set_random))
train_labels = torch.cat((train_labels_mcts, train_labels_step, train_labels_random))
# shuffle
train_set_labels = list(zip(train_set, train_labels))
random.shuffle(train_set_labels)
train_set, train_labels = zip(*train_set_labels)
train_set = torch.stack(train_set)
train_labels = torch.stack(train_labels)

# do hyperparameter search
# epochs, learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization_architecture(train_set, train_labels)
epochs, learning_rate, regularisation = hyper_param_optimization(train_set, train_labels)

# print(num_layers, hidden_sizes)
print(epochs, learning_rate, regularisation)

# output is: num_layers = 4, hidden_sizes = [16, 8]