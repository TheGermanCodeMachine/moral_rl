import pickle as pkl
import numpy as np
import torch
import os

models_random = []

# Load the models

# for each file, load the model
models_random = []

with open('datasets\\100random\\100\\results_sidebyside\80\hyperparameters.pkl', 'rb') as f:
    hyperparameters = pkl.load(f)
    num_layers = hyperparameters['num_layers']
    hidden_sizes = hyperparameters['hidden_sizes']
    layer_dims = [16] + hidden_sizes + [1]

    folder_path = 'datasets\\100random\\100\\results_sidebyside\80\saved_models'
    for path in os.listdir(folder_path):
        if path.endswith('.pt'):
            with open(os.path.join(folder_path, path), 'rb') as f:
                model = torch.nn.Sequential()
                model.add_module('linear' + str(0), torch.nn.Linear(layer_dims[0], layer_dims[1]))
                for i in range(1, num_layers-1):
                    model.add_module('relu' + str(i), torch.nn.ReLU())
                    model.add_module('linear' + str(i), torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
                loaded_state_dict = torch.load(f)
                model.load_state_dict(loaded_state_dict)
                models_random.append(model)
                # model.add_module('linear' + str(0), torch.nn.Linear(loaded_state_dict['linear0.weight'], layer_dims[1]))

models_mcts = []
folder_path = 'datasets\\100mcts\\100\\results_sidebyside\80\saved_models'
for path in os.listdir(folder_path):
    if path.endswith('.pt'):
        with open(os.path.join(folder_path, path), 'rb') as f:
            models_mcts.append(torch.load(f))

models_step = []
folder_path = 'datasets\\100step\\100\\results_sidebyside\80\saved_models'
for path in os.listdir(folder_path):
    if path.endswith('.pt'):
        with open(os.path.join(folder_path, path), 'rb') as f:
            models_step.append(torch.load(f))