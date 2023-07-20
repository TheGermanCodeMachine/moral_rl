import numpy as np
import torch

def extract_player_position(state):
    if len(state[0]==1):
        pos_tensor = np.argwhere(state[0][1] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[1] == 1).squeeze(0)
    return pos_tensor[0][0].item(), pos_tensor[1][0].item()

def extract_citizens_positions(state):
    if len(state[0]==1):
        pos_tensor = np.argwhere(state[0][2] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[2] == 1).squeeze(0)
    return pos_tensor[0], pos_tensor[1]

def count_citizens(state):
    if len(state[0]==1):
        return torch.sum(state[0][2], dim=(0,1))
    else:
        return np.sum(state[2])

def normalise_values(values):
    values = np.array(values)
    # check if all values are 0
    if np.all(values == 0):
        return values.tolist()
    mean = np.mean(values)
    std = np.std(values)
    normalised_values = ((values - mean) / std).tolist()
    return  normalised_values

# normalise values to [0,1]
def normalise_values_01(values):
    if np.all([v==0 for v in values]):
        return values
    minv = values - np.min(values)
    diff = np.max(values) - np.min(values)
    norm = minv / diff
    return (values - np.min(values)) / (np.max(values) - np.min(values))

# returns a partial trajectory from start to end (inclusive)
def partial_trajectory(full_traj, start, end):
    return {'states' : full_traj['states'][start:end+1],
            'actions': full_traj['actions'][start:end+1],
            'rewards': full_traj['rewards'][start:end+1]}