import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract_player_position(state):
    if state.shape[0]==1:
        pos_tensor = np.argwhere(state[0][1] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[1] == 1).squeeze(0)
    return pos_tensor[0][0].item(), pos_tensor[1][0].item()

def extract_house_positions(state):
    if state.shape[0]==1:
        pos_tensor = np.argwhere(state[0][3] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[3] == 1).squeeze(0)
    return pos_tensor[:,0].tolist(), pos_tensor[:,1].tolist()

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

def normalise_features(values):
    values = np.array(values)
    # check if all values are 0
    if np.all(values == 0):
        return values.tolist()
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values.tolist()
    normalised_values = ((values - mean) / std).tolist()
    return  normalised_values

def normalise_value(value, normalisation, criterion):
    return (value - normalisation[criterion][0]) / (normalisation[criterion][1] - normalisation[criterion][0])

def normalise_values(values, normalisation, criterion):
    normalised_values = []
    for val in values:
        normalised_values.append(normalise_value(val, normalisation, criterion))
    return normalised_values

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


def retrace_original(start, divergence, counterfactual_traj, org_traj, vec_env_counter, states_tensor, discriminator):
    # retrace the steps of original trajectory until the point of divergence
    for i in range(start, divergence):
        counterfactual_traj['states'].append(states_tensor)
        counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
        counterfactual_traj['actions'].append(org_traj['actions'][i])

        next_states, reward, done, info = vec_env_counter.step(org_traj['actions'][i])
        if done[0]:
            next_states = vec_env_counter.reset()
            break

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
    return counterfactual_traj, states_tensor