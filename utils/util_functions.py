import numpy as np
import os

def extract_player_position(state):
    if len(state[0]==1):
        pos_tensor = np.argwhere(state[0][1] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[1] == 1).squeeze(0)
    return pos_tensor[0][0].item(), pos_tensor[1][0].item()

def distance_subtrajectories(traj1, traj2):
    # calucalte the distance between every state-action pair in traj1 and traj2
    dist_table = np.zeros((len(traj1['states']), len(traj2['states'])))
    for i in range(len(traj1['states'])):
        for j in range(len(traj2['states'])):
            dist_table[i,j] = state_action_diff(traj1['states'][i], traj1['actions'][i], traj2['states'][j], traj2['actions'][j])

    # calculate the modified Hausdorff distance based on the distance table (see: Dubuisson, M. P., & Jain, A. K. (1994, October). A modified Hausdorff distance for object matching. In Proceedings of 12th international conference on pattern recognition (Vol. 1, pp. 566-568). IEEE.)
    dist_A_B = np.mean(np.min(dist_table, axis=1))
    dist_B_A = np.mean(np.min(dist_table, axis=0))
    return max(dist_A_B, dist_B_A)

# this method calculates how different two state-action pairs in the randomized_v2 environment are
# WARNING: this method works sensibly only for the randomized_v2 environment; other enivronments will need a different implementation
# TODO: this could be more environment agnostic by only treating the player position as a special feature and jsut doing some distance measure on the rest of the state features
def state_action_diff(s1, a1, s2, a2):
    dist = 0
    # add 1 if the action is different
    if a1 != a2: dist += 1
    # add the edit distance between the two matrices of unsaved citizens
    dist += np.sum(np.abs(s1[0][2] - s2[0][2]).detach().numpy())

    # manhattan distance between the player positions
    pos_1 = extract_player_position(s1)
    pos_2 = extract_player_position(s2)
    dist += abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])
    return dist

def extract_citizens_positions(state):
    if len(state[0]==1):
        pos_tensor = np.argwhere(state[0][2] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[2] == 1).squeeze(0)
    return pos_tensor[0], pos_tensor[1]

def normalise_values(values):
    values = np.array(values)
    # check if all values are 0
    if np.all(values == 0):
        return values.tolist()
    mean = np.mean(values)
    std = np.std(values)
    normalised_values = ((values - mean) / std).tolist()
    return  normalised_values

def iterate_through_folder(folder_path):
    # are there subfolders?
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    # if there are subfolders, iterate through them
    if subfolders:
        all_folder_base_paths = []
        for subfolder in subfolders:
            # subfolder shouldn't be called results
            if not subfolder.endswith('results'):
                subsubfolders = iterate_through_folder(subfolder)
                all_folder_base_paths.extend(subsubfolders)
            else:
                # in this case we have reached the relevant folder and don't want to go deeper
                return [folder_path]
        return all_folder_base_paths
    else:
        return [folder_path]