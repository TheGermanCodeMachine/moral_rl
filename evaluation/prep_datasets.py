import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
import extract_reward_features as erf
import random
import numpy as np
from copy import deepcopy
from utils.util_functions import normalise_values

def extract_features(trajectories):
    all_features = []
    citizens_saveds = []
    unsaved_citizenss = []
    distance_to_citizens = []
    standing_on_extinguishers = []
    lengths = []
    for traj in trajectories:
        features = []
        citizens_saveds.append(erf.citizens_saved(traj))
        # citizens_missed.append(erf.citizens_missed(cf_traj))
        unsaved_citizenss.append(erf.unsaved_citizens(traj))
        distance_to_citizens.append(erf.distance_to_citizen(traj))
        standing_on_extinguishers.append(erf.standing_on_extinguisher(traj))
        lengths.append(erf.length(traj))

    # normalise values
    citizens_saveds = normalise_values(citizens_saveds)
    unsaved_citizenss = normalise_values(unsaved_citizenss)
    distance_to_citizens = normalise_values(distance_to_citizens)
    standing_on_extinguishers = normalise_values(standing_on_extinguishers)
    lengths = normalise_values(lengths)

    all_features = [list(a) for a in zip(citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths)]
    return all_features

def part_trajectories_to_features(base_path, path_org, path_cf=None):
    # Load trajectories.
    with open(path_org, 'rb') as f:
        org_trajs = pickle.load(f)
    
    org_trajectories = [d[0] for d in org_trajs]
    org_rewards = [d[1] for d in org_trajs]

    org_features = extract_features(org_trajectories)
    org_features = [f + [r] for f,r in zip(org_features, org_rewards)]

    # Save features.
    with open(base_path + '\org_features.pkl', 'wb') as f:
        pickle.dump(org_features, f)
    
    if path_cf:
        with open(path_cf, 'rb') as f:
            cf_trajs = pickle.load(f)

        cf_trajectories = [d[0] for d in cf_trajs]
        cf_rewards = [d[1] for d in cf_trajs]

        # Extract features.
        cf_features = extract_features(cf_trajectories)
        cf_features = [f + [r] for f,r in zip(cf_features, cf_rewards)]
        
        with open(base_path + '\cf_features.pkl', 'wb') as f:
            pickle.dump(cf_features, f)



def full_trajectories_to_features(path_full, base_path):
    # Load trajectories.
    with open(path_full, 'rb') as f:
        full_trajs = pickle.load(f)
    
    trajectories = [d[0] for d in full_trajs]
    starts = [d[1] for d in full_trajs]
    lengths = [d[2] for d in full_trajs]

    # duplicate the trajectories (this is because we have n full trajectories, but n pairs of ctes)
    trajectories_copy = deepcopy(trajectories)
    starts_copy = deepcopy(starts)
    lengths_copy = deepcopy(lengths)

    # randomise starts and lengths togheter
    random.shuffle(trajectories)

    part_trajectories1 = []
    rewards1 = []
    for i in range(len(trajectories)):
        part_traj = {'states' : trajectories[i]['states'][starts[i]:starts[i]+lengths[i]],
                    'actions': trajectories[i]['actions'][starts[i]:starts[i]+lengths[i]],
                    'rewards': trajectories[i]['rewards'][starts[i]:starts[i]+lengths[i]]}
        part_trajectories1.append(part_traj)
        rewards1.append(sum(part_traj['rewards']))
    all_features1 = extract_features(part_trajectories1)
    part_features1 = [f + [r] for f,r in zip(all_features1, rewards1)]

    with open(base_path + '\\org_features_baseline.pkl', 'wb') as f:
        pickle.dump(part_features1, f)

    random.shuffle(trajectories_copy)

    part_trajectories2 = []
    rewards2 = []
    for i in range(len(trajectories_copy)):
        part_traj = {'states' : trajectories_copy[i]['states'][starts_copy[i]:starts_copy[i]+lengths_copy[i]],
                    'actions': trajectories_copy[i]['actions'][starts_copy[i]:starts_copy[i]+lengths_copy[i]],
                    'rewards': trajectories_copy[i]['rewards'][starts_copy[i]:starts_copy[i]+lengths_copy[i]]}
        part_trajectories2.append(part_traj)
        rewards2.append(sum(part_traj['rewards']))
    all_features2 = extract_features(part_trajectories2)
    part_features2 = [f + [r] for f,r in zip(all_features2, rewards2)]

    with open(base_path + '\\cf_features_baseline.pkl', 'wb') as f:
        pickle.dump(part_features2, f)

if __name__ == '__main__':
    # base_path = 'evaluation\datasets\\100_ablations\pvd100'

    # path_org = base_path + '\org_trajectories.pkl'
    # path_cf = base_path + '\cf_trajectories.pkl'
    # path_full = base_path + '\\full_trajectories.pkl'

    # part_trajectories_to_features(base_path, path_org, path_cf)
    # full_trajectories_to_features(path_full, base_path)

    # baseline
    base_path = 'evaluation\datasets\\100_ablations\\baseline'
    path_baseline_org = base_path + '\org_random_baselines.pkl'
    path_baseline_cf = base_path + '\\cf_random_baselines.pkl'

    part_trajectories_to_features(base_path, path_baseline_org, path_baseline_cf)