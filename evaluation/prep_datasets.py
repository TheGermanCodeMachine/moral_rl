import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
import extract_reward_features as erf
import random
from copy import deepcopy

def extract_features(trajectories):
    all_features = []
    for traj in trajectories:
        features = []
        features.append(erf.citizens_saved(traj))
        # citizens_missed.append(erf.citizens_missed(cf_traj))
        features.append(erf.unsaved_citizens(traj))
        features.append(erf.distance_to_citizen(traj))
        features.append(erf.standing_on_extinguisher(traj))
        features.append(erf.length(traj))
        all_features.append(features)
    return all_features

def part_trajectories_to_features(path_org, path_cf, base_path):
    # Load trajectories.
    with open(path_org, 'rb') as f:
        org_trajs = pickle.load(f)
    
    org_trajectories = [d[0] for d in org_trajs]
    org_rewards = [d[1] for d in org_trajs]
    
    with open(path_cf, 'rb') as f:
        cf_trajs = pickle.load(f)

    cf_trajectories = [d[0] for d in cf_trajs]
    cf_rewards = [d[1] for d in cf_trajs]

    # Extract features.
    org_features = extract_features(org_trajectories)
    cf_features = extract_features(cf_trajectories)
    org_features = [f + [r] for f,r in zip(org_features, org_rewards)]
    cf_features = [f + [r] for f,r in zip(cf_features, cf_rewards)]

    # Save features.
    with open(base_path + '\org_features.pkl', 'wb') as f:
        pickle.dump(org_features, f)
    
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
    base_path = 'evaluation\datasets\pv1000'

    path_org = base_path + '\org_trajectories.pkl'
    path_cf = base_path + '\cf_trajectories.pkl'
    path_full = base_path + '\\full_trajectories.pkl'

    part_trajectories_to_features(path_org, path_cf, base_path)
    full_trajectories_to_features(path_full, base_path)