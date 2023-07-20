import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
import reward_features as erf
import random
import numpy as np
from copy import deepcopy
from helpers.util_functions import normalise_values, partial_trajectory
from helpers.folder_util_functions import iterate_through_folder, write, read
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def extract_features(trajectories):
    all_features, citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens = [], [], [], [], [], [], [], [], []
    for traj in trajectories:
        citizens_saveds.append(erf.citizens_saved(traj))
        # citizens_missed.append(erf.citizens_missed(cf_traj))
        unsaved_citizenss.append(erf.unsaved_citizens(traj))
        distance_to_citizens.append(erf.distance_to_citizen(traj))
        standing_on_extinguishers.append(erf.standing_on_extinguisher(traj))
        lengths.append(erf.length(traj))
        could_have_saveds.append(erf.could_have_saved(traj))
        final_number_of_unsaved_citizenss.append(erf.final_number_of_unsaved_citizens(traj))
        moved_towards_closest_citizens.append(erf.moved_towards_closest_citizen(traj))

    average_features = [np.mean(citizens_saveds), np.mean(unsaved_citizenss), np.mean(distance_to_citizens), np.mean(standing_on_extinguishers), np.mean(lengths), np.mean(could_have_saveds), np.mean(final_number_of_unsaved_citizenss), np.mean(moved_towards_closest_citizens)]

    # normalise values
    citizens_saveds = normalise_values(citizens_saveds)
    unsaved_citizenss = normalise_values(unsaved_citizenss)
    distance_to_citizens = normalise_values(distance_to_citizens)
    standing_on_extinguishers = normalise_values(standing_on_extinguishers)
    lengths = normalise_values(lengths)
    could_have_saveds = normalise_values(could_have_saveds)
    final_number_of_unsaved_citizenss = normalise_values(final_number_of_unsaved_citizenss)
    moved_towards_closest_citizens = normalise_values(moved_towards_closest_citizens)

    all_features = [list(a) for a in zip(citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens)]
    return all_features, average_features

def part_trajectories_to_features(base_path, path_org, path_cf):
    # Load trajectories.
    print('part_orgs')
    org_trajs = read(path_org)
    org_trajectories = [d[0] for d in org_trajs]
    org_rewards = [d[1]/len(d[0]['rewards']) for d in org_trajs]

    org_features, average_org_features = extract_features(org_trajectories)
    average_org_features.append(np.mean(org_rewards))
    # explained_variance = calulate_PCA(org_features)
    org_features = [f + [r] for f,r in zip(org_features, org_rewards)]

    # Save features.
    write(org_features, base_path + '\org_features.pkl')
    write(average_org_features, base_path + '\statistics' + '\org_feature_stats.pkl')

    print('part_cfs')
    cf_trajs = read(path_cf)
    cf_trajectories = [d[0] for d in cf_trajs]
    cf_rewards = [d[1]/len(d[0]['rewards']) for d in cf_trajs]

    # Extract features.
    cf_features, average_cf_features = extract_features(cf_trajectories)
    average_cf_features.append(np.mean(cf_rewards))
    # explained_variance = calulate_PCA(cf_features)
    cf_features = [f + [r] for f,r in zip(cf_features, cf_rewards)]
    
    write(cf_features, base_path + '\cf_features.pkl')
    write(average_cf_features, base_path + '\statistics' + '\cf_feature_stats.csv')



def full_trajectories_to_features(path_full, base_path):
    # Load trajectories.
    full_trajs = read(path_full)
    
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
        part_traj = partial_trajectory(trajectories[i], starts[i], starts[i]+lengths[i])
        part_trajectories1.append(part_traj)
        rewards1.append(sum(part_traj['rewards']))
    all_features1 = extract_features(part_trajectories1)
    part_features1 = [f + [r] for f,r in zip(all_features1, rewards1)]

    write(part_features1, base_path + '\\org_features_baseline.pkl')

    random.shuffle(trajectories_copy)

    part_trajectories2 = []
    rewards2 = []
    for i in range(len(trajectories_copy)):
        part_traj = partial_trajectory(trajectories_copy[i], starts_copy[i], starts_copy[i]+lengths_copy[i])
        part_trajectories2.append(part_traj)
        rewards2.append(sum(part_traj['rewards']))
    all_features2 = extract_features(part_trajectories2)
    part_features2 = [f + [r] for f,r in zip(all_features2, rewards2)]

    write(part_features2, base_path + '\\cf_features_baseline.pkl')

def calulate_PCA(org_features):
    explained_variance = []
    for i in range(1, len(org_features[0])+1):
        pca = PCA(n_components=i)
        pca.fit(org_features)
        explained_variance.append(sum(pca.explained_variance_ratio_))
    plt.plot(explained_variance)
    plt.show()
    return explained_variance

if __name__ == '__main__':
    folder_path = 'datasets\\1000\\1000'

    all_folder_base_paths = iterate_through_folder(folder_path)
    all_folder_base_paths.reverse()

    for base_path in all_folder_base_paths:

        path_org = base_path + '\org_trajectories.pkl'
        path_cf = base_path + '\cf_trajectories.pkl'
        path_full = base_path + '\\full_trajectories.pkl'

        part_trajectories_to_features(base_path, path_org, path_cf)
        # full_trajectories_to_features(path_full, base_path)