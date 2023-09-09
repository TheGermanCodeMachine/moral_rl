import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
import evaluation.reward_features as erf
import random
import numpy as np
from copy import deepcopy
from helpers.util_functions import normalise_features, partial_trajectory
from helpers.folder_util_functions import iterate_through_folder, write, read
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

averaged = True

def extract_features_not_normalised(trajectories):
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
    all_features = [list(a) for a in zip(citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens)]
    return all_features

def extract_new_features(trajectories):
    all_features, citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens = [], [], [], [], [], [], [], [], []
    num_citizen_0s, num_citizen_1s, num_citizen_2s, num_citizen_3s, num_citizen_4s, num_citizen_5s, num_citizen_6s, num_citizen_7s = [], [], [], [], [], [], [], []
    avg_player_xs, avg_player_ys = [], []
    steps_outsides, step_betweens, step_insides = [], [], []
    dist_cit_0s, dist_cit_1s, dist_cit_2s, dist_cit_3s, dist_cit_4s, dist_cit_5s, dist_cit_6s, dist_cit_7s, dist_cit_8s, dist_cit_9s, dist_cit_10s = [], [], [], [], [], [], [], [], [], [], []
    dist_between_citizenss, avg_dist_extinguishers, dist_houses = [], [], []
    grab_actionss, walk_actionss, none_actionss = [], [], []
    left_of_closests, right_of_closests, down_of_closests, up_of_closests = [], [], [], []
    top_left_quadrants, top_right_quadrants, bottom_left_quadrants, bottom_right_quadrants = [], [], [], []

    for traj in trajectories:
        citizens_saveds.append(erf.citizens_saved(traj, averaged=averaged))
        # citizens_missed.append(erf.citizens_missed(cf_traj))
        unsaved_citizenss.append(erf.unsaved_citizens(traj, averaged=averaged))
        distance_to_citizens.append(erf.distance_to_citizen(traj, averaged=averaged))
        standing_on_extinguishers.append(erf.standing_on_extinguisher(traj, averaged=averaged))
        lengths.append(erf.length(traj))
        could_have_saveds.append(erf.could_have_saved(traj, averaged=averaged))
        final_number_of_unsaved_citizenss.append(erf.final_number_of_unsaved_citizens(traj))
        moved_towards_closest_citizens.append(erf.moved_towards_closest_citizen(traj, averaged=averaged))
        num_citizen_0, num_citizen_1, num_citizen_2, num_citizen_3, num_citizen_4, num_citizen_5, num_citizen_6, num_citizen_7 = erf.number_of_states_with_citizen_counts(traj, averaged=averaged)
        num_citizen_0s.append(num_citizen_0)
        num_citizen_1s.append(num_citizen_1)
        num_citizen_2s.append(num_citizen_2)
        num_citizen_3s.append(num_citizen_3)
        num_citizen_4s.append(num_citizen_4)
        num_citizen_5s.append(num_citizen_5)
        num_citizen_6s.append(num_citizen_6)
        num_citizen_7s.append(num_citizen_7)
        avg_player_x, avg_player_y = erf.average_player_position(traj)
        avg_player_xs.append(avg_player_x)
        avg_player_ys.append(avg_player_y)
        steps_outside, step_inside, step_between = erf.outside_inside_between(traj, averaged=averaged)
        steps_outsides.append(steps_outside)
        step_insides.append(step_inside)
        step_betweens.append(step_between)
        dist_cit_0, dist_cit_1, dist_cit_2, dist_cit_3, dist_cit_4, dist_cit_5, dist_cit_6, dist_cit_7, dist_cit_8, dist_cit_9, dist_cit_10 = erf.player_distances_from_closest_citizen(traj, averaged=averaged)
        dist_cit_0s.append(dist_cit_0)
        dist_cit_1s.append(dist_cit_1)
        dist_cit_2s.append(dist_cit_2)
        dist_cit_3s.append(dist_cit_3)
        dist_cit_4s.append(dist_cit_4)
        dist_cit_5s.append(dist_cit_5)
        dist_cit_6s.append(dist_cit_6)
        dist_cit_7s.append(dist_cit_7)
        dist_cit_8s.append(dist_cit_8)
        dist_cit_9s.append(dist_cit_9)
        dist_cit_10s.append(dist_cit_10)
        dist_between_citizenss.append(erf.average_distance_between_citizens(traj, averaged=averaged))
        avg_dist_extinguishers.append(erf.average_distance_to_extinguisher(traj, averaged=averaged))
        dist_houses.append(erf.distance_to_closest_house(traj, averaged=averaged))
        grab_actions, walk_actions, none_actions = erf.type_of_action(traj, averaged=averaged)
        grab_actionss.append(grab_actions)
        walk_actionss.append(walk_actions)
        none_actionss.append(none_actions)
        left_of_closest, right_of_closest, down_of_closest, up_of_closest = erf.relative_position_of_closest_citizen(traj, averaged=averaged)
        left_of_closests.append(left_of_closest)
        right_of_closests.append(right_of_closest)
        down_of_closests.append(down_of_closest)
        up_of_closests.append(up_of_closest)
        top_left_quadrant, top_right_quadrant, bottom_left_quadrant, bottom_right_quadrant = erf.quadrant_player_position(traj, averaged=averaged)
        top_left_quadrants.append(top_left_quadrant)
        top_right_quadrants.append(top_right_quadrant)
        bottom_left_quadrants.append(bottom_left_quadrant)
        bottom_right_quadrants.append(bottom_right_quadrant)

    # normalise values
    citizens_saveds = normalise_features(citizens_saveds)
    unsaved_citizenss = normalise_features(unsaved_citizenss)
    distance_to_citizens = normalise_features(distance_to_citizens)
    standing_on_extinguishers = normalise_features(standing_on_extinguishers)
    lengths = normalise_features(lengths)
    could_have_saveds = normalise_features(could_have_saveds)
    final_number_of_unsaved_citizenss = normalise_features(final_number_of_unsaved_citizenss)
    moved_towards_closest_citizens = normalise_features(moved_towards_closest_citizens)
    num_citizen_0s = normalise_features(num_citizen_0s)
    num_citizen_1s = normalise_features(num_citizen_1s)
    num_citizen_2s = normalise_features(num_citizen_2s)
    num_citizen_3s = normalise_features(num_citizen_3s)
    num_citizen_4s = normalise_features(num_citizen_4s)
    num_citizen_5s = normalise_features(num_citizen_5s)
    num_citizen_6s = normalise_features(num_citizen_6s)
    num_citizen_7s = normalise_features(num_citizen_7s)
    avg_player_xs = normalise_features(avg_player_xs)
    avg_player_ys = normalise_features(avg_player_ys)
    steps_outsides = normalise_features(steps_outsides)
    step_between = normalise_features(step_between)
    step_insides = normalise_features(step_insides)
    dist_cit_0s = normalise_features(dist_cit_0s)
    dist_cit_1s = normalise_features(dist_cit_1s)
    dist_cit_2s = normalise_features(dist_cit_2s)
    dist_cit_3s = normalise_features(dist_cit_3s)
    dist_cit_4s = normalise_features(dist_cit_4s)
    dist_cit_5s = normalise_features(dist_cit_5s)
    dist_cit_6s = normalise_features(dist_cit_6s)
    dist_cit_7s = normalise_features(dist_cit_7s)
    dist_cit_8s = normalise_features(dist_cit_8s)
    dist_cit_9s = normalise_features(dist_cit_9s)
    dist_cit_10s = normalise_features(dist_cit_10s)
    dist_between_citizenss = normalise_features(dist_between_citizenss)
    avg_dist_extinguishers = normalise_features(avg_dist_extinguishers)
    dist_houses = normalise_features(dist_houses)
    grab_actionss = normalise_features(grab_actionss)
    walk_actionss = normalise_features(walk_actionss)
    none_actionss = normalise_features(none_actionss)
    left_of_closests = normalise_features(left_of_closests)
    right_of_closests = normalise_features(right_of_closests)
    down_of_closests = normalise_features(down_of_closests)
    up_of_closests = normalise_features(up_of_closests)
    top_left_quadrants = normalise_features(top_left_quadrants)
    top_right_quadrants = normalise_features(top_right_quadrants)
    bottom_left_quadrants = normalise_features(bottom_left_quadrants)
    bottom_right_quadrants = normalise_features(bottom_right_quadrants)

    all_features = [list(a) for a in zip(citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens, num_citizen_0s, num_citizen_1s, num_citizen_2s, num_citizen_3s, num_citizen_4s, num_citizen_5s, num_citizen_6s, num_citizen_7s, avg_player_xs, avg_player_ys, steps_outsides, step_betweens, step_insides, dist_cit_0s, dist_cit_1s, dist_cit_2s, dist_cit_3s, dist_cit_4s, dist_cit_5s, dist_cit_6s, dist_cit_7s, dist_cit_8s, dist_cit_9s, dist_cit_10s, dist_between_citizenss, avg_dist_extinguishers, dist_houses, grab_actionss, walk_actionss, none_actionss, left_of_closests, right_of_closests, down_of_closests, up_of_closests, top_left_quadrants, top_right_quadrants, bottom_left_quadrants, bottom_right_quadrants)]
    return all_features

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
    citizens_saveds = normalise_features(citizens_saveds)
    unsaved_citizenss = normalise_features(unsaved_citizenss)
    distance_to_citizens = normalise_features(distance_to_citizens)
    standing_on_extinguishers = normalise_features(standing_on_extinguishers)
    lengths = normalise_features(lengths)
    could_have_saveds = normalise_features(could_have_saveds)
    final_number_of_unsaved_citizenss = normalise_features(final_number_of_unsaved_citizenss)
    moved_towards_closest_citizens = normalise_features(moved_towards_closest_citizens)

    all_features = [list(a) for a in zip(citizens_saveds, unsaved_citizenss, distance_to_citizens, standing_on_extinguishers, lengths, could_have_saveds, final_number_of_unsaved_citizenss, moved_towards_closest_citizens)]
    return all_features

def part_trajectories_to_features(base_path, path_org, path_cf):
    # Load trajectories.
    print('part_orgs')
    org_trajs = read(path_org)
    print(len(org_trajs))
    org_trajectories = [d[0] for d in org_trajs]
    org_rewards = [d[1]/len(d[0]['rewards']) for d in org_trajs]

    org_features = extract_new_features(org_trajectories)
    # explained_variance = calulate_PCA(org_features)
    org_features = [f + [r] for f,r in zip(org_features, org_rewards)]

    # Save features.
    write(org_features, base_path + '\org_features_new.pkl')

    print('part_cfs')
    cf_trajs = read(path_cf)
    cf_trajectories = [d[0] for d in cf_trajs]
    cf_rewards = [d[1]/len(d[0]['rewards']) for d in cf_trajs]

    # Extract features.
    cf_features = extract_new_features(cf_trajectories)
    # explained_variance = calulate_PCA(cf_features)
    cf_features = [f + [r] for f,r in zip(cf_features, cf_rewards)]
    
    write(cf_features, base_path + '\cf_features_new.pkl')



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
    folder_path = 'datasets\\1000_ablations\weights\\set5\\100'

    all_folder_base_paths = iterate_through_folder(folder_path)
    all_folder_base_paths.reverse()

    for base_path in all_folder_base_paths:
        print(base_path)
        path_org = base_path + '\org_trajectories.pkl'
        path_cf = base_path + '\cf_trajectories.pkl'
        path_full = base_path + '\\full_trajectories.pkl'

        part_trajectories_to_features(base_path, path_org, path_cf)
        # full_trajectories_to_features(path_full, base_path)