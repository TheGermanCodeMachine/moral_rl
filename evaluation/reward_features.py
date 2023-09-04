import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from helpers.util_functions import extract_player_position, extract_citizens_positions, extract_house_positions
import torch
import numpy as np

# HELPER: How many citizens are there in the state?
def number_of_citizens(state):
    return torch.sum(state[0][2], dim=(0,1))

# MEASURE: How many citizens were saved in the trajectory?
def citizens_saved(traj, averaged=True):
    num_citizens_start = number_of_citizens(traj['states'][0])
    num_citizens_end = number_of_citizens(traj['states'][-1])
    if averaged:
        return (num_citizens_start - num_citizens_end) / len(traj['states'])
    else:
        return num_citizens_start - num_citizens_end

# MEASURE: How long is the partial trajectory?
def length(traj):
    return len(traj['states'])

# MEASURE: Sums up over all states how many unsaved citizens there were
def unsaved_citizens(traj, averaged=True):
    uc = 0
    for state in traj['states']:
        uc += number_of_citizens(state)
    if averaged:
        return uc / len(traj['states'])
    else:
        return uc

# MEASURE: Sums up over all states how close the closest citizen is
def distance_to_citizen(traj, averaged=True):
    dtc = 0
    for state in traj['states']:
        dtc += distance_to_closest_citizen_state(state)
    if averaged:
        return dtc / len(traj['states'])
    else:
        return dtc

# HELPER: How close is the closest citizen?
def distance_to_closest_citizen_state(state):
    p_x, p_y = extract_player_position(state)
    c_xs, c_ys = extract_citizens_positions(state)
    if len(c_xs)==0: return 0
    distances = []
    for c_x, c_y in zip(c_xs, c_ys):
        d = abs(p_x - c_x) + abs(p_y - c_y)
        if d==0: d=1
        else: d=d-1
        distances.append(d)
    return min(distances)

# MEASURE: Sums up over the trajectory in how many states the player is standing on an extinguisher
def standing_on_extinguisher(traj, averaged=True):
    soe = 0
    for state in traj['states']:
        soe += standing_on_extinguisher_state(state)
    if averaged:
        return soe / len(traj['states'])
    else:
        return soe

# HELPER: Is the player standing on an extinguisher?
def standing_on_extinguisher_state(state):
    x,y  = extract_player_position(state)
    return (x,y) == (7,7)

# MEASURE: Sums up over the trajectory in how many states the player could have saved a citizen, but failed to do so
def could_have_saved(traj, averaged=True):
    chs = 0
    for i in range(len(traj['states'])-1):
        chs += could_have_saved_state(traj['states'][i])
    did_save = citizens_saved(traj)
    if averaged:
        return (chs - did_save)  / len(traj['states'])
    else:
        return chs - did_save

# HELPER: Could the player have saved a citizen in this state?
def could_have_saved_state(state):
    p_x, p_y = extract_player_position(state)
    c_xs, c_ys = extract_citizens_positions(state)
    if len(c_xs)==0: return 0
    for c_x, c_y in zip(c_xs, c_ys):
        if abs(p_x - c_x) + abs(p_y - c_y) == 1: return 1   
    return 0

# MEASURE: How many unsaved citizens are there at the end of the trajectory?
def final_number_of_unsaved_citizens(traj):
    return number_of_citizens(traj['states'][-1])

# MEASURE: Summed up over actions, did the player Move closer To a Citizen?
def moved_towards_closest_citizen(traj, averaged=True):
    diffs = []
    mtc = distance_to_closest_citizen_state(traj['states'][0])
    for i in range(1, len(traj['states'])):
        mtc_prev = mtc
        mtc = distance_to_closest_citizen_state(traj['states'][i])
        if number_of_citizens(traj['states'][i]) < number_of_citizens(traj['states'][i-1]):
            diffs.append(torch.tensor(0))
        else:
            diffs.append(mtc_prev - mtc)
    if averaged:
        return np.mean(diffs)
    else:
        return np.sum(diffs)

#### New ones:

# HELPER: In how many states of the trajectory are there still n citizens?
def number_of_states_with_n_citizens(traj, n):
    num = 0
    for state in traj['states']:#
        num_cit = number_of_citizens(state).detach().numpy().item()
        if number_of_citizens(state) == n:
            num += 1
    return num

# MEASURE: How many states are there with 7,6,5,4,3,2,1,0 citizens?
def number_of_states_with_citizen_counts(traj, averaged=True):
    num = []
    for i in range(8):
        num.append(number_of_states_with_n_citizens(traj, i))
    if averaged:
        return [n/len(traj['states']) for n in num]
    else:
        return num

# MEASURE: Average x,y position of the player
def average_player_position(traj):
    x = 0
    y = 0
    for state in traj['states']:
        p_x, p_y = extract_player_position(state)
        x += p_x
        y += p_y
    return x/len(traj['states']), y/len(traj['states'])


# MEASURE: How many steps the player is on the outside (the ring next to the wall), inside (the 4 squares in the middle) or between (the other states)
def outside_inside_between(traj, averaged=True):
    outside = 0
    inside = 0
    between = 0
    for state in traj['states']:
        p_x, p_y = extract_player_position(state)
        if p_x in [1,6] or p_y in [1,6]:
            outside += 1
        elif p_x in [3,4] and p_y in [3,4]:
            inside += 1
        else:
            between += 1
    if averaged:
        return outside/len(traj['states']), inside/len(traj['states']), between/len(traj['states'])
    else:
        return outside, inside, between

# MEASURE: Counts of player distances from closest citizen
def player_distances_from_closest_citizen(traj, averaged=True):
    num = [0,0,0,0,0,0,0,0,0,0,0]
    for state in traj['states']:
        d = distance_to_closest_citizen_state(state)
        # check if d is of type int
        if not isinstance(d, int):
            d = d.detach().numpy().item()
        num[d] += 1
    if averaged:
        return [n/len(traj['states']) for n in num]
    else:
        return num

# MEASURE: Average distance between citizens
def average_distance_between_citizens(traj, averaged=True):
    avg_distances = 0
    for state in traj['states']:
        c_xs, c_ys = extract_citizens_positions(state)
        if len(c_xs) > 1:
            distances = []
            for i in range(len(c_xs)):
                for j in range(i+1, len(c_xs)):
                    d = (abs(c_xs[i] - c_xs[j]) + abs(c_ys[i] - c_ys[j])).detach().numpy().item()
                    distances.append(d)
            avg_distances += np.mean(distances)
        else:
            avg_distances += 0
    if averaged:
        return avg_distances/len(traj['states'])
    else:
        return avg_distances


# MEASURE: Average distance to fire-extinguisher
def average_distance_to_extinguisher(traj, averaged=True):
    avg_distances = 0
    for state in traj['states']:
        p_x, p_y = extract_player_position(state)
        if (p_x, p_y) == (6,6):
            avg_distances += 0
        else:
            avg_distances += abs(p_x - 6) + abs(p_y - 6)
    if averaged:
        return avg_distances/len(traj['states'])
    else:
        return avg_distances

# MEASURE: How often a grab, walk or none action was taken
def type_of_action(traj, averaged=True):
    grabs = 0
    walks = 0
    none = 0
    for i in range(len(traj['states'])-1):
        if traj['actions'][i] in [0,1,2,3]:
            walks += 1
        elif traj['actions'][i] in [4]:
            none += 1
        elif traj['actions'][i] in [5,6,7,8]:
            grabs += 1
    if averaged:
        return grabs/(len(traj['states'])-1), walks/(len(traj['states'])-1), none/(len(traj['states'])-1)
    else:
        return grabs, walks, none

# MEASURE: Is the closest citizen to the right, left, higher or lower?
def relative_position_of_closest_citizen(traj, averaged=True):
    lefts = 0
    rights = 0
    ups = 0
    downs = 0
    for state in traj['states']:
        p_x, p_y = extract_player_position(state)
        c_xs, c_ys = extract_citizens_positions(state)
        closest_positions = ([], [])
        min_distance = 1000
        for i in range(len(c_xs)):
            d = abs(p_x - c_xs[i]) + abs(p_y - c_ys[i])
            if d == min_distance:
                closest_positions[0].append(c_xs[i])
                closest_positions[1].append(c_ys[i])
            elif d < min_distance:
                min_distance = d
                closest_positions = ([c_xs[i]], [c_ys[i]])
        left, right, up, down = 0, 0, 0, 0
        for i in range(len(closest_positions[0])):
            if closest_positions[0][i] < p_x:
                left += 1
            if closest_positions[0][i] > p_x:
                right += 1
            if closest_positions[1][i] < p_y:
                up += 1
            if closest_positions[1][i] > p_y:
                down += 1
        summ = sum([left, right, up, down])
        if summ != 0:
            lefts += left/summ
            rights += right/summ
            ups += up/summ
            downs += down/summ

    if averaged:
        return lefts/len(traj['states']), rights/len(traj['states']), ups/len(traj['states']), downs/len(traj['states'])
    else:
        return lefts, rights, ups, downs

# MEASURE: Which quadrant was the player in?
def quadrant_player_position(traj, averaged=True):
    top_left = 0
    top_right = 0
    bottom_left = 0 
    bottom_right = 0
    for state in traj['states']:
        p_x, p_y = extract_player_position(state)
        if p_x < 4 and p_y < 4:
            top_left += 1
        elif p_x < 4 and p_y > 3:
            bottom_left += 1
        elif p_x > 3 and p_y < 4:
            top_right += 1
        elif p_x > 3 and p_y > 3:
            bottom_right += 1
    if averaged:
        return top_left/len(traj['states']), top_right/len(traj['states']), bottom_left/len(traj['states']), bottom_right/len(traj['states'])
    else:
        return top_left, top_right, bottom_left, bottom_right

# HELPER: Distance to the closest house
def distance_to_closest_house_state(state):
    p_x, p_y = extract_player_position(state)
    c_xs, c_ys = extract_house_positions(state)
    if len(c_xs)==0: return 0
    distances = []
    for c_x, c_y in zip(c_xs, c_ys):
        d = abs(p_x - c_x) + abs(p_y - c_y)
        if d==0: d=1
        else: d=d-1
        distances.append(d)
    return min(distances)

# MEASURE: How close is the closest house on average?
def distance_to_closest_house(traj, averaged=True):
    dtc = 0
    for state in traj['states']:
        dtc += distance_to_closest_house_state(state)
    if averaged:
        return dtc / len(traj['states'])
    else:
        return dtc
