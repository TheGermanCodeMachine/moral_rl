import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from utils.util_functions import extract_player_position, extract_citizens_positions
import torch
import numpy as np

# HELPER: How many citizens are there in the state?
def number_of_citizens(state):
    return torch.sum(state[0][2], dim=(0,1))

# MEASURE: How many citizens were saved in the trajectory?
def citizens_saved(traj):
    num_citizens_start = number_of_citizens(traj['states'][0])
    num_citizens_end = number_of_citizens(traj['states'][-1])
    return num_citizens_start - num_citizens_end

# MEASURE: How long is the partial trajectory?
def length(traj):
    return len(traj['states'])

# MEASURE: Sums up over all states how many unsaved citizens there were
def unsaved_citizens(traj):
    uc = 0
    for state in traj['states']:
        uc += number_of_citizens(state)
    return uc

# MEASURE: Sums up over all states how close the closest citizen is
def distance_to_citizen(traj):
    dtc = 0
    for state in traj['states']:
        dtc += distance_to_closest_citizen_state(state)
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
def standing_on_extinguisher(traj):
    soe = 0
    for state in traj['states']:
        soe += standing_on_extinguisher_state(state)
    return soe

# HELPER: Is the player standing on an extinguisher?
def standing_on_extinguisher_state(state):
    x,y  = extract_player_position(state)
    return (x,y) == (7,7)

# MEASURE: Sums up over the trajectory in how many states the player could have saved a citizen, but failed to do so
def could_have_saved(traj):
    chs = 0
    for i in range(len(traj['states'])-1):
        chs += could_have_saved_state(traj['states'][i])
    did_save = citizens_saved(traj)
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

# MEASURE: Summed up over actions, did the player move closer to a citizen?
def moved_towards_closest_citizen(traj):
    diffs = []
    mtc = distance_to_closest_citizen_state(traj['states'][0])
    for i in range(1, len(traj['states'])):
        mtc_prev = mtc
        mtc = distance_to_closest_citizen_state(traj['states'][i])
        if number_of_citizens(traj['states'][i]) < number_of_citizens(traj['states'][i-1]):
            diffs.append(torch.tensor(0))
        else:
            diffs.append(mtc_prev - mtc)
    return np.mean(diffs)