import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from utils.util_functions import extract_player_position, extract_citizens_positions
import torch

def citizens_saved(traj):
    num_citizens_start = torch.sum(traj['states'][0][0][2], dim=(0,1))
    num_citizens_end = torch.sum(traj['states'][-1][0][2], dim=(0,1))
    if num_citizens_start - num_citizens_end < 0:
        a=0
    return num_citizens_start - num_citizens_end


# def citizens_saved2(traj):
#     saved = 0
#     for state in traj['states']:
#         player_row, player_col = extract_player_position(state)
#         if traj['actions'] == 5 and traj['states'][0][0][2][player_row-1, player_col]: # grab upward?
#             saved += 1
#         if traj['actions'] == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
#             saved += 1
#         if traj['actions'] == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
#             saved += 1
#         if traj['actions'] == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
#             saved += 1
#     return saved

# def citizens_missed(traj):

def length(traj):
    return len(traj['states'])

def unsaved_citizens(traj):
    uc = 0
    for state in traj['states']:
        uc += unsaved_citizens_state(state)
    return uc

def unsaved_citizens_state(state):
    return torch.sum(state[0][2], dim=(0,1))

def distance_to_citizen(traj):
    dtc = 0
    for state in traj['states']:
        dtc += distance_to_citizen_state(state)
    return dtc

def distance_to_citizen_state(state):
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

def standing_on_extinguisher(traj):
    soe = 0
    for state in traj['states']:
        soe += standing_on_extinguisher_state(state)
    return soe

def standing_on_extinguisher_state(state):
    x,y  = extract_player_position(state)
    return (x,y) == (7,7)