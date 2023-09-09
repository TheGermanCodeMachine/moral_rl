import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
from colorama import init, Back, Fore, Style
import numpy as np
import time
from helpers.util_functions import *
import keyboard

TYPE_TO_STRING = {0: {"letter": "#", "color": Back.BLACK},  # Walls.
                        1: {"letter": "P", "color": Back.BLUE},     # Player.
                        2: {"letter": "C", "color": Back.GREEN},     # Citizen.
                        3: {"letter": " ", "color": Back.WHITE}, # Floor.
                        4: {"letter": "F", "color": Back.YELLOW},   # Fireextinguisher.
                        5: {"letter": " ", "color": Back.RED},  # Interacting with that field
                        6: {"letter": "H", "color": Back.CYAN}}  # House.

def action_to_gray_position(action, state):
    #check if action is a numpy array
    if isinstance(action, np.ndarray):
        action = action[0].item()
    else:
        action = action[0]
    if action in {5,6,7,8}:
        x,y = extract_player_position(state)
        if action == 5: #up
            return x-1, y
        elif action == 6: #down
            return x+1, y
        elif action == 7: #left
            return x, y-1
        elif action == 8: #right
            return x, y+1
    else:
        return 100, 100

# returns what type of field is at position i,j, matching the numbers in TYPE_TO_STRING
def get_field(state, i, j, interact_x, interact_y):
    # test if the field is being interacted with
    if i == interact_x and j == interact_y:
        return 5

    # get at which layer the state is not zero, if there is no layer with value 1, then the field is a floor
    field = np.where(state[0,:, i, j] == 1)
    # check if field[0][0] is empty
    if field[0].size == 0:
        field = 3
    else:
        field = np.where(state[0,:, i, j] == 1)[0][0]

    if field not in {0, 1, 2, 3, 4, 5}:
        field == 3
    return field

# paints a single state (currently unused)
def paint_state(state):
    # get the size of the 3rd and 4th dimension of the tensor state
    x_length, y_length = state.shape[1:]
    for i in range(y_length):
        fields = []
        for j in range(x_length):
            # get what type is in the field
            fields.append(get_field(state, i, j))

        formatted_fields = [f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields]
        print("".join(formatted_fields))

# paints two states next to each other
def paint_two_states(org_state, cf_state, org_action=[0], cf_action=[0]):
    # get the size of the 3rd and 4th dimension of the tensor state
    org_state = org_state
    cf_state = cf_state
    x_length, y_length = org_state.shape[2:]

    interact_x_org, interact_y_org = action_to_gray_position(org_action, org_state)
    interact_x_cf, interact_y_cf = action_to_gray_position(cf_action, cf_state)

    for i in range(y_length):
        fields_org = []
        fields_cf = []
        for j in range(x_length):
            # get what types are in the fields
            fields_org.append(get_field(org_state, i, j, interact_x_org, interact_y_org))
            fields_cf.append(get_field(cf_state, i, j, interact_x_cf, interact_y_cf))
            

        formatted_fields = [f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_org]
        formatted_fields.append('    ')
        formatted_fields.extend([f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_cf])
        print("".join(formatted_fields))

# visualizes a single trajectory (currently unused)
def visualize_trajectory(org_traj):
    for i, state in enumerate(org_traj['states']):
        print('\033c')
        print('Step: {}'.format(i))
        paint_state(state)
        print("Press the space bar for the next step...")
        keyboard.wait("space")

# visualizes two trajectories next to each other
def visualize_two_part_trajectories(org_traj, cf_traj, start_part, end_part_cf, end_part_org):
    # choose the longer trajectory to iterate over
    end_part = max(end_part_cf, end_part_org)

    for i in range(start_part, end_part+1):
        i_cf = min(i, end_part_cf)
        i_org = min(i, end_part_org)
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        print('Reward this step - orginial: {} -counterfactual: {}'.format(org_traj['rewards'][i_org], cf_traj['rewards'][i_cf]))
        print('Total reward - original {} - counterfactual {}'.format(sum(org_traj['rewards'][start_part:end_part_org+1])/(end_part_org+1-start_part), sum(cf_traj['rewards'][start_part:end_part_cf+1])/(end_part_cf+1-start_part)))
        print('Original Trajectory      Counterfactual Trajectory')
        if i == start_part:
            paint_two_states(org_traj['states'][i_org], cf_traj['states'][i_cf])
        else:
            paint_two_states(org_traj['states'][i_org], cf_traj['states'][i_cf], org_traj['actions'][i_org-1], cf_traj['actions'][i_cf-1])
        # wait for user to press any key
        print("Press the any key for the next step...")
        keyboard.read_event()
        time.sleep(0.2)

def visualize_two_part_trajectories_part(org_traj, cf_traj):
    end = max(len(org_traj['states']), len(cf_traj['states']))
    
    for i in range(0, end):
        i_cf = min(i, len(cf_traj['states']))
        i_org = min(i, len(org_traj['states']))
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        # print('Reward this step - orginial: {} -counterfactual: {}'.format(org_traj['rewards'][i_org], cf_traj['rewards'][i_cf]))
        print('Average reward - original {} - counterfactual {}'.format(sum(org_traj['rewards'])/len(org_traj['states']), sum(cf_traj['rewards'])/len(cf_traj['states'])))
        print('Original Trajectory      Counterfactual Trajectory')
        if i == 0:
            paint_two_states(org_traj['states'][i_org], cf_traj['states'][i_cf])
        else:
            paint_two_states(org_traj['states'][i_org], cf_traj['states'][i_cf], org_traj['actions'][i_org-1], np.array([cf_traj['actions'][i_cf-1]]))
        # wait for user to press any key
        print("Press the any key for the next step...")
        keyboard.read_event()
        time.sleep(0.2)