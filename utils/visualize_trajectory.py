import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import pickle
from colorama import init, Back, Fore, Style
import numpy as np
import time
from utils.util_functions import *
import keyboard

WAREHOUSE_ART = \
    ['C#######',
     '#      #',
     '#      #',
     '#  P   #',
     '#      #',
     '#      #',
     '#     G#',
     '########']

BACKGROUND_ART = \
    ['########',
     '#      #',
     '#      #',
     '#      #',
     '#      #',
     '#      #',
     '#      #',
     '########']


WAREHOUSE_FG_COLOURS = {' ': (870, 838, 678),  # Floor.
                        '#': (428, 135, 0),    # Walls.
                        'C': (0, 600, 67),     # Citizen.
                        'x': (850, 603, 270),  # Unused.
                        'P': (388, 400, 999),  # The player.
                        'F': (300, 300, 300),  # Waste.
                        'G': (900, 300, 900),      # Street.
                        'H': (428, 135, 0)}    # House.

TYPE_TO_STRING = {0: {"letter": "#", "color": Back.BLACK},  # Walls.
                        1: {"letter": "P", "color": Back.BLUE},     # Player.
                        2: {"letter": "C", "color": Back.GREEN},     # Citizen.
                        3: {"letter": " ", "color": Back.WHITE}, # Floor.
                        4: {"letter": "F", "color": Back.YELLOW},   # Fireextinguisher.
                        5: {"letter": " ", "color": Back.RED}}  # Grabbing to that field

def action_to_gray_position(action, state):
    if action[0] in {5,6,7,8}:
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


def load_trajectory(trajectory_path):
    with open(trajectory_path, 'r') as f:
        trajectories = pickle.load(open(trajectory_path, 'rb'))
    return trajectories[0], trajectories[1]

def get_field(state, i, j, grab_x, grab_y):
    if i == grab_x and j == grab_y:
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

def paint_two_states(org_state, cf_state, org_action, cf_action):
    # get the size of the 3rd and 4th dimension of the tensor state
    org_state = org_state
    cf_state = cf_state
    x_length, y_length = org_state.shape[2:]

    grab_x_org, grab_y_org = action_to_gray_position(org_action, org_state)
    grab_x_cf, grab_y_cf = action_to_gray_position(cf_action, cf_state)

    for i in range(y_length):
        fields_org = []
        fields_cf = []
        for j in range(x_length):
            # get what types are in the fields
            fields_org.append(get_field(org_state, i, j, grab_x_org, grab_y_org))
            fields_cf.append(get_field(cf_state, i, j, grab_x_cf, grab_y_cf))
            

        formatted_fields = [f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_org]
        formatted_fields.append('    ')
        formatted_fields.extend([f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_cf])
        print("".join(formatted_fields))

def visualize_trajectory(org_traj):
    for i, state in enumerate(org_traj['states']):
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        # print('Step: {} Reward: {}'.format(i, org_traj['rewards'][i]))
        paint_state(state)
        # delay the next step for 0.5 seconds
        # time.sleep(0.5)
        print("Press the space bar for the next step...")
        keyboard.wait("space")

def visualize_two_trajectories(org_traj, cf_traj):
    for i, (org_state, cf_state) in enumerate(zip(org_traj['states'], cf_traj['states'])):
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        print('Reward this step - orginial: {} -counterfactual: {}'.format(org_traj['rewards'][i], cf_traj['rewards'][i]))
        print('Total reward - original {} - counterfactual {}'.format(sum(org_traj['rewards'])/len(org_traj['rewards']), sum(cf_traj['rewards'])/len(cf_traj['rewards'])))
        print('Original Trajectory Counterfactual Trajectory')
        if i>0:
            paint_two_states(org_traj['states'][i], cf_traj['states'][i], org_traj['actions'][i-1], cf_traj['actions'][i-1])
        else:
            paint_two_states(org_traj['states'][i], cf_traj['states'][i], [0], [0])
        # delay the next step for 0.5 seconds
        # time.sleep(0.5)
        # wait for user to press any key
        print("Press the any key for the next step...")
        keyboard.read_event()
        time.sleep(0.2)

# main function
if __name__ == '__main__':
    org_traj, cf_traj = load_trajectory('demonstrations\ppo_demos_v2_75_[0,1].pk')

    init()
    visualize_two_trajectories(org_traj, cf_traj)

    # iterate through the original trajectory
    # iterate_trajectory(org_traj)

    #clear the screen
    # print('\033c')

    # iterate through the counterfactual trajectory
    # iterate_trajectory(cf_traj)
    