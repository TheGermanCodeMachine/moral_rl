import pickle
from colorama import init, Back, Fore, Style
import numpy as np
import time

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

EMERGENCY_TYPE_TO_STRING = {0: 'H}',
                            1: 'C}',
                            2: 'G}',
                            3: 'P}'}

TYPE_TO_STRING = {0: {"letter": "#", "color": Back.RED},  # Walls.
                        1: {"letter": "P", "color": Back.BLUE},     # Player.
                        2: {"letter": "C", "color": Back.GREEN},     # Citizen.
                        3: {"letter": " ", "color": Back.WHITE}, # Floor.
                        4: {"letter": "F", "color": Back.YELLOW}}  # Fireextinguisher.

def load_trajectory(trajectory_path):
    with open(trajectory_path, 'r') as f:
        trajectories = pickle.load(open(trajectory_path, 'rb'))
    return trajectories[0], trajectories[1]

def get_field(state, i, j):
    # get at which layer the state is not zero, if there is no layer with value 1, then the field is a floor
    field = np.where(state[:, i, j] == 1)
    # check if field[0][0] is empty
    if field[0].size == 0:
        field = 3
    else:
        field = np.where(state[:, i, j] == 1)[0][0]

    if field not in {0, 1, 2, 3, 4}:
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

def paint_two_states(org_state, cf_state):
    # get the size of the 3rd and 4th dimension of the tensor state
    x_length, y_length = org_state.shape[1:]
    for i in range(y_length):
        fields_org = []
        fields_cf = []
        for j in range(x_length):
            # get what types are in the fields
            fields_org.append(get_field(org_state, i, j))
            fields_cf.append(get_field(cf_state, i, j))

        formatted_fields = [f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_org]
        formatted_fields.append(' ')
        formatted_fields.extend([f"{TYPE_TO_STRING[field]['color']}{TYPE_TO_STRING[field]['letter']}{Style.RESET_ALL}" for field in fields_cf])
        print("".join(formatted_fields))

def iterate_trajectory(org_traj):
    for i, state in enumerate(org_traj['states']):
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        # print('Step: {} Reward: {}'.format(i, org_traj['rewards'][i]))
        paint_state(state)
        # delay the next step for 0.5 seconds
        time.sleep(0.5)

def iterate_two_trajectories(org_traj, cf_traj):
    for i, (org_state, cf_state) in enumerate(zip(org_traj['states'], cf_traj['states'])):
        print('\033c')
        print('Step: {}'.format(i))
        # print the steps and rewards in one line
        # print('Step: {} Reward orginial: {} Reward counterfactual'.format(i, org_traj['rewards'][i], cf_traj['rewards'][i]))
        print('Original Trajectory Counterfactual Trajectory')
        paint_two_states(org_state, cf_state)
        # delay the next step for 0.5 seconds
        time.sleep(0.5)

# main function
if __name__ == '__main__':
    org_traj, cf_traj = load_trajectory('demonstrations\ppo_demos_v2_75_[0,1].pk')

    init()
    iterate_two_trajectories(org_traj, cf_traj)

    # iterate through the original trajectory
    # iterate_trajectory(org_traj)

    #clear the screen
    # print('\033c')

    # iterate through the counterfactual trajectory
    # iterate_trajectory(cf_traj)
    