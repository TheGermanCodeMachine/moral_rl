from helpers.util_functions import extract_player_position
from copy import deepcopy
import torch
'''
In randomized_v2 the objects contained are:
- Matrix 0: borders of the environment
- Matrix 1: the player
- Matrix 2: citizens
- Matrix 3: houses
- Matrix 4: fire extinguisher

actions:
- 0: go upwards
- 1: go downwards
- 2: go left
- 3: go right
- 4: do nothing
- 5: grab upwards
- 6: grab downwards
- 7: grab left
- 8: grab right
- 9: quit game
'''

# General Parameters
MAX_STEPS = 75

def legal_field(next_state, action, x, y):
    # collides with border
    if next_state[0][x][y] == 1:
        return False
    # collides with house
    elif next_state[3][x][y] == 1:
        return False
    return True

# returns the new state and whether the game is over
def step_v2(state, action, num_step):
    next_state = deepcopy(state).squeeze(dim=0)
    if num_step >= MAX_STEPS:
        return next_state, True, num_step
    else:
        num_step += 1
        p_x, p_y = extract_player_position(next_state)
        # go upward?
        if action == 0:
            # check if it's a legal move
            x, y = p_x, p_y - 1
            if legal_field(next_state, action, x, y):
                next_state[1][p_x][p_y] = 0
                next_state[1][x][y] = 1
        # go downward?
        elif action == 1:
            x, y = p_x, p_y + 1
            if legal_field(next_state, action, x, y):
                next_state[1][p_x][p_y] = 0
                next_state[1][x][y] = 1
        # go leftward?
        elif action == 2:
            x, y = p_x - 1, p_y
            if legal_field(next_state, action, x, y):
                next_state[1][p_x][p_y] = 0
                next_state[1][x][y] = 1
        # go rightward?
        elif action == 3:
            x, y = p_x + 1, p_y
            if legal_field(next_state, action, x, y):
                next_state[1][p_x][p_y] = 0
                next_state[1][x][y] = 1
        # do nothing?
        elif action == 4:
            pass
        # grab upwards?
        elif action == 5:
            x, y = p_x, p_y - 1
            next_state[2][x][y] = 0
        # grab downwards?
        elif action == 6:
            x, y = p_x, p_y + 1
            next_state[2][x][y] = 0
        # grab leftward?
        elif action == 7:
            x, y = p_x - 1, p_y
            next_state[2][x][y] = 0
        # grab rightward?
        elif action == 8:
            x, y = p_x + 1, p_y
            next_state[2][x][y] = 0
        
        next_state = next_state.unsqueeze(dim=0)
        # quit game?
        if action == 9 or num_step >= MAX_STEPS:
            return next_state, True, num_step
        return next_state, False, num_step


            
