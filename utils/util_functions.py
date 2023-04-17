import numpy as np

def extract_player_position(state):
    if len(state[0]==1):
        pos_tensor = np.argwhere(state[0][1] == 1).squeeze(0)
    else:
        pos_tensor = np.argwhere(state[1] == 1).squeeze(0)
    return pos_tensor[0][0].item(), pos_tensor[1][0].item()