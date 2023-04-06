from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
import matplotlib.pyplot as plt
from envs.gym_wrapper import *
from moral.preference_giver import *
from utils.evaluate_ppo import evaluate_ppo
import argparse
import sys
from copy import deepcopy
import random
from helper import *

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# number of steps between two coordinates (not including diagonal steps)
def num_steps(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def generate_concept_examples(config, ppo, concept_name = 'immediate_available', num_pos = 10, num_neg = 10):
    pos_states = []
    neg_states = []
    while len(pos_states)<num_pos or len(neg_states)< num_neg:
        vec_env = VecEnv(config.env_id, 1)
        states = vec_env.reset()
        for step in range(50):
            states_tensor = torch.tensor(states).float().to(device)
            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)
            

            pos_player = np.argwhere(states[0][1] == 1).squeeze(0)
            concept_present = test_concept(states, pos_player, concept_name)
            if concept_present and len(pos_states) < num_pos:
                pos_states.append((states, next_states))
            elif not concept_present and len(neg_states) < num_neg:
                neg_states.append((states, next_states))

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

    #cut off numbers

    return pos_states[:num_pos], neg_states[:num_neg]

def test_concept(states, pos_player, concept_name = 'immediate_available'):
    if concept_name == 'immediate_available':
        for i in range(pos_player[0]-1,pos_player[0]+2):
            for j in range(pos_player[1]-1,pos_player[1]+2):
                if i >=0 and i<16 and j >=0 and j<16:
                    if states[0][2][i][j] == 1 or states[0][3][i][j] == 1 or states[0][4][i][j] == 1:
                        return True

    elif concept_name == 'within_2_steps':
        for i in range(pos_player[0]-2,pos_player[0]+3):
            for j in range(pos_player[1]-2,pos_player[1]+3):
                if i >=0 and i<16 and j >=0 and j<16:
                    if states[0][2][i][j] == 1 or states[0][3][i][j] == 1 or states[0][4][i][j] == 1:
                        return True

    elif concept_name == 'vase_around':
        for i in range(pos_player[0]-1,pos_player[0]+2):
            for j in range(pos_player[1]-1,pos_player[1]+2):
                if i >=0 and i<16 and j >=0 and j<16:
                    if states[0][5][i][j] == 1:
                        return True

    elif concept_name == 'more_vases':
        num_vase = 0
        num_other = 0
        for i in range(pos_player[0]-3, pos_player[0]+4):
            for j in range(pos_player[1]-3,pos_player[1]+4):
                if i >=0 and i<16 and j >=0 and j<16:
                    if states[0][5][i][j] == 1:
                        num_vase += 1
                    elif states[0][2][i][j] == 1 or states[0][3][i][j] == 1 or states[0][4][i][j] == 1:
                        num_other += 1
        return num_vase >= num_other
    
    return False
    
        


def generate_counterfactuals(config, ppo, concept_name = 'immediate_available', num_examples = 10):
    return


def counterfactual_num_surrounding_types(config, ppo, test_type = 5, num_examples=10):
    if test_type not in range(2,6):
        raise ValueError('this type doesn\'t exist')
    
    zero_states = []
    one_states = []
    two_states = []
    three_states = []
    four_states = []

    while len(zero_states) < num_examples:
        vec_env = VecEnv(config.env_id, 1)
        states = vec_env.reset()
        for step in range(50):
            states_tensor = torch.tensor(states).float().to(device)
            
            pos_player = np.argwhere(states[0][1] == 1).squeeze(0)
            x,y = pos_player
            surrounding_coordinates = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]

            states_tensor0 = deepcopy(states_tensor)
            for (i,j) in surrounding_coordinates:
                states_tensor0[0][2][i][j] = 0
                states_tensor0[0][3][i][j] = 0
                states_tensor0[0][4][i][j] = 0
                states_tensor0[0][5][i][j] = 0
            actions0, log_probs = ppo.act(states_tensor0)
            next_states0, rewards, done, info = vec_env.step(actions0)
            zero_states.append((states_tensor0, next_states0))

            states_tensor1 = deepcopy(states_tensor)
            rand_coord = random.choice(surrounding_coordinates)
            states_tensor1[0][test_type][rand_coord[0]][rand_coord[1]] = 1
            actions1, log_probs = ppo.act(states_tensor1)
            next_states1, rewards, done, info = vec_env.step(actions1)
            one_states.append((states_tensor1, next_states1))

            states_tensor2 = deepcopy(states_tensor0)
            for (i,j) in random.sample(surrounding_coordinates, 2):
                states_tensor2[0][test_type][rand_coord[0]][rand_coord[1]] = 1
            actions2, log_probs = ppo.act(states_tensor2)
            next_states2, rewards, done, info = vec_env.step(actions2)
            two_states.append((states_tensor2, next_states2))

            states_tensor3 = deepcopy(states_tensor0)
            for (i,j) in random.sample(surrounding_coordinates, 3):
                states_tensor3[0][test_type][rand_coord[0]][rand_coord[1]] = 1
            actions3, log_probs = ppo.act(states_tensor3)
            next_states3, rewards, done, info = vec_env.step(actions3)
            three_states.append((states_tensor3, next_states3))

            states_tensor4 = deepcopy(states_tensor0)
            for (i,j) in surrounding_coordinates:
                states_tensor4[0][test_type][rand_coord[0]][rand_coord[1]] = 1
            actions4, log_probs = ppo.act(states_tensor4)
            next_states4, rewards, done, info = vec_env.step(actions4)
            four_states.append((states_tensor4, next_states4))
            
            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

    #cut off numbers

    return zero_states[:num_examples], one_states[:num_examples], two_states[:num_examples], three_states[:num_examples], four_states[:num_examples]


def counterfactuals_close_far_removal(config, ppo, test_type = 5, num_examples=10):
    # arrays to store the original board state, the state after a change closeby and after a change far away
    originals = []
    close_changes = []
    far_changes = []

    
    while len(originals) < num_examples:
        vec_env = VecEnv(config.env_id, 1)
        states = vec_env.reset()
        for step in range(50):
            # check if there is a vase adjacent to the player poisition
            states_tensor = torch.tensor(states).float().to(device)
            
            pos_player = np.argwhere(states[0][1] == 1).squeeze(0)
            x,y = pos_player
            surrounding_coordinates = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]

            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)

            if any([states[0][test_type][i[0]][i[1]] == 1 for i in surrounding_coordinates]):
                # if there is a vase adjacent to the player, we can remove it
                # we need to store the original state, the state after removing the vase closeby and the state after removing the vase far away

                originals.append((states_tensor, next_states))

                # remove a vase closeby
                states_tensor_close = deepcopy(states_tensor)
                # randomly choose one of the surrounding_coordinates that contains a vase
                rand_coord = random.choice([i for i in surrounding_coordinates if states[0][test_type][i[0]][i[1]] == 1])
                states_tensor_close[0][test_type][rand_coord[0]][rand_coord[1]] = 0
                close_changes.append((states_tensor_close, next_states))

                # remove a vase far away
                states_tensor_far = deepcopy(states_tensor)
                rand_coord = random.choice([(i,j) for i in range(16) for j in range(16) if num_steps((i,j), pos_player) > 3 and states[0][test_type][i][j] == 1])
                states_tensor_far[0][test_type][rand_coord[0]][rand_coord[1]] = 0
                far_changes.append((states_tensor_far, next_states))
            
            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)
            states = next_states.copy()

    return originals[:num_examples], close_changes[:num_examples], far_changes[:num_examples]


def counterfactual_all_possible_changes(config, state, next_state):
    original = (torch.tensor(state).to(device).float(), torch.tensor(next_state).to(device).float())
    changed = []
    pos_player = np.argwhere(state[0][1] == 1).squeeze(0)
    x,y = pos_player
    # all surrounding coordinates around the player within 2 steps
    surrounding_coordinates = [(i,j) for i in range(x-2,x+3) for j in range(y-2,y+3) if (i,j) != (x,y) and 0 <= i < 16 and 0 <= j < 16]
    
    for (i,j) in surrounding_coordinates:
        # if there is something in this position, we remove it from the state and next_state
        if any([state[0][k][i][j] == 1 for k in range(2,6)]):
            for k in range(2,6):
                if state[0][k][i][j] == 1:
                    changed_state = deepcopy(state)
                    changed_next_state = deepcopy(next_state)
                    changed_state[0][k][i][j] = 0
                    changed_next_state[0][k][i][j] = 0
                    changed.append((torch.tensor(changed_state).to(device).float(), torch.tensor(changed_next_state).to(device).float()))
        else:
            # if there is nothing in this position, we add all possible objects here
            for k in range(2,6):
                changed_state = deepcopy(state)
                changed_next_state = deepcopy(next_state)
                changed_state[0][k][i][j] = 1
                changed_next_state[0][k][i][j] = 1
                changed.append((torch.tensor(changed_state).to(device).float(), torch.tensor(changed_next_state).to(device).float()))

    return original, changed

def test_all_changes(config, ppo, discriminator):
    # generate new environment
    vec_env = VecEnv(config.env_id, 1)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    actions, log_probs = ppo.act(states_tensor)
    next_states, rewards, done, info = vec_env.step(actions)
    original, changed = counterfactual_all_possible_changes(config, states, next_states)
    pos_player = np.argwhere(states[0][1] == 1).squeeze(0)

    #show the original state and its reward
    reward_org = discriminator.forward(original[0], original[1], config.gamma).squeeze(1)
    largest_change_pos_r = -100
    largest_change_neg_r = +100
    largest_change_pos_s = changed[0]
    largest_change_neg_s = changed[0]
    for changes in changed:
        #show the changed state and its reward
        reward_change = discriminator.forward(changes[0], changes[1], config.gamma).squeeze(1)
        if reward_change < largest_change_neg_r:
            largest_change_neg_r = reward_change
            largest_change_neg_s = changes
        elif reward_change > largest_change_pos_r:
            largest_change_pos_r = reward_change
            largest_change_pos_s = changes
    # show_state_around_player_pos(pos_player, largest_change_s[0][0])

    #get player position in the next state
    next_pos_player = np.argwhere(next_states[0][1] == 1).squeeze(0)
    print_states(pos_player, next_pos_player, original[0][0], original[1][0], actions, largest_change_pos_s[0][0], largest_change_neg_s[0][0], reward_org, largest_change_pos_r, largest_change_neg_r)
        