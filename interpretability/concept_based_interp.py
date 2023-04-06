from tqdm import tqdm
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
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
from create_data_interp import generate_concept_examples
import tensorflow as tf
from linear_regression import *
import helper
from copy import deepcopy

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
    env_id= 'randomized_v3'
    env_steps= 8e6
    batchsize_ppo= 12
    n_queries= 50
    preference_noise= 0
    n_workers= 1
    lr_ppo= 3e-4
    entropy_reg= 0.25
    gamma= 0.999
    epsilon= 0.1
    ppo_epochs= 5
        


if __name__ == '__main__':

    # Fetch ratio args for automatic preferences

    parser = argparse.ArgumentParser(description='Number of negative and positive examples')
    parser.add_argument('-n', nargs=1, type=int)
    args = parser.parse_args()

    num_neg, num_pos = args.n[0], args.n[0]

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)

    pos_states, neg_states = generate_concept_examples(config, ppo, num_neg=num_neg, num_pos=num_pos, concept_name='vase_around')

    # Expert 0
    discriminator_0 = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator_0.load_state_dict(torch.load('saved_models/discriminator_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))
    ppo_0 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo_0.load_state_dict(torch.load('saved_models/ppo_airl_v3_[0,1,0,1].pt', map_location=torch.device('cpu')))
    utop_0 = discriminator_0.estimate_utopia(ppo_0, config)
    print(f'Reward Normalization 0: {utop_0}')
    discriminator_0.set_eval()


    all_cases = np.concatenate([pos_states, neg_states])
    all_labels = [1] * len(pos_states) + [0]* len(neg_states)
    all_labels = np.array(all_labels)
    shuffled_indices = np.arange(all_labels.shape[0])
    np.random.shuffle(shuffled_indices)
    all_cases = all_cases[shuffled_indices]
    all_labels = all_labels[shuffled_indices]

    POSITIONS_TO_CONSIDER = round((num_neg + num_pos)*0.6)

    activations_layer1 = []
    activations_layer2 = []
    activations_layer3 = []
    activations_layer4 = []
    activations_layer5 = []


    # calculate outputs
    for state, next_state in all_cases:

        # Environment interaction
        state_tensor = torch.tensor(state).float().to(device)

        # helper.print_board(state[0])

        # Fetch AIRL rewards
        airl_state = torch.tensor(state).to(device).float()
        airl_next_state = torch.tensor(next_state).to(device).float()
        airl_rewards_0 = discriminator_0.forward(airl_state, airl_next_state, config.gamma).squeeze(1)

        # print(airl_rewards_0)

        # record activations
        #TODO this should include the value_state functions as well
        act1, act2, act3, act4 = discriminator_0.activations(airl_state, airl_next_state, config.gamma)
        
        activations_layer1.append(act1.squeeze().detach().numpy())
        activations_layer2.append(act2.squeeze().detach().numpy())
        activations_layer3.append(act3.squeeze().detach().numpy())
        activations_layer4.append(act4.squeeze().detach().numpy())

    activations_layer1 = tf.convert_to_tensor(activations_layer1)
    train_acc, val_acc = perform_regression(activations_layer1[:POSITIONS_TO_CONSIDER], all_labels[:POSITIONS_TO_CONSIDER], activations_layer1[POSITIONS_TO_CONSIDER:], all_labels[POSITIONS_TO_CONSIDER:], is_binary=True)
    print("layer 1: train_acc - ", train_acc, "; val_acc - ", val_acc)

    activations_layer2 = tf.convert_to_tensor(activations_layer2)
    train_acc, val_acc = perform_regression(activations_layer2[:POSITIONS_TO_CONSIDER], all_labels[:POSITIONS_TO_CONSIDER], activations_layer2[POSITIONS_TO_CONSIDER:], all_labels[POSITIONS_TO_CONSIDER:], True)
    print("layer 2: train_acc - ", train_acc, "; val_acc - ", val_acc)

    activations_layer3 = tf.convert_to_tensor(activations_layer3)
    train_acc, val_acc = perform_regression(activations_layer3[:POSITIONS_TO_CONSIDER], all_labels[:POSITIONS_TO_CONSIDER], activations_layer3[POSITIONS_TO_CONSIDER:], all_labels[POSITIONS_TO_CONSIDER:], True)
    print("layer 3: train_acc - ", train_acc, "; val_acc - ", val_acc)

    random_baseline = tf.convert_to_tensor(np.random.uniform(size=activations_layer1.shape))
    train_acc, val_acc = perform_regression(random_baseline[:POSITIONS_TO_CONSIDER], all_labels[:POSITIONS_TO_CONSIDER], random_baseline[POSITIONS_TO_CONSIDER:], all_labels[POSITIONS_TO_CONSIDER:], is_binary=True)
    print("random baseline: train_acc - ", train_acc, "; val_acc - ", val_acc)

    fake_labels = deepcopy(all_labels)
    np.random.shuffle(shuffled_indices)
    fake_labels = fake_labels[shuffled_indices]

    activations_layer2 = tf.convert_to_tensor(activations_layer2)
    train_acc, val_acc = perform_regression(activations_layer2[:POSITIONS_TO_CONSIDER], fake_labels[:POSITIONS_TO_CONSIDER], activations_layer2[POSITIONS_TO_CONSIDER:], fake_labels[POSITIONS_TO_CONSIDER:], True)
    print("layer 2 (shuffeled): train_acc - ", train_acc, "; val_acc - ", val_acc)

    activations_layer4 = tf.convert_to_tensor(activations_layer4)
    train_acc, val_acc = perform_regression(activations_layer4[:POSITIONS_TO_CONSIDER], all_labels[:POSITIONS_TO_CONSIDER], activations_layer4[POSITIONS_TO_CONSIDER:], all_labels[POSITIONS_TO_CONSIDER:], True)
    print("layer 4: train_acc - ", train_acc, "; val_acc - ", val_acc)