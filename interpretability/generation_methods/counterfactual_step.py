import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from envs.gym_wrapper import *
import torch
from helpers.util_functions import *

def generate_counterfactual_step(org_traj, ppo, discriminator, seed_env, config):
    # Now we make counterfactual trajectories by changing one action at a time and see how the reward changes

    # we make copies of the original trajectory and each changes one action at a timestep step
    # for each copy, change one action; for each action, change it to the best action that is not the same as the original action
    counterfactual_trajs, counterfactual_rewards = [], []
    # the timestep where the counterfactual diverges from the original trajectory and rejoins again
    starts, end_orgs, end_cfs = [], [], []

    # create a new environment to make the counterfactual trajectory in; this has the same seed as the original so the board is the same
    vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env_cf.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # this loop is over all timesteps in the original and each loop creates one counterfactual with the action at that timestep changed
    for step in range(0, len(org_traj['actions'])-1):
        counterfactual_traj = {'states': [], 'actions': [], 'rewards': []}
        counterfactual_deviation = 0

        # follow the steps of the original trajectory until the point of divergence (called step here)
        counterfactual_traj, states_tensor = retrace_original(0, step, counterfactual_traj, org_traj, vec_env_cf, states_tensor, discriminator)

        # now we are at the point of divergence
        counterfactual_traj['states'].append(states_tensor)
        reward = discriminator.g(states_tensor)[0][0].item()
        counterfactual_traj['rewards'].append(reward)
        same_next_state = True
        while same_next_state:
            counterfact_action, log_probs = ppo.pick_another_action(states_tensor, org_traj['actions'][step])
            same_next_state = test_same_next_state(states_tensor, org_traj['states'][step+1], org_traj['actions'][step], counterfact_action)
        # take the best action that is not the same as the original action
        # remove the action which is the same as the orignal action
        counterfactual_traj['actions'].append(counterfact_action)

        next_states, reward, done, info = vec_env_cf.step(counterfact_action)
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

        # continue the counterfactual trajectory with the policy until the end of the trajectory or until it rejoins the original trajectory
        for t in range(step+1, config.max_steps-1):
            counterfactual_traj['states'].append(states_tensor)
            counterfactual_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
            actions, log_probs = ppo.act(states_tensor)
            counterfactual_traj['actions'].append(actions)

            # test if this next step will rejoin the original trajectory
            rejoin_step = test_rejoined_org_traj(org_traj, states_tensor, t, step)
            if rejoin_step and rejoin_step < len(org_traj['states']):
                end_part_cf = t
                # follow the steps of the original trajectory until the length of the original trajectory
                counterfactual_traj, states_tensor = retrace_original(rejoin_step, len(org_traj['states']), counterfactual_traj, org_traj, vec_env_cf, states_tensor, discriminator)
                break

            next_states, reward, done, info = vec_env_cf.step(actions)
            if done[0]:
                next_states = vec_env_cf.reset()
                break
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        if not rejoin_step:
            rejoin_step = len(org_traj['states']) - 1
            end_part_cf = len(counterfactual_traj['states']) - 1
            
        # if the rewards are the same, then the counterfactual is not informative and we don't include it
        # NOTE: By including 'test_same_next_state' this if-statement should always be true, because the next state should always be different
        if np.mean(counterfactual_traj['rewards'][step:end_part_cf+1]) - np.mean(org_traj['rewards'][step:rejoin_step+1]) != 0:
            counterfactual_trajs.append(counterfactual_traj)
            counterfactual_rewards.append(torch.mean(torch.tensor(counterfactual_traj['rewards'])))
            starts.append(step)
            end_orgs.append(rejoin_step)
            end_cfs.append(end_part_cf)

        vec_env_cf = VecEnv(config.env_id, config.n_workers, seed=seed_env)
        states = vec_env_cf.reset()
        states_tensor = torch.tensor(states).float().to(device)

    return counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs


# this tests that the states of the original and counterfactual are actually different after the deviation has been taken
def test_same_next_state(states_tensor, next_state, org_action, cf_action):
    # option 1 for difference: any of them takes a step (this guarantees different states since they can't take the same step)
    if org_action in [0,1,2,3] or cf_action in [0,1,2,3]:
        return False
    # option 2 for difference: a citizen has been grabbed in the original
    if count_citizens(states_tensor) != count_citizens(next_state):
        return False
    # option 3: a citizen will be grabbed in the counterfactual
    pp = extract_player_position(states_tensor)
    if cf_action == 5:
        if states_tensor[0][2][pp[0]][pp[1]+1] == 1:
            return False
    elif cf_action == 6:
        if states_tensor[0][2][pp[0]][pp[1]-1] == 1:
            return False
    elif cf_action == 7:
        if states_tensor[0][2][pp[0]+1][pp[1]] == 1:
            return False
    elif cf_action == 8:
        if states_tensor[0][2][pp[0]-1][pp[1]] == 1:
            return False
    return True

# tests whether the current state is in the set of states that have been visited in the orignial trajectory after timestep step
def test_rejoined_org_traj(org_traj, state, step, start):
    if step > start+1:
        (x,y) = extract_player_position(state)
        # ensure that the point of rejoining is not too far away in the future. This would otherwise make for unnatural rejoins. I consider 5 steps to be a reasonable limit
        # TODO Experiment with different values for this limit
        s = max(start+1, step-1)
        e = min(len(org_traj['states']), step+2)
        for t in range(s, e):
            # test whether position is the same
            (x_org, y_org) = extract_player_position(org_traj['states'][t])
            if x == x_org and y == y_org:
                return t
    return False