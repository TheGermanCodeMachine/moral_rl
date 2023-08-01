import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
from envs.gym_wrapper import *
from moral.preference_giver import *
import argparse
import sys
import tensorflow as tf
from copy import *
from helpers.visualize_trajectory import visualize_two_part_trajectories
from helpers.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, evaluate_qcs_for_cte, evaluate_qc
from quality_metrics.distance_measures import distance_all as distance_all
from quality_metrics.critical_state_measures import critical_state_all as critical_state
import pickle
from helpers.parsing import sort_args, parse_attributes
from collections import defaultdict
from helpers.util_functions import extract_player_position
from counterfactual_trajectories import config, retrace_original
from moral.ppo import PPO
from torch.distributions import Categorical
from helpers.util_functions import partial_trajectory

# 0-8 are the normal actions. 9 is an action which goes to the terminal state
set_of_actions = [0,1,2,3,4,5,6,7,8]
action_threshold = 0.05
likelihood_terminal = 0.2
discout_factor = 0.8
terminal_state = (-1,-1)
qc_criteria_to_use = ['proximity', 'sparsity', 'validity', 'realisticness']
normalisation = {'proximity':1, 'sparsity':1/40, 'validity':1/5, 'realisticness':20}

def run_mcts_from(vec_env_cf, org_traj, starting_position, ppo, discriminator, seed_env):
    chosen_action = -1
    done = False
    cf_trajectory = []
    root_node = None
    qfunction = QTable()

    i = 0
    while chosen_action!=9 and not done:
        mdp = MDP(vec_env_cf, org_traj, starting_position, ppo, discriminator, seed_env, cf_trajectory)
        root_node = SingleAgentMCTS(mdp, qfunction, UpperConfidenceBounds(), ppo).mcts(timeout=5, root_node=root_node)

        # choose the child node with the highest q value
        max_val = -np.inf
        string = ""
        for action in root_node.children:
            string += str(action) + ": " + str(qfunction.get_q_value(root_node.trajectory, action)) + "; "
            value = qfunction.get_q_value(root_node.trajectory, action)
            if value > max_val:
                max_val = value
                chosen_action = action

        print("Depth:", i, "; Chosen action:", chosen_action, string)
        cf_trajectory.append(chosen_action)
        for (child, _) in root_node.children[chosen_action]:
            if cf_trajectory == child.trajectory:
                root_node = child
        i +=1
        # action = qfunction.get_max_q(root_Node.trajectory, mdp.get_actions(root_Node.state))[0]
        # cf_trajectory.append(action)
        # (next_state, reward, done) = mdp.execute(action)

    cf_trajectory.remove(9)

    full_trajectory = {'states': [], 'actions': [], 'rewards': []}
    vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    full_trajectory, states_tensor = retrace_original(0, starting_position, full_trajectory, org_traj, vec_env, states_tensor, discriminator)
    cf_traj = {'states': [states_tensor], 'actions': [], 'rewards': [discriminator.g(states_tensor)[0][0].item()]}
    for action in cf_trajectory:
        cf_traj['actions'].append(action)
        state, reward, done, info = vec_env.step([action])
        states_tensor = torch.tensor(state).float().to(device)
        cf_traj['states'].append(states_tensor)
        cf_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
    org_traj = partial_trajectory(org_traj, starting_position, len(cf_traj['states'])-1+starting_position)
    org_traj['actions'] = org_traj['actions'][:-1]
    
    return org_traj, cf_traj, max_val


        

def get_viable_actions(state, ppo):
    # TODO: if it's the first state of the trajectory, it has to be a different action than the original action
    viable_actions = []
    actions, log_probs = ppo.act(state)
    for i in range(len(actions)):
        # only include actions above the threshold of ppo loglikelihood
        if log_probs[i] > action_threshold:
            # divied by 1-likelihood_terminal to account for the fact that the terminal state is not included in the ppo policy. Now likelihoods sum of to 1 again :)
            viable_actions.append(actions[i], log_probs[i]*(1-likelihood_terminal))
    viable_actions.append(9, likelihood_terminal)
    return viable_actions

# TODO: implement this
# It should sample many random trajectory and evaluate their qc_metrics to be able to normalise later values
def normalise_qc_values():
    pass

class MDP():
    def __init__(self, env, original_trajectory, starting_step, ppo, discriminator, seed_env, traj=[]) -> None:
        self.env = env
        self.seed_env = seed_env
        self.original_trajectory = original_trajectory
        self.starting_step = starting_step
        # this saves the state-action pairs of the trajectory with the states as full matrices as usually represented in the environment
        self.full_cf_traj = {'states': [], 'actions': [], 'rewards': []}
        # this saves the actions of a trajectory. With this we can uniquely identify a trajectory, but it is more efficient to use the full_cf_traj for the mcts
        self.start_cf_traj = deepcopy(traj) # this isn't changed throughout a simulation. We reset the environment back to this state of the trajectory after each simulation
        self.update_cf_traj = [] # this is updated throughout a simulation
        self.ppo = ppo
        self.discriminator = discriminator
        self.reset_env()

    def execute(self, action):
        next_state, done = self.apply_action_to_env(self.env, action)
        if done:
            reward = self.get_reward()
        else:
            reward = 0
        return next_state, reward, done

    def apply_action_to_env(self, env, action):
        if action == 9:
            self.update_cf_traj.append(action)
            return self.full_cf_traj['states'][-1] , True
        else:
            next_state, reward, done, info = env.step([action])
            state_tensor = torch.tensor(next_state).float().to(device)
            self.full_cf_traj['states'].append(state_tensor)
            self.full_cf_traj['actions'].append(action)
            self.full_cf_traj['rewards'].append(self.discriminator.g(state_tensor)[0][0].item())
            self.update_cf_traj.append(action)
            if done[0]:
                return state_tensor, True
            return state_tensor, False
        
    def get_reward(self):
        org_traj = partial_trajectory(self.original_trajectory, self.starting_step, len(self.update_cf_traj)-1+self.starting_step)
        org_traj = {'states': org_traj['states'][1:], 'actions': org_traj['actions'][:-1], 'rewards': org_traj['rewards'][1:]}
        cf_traj = {'states': self.full_cf_traj['states'][1:], 'actions': self.full_cf_traj['actions'], 'rewards': self.full_cf_traj['rewards'][1:]}
        return evaluate_qc(org_traj, cf_traj, qc_criteria_to_use, normalisation)
    

    def get_actions(self, state):
        valid_actions = []
        actions = self.ppo.action_probabilities(state)
        act_tresh = action_threshold
        # the first action of the coutnerfactual trajectory should not be the same as the first action of the original trajectory 
        if len(self.update_cf_traj) == 0:
            org_action = self.original_trajectory['actions'][self.starting_step]
            actions[org_action] = 0
        while len(valid_actions) < 3:
            valid_actions = []
            for action in set_of_actions:
                if actions[action] > act_tresh:
                    valid_actions.append(action)
            act_tresh = act_tresh * 0.1
        # the first action cannot be 9, since this would immediately end the counterfactual without making a difference
        if len(self.update_cf_traj) > 0:
            valid_actions.append(9)
        return valid_actions
    
    def get_initial_state(self):
        return self.full_cf_traj['states'][0]

    def is_terminal(self, traj):
        if len(traj) >= 1:
            return traj[-1]==9

    def reset_env(self):
        env = VecEnv(config.env_id, config.n_workers, seed=self.seed_env)
        states = env.reset()
        states_tensor = torch.tensor(states).float().to(device)
        tmp_traj = {'states': [], 'actions': [], 'rewards': []}
        tmp_traj, states_tensor = retrace_original(0, self.starting_step, tmp_traj, self.original_trajectory, env, states_tensor, self.discriminator)
        self.env = env
        self.full_cf_traj = {'states': [self.original_trajectory['states'][self.starting_step]], 'actions': [], 'rewards': [self.original_trajectory['rewards'][self.starting_step]]}
        self.update_cf_traj = []
        for action in self.start_cf_traj:
            next_state, done = self.apply_action_to_env(env, action)
    

class QTable():
    def __init__(self, default=0.0):
        self.trajectory_action_values = defaultdict(lambda: default)

    def get_q_value(self, trajectory, action):
        return self.trajectory_action_values[(tuple(trajectory), action)]
    
    def get_max_q(self, trajectory, actions):
        max_q_value = -np.inf
        max_action = None
        for action in actions:
            q_value = self.get_q_value(trajectory, action)
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return (max_action, max_q_value)

    def update(self, trajectory, action, delta):
        self.trajectory_action_values[(tuple(trajectory), action)] = self.trajectory_action_values[(tuple(trajectory), action)] + delta

class Node():
    
    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, trajectory, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.trajectory = trajectory
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return the value of this node """
    def get_value(self):
        (_, max_q_value) = self.qfunction.get_max_q(
            self.trajectory, self.mdp.get_actions(self.state)
        )
        return max_q_value
    
    """ Get the number of visits to this state """
    def get_visits(self):
        return Node.visits[tuple(self.trajectory)]
    
    """ Return true if and only if all child actions have been expanded """
    # TODO: exclude actions that are below a threshold according to the ppo policy
    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)  - self.children.keys()
        if len(valid_actions) == 0:
            return True
        else:
            return False
        
    """ Select a node that is not fully expanded """
    def select(self):
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.trajectory):
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.trajectory, actions, self.qfunction)
            return self.get_outcome_child(action).select()

    """ Expand a node if it is not a terminal node """
    # TODO: choose based on quality of state s' after action has been applied
    def expand(self):
        if not self.mdp.is_terminal(self.trajectory):
            # Randomly select an unexpanded action to expand
            actions = self.mdp.get_actions(self.state) - self.children.keys()
            if actions == {}:
                a=0
            action = random.choice(list(actions))

            self.children[action] = []
            return self.get_outcome_child(action)
        return self
    
    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[tuple(self.trajectory)] = Node.visits[tuple(self.trajectory)] + 1
        Node.visits[(tuple(self.trajectory), action)] = Node.visits[(tuple(self.trajectory), action)] + 1

        q_value = self.qfunction.get_q_value(self.trajectory, action)
        delta = (1 / (Node.visits[(tuple(self.trajectory), action)])) * (
            reward - q_value
        )
        self.qfunction.update(self.trajectory, action, delta)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)


    """ Simulate the outcome of an action, and return the child node """

    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        (next_state, reward, done) = self.mdp.execute(action)

        traj = self.trajectory + [action]
        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            if traj == child.trajectory:
                return child

        # This outcome has not occured from this state-action pair previously
        new_child = Node(
            self.mdp, self, next_state, self.trajectory + [action], self.qfunction, self.bandit, reward, action
        )

        # Find the probability of this outcome (only possible for model-based) for visualising tree
        self.children[action] += [(new_child, 1.0)]
        return new_child
            
    def create_root_node(self):
        return Node(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
        )
    
class MCTS:
    def __init__(self, mdp, qfunction, bandit, ppo):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit
        self.ppo = ppo

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """
    def mcts(self, timeout=1, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()
        else:
            root_node.mdp = self.mdp

        start_time = time.time()
        current_time = time.time()
        iteration = 0
        while current_time < start_time + timeout:
            iteration += 1
            # Find a state node to expand
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node.trajectory):

                child = selected_node.expand()
                reward = self.simulate(child)
                selected_node.back_propagate(reward, child)


            current_time = time.time()

        print("Number of iterations", iteration)
        return root_node
    
    """ Choose a random action. Heustics can be used here to improve simulations. """
    # TODO: add ppo policy as heuristic for choosing actions
    def choose(self, state):
        action_distribution = self.ppo.action_distribution_torch(state)
        # remove all actions below the threshold
        for i in range(action_distribution.shape[1]):
            if action_distribution[0][i] < action_threshold:
                action_distribution[0][i] = 0

        action_distribution = torch.cat((action_distribution, torch.tensor([likelihood_terminal]).view(1,1)), 1)
        m = Categorical(action_distribution)
        action = m.sample()
        return action.item()
    
    """ Simulate until a terminal state """
    # TODO: change how the reward is calculated to be based on quality criteria
    # TODO: change how the mdp is being simualted to be in the actual environment
    def simulate(self, node):
        trajectory = node.trajectory
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        done = False
        if node.action==9:
            cumulative_reward = self.mdp.get_reward()
        while not self.mdp.is_terminal(trajectory) and not done:
            # Choose an action to execute
            action = self.choose(state)

            # Execute the action
            (next_state, reward, done) = self.mdp.execute(action)

            # Discount the reward
            # cumulative_reward += pow(discout_factor, depth) * reward
            # Note: we don't discount the reward. Shorter trajectories are already incentivised by proximity and sparsity.
            cumulative_reward = reward
            depth += 1

            state = next_state
            trajectory = trajectory + [action]
        self.mdp.reset_env()
        return cumulative_reward

class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return Node(
            self.mdp, None, self.mdp.get_initial_state(), [], self.qfunction, self.bandit
        )


class UpperConfidenceBounds():
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, trajectory, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        # remove the item 9 from the actions if it exists. This avoids resampling the action 9. 9 does not have to be resampled, because there is not tree of actions below it
        if 9 in actions:
            actions.remove(9)
        for action in actions:
            value = qfunction.get_q_value(trajectory, action) + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)
        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1
        return result
    
    """ Reset a multi-armed bandit to its initial configuration """
    def reset(self):
        self.__init__()
