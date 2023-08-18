from tqdm import tqdm
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from moral.ppo import *
import torch
from envs.gym_wrapper import *
import numpy as np


def evaluate_ppo(ppo, config, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    scal_logs = []
    scalarised_returns = []

    for t in range(n_eval):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        obj_logs.append(reward)
        scal_logs.append(reward[0]*1 + reward[1]*10)
        

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            scal_logs = sum(scal_logs)
            scalarised_returns.append(scal_logs)
            scal_logs = []
            obj_logs = []

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)
    scalarised_returns = np.array(scalarised_returns)
    scalarised_mean = scalarised_returns.mean()
    scalarised_std = scalarised_returns.std()


    return list(obj_means), list(obj_std), scalarised_mean, scalarised_std