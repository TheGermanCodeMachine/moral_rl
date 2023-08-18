from tqdm import tqdm
from ppo import *
import torch
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from envs.gym_wrapper import *
import wandb
import argparse


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Fetch ratio args
    #parser = argparse.ArgumentParser(description='Preference Lambda.')
    #parser.add_argument('--lambd', nargs='+', type=float)
    #args = parser.parse_args()

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'randomized_v2',
        'env_steps': 6e6,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'lr_ppo': 1e-4,
        'entropy_reg': 0.1,
        'lambd': [1, 10],
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

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
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    ppo.load_state_dict(torch.load('saved_models\\tmp\ppo_v2_[1, 10]tmp_20000.pt', map_location=torch.device('cpu')))
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        # how many dimensions are there in rewards? If it's more than 1, just reduce it to the one about saving people
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)

        if train_ready:
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        if t % 10000 == 0:
            torch.save(ppo.state_dict(), 'saved_models/tmp/ppo_v2_' + str(config.lambd) + 'tmp_' + str(t) + '.pt')

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # vec_env.close()
    torch.save(ppo.state_dict(), 'saved_models/new_ppo_v2_' + str(config.lambd) + '.pt')