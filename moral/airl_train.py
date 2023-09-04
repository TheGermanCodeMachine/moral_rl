import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.gym_wrapper import *
from moral.airl import *
import torch
import numpy as np
import pickle
import wandb

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
        env_id = 'randomized_v2'
        env_steps = 3e6
        batchsize_discriminator = 512
        batchsize_ppo = 12
        n_workers = 12
        entropy_reg = 0
        gamma = 0.999
        epsilon = 0.1
        ppo_epochs = 5
        lambd = [1, 10]
    
if __name__ == '__main__':
    # Load demonstrations
    expert_trajectories = pickle.load(open('demonstrations/ppo_demos_v2_75_[1,10]_1000.pk', 'rb'))
    print(len(expert_trajectories))



    # Init WandB & Parameters
    wandb.init(project='AIRL', config={
        'env_id': 'randomized_v2',
        'env_steps': 3e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'entropy_reg': 0,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'lambd': [1,10]
    })
    

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    ppo.load_state_dict(torch.load('saved_models\\tmp\ppo_airl_v2_[1, 10]tmp_240000.pt', map_location=torch.device('cpu')))
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('saved_models\\tmp\discriminator_v2_[1, 10]tmp_240000.pt', map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(ppo.parameters(), lr=5e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # Logging
    objective_logs = []
    true_returns = []


    for t in tqdm(range(240000, (int(config.env_steps/config.n_workers)))):
        # at every 10% of the training process save a copy of the model
        # if t % (int(config.env_steps/config.n_workers)/10) == 0:
        #     torch.save(discriminator.state_dict(), 'saved_models/discriminator_v2_[0,1].pt')
        #     torch.save(ppo.state_dict(), 'saved_models/ppo_airl_v2_[0,1].pt')
        
        # Act
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        scalarised_reward = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        # Log Objectives
        objective_logs.append(rewards)
        true_returns.append(scalarised_reward)

        # Calculate (vectorized) AIRL reward
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(airl_state, airl_next_state, config.gamma, airl_action_prob)
        airl_rewards = list(airl_rewards.detach().cpu().numpy() * [0 if i else 1 for i in done])

        # Save Trajectory
        train_ready = dataset.write_tuple(states, actions, airl_rewards, done, log_probs)

        if train_ready:
            # Log Objectives
            objective_logs = np.array(objective_logs)
            objective_logs = objective_logs.sum(axis=0)
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            objective_logs = []
            true_returns = np.array(true_returns)
            true_returns = true_returns.sum(axis=0)
            wandb.log({'True Return': true_returns.mean()})
            true_returns = []

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=config.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories.copy(), ppo=ppo,
                                                              batch_size=config.batchsize_discriminator)

            # Log Loss Statsitics
            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc})
            airl_returns = dataset.log_returns()
            wandb.log({'avg_airl_reward': np.mean(airl_returns)})
            for ret in airl_returns:
                wandb.log({'AIRL Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
        if t % 5000 == 0:
            torch.save(discriminator.state_dict(), 'saved_models/tmp/discriminator_v2_' + str(config.lambd) + 'tmp_' + str(t) + '.pt')
            torch.save(ppo.state_dict(), 'saved_models/tmp/ppo_airl_v2_' + str(config.lambd) + 'tmp_' + str(t) + '.pt')

    vec_env.close()
    torch.save(discriminator.state_dict(), 'saved_models/discriminator_v2_[1,10]_new.pt')
    torch.save(ppo.state_dict(), 'saved_models/ppo_airl_v2_[1,10]_new.pt')
