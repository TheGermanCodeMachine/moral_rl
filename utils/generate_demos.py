from tqdm import tqdm
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from moral.ppo import PPO
import torch
from envs.gym_wrapper import GymWrapper
import pickle

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Select Environment
env_id = 'randomized_v2'

# Select n demos (need to set timer separately)
n_demos = 50
max_steps = 75

# Initialize Environment
env = GymWrapper(env_id)
states = env.reset()
states_tensor = torch.tensor(states).float().to(device)
dataset = []
episode = {'states': [], 'actions': []}
episode_cnt = 0

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape
state_shape = obs_shape[:-1]
in_channels = obs_shape[-1]

# Load Pretrained PPO
ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
ppo.load_state_dict(torch.load('saved_models\ppo_v2_[0, 1].pt'))


for t in tqdm(range((max_steps-1)*n_demos)):
    actions, log_probs = ppo.act(states_tensor)
    next_states, reward, done, info = env.step(actions)
    episode['states'].append(states)
    # Note: Actions currently append as arrays and not integers!
    episode['actions'].append(actions)

    if done:
        next_states = env.reset()
        dataset.append(episode)
        episode = {'states': [], 'actions': []}

    # Prepare state input for next time step
    states = next_states.copy()
    states_tensor = torch.tensor(states).float().to(device)

pickle.dump(dataset, open('demonstrations\ppo_demos_v2_75_[0,1]' + '.pk', 'wb'))