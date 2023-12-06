import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque  # data structures that're useful
from itertools import count

import torch
import torch.nn as nn  # neural networks
import torch.optim as optim  # optimisation
import torch.nn.functional as F
import numpy as np
from wordle import WordleEnv100
from wordle import WordleEnv200
from wordle import WordleEnv300
from wordle import WordleEnv400
from wordle import WordleEnv500
from wordle import WordleEnv600
from wordle import WordleEnv1000

import pandas as pd
import glob
import json

from graph_plotting import plot_experiment

import time

# for mac environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# set up matplotlib
# hecks if the string 'inline' is present in the name of the current backend.
is_ipython = 'inline' in matplotlib.get_backend()
# checks if code is running in an IPython environment (like Jupyter Notebook)
if is_ipython:
    # for controlling the display of output, including the display of Matplotlib plots
    from IPython import display

plt.ion()  # ion stands for "interactive mode." (display plots dynamically)

# if GPU is to be used
# checks if a CUDA-compatible GPU is available. If so, it sets the device to "cuda"; otherwise, it uses the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################################

# This is just a data structure whose 1st elt is a state, 2nd elt is an action, 3rd elt is next_state and 4th elt is reward
# if tuple_1 = <10, 2, 11, +1> then to access the reward, can either say tuple_1[3] or tuple_1.reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# We’ll be using experience replay memory for training our DQN.
# It stores the transitions that the agent observes in a deque data structure, allowing us to reuse this data later.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

####################################################################################


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 104)
        self.layer2 = nn.Linear(104, n_actions)
        # self.layer3 = nn.Linear(208, n_actions)
        # nn.init.normal_(self.layer1.weight)
        # nn.init.normal_(self.layer2.weight)
        # nn.init.uniform_(self.layer1.weight, a=0.0, b=.1)
        # nn.init.uniform_(self.layer2.weight, a=0.0, b=.1)
        # nn.init.kaiming_uniform_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):  # x can be thought of as a vector of our state features. This is an input to our function
        x = F.relu(self.layer1(x))
        # We return a tensor from the final layer, theres 1 elt for each action. Each element in this tensor holds the expected state-action value for the associated action.
        return self.layer2(x)

####################################################################################


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer # Shaun: This is kinda like the learning rate, alpha
BATCH_SIZE = 1024
GAMMA = 0.85
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 5e-5  # halve learning rate


load_path = './dqn_wordle_data500_18m.pth'


# Setting up seeds
seed = 42  # https://stackoverflow.com/questions/75943057/i-cant-find-how-to-reproducibly-run-a-python-gymnasium-taxi-v3-environment
np.random.seed(seed)
# https://www.w3schools.com/python/ref_random_seed.asp#:~:text=The%20random%20number%20generator%20needs,of%20the%20random%20number%20generator.
random.seed(seed)

# https://pytorch.org/docs/stable/generated/torch.manual_seed.html
torch.manual_seed(seed)

# SETUP WARM START ENVIRONMENT
warm_start = False
restart = True
if warm_start:
    env_orig = WordleEnv400()  # CHANGE THIS TO THE ENVIRONMENT YOU WANT TO WARM START FROM
    original_n_actions = env_orig.action_space.n
    orig_state = env_orig.reset(seed=seed)
    original_n_observations = len(orig_state)
    original_model = DQN(original_n_observations, original_n_actions).to(device)
    original_model.load_state_dict(torch.load(load_path))


# SETUP ENVIRONMENT TRAINING ENVIRONMENT
env = WordleEnv500() # CHANGE THIS TO THE ENVIRONMENT YOU WANT TO RUN
env.action_space.seed(seed)    # https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete.sample
n_actions = env.action_space.n # reset() should be called with a seed right after initialization and then never again. https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
state = env.reset(seed=seed)
n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)
if warm_start:
    print("Warm Starting")
    assert original_n_observations == n_observations, "observation space should be the same"
    policy_net.layer1.weight.data = original_model.layer1.weight.data.clone()
    policy_net.layer1.bias.data = original_model.layer1.bias.data.clone()
    # for param in policy_net.layer1.parameters():
    #     param.requires_grad = False
    # Redefine the second layer for the new action space
    # policy_net.layer2 = torch.nn.Linear(104, n_actions)  # Adjust the output size to the new action space

    # policy_net.layer2.weight.data = original_model.layer2.weight.data.clone()
    # policy_net.layer2.bias.data = original_model.layer2.bias.data.clone()
elif restart:
    print("Restarting")
    policy_net.load_state_dict(torch.load(load_path))


# policy_net.eval() # Evaluate model
# eval = True
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

#orig optimizer
# initializing an AdamW optimizer for the policy_net neural network.
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# initializing an AdamW optimizer for the trainable parameters of policy_net
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=LR, amsgrad=True)


# https://pytorch.org/docs/stable/optim.html
# optim is constructing an optimizer object that will hold the current state (e.g. wordle 416 array) and will update the parameters (i.e. the w1, w2, etc within our neural network) based on the computed gradients (want w1, w2 that minimizes objective func)
# 1st arg is an iterable containing the parameters (all should be 'Variable's) to optimize.
# For 2nd and 3rd args, you specify optimizer specific options e.g. learning rate etc
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
# TLDR AdamW algo is a way of finding the optimal theta, which is equivalent to w1 w2, I think.


memory = ReplayMemory(10000)


steps_done = 0

# For an input state, select_action returns the best action for that state


def select_action(state):
    global steps_done
    sample = random.random()  # Choose random number from 0 to 1
    # Basically, when steps_done = 0, this value is EPS_START. As steps_done --> infinity, eps_threshold = EPS_END
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # (1 - epsilon) chance of choosing greedy action i.e. what our policy network says the best action is
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:  # (epsilon) chance of choosing random action sampled from our action space.
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
data = torch.tensor(episode_durations, dtype=torch.float)

# this function plots the durations of episodes as a function of every 100 episodes and also plots the average duration over the last 1000 episodes.
def plot_durations(show_result=False):
    meanResults = torch.tensor(episode_durations, dtype=torch.float)
    plt.figure(1)  # creates a new figure for plotting
    # converts the list episode_durations to a PyTorch tensor named durations_t
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()  # clear current figure https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clf.html
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        # computes the rolling average of durations over a window of 100 episodes and plots it.
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        meanResults = means
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            # Clear the output of the current cell receiving output https://ipython.org/ipython-doc/3/api/generated/IPython.display.html#functions
            display.clear_output(wait=True)
        else:
            # plt.gcf gets current figure (or creates a fig if one doesnt exist) https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.gcf.html
            display.display(plt.gcf())
    if len(durations_t) >= 100:
        return meanResults

####################################################################################

# code for training our model.
# optimize_model performs a single step of the optimization.


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # transitions is an array/list of transitions e.g. [Transition(s2, a2, s3, r3), Transition(s35, a35, s36, r36), ....]. And here, s2 is like the 417 elt tensor representing wordle state, a2 is like a single-valued tensor representing guess word, etc.  reward is also a 1 elt tensor
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # batch is a single transition tuple i.e. Transition (S, A, S', R). But here, S is a tuple of states i.e. S = (s2, s35, ....). And A is a tuple of actions i.e. A = (a2, a35, ...). Each individual item e.g. s2 and a2, are tensors (a 4-elt state tensor and 1-elt action tensor), as before
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    # In above code, we use the map function along with a lambda function to apply the condition "s is not None" to each element in batch.next_state. Here, batch.next_state is a tuple of states.
    # The result is a tuple of Boolean values indicating whether each corresponding element in next_state is not None.
    # Converting to a torch tensor, we have non_final_mask = tensor(true, true, false, true) which was obtained from batch.next_state = (non-final-state-1, non-final-state-2, None, non-final-state-3)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    # What does .cat do? Imagine batch_state is a tuple of states (i.e. a tuple of tensors)
    # batch_state = (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]))

    # Then, torch.cat concatenates tensors along dimension 0
    # result_tensor = torch.cat(batch_state) # THIS GIVES tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_batch is the concatenated tuple of states (order retained) and action_batch is the concatenated tuple of actions (order retained). With this, can get Q(s, a) from our policy neural network.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * GAMMA) + reward_batch  # Q(S) = r + gamma*V(S')

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # Comparing Q(s,a) outputted by our policy network for a batch of sampled (s, a) AGAINST expected Q(s,a) (we get this from the Reward and S' data)
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model (MAKING OUR POLICY NN BETTER ABLE TO PREDICT THE BEST ACTION)

    optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch I think this is just boilerplate
    loss.backward()  # function that sends some information "backwards", which we need to run before running optimizer.step()
    # In-place gradient clipping
    # uses PyTorch's gradient clipping utility to clip the gradients of the parameters of the policy_net to a specified maximum value, in this case, 100. Gradient clipping is a technique used to prevent exploding gradients during training, which can lead to numerical instability.
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # If the gradients become too large, they can cause the model parameters to be updated by a very large value, potentially destabilizing the training process.
    optimizer.step()  # This is the key part- i picture this as helping to update the parameters (w1, w2) etc in our policy neural network backwards on how the good results that it spat out were

####################################################################################


# HEART OF CODE, KINDA LIKE THE main() function
if torch.cuda.is_available():  # If you installed the CUDA version of pytorch which makes use of GPU
    # Set the seed if cuda available https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed_all.html
    torch.cuda.manual_seed_all(seed)
    num_episodes = 200000
    print("Using GPU")
else:
    num_episodes = 40000


win_count = []
missed_words_dict = {}

start_time = time.time()

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()  # Reset the environment before you start an episode
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)

    won = False

    for t in count():  # This python loop loops forever until we reach the end of the episode (at which point we break outta the loop). t takes on values 0, 1, 2 3...
        # 1) Choose action using eps-greedy policy
        action = select_action(state)
        # 2) Get the r and s'.
        observation, reward, terminated, truncated = env.step(action.item())
        if reward == 10:
            won = True
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)  # 3) Store transition

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()  # 4) Very similar to PlanningUpdate() step in DynaQ pseudocode of Junfeng's notes. BATCH_SIZE = 128 is equiv to max_model_step = 10 in assignment 2. We're in dreamworld here

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # 5) To me this isnt super important, I care more about the policy network, not the target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if i_episode % 50 == 0 or i_episode == (num_episodes - 1):
                data = plot_durations()
            if i_episode % 10000 == 0:
                end_time = time.time()
                print("Episode: ", i_episode, "Time taken: ", end_time - start_time)

        if done:
            win_count.append(episode_durations[-1] if won else -1)
            if not won:
                missed_word = env.words[truncated["goal_id"]]
                if missed_word in missed_words_dict:
                    missed_words_dict[missed_word] += 1
                else:
                    missed_words_dict[missed_word] = 1
            break

end_time = time.time()
print("Time taken: ", end_time - start_time)

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()



# Saving the model https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-save-and-load-a-pytorch-model.md
save_path = './dqn_wordle_data500_2m.pth'
torch.save(policy_net.state_dict(), save_path)

# saving the average duration in a file
x_df = pd.DataFrame(data)
episode_durations_df = pd.DataFrame(episode_durations)
episode_durations_wins_df = pd.DataFrame(win_count)

# CHANGE the string based on the parameter you are assigned to experiement (epsilon, batch, reward, discount factor, Q-network weight, hidden layers, space)
experimentParameter = "Warm_Start500"
# CHANGE LR to experiment variable
WEIGHTS  = "env500-18-2m"
fileName = experimentParameter+str(WEIGHTS)
x_df.to_csv(fileName+".csv", index=False)
episode_durations_df.to_csv(fileName + "episode_durations.csv", index=False)
episode_durations_wins_df.to_csv(fileName + "episode_durations_wins.csv", index=False)

with open(fileName + 'missed_words.json', 'w') as file:
    json.dump(missed_words_dict, file, indent=4)

# final_mean_result = np.mean(x_df.to_numpy()[::-10][:10])

final_mean_result = np.mean(episode_durations[-1000:])
print('Average duration of last 1000 episodes: ', final_mean_result)

# This function will crash if the number of eppisode is less than 1000
# plot_experiment(experimentParameter) #CHANGE make a change inside this function - go to func for more details


####################################################################################
