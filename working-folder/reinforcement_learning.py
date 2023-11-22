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

# for mac environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = WordleEnv100()

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
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 5e-5  # halve learning rate

# Setting up seeds
seed = 42  # https://stackoverflow.com/questions/75943057/i-cant-find-how-to-reproducibly-run-a-python-gymnasium-taxi-v3-environment
np.random.seed(seed)
# https://www.w3schools.com/python/ref_random_seed.asp#:~:text=The%20random%20number%20generator%20needs,of%20the%20random%20number%20generator.
random.seed(seed)
# https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete.sample
env.action_space.seed(seed)
# https://pytorch.org/docs/stable/generated/torch.manual_seed.html
torch.manual_seed(seed)


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
# reset() should be called with a seed right after initialization and then never again. https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
state = env.reset(seed=seed)
n_observations = len(state)

# Policy network is kinda like a neural network which contains our policy data (i.e. optimal Q values, or at least how to get them)
# Target network is kinda like the "little brother" of the policy network, that learns at a slightly slower rate. It's a tool used during training to provide stable target values
policy_net = DQN(n_observations, n_actions).to(device)
# policy_net.load_state_dict(torch.load('./static_goal_100_words.pth')) # Load data in my saved file
# policy_net.eval() # Evaluate model
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# initializing an AdamW optimizer for the policy_net neural network.
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
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


def plot_durations(show_result=False):
    plt.figure(1)  # creates a new figure for plotting
    # converts the list episode_durations to a PyTorch tensor named durations_t
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()  # clear current figure https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clf.html
        plt.title('Training...')
    plt.xlabel('Every 10th Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        # computes the rolling average of durations over a window of 100 episodes and plots it.
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
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
    num_episodes = 600
else:
    num_episodes = 25000

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()  # Reset the environment before you start an episode
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)

    for t in count():  # This python loop loops forever until we reach the end of the episode (at which point we break outta the loop). t takes on values 0, 1, 2 3...
        # 1) Choose action using eps-greedy policy
        action = select_action(state)
        # 2) Get the r and s'.
        observation, reward, terminated, truncated = env.step(action.item())
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

        if done and i_episode % 10 == 0:
            episode_durations.append(t + 1)
            plot_durations()

        if done:
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# Saving the model https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-save-and-load-a-pytorch-model.md
save_path = './dqn_wordle_data.pth'
torch.save(policy_net.state_dict(), save_path)

####################################################################################
