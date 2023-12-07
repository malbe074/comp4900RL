import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical # https://pytorch.org/docs/stable/distributions.html#categorical
import matplotlib
import matplotlib.pyplot as plt
# Example:
# >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# >>> m.sample()  # equal probability of 0, 1, 2, 3
# tensor(3)

from wordle import WordleEnv100
import pandas as pd
import glob

# from graph_plotting import plot_experiment

# for mac environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args() # printing args gives # Namespace(gamma=0.99, seed=543, render=False, log_interval=10)

# Better graph recording
# Try changing neural network structure (dont share a common layer?)
# Gradient clipping?
# Encourage/discourage exploration via softmax temperatures
# Normalizing returns

env = WordleEnv100()
state = env.reset(seed=args.seed)
torch.manual_seed(args.seed)

# set up matplotlib
# hecks if the string 'inline' is present in the name of the current backend.
is_ipython = 'inline' in matplotlib.get_backend()
# checks if code is running in an IPython environment (like Jupyter Notebook)
if is_ipython:
    # for controlling the display of output, including the display of Matplotlib plots
    from IPython import display

plt.ion()  # ion stands for "interactive mode." (display plots dynamically)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) # The tuple is <'log_prob', 'value'> and represents a SavedAction

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, n_observations, n_action):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_observations, 128) # creates an instance of the nn.Linear module and assigns it to the attribute affine1 of the current class instance

        # actor's layer
        self.action_head = nn.Linear(128, n_action) # our theta
        # self.action_layer = nn.Linear(n_observations, 104)
        # self.action_head = nn.Linear(128, n_action) # our theta

        # critic's layer
        self.value_head = nn.Linear(128, 1) # state value v^, not q^
        # self.value_layer = nn.Linear(n_observations, 104)
        # self.value_head = nn.Linear(128, 1) # state value v^, I think, not q^

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        # a = F.relu(self.action_layer(x))
        # v = F.relu(self.value_layer(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1, dtype=torch.float64)
        # action_prob = F.softmax(self.action_head(a), dim=-1, dtype=torch.float64)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)
        # state_values = self.value_head(v)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

n_observations = len(state)
model = Policy(n_observations, env.action_space.n)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item() # uses NumPy to obtain the machine epsilon for the float32 data type. Machine epsilon is the difference between 1 and the smallest number greater than 1 that is representable in the given floating-point data type.


def select_action(state):
    state = torch.from_numpy(state).float() # converts the input state from a NumPy array to a PyTorch tensor and sets the data type to float.
    probs, state_value = model(state) # performs a forward pass through a PyTorch model (model) with the given state as input. The model is expected to have two output tensors: probs representing the probabilities of different actions, and state_value representing the estimated value of the state.

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs) # e.g. m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))

    # and sample an action using the distribution
    action = m.sample() # e.g. using my earlier example, equal probability of 0, 1, 2, 3

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value)) # the SavedAction tuple is <'log_prob', 'value'>

    # the action to take (left or right) (or for wordle, which word to guess)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    # returns for an episode looks like: tensor([-9.51, -9.606, -9.703, -9.801, -9.9, -10.0]). -10 reward in Wordle for not guessing word, earlier terms are discounted
    # IF you guess the word right away, your returns is a single-element tensor: tensor([0.]) 
    
    # if returns.size()[0] > 1: # If the returns tensor has more than one element, we can take .std() safely.
    #     returns = (returns - returns.mean()) / (returns.std() + eps) # eps is a v small number, can ignore. Essentially this is normalization- shifting the return values so the mean is 0. 
    # Im commenting out this step as I believe it's unnecessary based on https://ai.stackexchange.com/questions/10196/why-does-is-make-sense-to-normalize-rewards-per-episode-in-reinforcement-learnin
    # This line can be buggy, becaues taking std of a single elt tensor, tensor([0.]), causes NaN issues https://github.com/pytorch/pytorch/issues/29372
    # If I included this line, then new returns is tensor([ 1.3273,  0.8035,  0.2744, -0.2601, -0.7999, -1.3452])

    # saved_actions is a list of SavedAction tuples e.g. [<'log_prob', 'value'>, <'log_prob', 'value'>,...], where log_prob and value are both single-elt tensors. 
    # Specific example: [SavedAction(log_prob=tensor(-4.7819, dtype=torch.float64, grad_fn=<SqueezeBackward1>), value=tensor([0.0112], grad_fn=<ViewBackward0>)), SavedAction(log_prob=tensor(-4.4947, dtype=torch.float64, grad_fn=<SqueezeBackward1>), value=tensor([0.0309], grad_fn=<ViewBackward0>)), ...]
    # saved_actions is of length 6 if there were 6 steps in the episode
    # returns will also be of the same length as saved_actions
    # zipping takes the list of (6) SavedAction tuples and the list of (6) returns, and spits out 
    # a list of (6) 2-element tuples, where the first element is a SavedAction, and the 2nd elt is a return e.g. [<<'log_prob', 'value'>, 'return'>, <<'log_prob', 'value'>, 'return'>,..] 
    for (log_prob, value), R in zip(saved_actions, returns):
        # For each step in the episode, get the savedAction (specifically its log_prob and value) and its corresponding R
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()



    # perform backprop
    loss.backward()

    # In-place gradient clipping- My addition
    # uses PyTorch's gradient clipping utility to clip the gradients of the parameters of the policy_net to a specified maximum value, in this case, 100. Gradient clipping is a technique used to prevent exploding gradients during training, which can lead to numerical instability.
    # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

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

def main():
    running_reward = 10 # This is a running value, which starts out at 10, but gets updated with EACH EPISODE (not each step)! It's a weighted avg between the episode reward (NEW VALUE) and the running reward (OLD VALUE).
    num_episodes = 40000

    # run infinitely many episodes
    for i_episode in range(num_episodes): # count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning. BUT FOR WORDLE, ONLY RUN 6 Steps
        for t in count():

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, truncated = env.step(action)

            # if args.render: # Checks ~ args on line 36. Ignore, i dont care about visual rendering
            #     env.render()

            model.rewards.append(reward) # append to the model's list of rewards at EVERY step (lotta 0s)
            ep_reward += reward
            if done:
                episode_durations.append(t + 1)
                if i_episode % 50 == 0 or i_episode == (num_episodes - 1):
                    data = plot_durations()
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode() # Optimize at the end of the ep, not after each step

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()