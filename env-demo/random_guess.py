import gymnasium as gym

# import ShaunSim.A2codes as A2codes
# from A2helpers import FourRoom, plot_grid_world

# import gym_wordle

from wordle_env import WordleEnv
from exceptions import InvalidWordException

env = WordleEnv()  # gym.make('wordle_env')

obs = env.reset()
done = False
while not done:
    while True:
        try:
            # make a random guess
            act = env.action_space.sample()

            # take a step
            obs, reward, done, _ = env.step(act)
            break
        except InvalidWordException:
            pass

    env.render()