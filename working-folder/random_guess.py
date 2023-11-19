import gymnasium as gym

from wordle import WordleEnv1000

env = WordleEnv1000()

obs = env.reset()
done = False
while not done:
    # make a random guess
    act = env.action_space.sample()

    # take a step
    obs, reward, done, _ = env.step(act)
    env.render()