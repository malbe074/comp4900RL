import gymnasium as gym

from wordle import WordleEnvFull

env = WordleEnvFull()

obs = env.reset()
done = False

env.render(hide_goal_word=True)

while not done:
    while True:
        # make a user inputed guess
        act_str = input("Enter a guess: ")
        while act_str not in env.words:
            act_str = input("Invalid guess. Enter a guess: ")

        # convert the guess to an action
        act = env.words.index(act_str)

        # take a step
        obs, reward, done, _ = env.step(act)
        break
    
    env.render(hide_goal_word=True)

env.render(hide_goal_word=False)