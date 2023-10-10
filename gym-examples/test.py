import gymnasium as gym
import gym_examples
env = gym.make('gym_examples/GridWorld-v0', render_mode="human" ,size=10)
observation, info = env.reset(seed=42)
for _ in range(100000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()