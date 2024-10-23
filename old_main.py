import gymnasium as gym
import gym_examples
env = gym.make('Warhammer40k-v0', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()