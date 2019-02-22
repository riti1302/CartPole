import gym
from collections import Counter
import random

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 800

for episodes in range(50):
	env.reset()
	rewards = []
	for t in range(goal_steps):
		env.render()
		action = random.randrange(0,2)
		observation, reward, done, info = env.step(action)
		print(reward)
		print(done)
		rewards.append(reward)
		if done:
			break
print(rewards)
print(len(rewards))
print(Counter(rewards))
