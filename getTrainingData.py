import gym
import random
import os
import numpy as np
from statistics import median, mean
from collections import Counter

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 800
required_score = 50
episodes = 10000

def GetInitialPopulation():
	training_data = []
	scores = []
	accepted_scores = []
	for i in range(episodes):
		score = 0
		game_memory = []
		prev_observation = []
		for i in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation
			score += reward
			if done:
				break

		if score >= required_score:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]
				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	if not os.path.exists('Data/'):
		os.makedirs('Data/')

	Mean = mean(accepted_scores)
	Median = median(accepted_scores)

	name = 'saved-{}-{}-mean-{}-median-{}'.format(goal_steps, episodes, int(Mean),int(Median))
	
	training_data_save = np.array(training_data)
	np.save('Data/{}.npy'.format(name), training_data_save)

	print('Average accepted score: ', Mean)
	print('Median accepted scores: ', Median)
	print(Counter(accepted_scores))

	return training_data

GetInitialPopulation()