import gym
import random
import numpy as np
from getModel import train_model

env = gym.make('CartPole-v0')
env.reset()
population_size = 800
required_score = 50

training_data = np.load('Data/saved-800-10000-mean-61-median-57.npy')
model = train_model(training_data)

scores = []
choices = []
for episode in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(population_size):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(1,len(prev_obs))))

        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: 
            break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(required_score)
