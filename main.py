import gym
import random
import numpy as np
from tensorflow.keras.models import load_model

env = gym.make('CartPole-v1')
env.reset()
population_size = 500
required_score = 50

model = load_model('Data/model/new_model.model')

scores = []
choices = []
for episode in range(50):
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
    mean = sum(scores)/(episode+1)
    print(f"Episode {episode+1}   Score: {score}  Mean: {mean}")
    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(required_score)
