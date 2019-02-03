import gym

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 800

def random_check():
  for episodes in range(5):
    env.reset()
    for t in range(goal_steps):
      env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      if done:
        break

random_check()
