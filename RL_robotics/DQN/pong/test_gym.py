import gym
env = gym.make('Pong-v0')
env.reset()
count = 0
while True:
    count += 1
    env.render()
    ob, r, done, _ = env.step(env.action_space.sample()) # take a random action
    if done:
        print(done)
        print(count)
        env.reset()
env.close()