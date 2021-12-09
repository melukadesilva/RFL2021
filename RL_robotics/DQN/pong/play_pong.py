import torch
import numpy as np

from collections import Counter
import time

import wrapper
import dqn_model

### Play the pong game with a trained dqn agent
LOAD_PATH = './models/pong/400_pong_policy_net.pt'
RENDER = True
FPS = 25

## for playing first we initialize the env
env = wrapper.make_env("PongNoFrameskip-v4")

## initialize a model
policy_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).eval()
## load the trained model
#print(torch.load(LOAD_PATH))
policy_net.load_state_dict(torch.load(LOAD_PATH))

## get the initial state
state = env.reset()
state = torch.FloatTensor(state).unsqueeze(0)

total_reward = 0.0
action_count = Counter()
## play the game
#for i in range(10):
while True:
    ## get start time
    start_ts = time.time()
    ## if render the game
    if RENDER:
        env.render()
    ## get the actions from the policy net
    action = policy_net(state).max(1)[1].item()
    #print(action)
    ## perform the action
    new_state, reward, is_done, _ = env.step(action)
    ## assign the new state as the current state for the next iteration
    state = torch.FloatTensor(new_state).unsqueeze(0)
    action_count[action] += 1

    if is_done:
        break

    if RENDER:
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

print("Total Reward: {}".format(total_reward))
print("Number of actions: {}".format(action_count))