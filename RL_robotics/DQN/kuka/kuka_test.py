## do the necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import dqn_model

from collections import deque, Counter
import random
import os

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

def get_observation():
    obs = env._get_observation()#.transpose(2, 0, 1)
    #print(obs.shape)
    return obs

def convert_to_gray(color_state):
    return cv2.cvtColor(color_state, cv2.COLOR_BGR2GRAY) / 255.0

### Play the pong game with a trained dqn agent
LOAD_PATH = './models/kuka/243_kuka_policy_net.pt'
RENDER = True
STACK_SIZE = 5

## for playing first we initialize the env
#env = wrapper.make_env("PongNoFrameskip-v4")
env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20)
env.cid = p.connect(p.DIRECT)
env.reset()


## initialize a model
input_shape = [STACK_SIZE, 48, 48]
policy_net = dqn_model.DQN(input_shape, env.action_space.n).eval()
## load the trained model
#print(torch.load(LOAD_PATH))
policy_net.load_state_dict(torch.load(LOAD_PATH))

for e in range(10):
    ## get the initial state
    state = env.reset()

    ## make a state stack
    state_stack = deque([convert_to_gray(state)]*STACK_SIZE, maxlen=STACK_SIZE)
    state = torch.FloatTensor(state).unsqueeze(0)

    total_reward = 0.0
    action_count = Counter()
    ## play the game
    #for i in range(10):
    while True:
        ## make a tensor out of statestack
        state_stack_tensor = torch.FloatTensor(state_stack).unsqueeze(0)
        #print(state_stack_tensor.size())
        ## get the actions from the policy net
        action = policy_net(state_stack_tensor).max(1)[1].item()
        #print(action)
        ## perform the action
        new_state, reward, is_done, _ = env.step(action)
        #print(a.shape)
        #new_state = get_observation()
        ## assign the new state as the current state for the next iteration
        #state = torch.FloatTensor(new_state).unsqueeze(0)
        state_stack.append(convert_to_gray(new_state))
        action_count[action] += 1

        if is_done:
            total_reward = reward
            break

    print("Total Reward: {}".format(total_reward))
    print("Number of actions: {}".format(action_count))