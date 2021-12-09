## do the necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import dqn_model

from collections import deque
import random
import os

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p


## GLOBALS
## GLOBALS
STACK_SIZE = 5
NUM_EPISODES = 100

MEAN_REWARD_BOUND = 50.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 500 #150#1000
REPLAY_START_SIZE = BATCH_SIZE

EPSILON_DECAY_LAST_FRAME = 10**4#10**5
EPSILON_START = 0.9 #1.0
EPSILON_FINAL = 0.1

MODEL_PATH = './models/kuka'
MODEL_NAME = 'kuka_policy_net.pt'
device = 'cuda'

env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
env.cid = p.connect(p.DIRECT)
env.reset()
## get the camera image from the pybullet environment 
## (this is an observation of the enviroment)
def get_observation():
    obs = env._get_observation()#.transpose(2, 0, 1)
    #print(obs.shape)
    return obs

plt.imshow(get_observation())
#plt.show()

## define a replay buffer to store (s, a, r, s`) 
class ReplayBuffer:
    ## initialize the buffer
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    ## get buffer length
    def __len__(self):
        return len(self.buffer)

    ## append a sample of data to the buffer
    def push(self, experiance):
        self.buffer.append(experiance)
    
    ## sample data from the buffer
    def sample(self, batch_size):
        ## get a set of samples randomly
        batch_idx = np.random.choice(len(self.buffer), batch_size)
        state, action, reward, new_state = zip(*[self.buffer[idx] for idx in batch_idx])
        
        ## convert the state and new_state deques to a tensor
        ## converting the list to np array is a must, otherwise it
        ## introduces a bottleneck in performances
        state = torch.squeeze(torch.FloatTensor(np.array([state])), 0).to(device)
        new_state = torch.squeeze(torch.FloatTensor(np.array([new_state])), 0).to(device)

        ## convert the action batch to a tensor so it can gather
        action = torch.unsqueeze(torch.tensor(action), dim=1).to(device)

        ## convert rewards to tensor for bellman calculation
        reward = torch.tensor(reward).to(device)
        
        #print(reward)
        #print(new_state.size())
        #for i in range(0, batch_size, STACK_SIZE):
        #    state_batch.append(list(state[i:i+STACK_SIZE]))
        #state = torch.cat(state_batch, dim=1)
        
        return state, action, reward, new_state

## define the agent class
## agent can perform a random action or an action from the policy net
## agent should keep track of total reward for each episode so we know 
## if the agent improves or not
class Agent:
    ## Initialize the agent
    def __init__(self, policy_net, env):
        ## keep track of frames for debugging
        self.step_counter = 0
        self.policy_net = policy_net
        self.env = env
        self.total_reward = 0.0
        self.env.reset()

    ## play a step
    def play_step(self, state, epsilon):
        done_reward = None
        ## increment the step counter
        self.step_counter += 1
        rnd = random.random()
        if rnd > epsilon:
            state = torch.unsqueeze(state, 0).to(device)
            #print(state.size())
            action = self.policy_net(state).max(1)[1].view(1,1)
            action = action.item()
        else:
            #print("random")
            action = self.env.action_space.sample()
        
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        #print(is_done)
        #print(action)
        #print(reward)
        #print(action)
        ## check if the episode has ended
        #print(self.step_counter)
        if is_done:
            #print(self.step_counter)
            #self.step_counter = 0
            #print(is_done)
            done_reward = self.total_reward
            self.total_reward = 0.0
            #self.env.reset()

        return action, done_reward, reward, new_state, is_done

    def compute_loss(self, tgt_net, batch):
        ## unpack the databatch
        state, action, reward, new_state = batch
        #print(state_batch.size())
        #print(action_batch.size())

        ## convert the state and new_state deques to a tensor
        ## converting the list to np array is a must, otherwise it
        ## introduces a bottleneck in performances
        '''
        state = torch.squeeze(torch.FloatTensor(np.array([state])), 0).to(device)
        new_state = torch.squeeze(torch.FloatTensor(np.array([new_state])), 0).to(device)

        ## convert the action batch to a tensor so it can gather
        action = torch.unsqueeze(torch.tensor(action), dim=1).to(device)

        ## convert rewards to tensor for bellman calculation
        reward = torch.tensor(reward).to(device)
        '''
        ##to compute the loss we need Q(s,a) and Q(s`,a`)
        ## first get the Q(s,a) from the policy network
        ## then gather the values corresponds to target actions from the memory
        action_q_value = self.policy_net(state) \
                                            .gather(1, action.to(device)).squeeze(1)
        #print(action_q_value)

        ## Now find the Q(s`, a`)
        ## for this first we need to filter the done states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, new_state)))
        non_final_new_states = new_state[non_final_mask]
        ## now we use the target net and non final next state batch to get Q(s`, a`)
        ## detach the tensor from the graph to prevent gradients flowing to the
        ## target network
        next_state_q_value = torch.zeros(BATCH_SIZE, dtype=torch.float, device=device)
        next_state_q_value[non_final_mask] = tgt_net(non_final_new_states) \
                                                        .max(1)[0].detach()
        #print(next_state_q_value)
        ## now find the Q(s,a)_target using bellman equation
        expected_q_value = reward + GAMMA * next_state_q_value
        #print(action_q_value.size())
        #print(expected_q_value.size())
        ## compute the MSE loss
        loss = nn.SmoothL1Loss()(action_q_value, expected_q_value)
        return loss

    def optimize(self, optimizer, batch, tgt_net):
        ## init the optimizer
        ## we do this to prevent the gradients of one minibactch flowing
        ## to the next minibatch (pytorch accumilates gradients within a minibatch
        ## when you call loss.backward())
        optimizer.zero_grad()
        ## compute the loss
        loss = self.compute_loss(tgt_net, batch)
        ## compute gradients
        loss.backward()
        ## update parameters
        optimizer.step()

        ## return loss for monitoring
        return loss.item()

#def apply_action(env, action):
    ## perform the action on the env

def convert_to_gray(color_state):
    return cv2.cvtColor(color_state, cv2.COLOR_BGR2GRAY) / 255.0

## Main train loop
#env = gym.make('Pong-v0')
#env = wrapper.make_env("PongNoFrameskip-v4")
#print(env.observation_space.shape)
## init a replay buffer
memory_buffer = ReplayBuffer(REPLAY_SIZE)

## init the DQN networks (both policy and target nets)
input_shape = [STACK_SIZE, 48, 48]
#print(input_shape)
policy_net = dqn_model.DQN(input_shape, env.action_space.n).to(device)
target_net = dqn_model.DQN(input_shape, env.action_space.n).to(device)
## copy the policy net weights to target net
target_net.load_state_dict(policy_net.state_dict())

## define the optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

## init an agent
agent_dqn = Agent(policy_net, env)

## reward vars
ten_rewards = 0.0 ## reward for 10 episodes
total_rewards = list() ## total 
best_mean_reward = None
update_counter = 0

while True:
#for e in range(NUM_EPISODES):
    update_counter += 1
    ## get the training loss for an episode
    episode_loss = list()
    ## decay the epsilon value
    #epsilon = max(EPSILON_FINAL, EPSILON_START - e / EPSILON_DECAY_LAST_FRAME)

    current_state = env.reset()
    #print(current_state[0].shape)
    ## convert to gray scale
    current_state = convert_to_gray(current_state)
    #print(current_state.shape)
    ## deq the initial frames as a stak defined by STACK_SIZE
    current_state_stack = deque([current_state] * STACK_SIZE, maxlen=STACK_SIZE)
    #print(len(current_state_stack))
    
    ## Now play a game episode
    #for _ in range(10):
    #env.render()
    while True:
    #for i in range(1000):
        #print(agent_dqn.step_counter)
        epsilon = max(EPSILON_FINAL, EPSILON_START - update_counter / EPSILON_DECAY_LAST_FRAME)
        #env.render()
        ## let the agent take an action
        #print(torch.tensor(current_state_stack).size())
        #print(current_state_stack)
        #print()
        current_state_tensor = torch.FloatTensor(current_state_stack)
        #print(current_state_tensor.type())
        #print(current_state_tensor.size())

        action, done_reward, reward, _, is_done = agent_dqn.play_step(
                                                            current_state_tensor, epsilon)
        new_state = get_observation()
        #print(new_state.shape)
        if not is_done:
            ## make new state stack from the current states
            new_state_stack = current_state_stack
            ## add the new state as the last element in the stack
            ## so the stack now pops the oldest element
            new_state_stack.append(convert_to_gray(new_state))
            #new_state_stack = deque([new_state_frame for new_state_frame in new_state])
            #print(len(new_state_stack))
            ## update the buffer
            memory_buffer.push((current_state_stack, action, reward, new_state_stack))
        else:
            new_state_stack = None

        ### if episode is done: brake
        ### if reward is better than threshold: brake
        ### When the episode is done 
        #print(len(memory_buffer))
        #print()
        #print(reward)
        
        if done_reward is not None:
            total_rewards.append(done_reward)
            #print(reward)
            mean_reward = np.mean(total_rewards[-100:]) * 100.0
            num_played_games = len(total_rewards)
            '''
            print("Number of games: {}, mean reward: {}, best mean reward: {}, epsilon: {}" \
                                                        .format(num_played_games, 
                                                                mean_reward, 
                                                                best_mean_reward,
                                                                epsilon))
            '''
            if best_mean_reward is None or best_mean_reward < mean_reward:
                ## update the best mean reward
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward: {}".format(best_mean_reward))
                    SAVE_PATH = os.path.join(MODEL_PATH, str(num_played_games)+'_'+MODEL_NAME)  
                    torch.save(policy_net.state_dict(), SAVE_PATH)
            '''
            if num_played_games % 1000 == 0:
                ## save the model
                SAVE_PATH = os.path.join(MODEL_PATH, str(num_played_games)+'_'+MODEL_NAME)  
                torch.save(policy_net.state_dict(), SAVE_PATH)
            '''
            break

        ## make the current state the new state
        current_state_stack = new_state_stack
        #print(current_state.shape)
        ## check if we have enough data in the memory buffer
        #print(len(memory_buffer))
        if len(memory_buffer) < REPLAY_START_SIZE:
            continue
        
        ## if there are enough data we sample a batch
        train_batch = memory_buffer.sample(BATCH_SIZE)
        #agent_dqn.compute_loss(target_net, train_batch)
        loss = agent_dqn.optimize(optimizer, train_batch, target_net)
        episode_loss.append(loss)
        #print(deque_to_tensor(train_batch[0][0]).size())
        #print(train_batch[3][0])

        ## We update the target net at each SYNC_TARGET_FRAMES
        if update_counter % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    #print("episode loss: {}".format(np.mean(np.array(episode_loss))))
    
    if best_mean_reward>50:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(update_counter+1, best_mean_reward))
        break