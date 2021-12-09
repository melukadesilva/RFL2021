import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import tensorboardX

from itertools import count

class PolicyNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super(PolicyNet, self).__init__()
        self.fc_in = nn.Linear(input_shape, 128)
        self.drop = nn.Dropout(p=0.6)
        self.fc_out = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return Categorical(torch.softmax(x, dim=1))

## function that takes a list of rewards and reutrn the list of returns for each step
def discounted_returns(rewards, gamma=0.9):
    ## Init R
    R = 0
    returns = list()
    for reward in reversed(rewards):
        R = reward + gamma * R
        #print(R)
        returns.insert(0, R)
        #returns.append(R)

    returns = torch.tensor(returns)
    ## normalize the returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

## test the environment
env = gym.make('CartPole-v0')
input_shape = env.observation_space.shape[0]
action_size = env.action_space.n
print("Env reward threshold: {}".format(env.spec.reward_threshold))
reward_list = list()

## initialize the net
net = PolicyNet(input_shape, action_size)
## initialize an optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

writer = tensorboardX.SummaryWriter()
running_reward = 10
for e in count(1):
    action_log_probs = list()
    rewards = list()
    entropys = list()
    state = env.reset()
    #print(state.shape)
    for t in range(100000):
    #while True:
        #env.render()
        ## take an action sampled from a categorical distribution given the state
        action_prob = net(torch.FloatTensor(state).unsqueeze(0))
        action = action_prob.sample()
        action_log_probs.append(action_prob.log_prob(action))
        entropy = action_prob.entropy()
        entropys.append(entropy)
        #print(action)
        next_state, reward, is_done, _ = env.step(action.item()) # take a random action
        rewards.append(reward)
        ## current state is next state now
        state = next_state
        
        if is_done:
            #print(rewards)
            break

    ## Now we have the discounted reward + log_probs of the actions
    returns = discounted_returns(rewards)
    action_losses = list()
    ## collect the action losses to a list
    for ret, l_prob, ent in zip(returns, action_log_probs, entropys):
        action_losses.append(-(l_prob * ret))

    optimizer.zero_grad()
    ## accumulate the action losses
    action_loss = torch.cat(action_losses).sum()# - torch.cat(entropys).sum()
    action_loss.backward()

    ## step the optimizer
    optimizer.step()

    ## get stats
    ep_reward = sum(rewards)
    writer.add_scalar("episode_reward", ep_reward, e)
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    #print(e)
    if e % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  e, ep_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
        break

writer.close()    
env.close()