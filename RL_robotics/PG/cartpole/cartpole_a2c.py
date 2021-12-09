import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

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

class CriticNet(nn.Module):
    def __init__(self, input_shape):
        super(CriticNet, self).__init__()
        self.fc_in = nn.Linear(input_shape, 128)
        self.drop = nn.Dropout(p=0.6)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

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
pnet = PolicyNet(input_shape, action_size)
cnet = CriticNet(input_shape)
## initialize an optimizer
p_optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-2)
c_optimizer = torch.optim.Adam(cnet.parameters(), lr=1e-2)

running_reward = 10
for e in count(1):
    action_log_probs = list()
    rewards = list()
    values = list()
    state = env.reset()
    #print(state.shape)
    for t in range(100000):
    #while True:
        #env.render()
        ## take an action sampled from a categorical distribution given the state
        action_prob = pnet(torch.FloatTensor(state).unsqueeze(0))
        action = action_prob.sample()
        action_log_probs.append(action_prob.log_prob(action))
        
        #print(entropy)
        value = cnet(torch.FloatTensor(state).unsqueeze(0))
        values.append(value[0])
        #print(action)
        next_state, reward, is_done, _ = env.step(action.item()) # take a random action
        rewards.append(reward)
        
        ## current state is next state now
        state = next_state

        if is_done:
            #print(rewards)
            #print(values)
            break

    ## Now we have the discounted reward + log_probs of the actions
    returns = discounted_returns(rewards)
    #print(returns)
    action_losses = list()
    critic_losses = list()
    ## collect the action losses to a list
    for ret, l_prob, v in zip(returns, action_log_probs, values):
        advantage = ret - v
        #print(advantage)
        #print(-l_prob * ret)
        action_losses.append(-l_prob * advantage.detach())
        critic_losses.append(advantage.pow(2))

    p_optimizer.zero_grad()
    ## accumulate the action losses
    action_loss = torch.cat(action_losses).sum()
    action_loss.backward()
    ## step the optimizer
    p_optimizer.step()

    c_optimizer.zero_grad()
    critic_loss = torch.cat(critic_losses).mean()
    critic_loss.backward()
    c_optimizer.step()

    ## get stats
    ep_reward = sum(rewards)
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    if e % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  e, ep_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
        break