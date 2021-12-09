## From https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On.git chapter 05
#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float) ## tuple of (state, action, new_state): reward
        self.transits = collections.defaultdict(collections.Counter) # tuple of (state, action): new_state : count
        self.values = collections.defaultdict(float)

    ## play random steps and update the reward and transit dicts
    ## This is the exploration phase of the agent
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample() ## sample action
            new_state, reward, is_done, _ = self.env.step(action) ## play step
            self.rewards[(self.state, action, new_state)] = reward ## update reward dict
            self.transits[(self.state, action)][new_state] += 1 ## update transit dict
            self.state = self.env.reset() if is_done else new_state ## update the current state with new state

    ## calculate the action values using the bellman equation using state, action, target state tuple
    def calc_action_value(self, state, action):
        ## for a given state, action get the count of all possible states
        target_counts = self.transits[(state, action)] 
        #print(target_counts)
        total = sum(target_counts.values()) ## total possible states
        action_value = 0.0 ## init the action value
        for tgt_state, count in target_counts.items(): ## for each possible target state
            reward = self.rewards[(state, action, tgt_state)] ## get the immediate reward
            ## calculate the action value from bellman equation 
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state]) 
        return action_value

    ## Here we greedily take actions (optimal actions)
    ## this is the exploitation phase
    ## max_a(Q(s,a))
    def select_action(self, state):
        best_action, best_value = None, None
        ## for all possible actions
        for action in range(self.env.action_space.n):
            ## get the action value from bellman
            action_value = self.calc_action_value(state, action)
            ## update the best action value and the action that caused it
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        ## return the optimal action
        return best_action

    ## Now once the exploration is done using the random actions, we do a play the episodes with
    ## optimal actions
    def play_episode(self, env):
        total_reward = 0.0
        ## reset the env at each iteration
        state = env.reset()
        ## while the episode is not done
        while True:
            ## select the best action for the given state (starts from init state)
            action = self.select_action(state)
            ## play a step on the env using the best action
            new_state, reward, is_done, _ = env.step(action)
            ## update the obtain reward
            self.rewards[(state, action, new_state)] = reward
            ## update the counter of target state
            self.transits[(state, action)][new_state] += 1
            ## update the total reward
            total_reward += reward
            if is_done:
                break
            ## set the current state as the new state for the next iteration
            state = new_state
        ## return the total reward
        return total_reward
    
    ## This is the learning algorithm
    ## update the state: max(value) table
    ## this is the value iteration step
    def value_iteration(self):
        ## for all the possible state
        for state in range(self.env.observation_space.n):
            ## calculate the value for each action and produce a list
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            ## find the optimal value and update the dictionary
            self.values[state] = max(state_values)

## Algorithm
## 1. Play random episodes and explore the env
## 2. Update the value table for the values that maximises the state value
## 3. Exploit the env using the best actions and get rewarded
## 4. Check if we have reached the reward bound and if so terminate
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        #test_env.render()
        iter_no += 1
        ## Explore the env by playing random actions
        ## and update the transit and reward tables
        agent.play_n_random_steps(100)
        ## Find the new values of the states after random actions
        agent.value_iteration()
        #print(agent.values)
        #print(agent.transits)
        #print(agent.rewards)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            ## Play an episode, this is the exploitation stage
            ## in this stage, agent utilises the best actions that maximises the state value
            reward += agent.play_episode(test_env)
            #print(reward)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
