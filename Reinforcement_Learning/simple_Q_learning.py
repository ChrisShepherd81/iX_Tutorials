import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')
# Create Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

EPISODES = 10000
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.95
EPSILON = 0.2

# Epsilon-Greedy alogrithm to balance between random and already learned actions
def epsilon_greedy(Q, state):
    if random.random() < EPSILON:
        # return random action
        return env.action_space.sample()
    else:
        # already learned action
        return max(list(range(env.action_space.n)),key = lambda x : Q[state,x])

for i in range(EPISODES):
    done = False
    state = env.reset()

    while not done:
        # show output
        # env.render()
        action = epsilon_greedy(Q, state)
        next_state, reward, done,_ = env.step(action)
        next_Q = reward + DISCOUNT_FACTOR * np.max(Q[next_state,:])
        # update Q-table
        Q[state, action] = (1-LEARNING_RATE) * Q[state, action] + LEARNING_RATE * next_Q
        state = next_state
print(Q)

