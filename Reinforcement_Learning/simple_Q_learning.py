import gym
import numpy as np
import random

IS_SLIPPERY = True
PUNISH_STANDSTILL = False
env = gym.make('FrozenLake-v0', is_slippery=IS_SLIPPERY)

# Create Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

EPISODES = 100000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.2

def printSingleStep():
    # Print a single step
    state = env.reset()
    env.render()
    print("State: %d" % state)
    action = env.action_space.sample()
    print("Action: %d" % action)
    next_state, reward, done,_ = env.step(action)
    env.render()
    print("Next State: %d" % next_state)
    print("Reward: %d" % reward)

# Epsilon-Greedy alogrithm to balance between random and already learned actions
def epsilon_greedy(Q, state):
    if random.random() < EPSILON:
        # return random action
        return env.action_space.sample()
    else:
        # already learned action
        return max(list(range(env.action_space.n)),key = lambda x : Q[state,x])

#printSingleStep()

for i in range(EPISODES):
    done = False
    state = env.reset()

    while not done:
        # show output
        #env.render()
        action = epsilon_greedy(Q, state)
        #print(action)
        next_state, reward, done,_ = env.step(action)
        
        # punish border bouncing
        if PUNISH_STANDSTILL and state == next_state:
            reward -= 0.1
            
        next_Q = reward + DISCOUNT_FACTOR * np.max(Q[next_state,:])
        # update Q-table
        Q[state, action] = (1-LEARNING_RATE) * Q[state, action] + LEARNING_RATE * next_Q
        state = next_state
print(Q)

# Playground: S=Start, G=Goal, F=Frozen, H=Hole
# SFFF   0  1  2  3
# FHFH   4  5  6  7
# FFFH   8  9 10 11
# HFFG  12 13 14 15

# Example output with is_slippery=False:
# Action/Pos    Left       Down       Right      Up
#  *0 S     [[0.63509189 *0.77378094 0.6983373  0.63509189]
#   1        [0.73509189 0.          0.66342043 0.5983373 ]
#   2        [0.6983373  0.58568631  0.39561296 0.43322477]
#   3        [0.58112412 0.          0.01815476 0.01976981]
#  *4        [0.67378094 *0.81450625 0.         0.73509189]
#   5 H      [0.         0.          0.         0.        ]
#   6        [0.         0.9025      0.         0.60291699]
#   7 H      [0.         0.          0.         0.        ]
#   *8       [0.71450625 0.          *0.857375  0.77378094]
#   *9       [0.81450625 *0.9025     0.9025     0.        ]
#  10        [0.857375   0.95        0.         0.857375  ]
#  11 H      [0.         0.          0.         0.        ]
#  12 H      [0.         0.          0.         0.        ]
#  *13       [0.         0.8025      *0.95      0.857375  ]
#  *14       [0.9025     0.85        *1.        0.9025    ]
#  *15 G     [0.         0.          0.         0.        ]]
# Predicted path: S,4,8,9,13,14,G
