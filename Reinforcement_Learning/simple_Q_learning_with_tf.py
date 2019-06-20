import gym
import numpy as np
import tensorflow.compat.v1 as tf

IS_SLIPPERY = False
env = gym.make('FrozenLake-v0', is_slippery=IS_SLIPPERY)
# Playground: S=Start, G=Goal, F=Frozen, H=Hole
# SFFF
# FHFH
# FFFH
# HFFG

# hyper parameters
EPISODES = 20000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1

# 16x4 network definition
input_state = tf.placeholder(tf.float32, shape=(1,16))
weights = tf.Variable(tf.random_uniform([16,4],0,0.01))
output_Q = tf.matmul(input_state, weights)
predicted_action = tf.argmax(output_Q, 1)

# loss function
next_Q = tf.placeholder(tf.float32, shape=(1,4))
loss = tf.reduce_sum(tf.square(next_Q - output_Q))

# optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

# training
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for i in range(EPISODES):
        state = env.reset()
        done = False

        while not done:
            next_input = np.identity(16)[state:state+1]
            action, predicted_Q = session.run([predicted_action, output_Q], feed_dict={input_state:next_input})
    
            # epsilon greedy
            if np.random.rand(1) < EPSILON:
                action[0] = env.action_space.sample()
    
            state_next, reward, done,_ = env.step(action[0])
    
            next_input = np.identity(16)[state_next:state_next+1]
    
            q_next = session.run(output_Q, feed_dict={input_state:next_input})
            max_q_next = np.max(q_next)
            target_Q = predicted_Q
            target_Q[0, action[0]] = reward + DISCOUNT_FACTOR * max_q_next
    
            _,w1 = session.run([train, weights], feed_dict={input_state:np.identity(16)[state:state+1], next_Q:target_Q})
            state=state_next

    # result
    for i in range(16):
        next_input = np.identity(16)[i:i+1]
        all_Q = session.run([output_Q], feed_dict={input_state:next_input})
        print(i, all_Q)

# Playground: S=Start, G=Goal, F=Frozen, H=Hole
# SFFF   0  1  2  3
# FHFH   4  5  6  7
# FFFH   8  9 10 11
# HFFG  12 13 14 15

# Example output with is_slippery=False:
# Action/Pos    Left       Down        Right       Up
# 0S  [array([[0.94854736, 0.9581287*, 0.93906176, 0.94854736]], dtype=float32)]
# 1   [array([[0.94854736, 0.00680895, 0.9296691 , 0.93906176]], dtype=float32)]
# 2   [array([[0.93906176, 0.5455922 , 0.0069613 , 0.19149894]], dtype=float32)]
# 3   [array([[0.0069575 , 0.00550618, 0.00277566, 0.00609652]], dtype=float32)]
# 4*  [array([[0.95812875, 0.9678069*, 0.00680895, 0.94854736]], dtype=float32)]
# 5H  [array([[0.0066596 , 0.0023571 , 0.00618639, 0.00687773]], dtype=float32)]
# 6   [array([[0.00338641, 0.9874576 , 0.00462531, 0.43863657]], dtype=float32)]
# 7H  [array([[0.00011377, 0.00571743, 0.00410145, 0.00051179]], dtype=float32)]
# 8*  [array([[0.96780694, 0.00436132, 0.9775829*, 0.95812875]], dtype=float32)]
# 9*  [array([[0.96780694, 0.9874576*, 0.9874576 , 0.00680895]], dtype=float32)]
# 10  [array([[0.9775829 , 0.997432  , 0.00912546, 0.9775829 ]], dtype=float32)]
# 11H [array([[0.00718496, 0.00921764, 0.00774916, 0.00135066]], dtype=float32)]
# 12H [array([[0.00205899, 0.00440538, 0.00236635, 0.00015437]], dtype=float32)]
# 13* [array([[0.00436132, 0.9874576 , 0.997432* , 0.9775829 ]], dtype=float32)]
# 14* [array([[0.9874576,  0.997432 ,  1.0075072*, 0.9874576 ]], dtype=float32)]
# 15G [array([[0.00080629, 0.00758324, 0.00481077, 0.00160158]], dtype=float32)]
# Predicted path: S,4,8,9,13,14,G