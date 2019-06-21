import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, flatten, fully_connected
from collections import deque
from random import randint
import os

env = gym.make("MsPacman-v0")

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# hyper parameters 
GAMES_TO_PLAY = 100
MAX_FRAMES_PER_GAME = 18000
MINIBATCH_SIZE = 32          
REPLAY_MEMORY_SIZE = 20000   
CHANNELS = 3
LEARNING_RATE = 0.01         
DISCOUNT_FACTOR = 0.99       

TARGET_NETWORK_UPDATE_FREQUENCY = 100
REPLAY_START_SIZE = 64              
UPDATE_FREQUENCY = 4                 

NUM_ACTION = env.action_space.n
INPUT_SHAPE = (None, 84, 84, CHANNELS)

# preprocessing
preprocess_input = tf.placeholder(shape=[210, 160, 3], dtype=tf.float32)
p_frame = tf.image.crop_to_bounding_box(preprocess_input,0,0,176,160)
p_frame = tf.image.resize_images(p_frame, [84, 84],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

preprocess_op = (p_frame - 128.0)/ 128.0 - 1
def preprocessing(input_frame):
    return session.run(preprocess_op, feed_dict = {preprocess_input:input_frame})

# neural network
class Network:
    def __init__(self, scope_name):

        self.input_s = tf.placeholder(tf.float32, shape=INPUT_SHAPE)

        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(scope_name) as scope:
        # Inspired by Network from DQN (Mnih 2015)
            conv_layer_1 = conv2d(self.input_s, num_outputs=32, kernel_size=(8,8),
                                  stride=4, padding='SAME', weights_initializer=initializer)
            conv_layer_2 = conv2d(conv_layer_1, num_outputs=64, kernel_size=(4,4),
                                  stride=2, padding='SAME', weights_initializer=initializer)
            conv_layer_3 = conv2d(conv_layer_2, num_outputs=64, kernel_size=(3,3),
                                  stride=1, padding='SAME', weights_initializer=initializer)

            flat = flatten(conv_layer_3)

            fc_layer_4 = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
            fc_layer_5 = fully_connected(fc_layer_4, num_outputs=NUM_ACTION, activation_fn=None,
                                         weights_initializer = initializer)

            self.output_q = fc_layer_5

            self._init_vars(scope_name)

    def _init_vars(self,scope_name):
        self.vars = tf.trainable_variables(scope=scope_name)

    def add_learning(self,target):
        self.input_action = tf.placeholder(tf.int32, shape=(None,))
        self.target_q = tf.reduce_sum(target.output_q * tf.one_hot(self.input_action, NUM_ACTION), axis=-1, keepdims=True)

        self.current_output = tf.placeholder(tf.float32, shape=(None, 1))

        # Loss Function 
        self.loss = tf.reduce_mean(tf.square(self.current_output - self.target_q))

        # Optimierer 
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.train_op = optimizer.minimize(self.loss)


# Epsilon Greedy
INITIAL_EXPLORATION     = 0.8    
FINAL_EXPLORATION       = 0.01   
FINAL_EXPLORATION_FRAME = 10000 

def get_greedy(best_action, frame):
    epsilon = FINAL_EXPLORATION

	#adjust epsilon value
    if frame < FINAL_EXPLORATION_FRAME:
        epsilon = (FINAL_EXPLORATION - INITIAL_EXPLORATION)/ FINAL_EXPLORATION_FRAME * frame + INITIAL_EXPLORATION

    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTION)
    else:
        return best_action

# Replay Memory
class Replay_memory:
    def __init__(self, capacity):
        self._currentes = deque(maxlen=capacity)
        self._actions = deque(maxlen=capacity)
        self._next_currentes = deque(maxlen=capacity)
        self._reward = deque(maxlen=capacity)
        self._dones = deque(maxlen=capacity)

    def append(self, s, a, next_s, r, done):
        self._currentes.append(s)
        self._actions.append(a)
        self._next_currentes.append(next_s)
        self._reward.append(r)
        self._dones.append(done)

    def get_batch(self, batch_size):
        s = []
        a = np.empty(batch_size, dtype=np.int32)
        s_next = []
        r = np.empty(batch_size,dtype=np.float32)
        done = np.empty(batch_size, dtype=np.bool)

        for i in range(batch_size):
            x = randint(0,len(self._currentes)-1)
            s.append(self._currentes[x])
            a[i] = self._actions[x]
            s_next.append(self._next_currentes[x])
            r[i] = self._reward[x]
            done[i] = self._dones[x]

        return s, a, s_next, r, done

# copy operator current --> target
def copy_vars_to_target(session, current_vars, target_vars):
    for i, var in enumerate(current_vars):
        copy_op = target_vars[i].assign(var.value())
        session.run(copy_op)

# print current statistics
average_reward = deque(maxlen=100)
def print_game_stats(episode, frame_number, episode_frame_number, episode_reward, episode_loss):
    average_reward.append(episode_reward)
    print(f'episode {episode} frame {frame_number} episode_frame {episode_frame_number} reward {episode_reward:.2f} average reward {sum(average_reward) / len(average_reward):.2f} episode loss {episode_loss:.2f}')


# initialization
current =  Network('current')
target = Network('target')
current.add_learning(target)

replay_memory = Replay_memory(REPLAY_MEMORY_SIZE)

# Training
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)

    frame_number = 0

    for i in range(GAMES_TO_PLAY):
        
        s = env.reset()
        
        done = False
        episode_frame_number = 0
        episode_reward = 0
        episode_loss = []
        
        while not done and episode_frame_number <= MAX_FRAMES_PER_GAME:
            
            s = preprocessing(s)
            
            # find an action
            actions = current.output_q.eval(feed_dict={current.input_s:[s]})
            best_action = np.argmax(actions, axis=-1)
            action = get_greedy(best_action, frame_number)
            
            # perform action
            s_next, r, done, _ = env.step(action)
            
            # show output
            env.render()
            
            # save new experience for later training
            replay_memory.append(s, action, preprocessing(s_next),r,done)
           
            # train
            if frame_number % UPDATE_FREQUENCY == 0 and frame_number > REPLAY_START_SIZE:
                sample_s, sample_a, sample_next_s, sample_r, sample_done = replay_memory.get_batch(MINIBATCH_SIZE)

                next_a = current.output_q.eval(feed_dict={current.input_s:sample_next_s})
                batch = sample_r + DISCOUNT_FACTOR * np.max(next_a, axis=-1) * (1 - sample_done)
                
                #Trainieren
                train_loss, _ = session.run([current.loss, current.train_op], feed_dict={current.input_s:sample_s,
                                                                         target.input_s:sample_s,
                                                                         current.current_output:np.expand_dims(batch, axis=-1),
                                                                         current.input_action:sample_a})
                episode_loss.append(train_loss)

            # copy network
            if frame_number % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and frame_number > REPLAY_START_SIZE:
                copy_vars_to_target(session,current.vars, target.vars)
                
            s = s_next
            episode_frame_number += 1
            frame_number += 1
            episode_reward += r

        print_game_stats(i,frame_number,episode_frame_number,episode_reward, sum(episode_loss) / len(episode_loss))

