import gym
import time
env = gym.make("MsPacman-v0")

s = env.reset()
done = False

while not done:
    env.render()
    a = env.action_space.sample()
    s,r,done,info = env.step(a) 
    time.sleep(0.1)
