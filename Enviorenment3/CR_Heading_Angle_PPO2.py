# custom utilies for displaying animation, collecting rollouts and more
import pong_utils
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time
import Enviorenment3 as Env
import progressbar as pb
from parallelEnv import parallelEnv
import numpy as np

envs = Env.envs()

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device
print("using device: ",device)

policy=pong_utils.Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

#envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
'''prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=200)
Lsur= clipped_surrogate(policy, prob, state, action, reward)
print(Lsur)'''

# training loop max iterations
episode = 100000

# widget bar to display progress
'''widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()'''

epsilon = 0.1
beta = .01
tmax = 500
SGD_epoch = 4

# keep track of progress
mean_rewards = []

for e in range(episode):
    # collect trajectories
    old_probs, states, actions, rewards = \
        pong_utils.collect_trajectories(envs, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis=0)
    # gradient ascent step
    for _ in range(SGD_epoch):
        L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
                                          epsilon=epsilon, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

    # the clipping parameter reduces as time goes on
    epsilon *= .999
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta *= .995
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))

    # display some progress every 20 iterations
    if (e + 1) % 20 == 0:
        print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
        #print(total_rewards)

    # update progress widget bar
    #timer.update(e + 1)

#timer.finish()

#pong_utils.play(env, policy, time=200)
torch.save(policy, 'PPO.policy')
