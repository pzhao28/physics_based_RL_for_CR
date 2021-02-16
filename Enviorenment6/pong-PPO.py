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
import Enviorenment6 as Env

envs = Env.envs()

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device
print("using device: ", device)

# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
# env = gym.make('PongDeterministic-v4')

# print("List of available actions: ", env.unwrapped.get_action_meanings())

policy = pong_utils.Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


def discounted_future_rewards(rewards, ratio=0.999):
    n = rewards.shape[1]
    step = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    ones = torch.ones_like(step)
    zeros = torch.zeros_like(step)

    target = torch.where(step >= 0, ones, zeros)
    step = torch.where(step >= 0, step, zeros)
    discount = target * (ratio ** step)
    discount = discount.float().to(device)

    rewards_discounted = torch.mm(rewards, discount)
    return rewards_discounted


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states, actions)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

    # discounted cumulative reward
    R_future = discounted_future_rewards(rewards, discount)

    # subtract baseline (= mean of reward)
    R_mean = torch.mean(R_future)
    R_future -= R_mean

    ratio = new_probs / (old_probs + 1e-6)
    ratio_clamped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    ratio_PPO = torch.where(ratio < ratio_clamped, ratio, ratio_clamped)

    # policy gradient maxmize target
    surrogates = (R_future * ratio_PPO).mean()

    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    # entropy = -(new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    # surrogates += torch.mean(beta*entropy)

    return surrogates


# envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
'''prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)
Lsur= clipped_surrogate(policy, prob, state, action, reward)
print(Lsur)'''

from parallelEnv import parallelEnv
import numpy as np

# keep track of how long training takes
# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
episode = 500

# widget bar to display progress
import progressbar as pb

'''widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()'''

# envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
eps_start = .5
eps_end = .01
eps_decay = 0.98
eps = eps_start # epsilon greedy
beta = .01
tmax = 200
SGD_epoch = 4
epsilon = 0.1

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards = \
        pong_utils.collect_trajectories(envs, policy, tmax=tmax, eps=eps)
    eps = max(eps_end, eps_decay*eps)

    total_rewards = np.sum(rewards, axis=0)

    # gradient ascent step
    for _ in range(SGD_epoch):
        # uncomment to utilize your own clipped function!
        # L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

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
        print(total_rewards)

    # update progress widget bar
#    timer.update(e + 1)

#timer.finish()

#env = envs.ps[0]
#pong_utils.play(env, policy, time=200)
torch.save(policy, 'PPO.policy')
