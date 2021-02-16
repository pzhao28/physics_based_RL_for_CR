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
import Env
import numpy as np
import progressbar as pb

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

    surrogates = (R_future * ratio_PPO).mean()
    return surrogates




def main(loop):
    beta = .01
    tmax = 200
    SGD_epoch = 4
    epsilon = 0.1
    episode = 500
    envs = Env.envs()
    # check which device is being used.
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ", device)
    # keep track of progress
    mean_rewards = []
    policy = pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)

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


    env = envs.ps[0]
    mean_rewards = np.array(mean_rewards)
    np.savetxt('data_discrete_{}.csv'.format(loop), mean_rewards, newline = '\n')

