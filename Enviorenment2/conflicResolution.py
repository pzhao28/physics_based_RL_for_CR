import numpy as np
from environment2 import env
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpg_agent import Agent
from collections import deque
import time

episode = 500
#!pip install progressbar
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()
env = env()

def discounted_future_rewards(rewards, ratio=.999):
    n = rewards.shape[1]
    step = torch.arange(n)[:,None] - torch.arange(n)[None,:]
    ones = torch.ones_like(step)
    zeros = torch.zeros_like(step)

    target = torch.where(step >= 0, ones, zeros)
    step = torch.where(step >= 0, step, zeros)
    discount = target * (ratio ** step)
    discount = discount.to(device)

    rewards_discounted = torch.mm(rewards, discount)
    return rewards_discounted


def surrogate(policy, old_probs, states, actions, rewards, discount=.995, beta=.01):
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)

    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
    R_future = discounted_future_rewards(rewards, discount)
    R_mean = torch.mean(R_future)
    R_future -= R_mean

    surrogates = (R_future * torch.log(new_probs)).mean()

    return surrogates


def ddpg(n_episodes=500, max_t=10000, solved_score=30.0, consec_episodes=100, print_every=1, train_mode=True, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth'):
    mean_score = []
    best_score = -np.inf
    score_window = deque(maxlen=consec_episodes)
    moving_avg = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset()
        state = env_info.vector_observation
        state_input = preprocess(state)
        state_input = conv(state_input)
        # need pre-process the states to go through the conv layers, refer to surrogate policy
        score = 0
        agent.reset()
        start_time = time.time()
        for t in range(max_t):

            action = agent.act(state_input, add_noise=True)
            action = np.squeeze(action)
            moving = env.move(action) # action = (delta_theta, velocity_magnitude)
            env_info = env.step(moving) # moving = (delta_x, delta_y)
            next_state = env_info.next_state
            next_state_input = preprocess(next_state)
            next_state_input = conv(next_state_input)
            reward = env_info.reward
            done = env_info.local_done # use this to re-write the enviorenment
            #for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            agent.step(state_input, action, reward, next_state_input, done, t) # Save experience in repaly memory, and use random sample from buffer to learn
            state_input = next_state_input
            score += reward
            if np.any(done):
                break
        mean_score.append(np.mean(score))
        score_window.append(mean_score[-1])
        moving_avg.append(np.mean(score_window))

        if i_episode % print_every == 0:
                print('\rEpisode {} -- \tMean: {:.1f}\tMov. Avg: {:.1f}'.format(i_episode, mean_score[-1], moving_avg[-1]))
        if moving_avg[-1] >= solved_score and i_episode >= consec_episodes:
            print('\n, Environment SOLVED in {} episodes!\tMoving Average = {:.1f} over last {} episodes'.format(i_episode, moving_avg[-1], consec_episodes))
            if train_mode:
                torch.save(agent.actor_local.state_dict(), actor_path)
                torch.save(agent.critic_local.state_dict(), critic_path)
            break

    return mean_score, moving_avg

def preprocess(state):
    state = state/255
    return state.view(1, 1, state.shape[0], state.shape[1])

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        #430x430 to 143x143
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=3, bias=False)
        #143x143 to 47x47
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5, stride=3)
        #47x47 to 15x15
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.size = 15*15*32

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.size)
        return x

conv = convNet()
state_size = 15*15*32
action_size = 2
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)
score, avg = ddpg()
# state_size(int): Dimension of each state
# action_size(int): Dimension of each action
# seed(int): Random seed









'''discount_rate = .99
beta = .01
tmax = 320

mean_rewards = []


for e in range(episode):
    old_probs, states, actions, rewards = CD_utils.collect_trajectories(env, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis = 0)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L

    beta*=.995

    mean_rewards.append(np.mean(total_rewards))

    if (e+1)%20 == 0:
        print("Episode: {0:d}, score: {1:f}".format(e+1, np.mean(total_rewards)))
        print(total_rewards)

    timer.update(e+1)'''

timer.finish()
