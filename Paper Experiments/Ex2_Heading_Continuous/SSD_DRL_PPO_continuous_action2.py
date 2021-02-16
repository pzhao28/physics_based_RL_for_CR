import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import Env2 as Env
import envDest as envDest
import csv
import testIntruderConfig as T
import pygame
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_single(image, bkg_color = np.array([146, 73, 0])):
    img = np.mean(image-bkg_color, axis=-1)/255.
    img = np.expand_dims(img, axis=1)
    return torch.from_numpy(img).float().to(device)

def preprocess_batch(images, bkg_color = np.array([146, 73, 0])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std):
        super(ActorCritic, self).__init__()
        #action mean range -1 to 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False),
            nn.Conv2d(4, 16, kernel_size=6, stride=4)
        )
        self.size = 9*9*16
        self.actor = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        self.action_dim = action_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state_input = self.conv(state).view(-1, self.size)
        action_mean = self.actor(state_input)
        cov_mat = torch.diag(self.action_var)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), action_mean.detach()

    def evaluate(self, states, actions):
        #states = torch.stack(states)
        states_input = states.view(-1,*states.shape[-3:])
        actor_critic_input = self.conv(states_input).view(-1, self.size)
        action_mean = self.actor(actor_critic_input)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        actions = actions.view(-1, self.action_dim)
        action_logprobs = dist.log_prob(actions).view(states.shape[:-3])

        dist_entropy = dist.entropy().view(states.shape[:-3])
        state_value = self.critic(actor_critic_input).view(states.shape[:-3])
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_dim = action_dim

        self.policy = ActorCritic(action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        #state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action, action_mean = self.policy_old.act(state, memory)
        return action.cpu().data.numpy().flatten(), action_mean.cpu().data.numpy().flatten()

    def update(self, memory):
        #Mote Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal.any():
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizeing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs)
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())



def plt_path(moving):
    path_x = np.tri(len(moving), len(moving), 0).dot(moving[:, 0].T)
    path_y = np.tri(len(moving), len(moving), 0).dot(moving[:, 1].T)
    #path = np.concatenate((path_x, path_y), axis=1)
    plt.plot(path_x, path_y)
    plt.show()

def main():
    ################### Hyperparameters ##################
    solved_reward = -0.5
    log_interval = 20 # print avg reward in the interval
    max_episodes = 100 # max training episodes
    max_timesteps = 200 # max timesteps in one episode
    update_timestep = 500 # update policy every n timesteps
    action_std = 0.5 #0.5 # constant std for action distribution (Multivariate Normal)
    K_epochs = 10 # update policy for K epochs
    eps_clip = 0.2 # clip parameter for PPO
    gamma = 0.99 # discount factor

    lr = 0.0003 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    ######################################################

    envs = Env.envs() # pixels = 80*80
    envs_dest = envDest.envs()

    action_dim = envs.action_size
    agent_num = envs.num_agents

    memory = Memory()
    ppo = PPO(action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    #logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    record_reward = []

    #training loop
    for i_episode in range(1, max_episodes + 1):
        envs.reset()
        envs_dest.reset()
        frame, _, reward, _ = envs.step([2]*agent_num)
        _, frame_dest, _, _ = envs_dest.step([2]*agent_num)



        state = preprocess_batch([frame, frame_dest])
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old
            action, _ = ppo.select_action(state, memory)
            frame, frame_dest, reward, done = envs.step(action)
            _, frame_dest, _, _ = envs_dest.step(action)
            state = preprocess_batch([frame, frame_dest])

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if done.any():
                break

        avg_length += t
        #plt.imshow(frame_dest[0,:,:,:])

        # stop training if avg_reward > solved_reward

        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous.pth')

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = (np.mean(running_reward) / log_interval)
            record_reward.append(running_reward)
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))

            if running_reward > solved_reward:
                print("########## Sloved! ##########")
                torch.save(ppo.policy.state_dict(), './PPO_continuous_solved.pth')
                break

            running_reward = 0
            avg_length = 0

    np.savetxt('data_no_action_penalty_2.csv', record_reward)
    Test = test()
    moving = Test.run(ppo)
    np.savetxt('test_moving.csv', moving)
    plt_path(moving)

if __name__ == '__main__':
    main()
