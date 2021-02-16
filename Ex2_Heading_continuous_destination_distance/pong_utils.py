from parallelEnv import parallelEnv 
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import random as rand
import torch
import torch.nn as nn
import torch.nn.functional as F
import Env2 as Env
from torch.distributions import MultivariateNormal

envs = Env.envs()
action_size = 1
action_std = 0.5
agent_num = envs.num_agents

Mseloss = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_single(image, bkg_color = np.array([146, 73, 0])):
    img = np.mean(image-bkg_color, axis=-1)/255.
    img = np.expand_dims(img, axis=1)
    return torch.from_numpy(img).float().to(device)

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([146, 73, 0])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)



# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([2]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _ = envs.step(np.random.uniform(-1, 1, n))
        #fr2, re2, _ = envs.step([0]*n)
    
    for t in range(tmax):
        # prepare the input
        # preprocess_batch properly converts two frames into 
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        single_input = preprocess_single(fr1)


        #plt.imshow(single_input[0,0,:,:].cpu().numpy())
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        actions, logprobs = policy.act(single_input)

        action_id = (actions[0] + 1) * 90

        logprobs = logprobs.squeeze().cpu().detach().numpy()
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done = envs.step(actions.cpu().numpy())
        #fr2, re2, is_done = envs.step([0]*n)
        reward = re1
        # store the result
        state_list.append(single_input)
        reward_list.append(reward)
        prob_list.append(logprobs)
        action_list.append(actions.cpu().numpy())
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
        action_list, reward_list

# convert states to probability, passing through the policy
'''def states_to_prob(policy, states, actions):
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-2])
    action_mean = policy(policy_input)
    action_var = torch.full((action_dim,), action_std*action_std).to(device)
    action_var = action_var.expand_as(action_mean)
    cov_mat = torch.diag_embed(action_var).to(device)
    dist = torch.distributions.MultiNormal(actor_mean, cov_mat)
    return dist.prob(actions)'''


def clipped_surrogate(policy, old_logprobs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01, value_coef=0.5):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_logprobs = torch.tensor(old_logprobs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_logprobs, state_values, dist_entropy = policy.evaluate(states, actions)

    # ratio for clipping
    ratios = torch.exp(new_logprobs - old_logprobs)
    advantages = rewards - state_values
    # clipped function
    clip = torch.clamp(ratios, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratios*advantages, clip*advantages)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    '''entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))'''

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*dist_entropy - value_coef*Mseloss(state_values, rewards))



class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False),
            nn.Conv2d(4, 16, kernel_size=6, stride=4)
        )
        self.size=9*9*16
        self.actor = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.action_var = torch.full((agent_num, action_size), action_std*action_std).to(device)
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        actor_critic_input = self.conv(state).view(-1, self.size)
        action_mean = self.actor(actor_critic_input)
        cov_mat = torch.diag_embed(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob

    def evaluate(self, states, action):
        states = torch.stack(states)
        states = states.view(-1, *states.shape[-3:])
        actor_critic_input = self.conv(states).view(-1, self.size)
        action_mean = self.actor(actor_critic_input)
        action_var = self.action_var.repeat(states.shape[0], 1)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = action.view(-1, action_size)
        action_logprobs = dist.log_prob(action).view(states.shape[:-3])
        dist_entropy = dist.entropy().view(states.shape[:-3])
        state_value = self.critic(actor_critic_input).view(states.shape[:-3])
        return action_logprobs, torch.squeeze(state_value), dist_entropy

