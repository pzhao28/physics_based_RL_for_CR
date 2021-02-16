import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import JSAnimation.IPython_display
import display_animation
from matplotlib import animation
from IPython.display import display
import random as rand
import environment2.0 as env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2]-bkg_color, axis=-1)/255.
    return img

def preprocess_batch(images, bkg_color = np.array([142, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.mean(list_of_images, 1)

    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color, axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return torch.from_numpy(batch_input).float().to(device)

def collect_trajectories(envs, policy, tmax = 200, nrand = 5):
    # number of parallel instances
    n = len(envs.ps)
    #initialize retruning lists and start the game
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    envs.reset()
    #start all parallel agents
    envs.step([1]*n)
    #perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
        fr2, re2, _, _ = envs.step([0]*n)

    for t in range(tmax):
        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1, fr2])

        # probs will only be suded as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)

        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re1, is_done, _ = envs.step([0]*n)
        reward = re1 + re2
        #store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if is_done.any():
            break

        #return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list

    def states_to_prob(policy, states):
        states = torch.stack(states)
        policy_input = states.view(-1, *states.shape[-3:])
        return policy(policy_input).view(states.shape[:-3])

class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
