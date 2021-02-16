import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import csv
import Env2 as Env
import testIntruderConfig as T
import pygame
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

max_x = 80
max_y = 80
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_single(image, bkg_color = np.array([0, 0, 0])):
    img = np.mean(image-bkg_color, axis=-1)/255.
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img).float().to(device)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.distance = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.distance[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std):
        super(ActorCritic, self).__init__()
        #action mean range -1 to 1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False),
            nn.Conv2d(4, 16, kernel_size=6, stride=4)
        )
        self.size = 9*9*16
        self.actor = nn.Sequential(
            nn.Linear(self.size+1, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.size+1, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        self.action_dim = action_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, distance, memory):
        #state_input = self.conv(state).view(-1, self.size)
        state_input = torch.cat((self.conv(state).view(-1, self.size), distance), 1).to(device)
        action_mean = self.actor(state_input)
        cov_mat = torch.diag(self.action_var)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.distance.append(distance)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), action_mean.detach()


class test:
    def __init__(self):
        self.intruder = []
        self.intruders = []
        self.state = []
        self.det_x = Env.max_x/2
        self.det_y = -100

    def run(self, policy):
        test_memory = Memory()
        testConf = T.testIntruderConfig()
        max_x = Env.max_x
        max_y = Env.max_y
        vel = 4
        moving_mem = []
        count = 0
        j = 0
        intruder_mem = []
        fig, ax = plt.subplots(2,4,sharex=True, sharey=True, figsize=(8,4.3))
        fig.text(0.5, 0, 'nautical mile', ha='center')
        fig.text(0, 0.5, 'nautical mile', va='center', rotation='vertical')
        for i in range(testConf.num):
            for t in range(testConf.steps[i]):
                if t == 0:
                    if i == 0:
                        moving = np.array([0, 0])
                    self.intruder = Env.intruder()
                    self.intruder.vel = testConf.vels[i]
                    self.intruder.start = testConf.starts[i]
                    self.intruder.end = testConf.ends[i]
                    self.intruder.x = self.intruder.start[0]
                    self.intruder.y = self.intruder.start[1]
                    self.intruders.append(self.intruder)
                Env.win.fill((0, 0, 0))

                for intruder in self.intruders:
                    intruder.update()
                    intruder.x, intruder.y = np.array([intruder.x, intruder.y]) - moving

                    if intruder.x > max_x + intruder.radius or intruder.x < 0 - intruder.radius or intruder.y > max_y + intruder.radius or intruder.y < 0 - intruder.radius:
                        self.intruders.pop(self.intruders.index(intruder))
                    if np.sqrt((intruder.x - max_x/2)**2 + (intruder.y - max_y)**2) < 5:
                        count += 1

                    intruder.shade()
                    intruder.draw(Env.win)

                self.state = pygame.surfarray.array3d(Env.win)
                self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - moving
                self.distance = np.sqrt((self.det_x-max_x/2)**2+(self.det_y-max_y/2)**2)
                self.distance = torch.tensor(self.distance).view(1,1).to(device)
                _, action_mean = self.act(policy, test_memory)
                theta = (action_mean-1)*np.pi/2
                moving = np.squeeze([vel * np.cos(theta), vel * np.sin(theta)])
                '''if j>=5:
                    moving = np.array([-4,0])'''
                moving_mem.append(moving)

                '''if i >= 2 and i <=4:
                    plt_real(ax[j//4][j-4*(j//4)], self.intruders, moving, np.array([self.det_x-max_x/2, self.det_y-max_y/2])*vel/self.distance.cpu().numpy(), j)
                    ax[j//4][j-4*(j//4)].set_title('time step {}'.format(j+1))
                    #ax[j].set_xlabel('nautical mile')
                    plt.tight_layout()
                    plt.show()
                    j += 1'''
                #ax[0].set_ylabel('nautical mile')


        return np.array(moving_mem), count

    def act(self, policy, test_memory):
        state_input = preprocess_single(self.state)
        action, action_mean = policy.act(state_input, self.distance, test_memory)
        return action.cpu().data.numpy().flatten(), action_mean.cpu().data.numpy().flatten()



def plt_path(moving):
    path_x = np.tri(len(moving), len(moving), 0).dot(moving[:, 0].T)
    path_y = np.tri(len(moving), len(moving), 0).dot(moving[:, 1].T)
    #path = np.concatenate((path_x, path_y), axis=1)
    plt.plot(path_x, path_y, 'o')
    #plt.axis("equal")
    plt.show()

def plt_real(ax, intruders, moving, d_vector, j):
    cx = max_x/2
    cy = max_y/2
    patches = []
    for intruder in intruders:
        T1, T2 = polygon_points(intruder)
        polygon = Polygon(np.array([[cx,cy], T1, T2]) + intruder.vector, True)
        patches.append(polygon)
        circle = plt.Circle((intruder.x, intruder.y), 4.5)
        ax.add_artist(circle)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
    #colors = np.array([0.1,0.5,0.9])
    #p.set_array(colors)
    #colors = 100 * np.random.rand(len(patches))
    #colors = np.array([255,0,0])
    #p.set_array(np.array(colors))
    p.set_color([1, 0, 0])
    ax.add_collection(p)

    if j<=4:
        ax.arrow(40, 40, moving[0], moving[1], head_width=1, length_includes_head=True, head_length=1, fc='k', ec='k')
    ax.arrow(40, 40, np.squeeze(d_vector)[0], np.squeeze(d_vector)[1], head_width=1, length_includes_head=True, head_length=1, fc='b', ec='b')

    #ax.axis('equal')
    ax.set_xlim(20,60)
    ax.set_ylim(20,60)
    ax.invert_yaxis()





def polygon_points(intruder):
    cx = max_x/2
    cy = max_y/2
    R = 10 * np.sqrt(cx**2 + cy**2)
    d = np.sqrt((intruder.x - cx)**2 + (intruder.y - cy)**2) + 0.1
    sin_alpha = (intruder.y - cy)/d
    cos_alpha = (intruder.x - cx)/d
    sin_theta = intruder.radius/d
    #if d > intruder.radius:
    cos_theta = np.sqrt(d**2 - intruder.radius**2) / d
    sin_beta1 = sin_alpha * cos_theta - cos_alpha * sin_theta  # beta1 = alpha - theta
    cos_beta1 = cos_alpha * cos_theta + sin_alpha * sin_theta
    sin_beta2 = sin_alpha * cos_theta + cos_alpha * sin_theta  # beta2 = alpha + theta
    cos_beta2 = cos_alpha * cos_theta - sin_alpha * sin_theta
    T1x = cx + R * cos_beta1
    T1y = cy + R * sin_beta1
    T2x = cx + R * cos_beta2
    T2y = cy + R * sin_beta2
    T1 = (T1x, T1y)
    T2 = (T2x, T2y)

    return T1, T2


if __name__ =="__main__":
    action_std = 0.5
    envs = Env.envs()  # pixels = 80*80
    action_dim = envs.action_size
    agent_num = envs.num_agents
    Test = test()
    ppo = ActorCritic(action_dim, action_std).to(device)
    ppo.load_state_dict(torch.load('./PPO_continuous.pth'))
    moving, count = Test.run(ppo)
    np.savetxt('test_moving.csv', moving)
    plt_path(moving)
    print("conflict number: {}".format(count))
