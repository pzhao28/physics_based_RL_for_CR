import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import csv
import Env2 as Env
import envDest as envDest
import testIntruderConfig as T
import pygame
import matplotlib.pyplot as plt
import numpy as np
from SSD_DRL_PPO_continuous_action2 import *


class test:
    def __init__(self):
        self.intruder = []
        self.intruders = []
        self.state = []
        self.dest = []
        self.det_x = envDest.max_x/2
        self.det_y = 1000

    def run(self, policy):
        test_memory = Memory()
        testConf = T.testIntruderConfig()
        max_x = Env.max_x
        max_y = Env.max_y
        vel = 4
        moving_mem = []
        count = 0

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
                Env.win.fill((144, 72, 17))
                envDest.win_dest.fill((144, 72, 17))

                for intruder in self.intruders:
                    intruder.update()
                    self.intruder.x, self.intruder.y = np.array([self.intruder.x, self.intruder.y]) - moving
                    self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - moving
                    if intruder.x > max_x + intruder.radius or intruder.x < 0 - intruder.radius or intruder.y > max_y + intruder.radius or intruder.y < 0 - intruder.radius:
                        self.intruders.pop(self.intruders.index(intruder))
                    if np.sqrt((intruder.x - max_x / 2) ** 2 + (intruder.y - max_y) ** 2) < 5:
                        count += 1
                    intruder.shade()
                    intruder.draw(Env.win)

                self.drawDestination(envDest.win_dest)
                self.state = pygame.surfarray.array3d(Env.win)
                self.dest = pygame.surfarray.array3d(envDest.win_dest)
                _, action_mean = self.act(policy, test_memory)
                theta = (action_mean+1)*np.pi/2
                moving = np.squeeze([vel * np.cos(theta), vel * np.sin(theta)])
                moving_mem.append(moving)



        return np.array(moving_mem), count

    def act(self, policy, test_memory):
        state_input = preprocess_batch([self.state, self.dest])
        action, action_mean = policy.act(state_input, test_memory)
        return action.cpu().data.numpy().flatten(), action_mean.cpu().data.numpy().flatten()

    def drawDestination(self, window):
        dest = envDest.intruder(radius=20 * envDest.scale, color=(0, 255, 0))
        dest.x = self.det_x
        dest.y = self.det_y
        dest.shade()
        dest.draw(window)

def plt_path(moving):
    path_x = np.tri(len(moving), len(moving), 0).dot(moving[:, 0].T)
    path_y = np.tri(len(moving), len(moving), 0).dot(moving[:, 1].T)
    #path = np.concatenate((path_x, path_y), axis=1)
    plt.plot(path_x, path_y)
    plt.xlim([-40,40])
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
    p.set_color([1, 0, 0])
    ax.add_collection(p)

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
    envs_dest = envDest.envs()
    action_dim = envs.action_size
    agent_num = envs.num_agents
    Test = test()
    ppo = ActorCritic(action_dim, action_std).to(device)
    ppo.load_state_dict(torch.load('./PPO_continuous.pth'))
    moving, count = Test.run(ppo)
    np.savetxt('test_moving.csv', moving)
    plt_path(moving)
    print("conflict number: {}".format(count))


