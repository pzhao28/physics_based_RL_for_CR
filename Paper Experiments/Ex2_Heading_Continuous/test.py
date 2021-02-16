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

                    intruder.shade()
                    intruder.draw(Env.win)

                self.drawDestination(envDest.win_dest)
                self.state = pygame.surfarray.array3d(Env.win)
                self.dest = pygame.surfarray.array3d(envDest.win_dest)
                _, action_mean = self.act(policy, test_memory)
                theta = (action_mean+1)*np.pi/2
                moving = np.squeeze([vel * np.cos(theta), vel * np.sin(theta)])
                moving_mem.append(moving)
        return np.array(moving_mem)

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

if __name__ =="__main__":
    action_std = 0.5
    envs = Env.envs()  # pixels = 80*80
    envs_dest = envDest.envs()
    action_dim = envs.action_size
    agent_num = envs.num_agents
    Test = test()
    ppo = ActorCritic(action_dim, action_std).to(device)
    ppo.load_state_dict(torch.load('./PPO_continuous.pth'))
    moving = Test.run(ppo)
    np.savetxt('test_moving.csv', moving)
    plt_path(moving)
