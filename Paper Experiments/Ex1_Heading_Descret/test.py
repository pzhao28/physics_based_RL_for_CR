import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import csv
import Env
import testIntruderConfig as T
import pygame
import matplotlib.pyplot as plt
import numpy as np
import pong_utils
#from SSD_DRL_PPO_continuous_action2 import *
max_x = Env.max_x
max_y = Env.max_y
vel = 4



class test:
    def __init__(self):
        self.intruder = []
        self.intruders = []
        self.f1 = []
        self.f2 = []
        self.det_x = 40
        self.det_y = -10000

    def run(self, policy):
        testConf = T.testIntruderConfig()

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


                self.intruder_update(moving)
                self.f1 = pygame.surfarray.array3d(Env.win)
                self.intruder_update([0,0])
                self.f2 = pygame.surfarray.array3d(Env.win)
                moving = self.act(policy)
                moving_mem.append(moving)
        return np.array(moving_mem)

    def intruder_update(self,moving):
        Env.win.fill((144, 72, 17))

        for intruder in self.intruders:
            intruder.update()
            self.intruder.x, self.intruder.y = np.array([self.intruder.x, self.intruder.y]) - moving
            self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - moving
            if intruder.x > max_x + intruder.radius or intruder.x < 0 - intruder.radius or intruder.y > max_y + intruder.radius or intruder.y < 0 - intruder.radius:
                self.intruders.pop(self.intruders.index(intruder))

            intruder.shade()
            intruder.draw(Env.win)
        self.draw_flight_plan()

    def act(self, policy):
        batch_input = pong_utils.preprocess_batch([self.f1, self.f2])
        probs = policy(batch_input)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample
        if action == 0:
            moving = [0, -5]
        elif action == 1:
            moving = [4, -3]
        else:
            moving = [-4, -3]
        return moving

    def draw_flight_plan(self):
        if self.det_x < 1:
            line_start_x = 1
        elif self.det_x > 80:
            line_start_x = 80
        else:
            line_start_x = self.det_x
        line_start = [line_start_x, 80]
        line_end = [line_start_x, 0]
        pygame.draw.line(Env.win, (0, 200, 0), line_start, line_end, 1)

def plt_path(moving):
    path_x = np.tri(len(moving), len(moving), 0).dot(moving[:, 0].T)
    path_y = np.tri(len(moving), len(moving), 0).dot(moving[:, 1].T)
    #path = np.concatenate((path_x, path_y), axis=1)
    plt.plot(path_x, path_y)
    plt.xlim([-40,40])
    #plt.axis("equal")
    plt.show()

if __name__ =="__main__":
    envs = Env.envs()  # pixels = 80*80
    #action_dim = envs.action_size
    #agent_num = envs.num_agents
    Test = test()
    device = pong_utils.device
    policy = pong_utils.Policy().to(device)
    policy.load_state_dict(torch.load('./policy.pth'))
    moving = Test.run(policy)
    np.savetxt('test_moving.csv', moving)
    plt_path(moving)
