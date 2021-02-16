import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import csv
import Env
import pygame
import matplotlib.pyplot as plt
import numpy as np
import pong_utils
#from SSD_DRL_PPO_continuous_action2 import *
max_x = Env.max_x
max_y = Env.max_y
vel = 4
num_intruder = 10



class test:
    def __init__(self):
        self.intruder = []
        self.intruders = []
        self.f1 = []
        self.f2 = []
        self.det_x = 40
        self.det_y = -10000

    def run(self, policy):

        moving_mem = []
        conflict_num = 0
        for i in range(num_intruder):
            '''for t in range(testConf.steps[i]):
                if t == 0:
                    if i == 0:
                        moving = np.array([0, 0])
                    self.intruder = Env.intruder()
                    self.intruder.vel = testConf.vels[i]
                    self.intruder.start = testConf.starts[i]
                    self.intruder.end = testConf.ends[i]
                    self.intruder.x = self.intruder.start[0]
                    self.intruder.y = self.intruder.start[1]
                    self.intruders.append(self.intruder)'''
            self.intruders.append(Env.intruder())
        moving = np.array([0,0])
        for _ in range(200):
            conflict_num += self.intruder_update(moving)
            self.frame = pygame.surfarray.array3d(Env.win)
            moving = self.act(policy)
            moving_mem.append(moving)
        return np.array(moving_mem), conflict_num

    def intruder_update(self,moving):
        Env.win.fill((144, 72, 17))
        conflict = 0

        for intruder in self.intruders:
            intruder.update()
            intruder.x, intruder.y = np.array([intruder.x, intruder.y]) - moving
            self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - moving
            if intruder.x > max_x + intruder.radius or intruder.x < 0 - intruder.radius or intruder.y > max_y + intruder.radius or intruder.y < 0 - intruder.radius:
                self.intruders.pop(self.intruders.index(intruder))
                self.intruders.append(Env.intruder())
            if np.sqrt((intruder.x - max_x/2)**2 + (intruder.y - max_y/2)**2) <= 5:
                conflict += 1
            intruder.shade()
            intruder.draw(Env.win)
        self.draw_flight_plan()
        return conflict

    def act(self, policy):
        batch_input = pong_utils.preprocess_single(self.frame)
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
    moving, conflict_num = Test.run(policy)
    np.savetxt('test_moving.csv', moving)
    #plt_path(moving)
