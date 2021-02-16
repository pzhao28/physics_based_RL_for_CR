import pygame
import numpy as np
import random as rd
from math import sqrt, acos, asin, atan2, sin, cos
import torch

pygame.init()

max_x = 430
max_y = 430
own_x = max_x/2
own_y = max_y/2
scale = max_x / 160
DET_x = own_x
DET_y = -100*scale
time_step = 1
win = pygame.display.set_mode((max_x, max_y),depth=8)
pygame.display.set_caption('Collision Avoidance System')

bg = pygame.image.load('bg.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def random_border_point():
    # Generate a point on border
    surv_range = (0, max_x)
    p1 = np.random.choice(surv_range)
    p2 = np.random.uniform(surv_range[0], surv_range[1])
    point = np.random.choice([p1, p2], 2, replace=False)
    return point

def random_inner_point():
    x = np.random.uniform(0, max_x)
    y = np.random.uniform(0, max_y)
    point = (x, y)
    return point

class intruder(object):
    def __init__(self, moving, vel_range = np.arange(7,9,0.1)*scale, radius = 5*scale, color = (255, 0, 0)): #radius=5*scale means 5 nautical miles; vel = 8 nautical miles per minitue
        self.moving = moving # action is a vector (v_x, v_y), indicates the velocity of the own aircraft
        self.vel = np.random.choice(vel_range)
        self.start = random_border_point()
        self.end = random_border_point()
        self.x = self.start[0]
        self.y = self.start[1]
        self.radius = radius
        self.color = color
        self.T1 = (0,0)
        self.T2 = (0,0)
        self.d = 0

    def update(self):
        # Generate the path
        d = np.sqrt(np.sum((self.end - self.start)**2))
        (cos_theta, sin_theta) = (self.end - self.start)/d
        self.vector = np.array([cos_theta, sin_theta])*self.vel
        self.x, self.y = np.array([self.x, self.y]) + self.vector - self.moving

    def shade(self):
        self.color = (255, 0, 0)
        cx = max_x/2
        cy = max_y/2
        R = 10 * np.sqrt(cx**2 + cy**2)
        self.d = sqrt((self.x - cx)**2 + (self.y - cy)**2)
        sin_alpha = (self.y - cy)/self.d
        cos_alpha = (self.x - cx)/self.d
        sin_theta = self.radius/self.d
        if self.d > self.radius:
            cos_theta = sqrt(self.d**2 - self.radius**2)/self.d
            sin_beta1 = sin_alpha*cos_theta - cos_alpha*sin_theta #beta1 = alpha - theta
            cos_beta1 = cos_alpha*cos_theta + sin_alpha*sin_theta
            sin_beta2 = sin_alpha*cos_theta + cos_alpha*sin_theta #beta2 = alpha + theta
            cos_beta2 = cos_alpha*cos_theta - sin_alpha*sin_theta
            T1x = cx + R*cos_beta1
            T1y = cy + R*sin_beta1
            T2x = cx + R*cos_beta2
            T2y = cy + R*sin_beta2
            self.T1 = (T1x, T1y)
            self.T2 = (T2x, T2y)

        else:
            self.color = (200, 0, 0)

    def draw(self, win):
        pygame.draw.polygon(win,(200,0,0), [(max_x/2, max_y/2), self.T1, self.T2] + self.vector)
        if self.d <= self.radius:
            pygame.draw.circle(win, self.color, (int(round(self.x)), int(round(self.y))), 100*int(round(self.radius)))

class own:
    def __init__(self):
        self.numberLoop = 0
        self.N = 5
        self.aircrafts = []
        self.max_step = 10000
        self.reward = 0
        self.done = False
        self.state = []
        self.next_state = []
        self.detx = DET_x
        self.dety = DET_y
        self.theta = -90 * np.pi/180
        self.vel = 8

    def reset(self):
        self.__init__()
        return self.step((0,0))

    def step(self, moving):
        self.state = self.next_state
        reward = 0

        #pygame.time.delay() # This will delay the game the given amount of milli seconds. In our case 0.1 seconds will be the delay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        #aircrafts.append(intruder())
        #aircraft.update()
        if self.numberLoop >= 0:
            self.numberLoop += 1
        if self.numberLoop > self.N:
            self.numberLoop = 0
            self.N = rd.randint(10,15)
            self.aircrafts.append(intruder(moving))

        for aircraft in self.aircrafts:
            if sqrt((aircraft.x - own_x)**2 + (aircraft.y - own_y)**2) < 5*scale:
                reward += -1

            if aircraft.x > max_x + aircraft.radius or aircraft.x < 0 - aircraft.radius or aircraft.y > max_y + aircraft.radius or aircraft.y < 0 - aircraft.radius:
                self.aircrafts.pop(self.aircrafts.index(aircraft))

        #aircrafts.append(aircraft)
        self.redrawDisplayWindow()
        self.next_state = pygame.surfarray.array2d(win) #Todo: surf needs another process to go through conv

        # check arivial
        self.detx -= moving[0] * time_step
        self.dety -= moving[1] * time_step

        if sqrt((self.detx - own_x)**2 + (self.dety - own_y)**2) < 10*scale:
            self.done = True
            reward += 100

        self.reward = reward
        if self.state == []:
            self.state = self.next_state
        return env_info(self.state, self.reward, self.done, self.next_state)

    def move(self, action):
        self.theta += action[0]*np.pi/4
        self.vel = 2*action[1] + 7
        moving = (self.vel*np.cos(self.theta), self.vel*np.sin(self.theta))
        return moving

    def redrawDisplayWindow(self):
        win.blit(bg, (0, -14))
        for aircraft in self.aircrafts:
            aircraft.update()
            aircraft.shade()
            aircraft.draw(win)

    def close(self):
        pygame.quit()

class env_info(env):
    def __init__(self, state, reward, done, next_state):
        self.vector_observation = torch.from_numpy(state).float().to(device)
        self.reward = reward
        self.local_done = done
        self.next_state = torch.from_numpy(next_state).float().to(device)

class envs:
    def __int__(self, n=8):
        self.ps = n # number of paralell instances
        for i in range(n):
            self.agent[i] = own()

    def reset(self):
        for i in range(n):
            self.agent[i].reset()

    def step(self, movings):
        for i in range(n):
            self.anent[i].step(movings[i])

    def move(self, actions):
        for i in range(n):
            self.agent[i].move(actions[i])

    def redrawDisplayWindow(self):
        for i in range(n):
            self.agent[i].redrawDisplayWindow()

    def close(self):
        for i in range(n):
            self.agent[i].close()








if __name__ == "__main__":
    run = True
    Environment = env()

    while run:
        Environment.step((0,0))

    pygame.quit()
