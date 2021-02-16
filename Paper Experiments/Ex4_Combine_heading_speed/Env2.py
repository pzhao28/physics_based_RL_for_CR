import pygame
import numpy as np
import random as rd
from math import sqrt, acos, asin, atan2, sin, cos
import torch

pygame.init()
action_size = 2
max_x = 80
max_y = 80
own_x = max_x/2
own_y = max_y/2
#own_y = max_y
#scale = max_x / 160
scale = 1
time_step = 1
win = pygame.display.set_mode((max_x, max_y),depth=8)
pygame.display.set_caption('Collision Avoidance System')

#bg = pygame.image.load('bg.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def random_border_point():
    # Generate a point on border
    surv_range = (0, max_x)
    p1 = np.random.choice(surv_range)
    p2 = np.random.uniform(surv_range[0], surv_range[1])
    point = np.random.choice([p1, p2], 2, replace=False)
    return point

def random_upper_border_point():
    surv_range = (0, max_x)
    p1 = np.random.uniform(0, max_x)
    p2 = 0
    point = np.array([p1,p2])
    return point

def random_lower_border_point():
    surv_range = (0, max_x)
    p1 = np.random.uniform(0, max_x)
    p2 = max_y
    point = np.array([p1, p2])
    return point

def random_inner_point():
    x = np.random.uniform(0, max_x)
    y = np.random.uniform(0, max_y)
    point = np.array([x, y])
    return point

class intruder(object):
    def __init__(self, vel_range = np.arange(3,5,0.1)*scale, radius = 5*scale, color = (255, 0, 0)): #radius=5*scale means 5 nautical miles; vel = 8 nautical miles per minitue
        #self.moving = moving # action is a vector (v_x, v_y), indicates the velocity of the own aircraft
        self.vel = np.random.choice(vel_range)
        #self.vel = 4
        self.start = random_border_point()
        #self.end = random_inner_point()
        #self.start = random_upper_border_point()
        #self.end = np.array([own_x, own_y])
        self.end = random_lower_border_point()
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
        self.x, self.y = np.array([self.x, self.y]) + self.vector #- self.moving
        #self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - self.moving

    def shade(self):
        self.color = (255, 0, 0)
        cx = max_x/2
        cy = max_y/2
        R = 10 * np.sqrt(cx**2 + cy**2)
        self.d = sqrt((self.x - cx)**2 + (self.y - cy)**2) + 0.1
        sin_alpha = (self.y - cy)/self.d
        cos_alpha = (self.x - cx)/self.d
        sin_theta = self.radius/self.d
        if self.d > self.radius:
            cos_theta = sqrt(self.d ** 2 - self.radius ** 2) / self.d
            sin_beta1 = sin_alpha * cos_theta - cos_alpha * sin_theta  # beta1 = alpha - theta
            cos_beta1 = cos_alpha * cos_theta + sin_alpha * sin_theta
            sin_beta2 = sin_alpha * cos_theta + cos_alpha * sin_theta  # beta2 = alpha + theta
            cos_beta2 = cos_alpha * cos_theta - sin_alpha * sin_theta
            T1x = cx + R * cos_beta1
            T1y = cy + R * sin_beta1
            T2x = cx + R * cos_beta2
            T2y = cy + R * sin_beta2
            self.T1 = (T1x, T1y)
            self.T2 = (T2x, T2y)

    def draw(self, win):
        pygame.draw.polygon(win,(200,0,0), [(max_x/2, max_y/2), self.T1, self.T2] + self.vector)
        #pygame.draw.circle(win, self.color, (int(round(self.x)), int(round(self.y))), self.radius)


class envs():
    def __init__(self, num_agents=4):

        self.ps = []
        self.num_agents = num_agents
        self.action_size = action_size
        for i in np.arange(num_agents):
            self.ps.append(env())

    def step(self, actions):
        self.fr = []
        self.re = []
        self.done = []
        for i in np.arange(self.num_agents):
            env_info = self.ps[i].step(actions[i,:])
            self.fr.append(env_info.next_state)
            self.re.append(env_info.reward)
            self.done.append(env_info.local_done)
        return np.asarray(self.fr), np.asarray(self.re), np.asarray(self.done)

    def reset(self):
        for i in np.arange(self.num_agents):
            self.ps[i].reset()


class env():
    def __init__(self):
        self.numberLoop = 0
        self.N = 5
        self.aircrafts = []
        self.max_step = 10000
        self.reward = 0
        self.done = False
        self.state = []
        self.next_state = []
        self.theta = -90 * np.pi/180
        self.own_vel = 4
        self.color = (255, 0, 0)
        self.det_x = 35
        self.det_y = 80 #destnation = (40,5)

    def reset(self):
        self.__init__()
        #return self.step(0)

    def step(self, action):
        if action == [2,2]:
            moving = np.array([0,0])
        else:
            moving = self.move(action.cpu().numpy())


        self.state = self.next_state
        reward = 0
        self.det_x, self.det_y = np.array([self.det_x, self.det_y]) - moving

        #pygame.time.delay(100) # This will delay the game the given amount of milli seconds. In our case 0.1 seconds will be the delay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        #aircrafts.append(intruder())
        #aircraft.update()
        if self.numberLoop >= 0:
            self.numberLoop += 1
        if self.numberLoop > self.N:
            self.numberLoop = 0
            self.N = rd.randint(1,5)
            #self.N = rd.randint(10,15)
            #self.N = 10
            self.aircrafts.append(intruder())

        if self.aircrafts == []:
            self.aircrafts.append(intruder())

        for aircraft in self.aircrafts:
            aircraft.x, aircraft.y = np.array([aircraft.x, aircraft.y]) - moving
            if sqrt((aircraft.x - own_x)**2 + (aircraft.y - own_y)**2) < 5*scale:
                reward -= 1
                self.aircrafts.pop(self.aircrafts.index(aircraft))
            if aircraft.x > max_x + aircraft.radius or aircraft.x < 0 - aircraft.radius or aircraft.y > max_y + aircraft.radius or aircraft.y < 0 - aircraft.radius:
                self.aircrafts.pop(self.aircrafts.index(aircraft))
                #reward += 1

        #aircrafts.append(aircraft)
        self.redrawDisplayWindow()
        self.next_state = pygame.surfarray.array3d(win)

        # check arivial
        '''if np.abs(self.det_x - own_x) < 5*scale and np.abs(self.det_y - own_y) < 5*scale:
            self.done = True
            reward += 100'''


        self.reward = reward
        if self.state == []:
            self.state = self.next_state
        return env_info(self.state, self.reward, self.done, self.next_state)

    def move(self, action): # action: changing of heading angle
        self.theta = (action[0]+1) * np.pi
        self.own_vel = action[1]*2 + 4
        moving = np.array([self.own_vel*np.cos(self.theta), self.own_vel*np.sin(self.theta)])
        return np.squeeze(moving)

    def redrawDisplayWindow(self):
        #win.blit(bg, (0, -14))
        win.fill((144,72,17))

        for aircraft in self.aircrafts:
            aircraft.update()
            aircraft.shade()
            aircraft.draw(win)
        #pygame.draw.rect(win, self.color, [self.det_x-5*scale, self.det_y-5*scale, 10 * scale, 10 * scale], 0)


    def close(self):
        pygame.quit()

class env_info(envs):
    def __init__(self, state, reward, done, next_state):
        self.vector_observation = torch.from_numpy(state).float().to(device)
        self.reward = reward
        self.local_done = done
        #self.next_state = torch.from_numpy(next_state).float().to(device)
        self.next_state = next_state
if __name__ == "__main__":
    run = True
    Environment = env()

    while run:
        Environment.step((0,0))

    pygame.quit()
