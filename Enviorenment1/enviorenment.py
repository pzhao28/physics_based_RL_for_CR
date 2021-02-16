import pygame
import numpy as np
import random as rd
from math import sqrt, acos, asin, atan2, sin, cos

pygame.init()

max_x = 430
max_y = 430
scale = max_x / 160
win = pygame.display.set_mode((max_x, max_y))
pygame.display.set_caption('Collision Avoidance System')

bg = pygame.image.load('bg.png')
ac = pygame.image.load('aircraft.jpg')

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
    def __init__(self, vel_range = np.arange(7,9,0.1)*scale, radius = 5*scale, color = (255, 0, 0)): #radius=8*scale means 5 nautical miles; vel = 8 nautical miles per minitue
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
        self.x, self.y = np.array([self.x, self.y]) + self.vector

    def shade(self):
        self.color = (255, 0, 0)
        cx = max_x/2
        cy = max_y/2
        R = 10 * np.sqrt(cx**2 + cy**2)
        self.d = sqrt((self.x - cx)**2 + (self.y - cy)**2)
        sin_alpha = (self.y - cy)/self.d
        cos_alpha = (self.x - cx)/self.d
        sin_theta = self.radius/self.d
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
        '''if self.d > self.radius:
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
            self.radius = int(R/5)
            self.color = (200, 0, 0)'''

    def draw(self, win):
        #pygame.draw.polygon(win,(200,0,0), [(max_x/2, max_y/2), self.T1, self.T2] + self.vector)
        pygame.draw.circle(win, self.color, (int(round(self.x)), int(round(self.y))), int(round(self.radius)))


def redrawDisplayWindow():
    win.blit(bg, (0, -14))

    for aircraft in aircrafts:
        aircraft.update()
        #aircraft.shade()
        aircraft.draw(win)

    #pygame.draw.aalines(win, (128,0,0), True, [(0,0),(100,100),(200,300),(300,100), (400,300)])
    pygame.display.update()


aircrafts = []
#aircraft = intruder()
numberLoop = 0

run = True
N = 5
while run:
    pygame.time.delay(100) # This will delay the game the given amount of milli seconds. In our case 0.1 seconds will be the delay
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    #aircrafts.append(intruder())

    #aircraft.update()
    if numberLoop >= 0:
        numberLoop += 1
    if numberLoop > N:
        numberLoop = 0
        N = rd.randint(1,15)
        aircrafts.append(intruder())

    for aircraft in aircrafts:
        if aircraft.x > max_x + aircraft.radius or aircraft.x < 0 - aircraft.radius or aircraft.y > max_y + aircraft.radius or aircraft.y < 0 - aircraft.radius:
            aircrafts.pop(aircrafts.index(aircraft))

    #aircrafts.append(aircraft)
    redrawDisplayWindow()
pygame.quit()
