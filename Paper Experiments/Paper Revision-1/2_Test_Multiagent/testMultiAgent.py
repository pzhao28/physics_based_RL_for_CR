import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import csv
import pygame
import numpy as np
import random as rd
from math import sqrt, acos, asin, atan2, sin, cos

pygame.init()
max_x, max_y = 80, 80
protection_level = 5
color = (255, 0, 0)
win = pygame.display.set_mode((max_x, max_y), depth=8)
win.fill((144, 72, 17))
pygame.display.set_caption('Conflict Resolution System')

class aircraft():
    def __init__(self, pos, speed, angle):
        self.pos = np.array(pos, dtype=np.float64)
        self.speed = speed
        self.angle = angle
        self.vel_vec = np.array([self.speed * cos(self.angle), self.speed * sin(self.angle)])

    def update(self, next_speed, next_angle):
        self.speed = next_speed
        self.angle = next_angle
        self.vel_vec = np.array([self.speed * cos(self.angle), self.speed * sin(self.angle)])
        update_x = self.speed * cos(self.angle)
        update_y = self.speed * sin(self.angle)
        self.pos += (update_x, update_y)



    def draw(self):
        pygame.draw.circle(win, color, (int(round(self.pos[0])), int(round(self.pos[1]))),5)

    def draw_intruder_shade(self, rel_distance, rel_angle, intruder_vel_vec):
        sin_alpha = sin(rel_angle)
        cos_alpha = cos(rel_angle)
        sin_theta = protection_level/rel_distance
        if rel_distance >protection_level:
            cos_theta = np.sqrt(rel_distance**2 - protection_level**2)/rel_distance
            sin_beta1 = sin_alpha * cos_theta - cos_alpha * sin_theta  # beta1 = alpha - theta
            cos_beta1 = cos_alpha * cos_theta + sin_alpha * sin_theta
            sin_beta2 = sin_alpha * cos_theta + cos_alpha * sin_theta  # beta2 = alpha + theta
            cos_beta2 = cos_alpha * cos_theta - sin_alpha * sin_theta
            T1x = 100 * cos_beta1
            T1y = 100 * sin_beta1
            T2x = 100 * cos_beta2
            T2y = 100 * sin_beta2
            ra = (90 - self.angle)*np.pi/180 # rotate angle
            T1 = np.matmul(np.array([[cos(ra), -sin(ra)], [sin(ra), cos(ra)]]), np.array([T1x, T1y])) + 40
            T2 = np.matmul(np.array([[cos(ra), -sin(ra)], [sin(ra), cos(ra)]]), np.array([T2x, T2y])) + 40
            vec = np.matmul(np.array([[cos(ra), -sin(ra)], [sin(ra), cos(ra)]]), intruder_vel_vec)
            pygame.draw.polygon(win, color, [(max_x/2, max_y/2), T1, T2] + vec)


def relative_distance(own_position, intruder_position):
    return np.sqrt((own_position[0] - intruder_position[0])**2 + (own_position[1] - intruder_position[1])**2)

def relative_angle(own_position, intruder_position):
    distance = relative_distance(own_position, intruder_position)
    if intruder_position[1] - own_position[1] > 0:
        k = 0
    else:
        k = 1
    return np.abs(k*2*np.pi - acos((intruder_position[0] - own_position[0])/distance))

def polar2cartesian(distance, angle):
    return np.array([distance*cos(angle), distance*sin(angle)])


def main():
    ac1 = aircraft((0,0), 5, 0)
    ac2 = aircraft((60, 20*np.sqrt(3)), 5, 240*np.pi/180)
    ac3 = aircraft((60, -20*np.sqrt(3)), 5, 120*np.pi/180)



    for _ in range(10):
        win.fill((144,72,17))
        ac1.update(5,0)
        ac2.update(5,240*np.pi/180)
        ac3.update(5,120*np.pi/180)
        distance21 = relative_distance(ac2.pos, ac1.pos)
        angle21 = relative_angle(ac2.pos, ac1.pos)
        distance23 = relative_distance(ac2.pos, ac3.pos)
        angle23 = relative_angle(ac2.pos, ac3.pos)
        # ac2_in_ac1 = polar2cartesian(distance,angle)
        ac2.draw_intruder_shade(distance21, angle21, ac1.vel_vec)
        ac2.draw_intruder_shade(distance23, angle23, ac3.vel_vec)
        pygame.time.delay(1000)
        pygame.display.update()


if __name__ == '__main__':
    main()


