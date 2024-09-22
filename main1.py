#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:16:12 2024

@author: johannes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d

diff = lambda vec, mat, coord: vec[None,coord,:] - mat[:,coord]

def g(w_free, w_current, k=0.001):
    _g = np.zeros(w_free.shape)
    
    d = diff(w_free, w_current, 0)
    d3 = np.repeat(np.linalg.norm(d, axis=1)[:,None], 3, axis=1)**3
    _g[1] = np.sum(k * np.divide(d, d3, out=np.zeros_like(d), where=d3!=0))
    return _g

class Field:
    def __init__(self, num, speed, length, width, x_f, v_f):
        r = np.random.uniform(-width, width, (num, 2))
        z = np.linspace(0, length, num)
        x = np.append(r, z[:,None], axis=1)
        
        v_neg = np.repeat(np.array([0,0,speed])[None,:], num, axis=0)
        w_neg = np.stack([x, v_neg], axis=1)
        self.w_neg = w_neg
        
        v_pos = np.zeros((num, 3))
        w_pos = np.stack([x, v_pos], axis=1)
        self.w_pos = w_pos
        
        self.l = length
        self.n = num
        self.w_free = np.stack([x_f, v_f], axis=0)
        
    def gamma(self, vec, mat, c=10.1): 
        v_d = diff(vec, mat, 1)
        v_n = np.repeat(np.linalg.norm(v_d, axis=1)[:,None], 3, axis=1)
        v = np.divide(v_d, v_n, out=np.zeros_like(v_d), where=v_n!=0)
        return np.sqrt(1 - v**2 / c**2)
    
    def rk4(self, dt, lorentz=False):
        self.w_neg[:,0] += dt * self.w_neg[:,1]
        self.w_neg[:,0,2] %= self.l
        
        w_pos = np.copy(self.w_pos)
        w_neg = np.copy(self.w_neg)
        if lorentz:
            w_pos[:,0] *= self.gamma(self.w_free, w_pos)
            w_neg[:,0] *= self.gamma(self.w_free, w_neg)
        
        self.w_free += dt * (g(self.w_free, w_neg) - g(self.w_free, w_pos))
        
        self.w_free[0] += dt * self.w_free[1]
        self.w_free[0, 2] %= self.l

def animate(i):
    ax.clear()
    ax.scatter3D(field.w_neg[:,0,0], field.w_neg[:,0,1], field.w_neg[:,0,2], c='b', s=5)
    ax.scatter3D(field.w_free[0,0], field.w_free[0,1], field.w_free[0,2], s=5)
    field.rk4(dt, lorentz=True)
    
    plt.axis('off')
    ax.set_xlim([0, .3])
    ax.set_ylim([0, .3])
    ax.set_zlim([20, 100])
    ax.set_aspect('auto')

if __name__ == '__main__':
    x_free = np.array([0.3, 0, 40])
    v_free = np.array([0, 0, 5.])
    field = Field(num=100, speed=5, length=200, width=0.001, x_f=x_free, v_f=v_free)
    dt = 0.1
    
    plt.style.use('dark_background')
    fig = plt.figure('magnetism')
    ax = plt.axes(projection="3d")
    plt.axis('off')

    anim = animation.FuncAnimation(fig, animate, interval=0, cache_frame_data=False)
    plt.show()