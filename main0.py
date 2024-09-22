#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:44:20 2024

@author: johannes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d

rand_pos = lambda n, r: np.random.uniform(-r, r, (n,3))

def g(w, k=500):
    _g = np.zeros(w.shape)
    _g[:,0] = w[:,1]
    d = w[:,None,0] - w[None,:,0]
    d3 = np.repeat(((np.linalg.norm(d, axis=2))[:,:,None]), 3, axis=2)**3
    _g[:,1] = np.sum((k * np.divide(d, d3, out=np.zeros_like(d), where=d3!=0)), axis=1)
    _g[:,1] = _g[:,1] - _g[0][1]
    return _g

class Field:
    def __init__(self, x, v):
        self.w = [np.stack([x, v], axis=1)]
    
    def rk4(self, dt):
        f_a = g(self.w[-1])
        w_b = self.w[-1] + (dt/2)*f_a
        f_b = g(w_b)
        w_c = self.w[-1] + (dt/2)*f_b
        f_c = g(w_c)
        w_d = self.w[-1] + dt*f_c
        f_d = g(w_d)
        
        w = self.w[-1] + (dt/6)*(f_a + 2*f_b + 2*f_c + f_d)
        self.w.append(w)

def animate(i):
    ax.clear()
    ax.scatter3D(field.w[-1][:,0,0], field.w[-1][:,0,1], field.w[-1][:,0,2])
    field.rk4(dt)
    
    plt.axis('off')
    r = 1000
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    ax.set_aspect('auto')

if __name__ == '__main__':
    n = 100
    dt = 0.1
    
    plt.style.use('dark_background')
    fig = plt.figure('electricity')
    ax = plt.axes(projection="3d")
    plt.axis('off')
    
    field = Field(rand_pos(n, 20), np.zeros((n, 3)))
    anim = animation.FuncAnimation(fig, animate, interval=0, cache_frame_data=False)
    plt.show()