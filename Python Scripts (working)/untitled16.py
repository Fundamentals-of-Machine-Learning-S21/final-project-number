# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:39:41 2021

@author: carte
"""
import numpy as np

w = np.array([[4,1]])
x = np.array([[-2,1]])
t = -1-1
b = 2

n=0.25

w_new = w + n*x*t

b_new = b + n*t

z_new = np.dot(x, w_new.T) +b_new

print('w',w_new)
print('b',b_new)
print(z_new)