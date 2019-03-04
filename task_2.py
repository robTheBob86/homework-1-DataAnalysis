#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:37:14 2018

@author: robertbaumgartner

Answers to the questions: 
    a) Breakpoint should be at t=0.6 (see code below)
"""
import numpy as np
import matplotlib.pyplot as plt

# do the array from the task
x = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.9, 1.01, 1.05, 0.97, 0.98, 0.95, 0.01, -0.1, 0.02, -0.1, 0.0]])

# -----------------Task a)------------------------

# plot the data (obviously)
fig1, ax1 = plt.subplots()
ax1.plot(x[0,:], x[1,:])
ax1.set_title("Original Data")
ax1.set_xlabel("t")
ax1.set_ylabel("b")

# select a breakpoint and extract the relevant data
breakpoint = 6
x_1 = x[0, 0:breakpoint]
x_2 = x[0, breakpoint:]
y_1 = x[1, 0:breakpoint]
y_2 = x[1, breakpoint:]

# -----------------Task b)------------------------

# generate the matrices for regression
X_1 = np.ones([len(x_1), 2])
X_1[:,1] = x_1.T
X_2 = np.ones([len(x_2), 2])
X_2[:,1] = x_2.T

X_1_inv = np.linalg.pinv(X_1)
X_2_inv = np.linalg.pinv(X_2)

b_1 = np.dot(X_1_inv,y_1)
b_2 = np.dot(X_2_inv,y_2)

fig2, ax2 = plt.subplots()
ax2.plot(x_1, y_1)
ax2.plot(x_1, np.dot(X_1,b_1))
ax2.plot(x_2, y_2)
ax2.plot(x_2, np.dot(X_2,b_2))
ax2.set_title("Data after linear regression vs. Original")
ax2.set_xlabel("t")
ax2.set_ylabel("b")