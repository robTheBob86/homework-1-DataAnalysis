#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:06:14 2018

@author: robertbaumgartner

Answers to the questions: 
    a) The polynomial fit can make sense. However, the data should be closely 
    observed and an appropriate degree should be chosen (possibly as low as
    possible, to avoid overfitting and to have the best generalizablity).
    
    b) I don't want to say that the interpolation can never make sense, but 
    it is most likely to overfit and to have a poor generalizablitiy. It should
    not be done here in this case (or in most cases), and it's use should be considered
    very carefully. The only case I can imagine it to be useful is a case with a 
    deterministic, robust outcome, when it is important that the points that 
    should be interpolated are crucial to hit, which is kind of an artificial 
    example. Note here: 
    
    c) As a cubic function it is just a case of a polynom, hence everything that 
    has been said in a) applies here. Here in this case specifically, it may be the 
    best choice, since its degree is low and simple, and it somehow matches the
    function. A possible flaw here is that it is ignoring the trend of rising 
    variance over the abscissa, which may have a meaning. Here one needs additional
    knowledge about what the data represent. 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

interval = np.linspace(0.9, 4.1, 33)
samples = -11 + 55/3*interval -17/2*interval**2 + 7/6*interval**3
fig0, ax0 = plt.subplots()
ax0.plot(interval, samples)
#samples = np.zeros(33)


for i in range(len(interval)):
    #samples[i] = -11 + 55/3*interval[i] -17/2*interval[i]**2 + 7/6*interval[i]**3
    samples[i] = samples[i] + 0.3*np.random.randn()*samples[i]
    
fig1, ax1 = plt.subplots() 
ax1.plot(interval, samples)
ax1.set_xlabel("x_values")
ax1.set_ylabel("y_values")
ax1.set_title("Original Data")

# -----------------Task a)------------------------

poly = np.polyfit(interval, samples,  32)
polyvals = np.polyval(poly, interval)

fig2, ax2 = plt.subplots() 
ax2.plot(interval, samples, label = 'Original')
ax2.plot(interval, polyvals, label = 'Polynom')
ax2.set_xlabel("x_values")
ax2.set_ylabel("y_values")
ax2.set_title("Polynomial fitting")

# -----------------Task b)------------------------
    
# Note here: If cubic spline is being interpolated not at same points, but on finer sampled
# grid, then the approximation deviates more
 
cub_spline = CubicSpline(interval, samples)

fig3, ax3 = plt.subplots() 
ax3.plot(interval, samples, label = 'Original')
#ax3.plot(interval, cub_spline(interval), label = 'Cubic Spline')
ax3.plot(np.linspace(0.9, 4.1, 66), cub_spline(np.linspace(0.9, 4.1, 66)), label = 'Cubic Spline')
ax3.set_xlabel("x_values")
ax3.set_ylabel("y_values")
ax3.set_title("Fitting with Cubic Spline")

# -----------------Task c)------------------------

# assumption: f(x) = w0 + w1*x1 + w2*x1^2
# therefore, define the X-matrix (see polynomial regression):
X = np.ones([len(interval), 3])
X[:, 1] = interval.T
X[:, 2] = np.array(map(lambda x: x**2, interval)).T

# calculate weights using the Penrose-Moore-Peusodinverse
weights = np.dot(np.linalg.pinv(X), samples)

fig4, ax4 = plt.subplots() 
ax4.plot(interval, samples, label = 'Original')
ax4.plot(interval, np.dot(X, weights), label = 'Cubic Polynom')
ax4.set_xlabel("x_values")
ax4.set_ylabel("y_values")
ax4.set_title("Cubic Regression (Least Squares Method)") 