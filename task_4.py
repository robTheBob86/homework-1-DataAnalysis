#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:33:11 2018

@author: robertbaumgartner

Answers to the questions: 
    Yes, there is a steady state. It is 0.66666*u0 and 1.33333*v0, see also the print 
    output of this script.
    
    NOTE: One can try more values for the iteration, as this converges really
    fast
    
"""

# Iterative approach, first step as example (syntax taken fro MATLAB):
# (u1 v1) = (8/10 1/10; 2/10 9/10) * (u0 v0)
# ===> we can multiply the matrix in the middle and see if its iteratively
# applied product with itself will converge to a steady matrix 

import numpy as np

X = np.array([[8./10, 1./10], [2./10, 9./10]])

rank = np.linalg.matrix_rank(X)
e, V = np.linalg.eig(X)
e = np.sort(e)
print 'Eigenvalue decomposition'
print('Eigenvalues: ', e)

# now, because none of the eigenvalues is larger than one, and because 
# X = np.inv(V)*np.diag(e)*V, we can expect X*X*X*... to converge
iterations = 100
e_power_matrix = np.diag(e)

for i in range(iterations):
    e_power_matrix = np.dot(np.diag(e), e_power_matrix)

X_power_iterations = np.dot(np.dot(V, e_power_matrix), np.linalg.inv(V))

res_vector = np.dot(X_power_iterations, np.ones(2).T)

print("The steady state is [{}*u0, {}*v0]]".format(res_vector[0], res_vector[1]))
print()
print("Footnote: The number of iterations can be reduced, 100 is clearly too \
much, since the matrix converges rapidly")