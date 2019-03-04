#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:29:10 2018

@author: robertbaumgartner

Answers to the questions: 
    
    a) I observe deviations in the recontructability. The first equation A_1
    is often more accurate when inverted than the second equation. Most time the
    values deviate, though.
    
    b) The same thing as in question a) applies here. Additionally it can be stated
    that this approach performs most times better than in a) on b_1, which may be because
    the function pinv() utilizes the Moore-Penrose-pseudoinverse, which utilizes SVD.
    As an interesting fact here it happened to me very often (I am not sure whether
    this holds true every time, but every time I tried) that the two inverses of A_2 
    are the same in both question a) and question b). 
    
    c) Checking the result by recomputing the original matrices shows very 
    accurate results. 
    
    d) Checking the result by recomputing the original matrices shows very 
    accurate results
    By some research I have found out that these methods are better for 
    inverting matrices, due to the structure of the factors. For example, an uper 
    or lower triangle matrix is easy to invert.
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

# prepare the matrices

A_1 = np.dot(np.random.randn(5,3),np.random.randn(3,4))
A_2 = np.random.randn(5,4)

print('Matrix A_1: ')
print(A_1)
print()
print('Matrix A_2: ')
print(A_2)
print()

x = np.random.randn(4,1)
b_1 = np.dot(A_1, x)
b_2 = np.dot(A_2, x) + np.random.randn(5,1)
 

# -----------------Task a)------------------------
 
""" a) First use the pseudo inverse"""
A_1_inverse_1 = np.dot(np.linalg.inv(np.dot(A_1.T, A_1)), A_1.T)
A_2_inverse_1 = np.dot(np.linalg.inv(np.dot(A_2.T, A_2)), A_2.T)
 
newX_1 = np.dot(A_1_inverse_1, b_1)
newX_2 = np.dot(A_2_inverse_1, b_2)

fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(1, 4, 4), x, color = 'black', label = 'x')
ax1.plot(np.linspace(1, 4, 4), newX_1, color = 'green', linestyle = 'dashed', marker = 'o', label = 'newX_1')
ax1.plot(np.linspace(1, 4, 4), newX_2, color = 'blue', marker = 'o', linestyle = 'dashed', label = 'newX_2')

# I observe deviations in the vectors newX_1 and newX_2


# -----------------Task b)------------------------

""" b) Secondly use the pinv() function """

A_1_inverse_2 = np.linalg.pinv(A_1)
A_2_inverse_2 = np.linalg.pinv(A_2)

newX_1 = np.dot(A_1_inverse_2, b_1)
newX_2 = np.dot(A_2_inverse_2, b_2)

fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(1, 4, 4), x, color = 'black', linestyle = 'dashed', label = 'x')
ax2.plot(np.linspace(1, 4, 4), newX_1, color = 'green', marker = 'o', label = 'newX_1')
ax2.plot(np.linspace(1, 4, 4), newX_2, color = 'blue', marker = 'o', linestyle = 'dashed', label = 'newX_2')

# I observe deviations again, but this time x is found better. The norm is 
# better matched by the first equation, i.e. using A_1 and b_1


# -----------------Task c)------------------------

""" c) Compute the LU-factorization of A_1 and A_2"""
(A1_LU_1, A1_LU_2, A1_LU_3) = scipy.linalg.lu(A_1)
(A2_LU_1, A2_LU_2, A2_LU_3) = scipy.linalg.lu(A_2)

print('A_1 reconstructed using the LU-factorization: ')
print( np.dot(np.dot(A1_LU_1, A1_LU_2), A1_LU_3))
print()
print('A_2 reconstructed using the LU-factorization: ')
print(np.dot(np.dot(A2_LU_1, A2_LU_2), A2_LU_3))
print()

# -----------------Task d)------------------------

""" d) Compute the QR-factorization of A_1 and A_2"""

(A1_QR_1, A1_QR_2) = scipy.linalg.qr(A_1)
(A2_QR_1, A2_QR_2) = scipy.linalg.qr(A_2)

print('A_1 reconstructed using the QR-factorization: ')
print(np.dot(A1_QR_1, A1_QR_2))
print()
print('A_2 reconstructed using the QR-factorization: ')
print(np.dot(A2_QR_1, A2_QR_2))
print()