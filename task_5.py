# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:57:18 2018

@author: robertbaumgartner

Answers to the questions: 
    b) The SVD works better on the picture of the Mandrill than on the Durer in the 
    sense that the picture can be reconstructed much better with less components. 
    As for the memory, I personally do not really know about compression techniques 
    of images, but let's just assume the each of the pixels in this case would be 
    stored individually. Here in this script this is true, because we have a 2-dimensional
    matrix  with intesities at each of the pixels. Hence, for a picture of dimension 
    (n, m) we would need n*m pixel units. Assuming that these are 3*n*m*8 bits (without research,
    again, I do not know that much about it), because 3 colors per pixel and 256 intesities
    per color. 
    Via SVD we do obtain a compact representation in the sense that we do only have
    to store scores, loadings and the singular values. This would be for our
    (n, m) picture a matrix of (n, n_components) for the scores, n_components
    for the singular values and (n_components, m) for the scores. Hence, we 
    save memory by a factor of (n*n_components + n_components + m*n_components)/n*m,
    which is n_components*(n+m+1)/n*m NOTE: n_components = r in the question , 
    so in notation of the queestion:
        
    memory(SVD) = r*(n+m+1)
    memory(original picure) = m*n
     
    SVD works much better on the Mandrill than of the Durer because the information 
    is sparser. Whereas the Durer has many detail and uncorrelated neighbouring pixels,
    the Mandrill has many properties that ease a sparser representation.
    For example, it is approximately symmetric arounf the vertical exis through the 
    middle, and is has many approximately homogeneous areas, such as the nose or the 
    "white" fields. It hence works better on smaller r for the SVD. 
"""

import matplotlib.pyplot as plt
import numpy as np

durer = np.load('durer.npy')
mandrill = np.load('mandrill.npy')

plt.figure(1)                       
plt.imshow(durer, cmap = 'gray')
plt.figure(2)
plt.imshow(mandrill, cmap = 'gray')

[durer_scores, durer_svs, durer_loadings] = np.linalg.svd(durer)
[mandrill_scores, mandrill_svs, mandrill_loadings] = np.linalg.svd(mandrill)

n_compon = 2
figure_counter = 1
plt.figure(3, figsize = (10,10))

# do the svd for durer and plot that in a grid of subplots
while n_compon <= 64:
     
    reconstructed_durer = np.dot(np.dot(durer_scores[:,0:n_compon-1], np.diag(durer_svs[0:n_compon-1])), durer_loadings[0:n_compon-1, :])
    plt.subplot(3,2,figure_counter)
    #plt.figure(figure_counter)
    plt.imshow(reconstructed_durer, cmap= 'gray')

    figure_counter = figure_counter + 1
    
    n_compon = n_compon * 2
    
n_compon = 2
figure_counter = 1
plt.figure(4, figsize = (10,10))

# do the svd for durer and plot that in a grid of subplots
while n_compon <= 64:

    reconstructed_mandrill = np.dot(np.dot(mandrill_scores[:,0:n_compon-1], np.diag(mandrill_svs[0:n_compon-1])), mandrill_loadings[0:n_compon-1, :])
    plt.subplot(3,2,figure_counter)
    #plt.figure(figure_counter)
    plt.imshow(reconstructed_mandrill, cmap= 'gray')

    figure_counter = figure_counter + 1
    
    n_compon = n_compon * 2