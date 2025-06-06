#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import time
import numpy as np
import pylab
from pylab import plt
from GMMInference import GMMInference



img = pylab.imread('..\..\landscape.jpg')
img=img[::10,::10]
img=img.astype(np.float)/255
N = img.shape[0]*img.shape[1]
print ('N:',N)

# data
x = img.reshape(N,3).T.copy()
data_dim=x.shape[0]


K = 3
hyperparams=dict()
p = 0.1**2
hyperparams['Psi']=np.asarray([[0.1**2, 0, 0],
                               [0, 0.1**2, 0],
                               [0, 0, 0.1**2]])
hyperparams['m'] = np.ones(data_dim)*.5 # the image is in [0,1], so take the mid value.
hyperparams['kappa']=100000
hyperparams['nu']=1000
hyperparams['alphas']= np.asarray([100, 100, 100])
 
colors = np.random.rand(K,3)    

gmm_inference = GMMInference(data_dim=data_dim,K=K,hyperparams=hyperparams,colors=colors)

tic = time.process_time()
gmm_inference.gibbs_sampling(x=x,nIters=40,N=N,tf_savefig=False)
toc = time.process_time()
print ('time:',toc-tic)


plt.show()

labels=gmm_inference.z_est

seg = labels.reshape(img.shape[0],img.shape[1])
plt.figure(1)
plt.subplot(131)
plt.imshow(img,interpolation="Nearest")
plt.title('Image')
plt.subplot(132)
plt.imshow(seg,interpolation="Nearest")
plt.title("Segmentation")

plt.subplot(133)
palette = np.array([comp.m_posterior for comp in gmm_inference.comps])
plt.imshow(palette[seg],interpolation="Nearest")
plt.title("Means")

plt.show()