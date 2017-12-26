# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:58:12 2017

@author: teja
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:22:28 2017

@author: teja
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.color import rgb2gray

###############################################################################

directory = 'D:\\iNFOTECH\\MachineLearning\\Exercises\\ex06\\yalefaces_cropBackground'

def read_imgs(inp):
    i = 0
    for file in os.listdir(directory):
        temp = np.array(rgb2gray(plt.imread(directory + '\\' + file)))
        temp = np.reshape(temp,(1,38880))
        inp[i] = temp
        i += 1
    shape = np.array(plt.imread(directory + '\\' + file)).shape
    return shape

def eu_dist(x,mu):
    return np.sum((x-mu)*(x-mu))

def clstering(inp,mn,k):
    x = np.empty((k))
    i = 0
    for p in inp:
        #print(p)
        j = 0
        for m in mn:
            x[j] = eu_dist(p,m)
            j +=1
        
        cl[i] = np.argmin(x)
        i += 1
    return cl

def clust_error(inp,mn):
    i = 0
    cl_err = 0
    for p in inp:
        j = cl[i]
        cl_err += eu_dist(p,mn[j])
        i += 1
    return cl_err   

def mean_cal(inp,mn,cl,k):
    ind_run = np.zeros((k))
    i = 0
    for p in inp:
        j = cl[i]
        if ind_run[j] == 0:
            mn[j] = p
            ind_run += 1
        else:
            mn[j] = ((ind_run[j]*mn[j])+p)/(ind_run[j]+1)
            ind_run += 1
        i += 1
        
    return mn

###############################################################################

inp = np.empty((136,38880))
img_dim = read_imgs(inp)
print(img_dim)
print(max(inp[0]))
print(np.mean(inp[0]))   


k = 4

#k-means to be repeated 10 times
random_means = np.random.rand(k*10,np.size(inp,1))
print(random_means.shape)

print(np.shape(eu_dist(inp[0],random_means[0])))
print(eu_dist(inp[0],random_means[0]))

cl = np.empty((np.size(inp,0)))

i = 0
print(random_means[i:(k+i),:])


clerr = np.zeros((10,3))
for i in range(10):
    j = 0
    c = 0
    print("Repeat No. ",i)
    
    while True:
        cl = clstering(inp,random_means[(k*i):((i*k)+k),:],k)
        
        clerr[i][(j%2)] = clust_error(inp,random_means[(k*i):((i*k)+k),:])
        
        random_means[(k*i):((i*k)+k),:] = mean_cal(inp,random_means[(k*i):((i*k)+k),:],cl,k)
        
        if clerr[i][0] == clerr[i][1]:
            c += 1
            
        if c == 5:
            break
        j += 1
    
    clerr[i][2] = j
    print("No. of iterations before convergence ",j)
    print(clerr[i])
    
        

print(np.argmin(clerr,axis=0))

opt_ind = np.argmin(clerr,axis=0)[0]

print(random_means[(k*opt_ind):(k*opt_ind)+k,:])


  
    
fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
img = random_means[(k*opt_ind),:]
img = np.reshape(img,(243,160))
plt.imshow(img)
fig.tight_layout()
a.set_title('mean-1')
a = fig.add_subplot(2, 2, 2)
img = random_means[(k*opt_ind)+1,:]
img = np.reshape(img,(243,160))
plt.imshow(img)
fig.tight_layout()
a.set_title('mean-2')
a = fig.add_subplot(2, 2, 3)
img = random_means[(k*opt_ind)+2,:]
img = np.reshape(img,(243,160))
plt.imshow(img)
fig.tight_layout()
a.set_title('mean-3')
a = fig.add_subplot(2, 2, 4)
img = random_means[(k*opt_ind)+3,:]
img = np.reshape(img,(243,160))
plt.imshow(img)
fig.tight_layout()
a.set_title('mean-4')
plt.show()

