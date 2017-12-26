# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:42:18 2017

@author: teja
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:39:33 2017

@author: teja
"""

import numpy as np
import math
from functools import reduce

def mdot(*args):
    return reduce(np.dot, args)


def covar(data_mat):
    cov_mat = np.zeros((np.shape(data_mat)[1],np.shape(data_mat)[1]))
    
    n = np.shape(data_mat)[0]
    
    cov_mat = (1/n)*(np.dot(data_mat.T,data_mat))
    return cov_mat

def random_mean_gen(dataset,k):
    random_means = np.zeros((k,np.shape(dataset)[1]))
    for i in range(k):
        random_means[i] = dataset[np.random.randint(0,np.shape(dataset)[0])]
        
    return random_means

def likelihood_cal(dataset,random_means,covar_datastr):
    k = np.shape(random_means)[0]
    likelihood_mat = np.zeros((np.shape(dataset)[0],k))
    
    i = 0
    for x in dataset:
        j = 0
        for lh_entry in range(k):
            coeff = (1/((math.sqrt(2*(math.pi)))*(np.linalg.det(covar_datastr[(2*j):(2*j)+2,:]))))
            
            x_mu = x-random_means[j]
            
            sig = covar_datastr[(2*j):(2*j)+2,:]
            
            exp_val = (-0.5)*mdot(x_mu,np.linalg.inv(sig),x_mu.T)
            
            likelihood_mat[i][j] = coeff*(math.exp(exp_val))
            j+=1
        i+=1
    return likelihood_mat                          

def posterior_denom(likelihood_elm,prior_prob):
    
    post_denom = np.dot(likelihood_elm,prior_prob)
    return post_denom

    
def posterior_cal(likelihood_mat,prior_prob):
    posterior_mat = np.zeros(np.shape(likelihood_mat))
    
    i = 0
    for likelihood_elm in likelihood_mat:
        posterior_mat[i] = (likelihood_elm*prior_prob)/posterior_denom(likelihood_elm,prior_prob)
        i+=1
        
    return posterior_mat

def mean_cal(dataset,posterior_mat,k):
    mean_mat = np.zeros((np.shape(posterior_mat)[1],np.shape(dataset)[1]))
    
    for i in range(k):
        post_col = posterior_mat[:,i]
        mean_mat[i] = np.dot(post_col,dataset)/np.sum(post_col)
        
    return mean_mat

def covar_cal(dataset,posterior_mat,mean_mat,cls_index):
    covar_mat = np.zeros((np.shape(dataset)[1],np.shape(dataset)[1]))
    diagcoeff_mat = np.zeros((np.shape(dataset)[0],np.shape(dataset)[0]))
    post_col = posterior_mat[:,cls_index]
    centred_dataset = dataset-mean_mat[cls_index]
    
    for i in range(np.shape(diagcoeff_mat)[0]):
        diagcoeff_mat[i][i] = posterior_mat[i][cls_index]/np.sum(post_col)
        
    covar_mat = mdot(centred_dataset.T,diagcoeff_mat,centred_dataset)
    
    return covar_mat
    



dataset = np.loadtxt("mixture.txt")
print ("data.shape:", dataset.shape)

k = 3

prior_prob = (1/k)*np.ones((k))

mean_mat = random_mean_gen(dataset,k)

covar_datastr = np.concatenate((covar(dataset),covar(dataset),covar(dataset)),axis = 0)

compare_datastr = np.zeros((2,np.shape(dataset)[0]))

i = 0
p = 0
for iteration in range(50):
    
    print("Iteration no. ",i)
    likelihood_mat = likelihood_cal(dataset,mean_mat,covar_datastr)
    
    posterior_mat = posterior_cal(likelihood_mat,prior_prob)
    
    compare_datastr[(i%2)] = np.argmax(posterior_mat,axis=1)
    
    if i>0:
        c = compare_datastr[0] - compare_datastr[1]
        cmax = np.amax(c,axis=0)
        cmin = np.amin(c,axis=0)
        
        if cmax == cmin:
            print("datapoints didn't change the class")
            p+=1
            
        if p == 5:
            print("CONVERGENCE REACHED")
            break
    i+=1    
    
    mean_mat = mean_cal(dataset,posterior_mat,k)
    
    for ci in range(k):
        covar_datastr[(2*ci):(2*ci)+2,:] = covar_cal(dataset,posterior_mat,mean_mat,ci)
        
    


print(mean_mat)
print(covar_datastr)
print(compare_datastr)