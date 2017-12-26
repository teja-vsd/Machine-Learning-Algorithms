#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D plotting
from functools import reduce
import math
###############################################################################
# Helper functions
def mdot(*args):
	"""Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
	return reduce(np.dot, args)
def prepend_one(X):
	"""prepend a one vector to X."""
	return np.column_stack([np.ones(X.shape[0]), X])
def grid2d(start, end, num=50):
	"""Create an 2D array where each row is a 2D coordinate.
	1
	Machine Learning
	Exercise 2, Marc Toussaint—May 2, 2017 
	2
	np.meshgrid is pretty annoying!
	"""
	dom = np.linspace(start, end, num)
	X0, X1 = np.meshgrid(dom, dom)
	
	#addition of quadratic features
	return np.column_stack([X0.flatten(), X1.flatten(), X0.flatten()*X0.flatten(), X1.flatten()*X1.flatten(), X0.flatten()*X1.flatten()])

def cv_func(X, y, lamb):
	'''function that takes feature matrix, output vector and lambda(regularization). Performs k-fold cross validation and returns training error for
	the input lambda, mean cross validation error.	
	'''	
	splt = np.vsplit(X,5)

	ny = [0,0,0,0,0]

	for j in range(0,5):
		ny[j] = y[(10*j):(10*j)+10]
	
	cv_error = [0,0,0,0,0]
	
	for i in range(0,5):
		#X divided into n parts and n-1 parts are concatenated sucessively and considered as training data and left out part as test data.
		newX = np.concatenate((splt[(i+1)%5],splt[(i+2)%5],splt[(i+3)%5],splt[(i+4)%5]), axis = 0)
		print(newX.shape)
	
		#similar division and concatenatoin performed on output vector y.
		newY = np.concatenate([ny[(i+1)%5],ny[(i+2)%5],ny[(i+3)%5],ny[(i+4)%5]])
		print(newY.shape)
		
		#beta computed with regularization parameter.
		beta_ = mdot(inv(dot(newX.T, newX)+lamb*np.identity(6)), newX.T, newY)
		print ("Optimal beta:", beta_)
		
		#output vector for test data.
		yhat = mdot(splt[i],beta_)
		
		#cross validation error. this is ordinary least square error.
		cv_error[i] = np.sum((yhat - ny[i]) ** 2)
		#print("cv error :", cv_error[i])
	
	#ordiary square error for training data.
	se = np.sum(((dot(X, beta_.T) - y) ** 2)) + lamb * (np.sum(beta_*beta_))
	print("squared error :", se)


	print(np.mean(cv_error))
	
	return se, np.mean(cv_error)
	

###############################################################################
# load the data
data = np.loadtxt("dataQuadReg2D_noisy.txt")
print ("data.shape:", data.shape)

# split into features and labels
X, y = data[:, :2], data[:, 2]

print ("X.shape:", X.shape)
print ("y.shape:", y.shape)


# prep for linear reg.
X = prepend_one(X)
print ("X.shape:", X.shape)

a = (X[:,1])*(X[:,1]).T
b = (X[:,2])*(X[:,2]).T
c = (X[:,1])*(X[:,2]).T

X = np. column_stack((X,a,b,c))


sq_er = []
cv_er = []
lamb_val = []

lamb = 0.0001
i = 0

#computing cross validation for different lambda.
while lamb <= 100000:
	#sq_er[i], cv_er[i] = cv_func(X, y, lamb)
	a,b = cv_func(X, y, lamb)
	sq_er.append(a)
	cv_er.append(b)
	print("i and lamb :", i,lamb)
	i += 1
	lamb *= 10
	lamb_val.append(math.log10(lamb))
	

print(sq_er)
print(cv_er)
print(lamb_val)

#plotting squared error and cross validation error.	
plt.plot(lamb_val, sq_er, label='squared error')
plt.plot(lamb_val, cv_er, label='cross validation error')
plt.legend()
plt.show()

