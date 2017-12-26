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
	Exercise 2, Marc Toussaint—May 2, 2017 2
	np.meshgrid is pretty annoying!
	"""
	dom = np.linspace(start, end, num)
	X0, X1 = np.meshgrid(dom, dom)
	
	return np.column_stack([X0.flatten(), X1.flatten()])

def sigmoid(x , beta):
	return 1/(1+math.exp(-1*(dot(x,beta))))	
	
	
def negloglike(X , y , lamb , beta):
	loss = np.zeros(np.shape(X)[0])
	for i in range(0 , np.shape(X)[0]):
		loss[i] = (-1)*((y[i]*math.log(sigmoid(X[i],beta)))+((1-y[i])*math.log(1-sigmoid(X[i],beta))))
	print(loss)	
	l = np.sum(loss)+lamb*(np.sum(dot(beta,beta)))
	print(l)	
	
def wele(x , beta):
	return sigmoid(x,beta)*(1-sigmoid(x,beta))
	
	
###############################################################################
# load the data
data = np.loadtxt("data2Class.txt")
print ("data.shape:", data.shape)

# split into features and labels
X, y = data[:, :2], data[:, 2]
print ("X.shape:", X.shape)
print ("y.shape:", y.shape)
print (np.shape(X)[0])

# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # the projection arg is important!
ax.scatter(X[:, 0], X[:, 1], y, color="red")
ax.set_title("raw data")
plt.draw() # show, use plt.show() for blocking



lamb = 10
beta = np.ones(np.shape(X)[1])

print (beta)

#a = [0.1,0.1]

#print(sigmoid(a,beta))

#negloglike(X,y,lamb,beta)
p = np.zeros(np.shape(y)[0])
for j in range(0,np.shape(y)[0]):
	p[j] = sigmoid(X[j],beta)
	
#print (p)	

firstder = mdot((X.T),(p-y))+(2*lamb*beta)

print (firstder)

wtemp = np.zeros(np.shape(y)[0])
W = np.identity(np.shape(y)[0])
for i in range(np.shape(y)[0]):
	#wtemp[i] = wele(X[i],beta)
	W[i][i] = wele(X[i],beta)






print (W)


secondder = np.zeros((2, 2))
print(secondder)
print(2*lamb*np.identity(np.shape(beta)[0]))
print(np.shape(X))
print(np.shape(W))

secondder = mdot((X.T),W,X)+2*lamb*np.identity(np.shape(beta)[0])

print(secondder)

for i in range(0,10):
	beta = beta - np.dot(inv(secondder),firstder)
	print(beta)
	
# prep for prediction
X_grid = grid2d(-3, 3, num=30)
print ("X_grid.shape:", X_grid.shape)

# Predict with trained model

y_grid = np.zeros(np.shape(X_grid)[0])
for j in range(0,np.shape(X_grid)[0]):
	y_grid[j] = sigmoid(X_grid[j],beta)

print ("Y_grid.shape", y_grid.shape)
# vis the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # the projection part is important
ax.scatter(X_grid[:, 0], X_grid[:, 1], y_grid) # don’t use the 1 infront
ax.scatter(X[:, 0], X[:, 1], y, color="red") # also show the real data
ax.set_title("predicted data")
plt.show()







