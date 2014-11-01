'''
Created on 24 Oct 2014
@author: Etienne
'''

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from Classifier import Classifier
from math import tanh

#TEST 1

#We create two columns of points with 4 points in each for each class
cols = 2 #columns of points not cols of matrix
rows = 4
y0 = 1#Height of lowest point
distCols = 2#distance between columns
#Class 1
X1 = np.zeros((cols*rows,2))
for j in xrange(cols):
    for i in xrange(rows): 
        X1[i+j*rows][0] = distCols*(j+1)
        X1[i+j*rows][1] = i+y0
Y1 = np.ones(cols*rows)
# print Y1

#Class 2
X2 = np.zeros((cols*rows,2))
for j in xrange(cols):
    for i in xrange(rows):
        X2[i+j*rows][0] = distCols*(j+1) + distCols*2 # We put the second class further to the right
        X2[i+j*rows][1] = i+y0
Y2 = -np.ones(cols*rows)

X = np.concatenate((X1,X2))
Y = np.concatenate((Y1,Y2))

#TEST 2
Y = np.array([1.,1.,-1.,-1.,1.,1.,-1.,-1.,1.,1.,-1.,-1.,1.,1.,-1.,-1.]) 
#TEST 3  
Y = np.array([1.,1.,1.,1.,1.,1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.])
#TEST 4
# Y = np.array([1.,1.,-1.,-1.,1.,1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.])
#TEST 5: circle
Y = np.array([-1.,-1.,-1.,-1.,-1.,1.,1.,-1.,-1.,1.,1.,-1.,-1.,-1.,-1.,-1.])
#TEST 6
Y = np.array([1.,-1.,-1.,-1.,1.,1.,1.,-1.,1.,-1.,-1.,-1.,1.,1.,1.,-1.])
#TEST 7
Y = np.array([1.,1.,-1.,-1.,1.,1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.])
#TEST 8
Y = np.array([1.,1.,-1.,-1.,1.,1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.])




X1 = X[Y==1.]
X2 = X[Y==-1.]








C = Classifier()
kernel = C.calcExpKernel(X)
alpha = C.solveAlpha(kernel, Y)
b = C.calcOffset(kernel, Y, alpha)


sX = X[C.findIndexes(alpha)]
sAlpha = alpha[C.findIndexes(alpha)]
sY = Y[C.findIndexes(alpha)]

# def f(x,y): return (w[0]*x+w[1]*y + b) #old method
 
#For non linear classification we don't want to actually map x to a higher space
#because it's too computationally expensive
#so instead of calculating w (which requires the higher space) we use the kernel directly
def fPoly(x,y): 
    temp = sAlpha.T*sY
    r = 0
    for i in xrange(len(temp[0])):
        r += temp[0][i]*((sX[i][0]*x + sX[i][1]*y + 1)**2)    
    return r + b

def fSig(x,y):    
    temp = sAlpha.T*sY
    r = 0
    for i in xrange(len(temp[0])):
        r += temp[0][i]*(np.tanh(sX[i][0]*x + sX[i][1]*y + 1))    
    return r + b

def fExp(x,y):     
    temp = sAlpha.T*sY
    r = 0
    for i in xrange(len(temp[0])):
#         r += temp[0][i]*(np.exp( (sX[i][0]-x)**2 + (sX[i][1]-y)**2 ))    
        r += temp[0][i]*(np.exp( -0.5*( (sX[i][0]-x)**2 + (sX[i][1]-y)**2))) 
    return r + b

n = 256
x = np.linspace(0,10,n)
y = np.linspace(0,10,n)
Xm,Ym = np.meshgrid(x,y)

c = contourf(Xm, Ym, fExp(Xm,Ym), 8, alpha=.75, cmap='jet')
cont = contour(Xm, Ym, fExp(Xm,Ym), 8, colors='black', linewidth=.5)
plt.colorbar(c)
plt.plot(X1.T[0],X1.T[1],"ro")
plt.plot(X2.T[0],X2.T[1],"bo")
plt.axis([0,10,0,6])
plt.show()
