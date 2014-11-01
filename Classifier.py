'''
Created on 23 Oct 2014
@author: Etienne
'''

import numpy as np
import cvxopt
from math import cos,atan,pi,sqrt,tanh
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self):
        pass
    def calcLinearKernel(self,X):
        return (X.dot(X.T))
    def calcPolyKernel(self,X):
        return (X.dot(X.T) + 1)**2
    def calcSigKernel(self,X):
        return np.tanh(X.dot(X.T) + 1)
    def calcExpKernel(self,X):
        d = X[:,None,...] - X[None,...]
        return np.exp( -0.5*(d**2.0).sum(axis=-1) )
    def calcAlpha(self,alpha,kernel,Y,sign=1.0): #for use with scipy solvers
        return sign*(sum(alpha) -0.5*alpha.T*np.outer(Y,Y)*kernel*alpha)
    def calcNormal(self,X,Y,alpha):
        return (alpha.T*Y).dot(X)
    def calcOffset(self,kernel,Y,alpha):
        support = self.findIndexes(alpha)
        #Following couple of lines was the initial way for linearly separable data
        #equivalent to new method and unnecessary, I keep it to remember how I got there
#         innerSum =  (alpha[support].T*Y[support]).dot(X[support]) 
#         outerSum = sum(Y[support] - innerSum.dot(X[support].T)[0])
        innerSum =  (alpha[support].T*Y[support])
        outerSum = sum(Y[support] - innerSum.dot(kernel[support].T[support].T)[0])
        b = outerSum/len(Y[support])
        return b
    def solveAlpha(self,kernel,Y):
        dim = kernel.shape[0]
        
        P = cvxopt.matrix(np.outer(Y,Y)*kernel)
        q = cvxopt.matrix(-np.ones(dim))
        #a>=0 : Gx <= h
        G = cvxopt.matrix(-np.identity(dim))
        h = cvxopt.matrix(np.zeros(dim))
        #sum(a[i]y[i]) : Ax=b
        A = cvxopt.matrix(Y, (1,dim))
        b = cvxopt.matrix(0.0)        
        
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.array(solution['x'])
    def findIndexes(self,alpha):
        return (alpha>1e-4).T[0]
    def plotLinearBoundary(self,w,b):
        #Calc intersections with axis
        theta = atan(w[1]/w[0])
        normW = sqrt(w[0]**2+w[1]**2)
        x = (b/normW)/cos(theta)
        point1 = (x,0)
        
        theta = 90*pi/180.0 - theta
        y = abs((b/normW)/cos(theta))
        point2 = (0,y)
        
        plt.plot(point1,point2,"b-")