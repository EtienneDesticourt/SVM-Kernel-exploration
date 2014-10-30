'''
Created on 23 Oct 2014
@author: Etienne
'''

import numpy as np
import cvxopt
from math import cos,atan,pi,sqrt
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self):
        pass
    def calcKernel(self,X,Y):
        return np.outer(Y,Y)*(X.dot(X.T))
    def calcAlpha(self,alpha,kernel,sign=1.0): #for use with scipy solvers
        return sign*(sum(alpha) -0.5*alpha.T*kernel*alpha)
    def calcNormal(self,X,Y,alpha):
        return (alpha.T*Y).dot(X)
    def calcOffset(self,X,Y,alpha):
        support = self.findIndexes(alpha)
        innerSum =  (alpha[support].T*Y[support]).dot(X[support])
        outerSum = sum(Y[support] - innerSum.dot(X[support].T)[0])
        b = outerSum/len(X[support])
        return b       
    def solveAlpha(self,kernel,y):
        dim = kernel.shape[0]
        
        P = cvxopt.matrix(kernel)
        q = cvxopt.matrix(-np.ones(dim))
        #a>=0 : Gx <= h
        G = cvxopt.matrix(-np.identity(dim))
        h = cvxopt.matrix(np.zeros(dim))
        #sum(a[i]y[i]) : Ax=b
        A = cvxopt.matrix(y, (1,dim))
        b = cvxopt.matrix(0.0)        
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.array(solution['x'])
    def findIndexes(self,alpha):
        return (alpha>1e-4).T[0]
    def findHyperplane(self,X,Y):
        kernel = self.calcKernel(X, Y)
        alpha = self.solveAlpha(kernel, Y)
        w = self.calcNormal(X,Y,alpha)
        return (w,alpha)
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