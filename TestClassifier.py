'''
Created on 24 Oct 2014
@author: Etienne
'''

import numpy as np
import matplotlib.pyplot as plt




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

print X.shape

result = 0
print X
for i in xrange(X.shape[0]):
    temp = Y[i]*X[i]
#     print "Temp:",temp
    temp2 = 0
    for j in xrange(X.shape[0]):
        temp2 += Y[j]*X[j]
#         print "Product:",Y[j]*X[j]
#         print "Temp2",temp2
#     print "Temp2",temp2
    temp *= temp2
#     print "Temp:",temp
#     print "-------------------"
#     raw_input()

print temp
print (Y*X.T).T*temp2

print "End test one."

# print X,Y
print Y
np.dot(X,X.T)
print sum(Y*Y.dot(np.dot(X,X.T)))
# print np.dot(X1,X2)

plt.plot(X1.T[0],X1.T[1],"ro")
plt.plot(X2.T[0],X2.T[1],"bo")
plt.axis([0,10,0,6])
plt.show()

#We want the hyperplane to be a vertical line with x = 5