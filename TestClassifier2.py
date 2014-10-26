'''
Created on 25 Oct 2014
@author: Etienne
'''
import numpy as np

X = np.array([[1,2],[3,4]])
Y = np.array([1,-1])
result = 0

print np.array_repr(X)

innerSum = 0
for j in xrange(X.shape[0]):
    innerSum += Y[j]*X[j]
    print "Xi*Yi:",Y[j]*X[j]
print "Inner sum:",innerSum
print sum((X.T*Y).T)




print "--------------------------"
outerSum = 0
for i in xrange(X.shape[0]):
    outerSum += Y[i]*X[i] * innerSum
print "Outer sum:",outerSum


print sum((Y*X.T).T*innerSum)


print "---------------------"
# 
# print X.T
# print X.T*Y
# print X.dot(X)
# print (Y*Y*X).dot(X)

print np.inner(np.inner(Y,Y),X.dot(X))
print np.inner(X,X.T)
print X.dot(X)


formula = (X.T*Y).T.dot((X.T*Y).T)
formula = (X.T*Y).T.dot((X.T*Y).T)
formula2 = Y*Y.T*(X.dot(X)).T
print formula
print sum(formula)
# print sum(formula2)

# A*sum(B) = A.dot(B) 

