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
    outerSum += Y[i]*X[i]
print "Outer sum:",outerSum

print outerSum.dot(innerSum)


print sum((Y*X.T).T.dot(innerSum))


print "---------------------"

print sum(sum(   (Y**2)*(X**2) ))
print Y*Y*X.dot(X)
print (Y*Y)*(X.T).dot((X.T))
print Y*(Y*X.T).dot((X.T))
print sum(   sum(  Y*(Y*X.T).dot((X.T))  )  )
print sum( sum(np.outer(Y,Y)*X.T.dot(X.T)) )



print Y*(Y*X.T)
print np.outer(Y,Y)*X.T


#A.dot(B*C) = ?


# A*sum(B) = A.dot(B) 

