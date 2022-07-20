from turtle import shape
import numpy as np
import os
x=np.array([[-1],[2],[-3],[4]])
y=np.array([[1,2,3,4],[1,2,3,4]])
print(np.dot(y,x))
x=(x>0)*x
x=x.T
print(x*np.array([1,3,2,4]))