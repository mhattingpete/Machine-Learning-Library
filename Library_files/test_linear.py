import MLL as ml
import numpy as np
from scipy import optimize

mdim = 5
ndim = 3
X = np.zeros((mdim,ndim))
X[:,0] = np.ones(mdim)
X[:,1] = np.array(ml.seq(1,5,1))
X[:,2] = np.array(ml.seq(1,5,1))**2

print("X:")
print(X)

res = ml.featureNormalize(X[:,1:])
mu = res.mean
sigma = res.sd
X[:,1:] = res.X
print("X:")
print(X)
print("mu:")
print(mu)
print("sigma:")
print(sigma)

y = np.random.rand(mdim) #np.array(ml.seq(6,10,1))
print("y:")
print(y)

theta0 = np.zeros(X.shape[1])
theta0 = theta0.T
alpha = 0.3
reg = 1

theta = ml.gradientDescent(ml.LinearRegCost,X,y,theta0,alpha,reg)
print("theta:")
print(theta)

res = ml.LinearRegCost(theta,X,y,reg)
J = res.J
grad = res.grad
print("J:")
print(J)
print("grad:")
print(grad)

print("Testing... \n")
args = (X,y,reg)
theta2 = ml.fmincg(ml.LinearRegCost,theta0,*args)
print("Comparison of optimzation algorithms:")
print("Old theta:",theta,", New theta:",theta2)

ml.plotDataFit(X,y,theta)

