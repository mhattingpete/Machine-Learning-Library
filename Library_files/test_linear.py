import MLL as ml
import numpy as np

X = np.zeros((5,3))
X[:,0] = np.ones(5)
X[:,1] = np.array(ml.seq(1,5,1))
X[:,2] = np.array(ml.seq(1,5,1))**2

print("X:")
print(X)

res = ml.featureNormalize(X[:,1:])
mu = res[-2,:]
sigma = res[-1,:]
X[:,1:] = res[np.array(ml.seq(0,X.shape[0]-1,1)),:]
print("X:")
print(X)
print("mu:")
print(mu)
print("sigma:")
print(sigma)

y = np.array(ml.seq(6,10,1))
print("y:")
print(y)

theta0 = np.array([0,0,0])
theta0 = theta0.T
alpha = 0.3
reg = 1

theta = ml.gradientDescent(ml.LinearRegCost,X,y,theta0,alpha,reg)
print("theta:")
print(theta)

res = ml.LinearRegCost(X,y,theta,reg)
J = res[0]
grad = res[1:]
print("J:")
print(J)
print("grad:")
print(grad)

