import MLL as ml
import numpy as np

ndim = 3
mdim = 20
X = np.zeros((mdim,ndim))
X[:,0] = np.ones(mdim)
X[:,1] = np.random.rand(mdim)*10 + 10#np.array(ml.seq(1,10,1))
X[:,2] = np.random.rand(mdim)*50 + 5

print("X:")
print(X)

y = np.array(1.0*(np.random.rand(mdim)>0.5))
print("y:")
print(y)

theta0 = np.zeros(ndim)
theta0 = theta0.T
alpha = 0.03
reg = 1

theta = ml.gradientDescent(ml.LogisticRegCost,X,y,theta0,alpha,reg)
print("theta:")
print(theta)
res = ml.LogisticRegCost(X,y,theta,reg)
J = res[0]
grad = res[1:]
print("J:")
print(J)
print("grad:")
print(grad)

pred = ml.LogisticPredict(X,theta)
print("Precision:")
print(np.mean((pred==y))*100)

ml.plotDecisionBoundary(X,y,theta)


