import MLL as ml
import numpy as np

ndim = 10
mdim = 1000
num_labels = 3
X = np.random.rand(mdim,ndim)*10 + 10
X[:,0] = np.ones(mdim)



print("X:")
print(X)

y =  np.random.randint(0,high=num_labels,size=mdim) #np.array([2,1,2,0,1,2,0,1,2])
print("y:")
print(y)

reg = 1
alpha = 0.03

all_theta = ml.oneVsAll(X,y,num_labels,reg,alpha)
print("theta:")
print(all_theta)

pred = ml.oneVsAllPredict(X,all_theta)
print("Precision:")
print(np.mean((pred==y))*100)


