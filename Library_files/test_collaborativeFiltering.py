import MLL as ml
import numpy as np
import matplotlib.pyplot as plt

Y=np.array([[4,5,4,0],[0,3,0,0],[0,4,0,0],[0,3,0,0],[0,3,0,0],[0,5,0,0]])
R=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]])
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 3 # just select a number
X = np.random.randn(num_movies,num_features)
Theta = np.random.randn(num_users,num_features)

print("Y:",Y)
print("R:",R)

initial_parameters = np.concatenate((np.reshape(X,X.size),np.reshape(Theta,Theta.size)))
reg = 10

args = (Y,R,num_users,num_movies,num_features,reg)
params = ml.fmincg(ml.cofiCostFun,initial_parameters,*args)
X = np.reshape(params[0:((num_movies*num_features)-1)],(num_movies,num_features))
Theta = np.reshape(params[(num_movies*num_features):],(num_users,num_features))

print("X:",X)
print("Theta:",Theta)
