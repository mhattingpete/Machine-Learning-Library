import MLL as ml
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[14.3004,14.5584,14.2249,12.0403,13.0793,21.7271,
               12.4766,19.5826,23.3399,18.2612,4.7526],
              [15.2664,15.4869,15.8612,13.3448,9.3479,4.1262,
               14.4594,10.4116,16.2989,17.9783,24.3504]]) # training set
Xval = np.array([[14.2249,14.2933,14.9525,14.7340,19.2895,
                  8.7386,0.3079,28.5418,19.0350,14.3758,8.7781],
                 [16.2842,15.9914,14.3641,15.8183,10.6757,
                  16.7958,5.3914,21.5998,12.0289,23.3560,16.6895]]) # validation set
yval = np.array([0,0,0,0,1,1,1,1,1,1,1])

X = X.T
Xval = Xval.T
yval = yval.T

print("X:","\n",X,"\n")
print("Xval:","\n",Xval,"\n")
print("yval:",yval,"\n")

res = ml.estimateGaussian(X)
mu = res.mu
sigma2 = res.sigma2

print("mu:",mu,"\n")
print("sigma2:",sigma2,"\n")

p = ml.multivariateGaussian(X,mu,sigma2)
print("p:",p,"\n")

pval = ml.multivariateGaussian(Xval,mu,sigma2)
res = ml.selectThreshold(yval,pval)
epsilon = res.epsilon
print("epsilon:",epsilon,"\n")

F1 = res.F1
print("F1:",F1,"\n")

outliers = np.where(p<epsilon)[0]
print("outliers:",outliers,"\n")

print("Now we test the function that does it all.")
res = ml.findAnomalies(X,Xval,yval)
outliers2 = res.out_index
F12 = res.F1
epsilon2 = res.epsilon

print("Comparison of values:")
print("Epsilon old:",epsilon,", Epsilon new:",epsilon2)
print("F1 old:",F1,", F1 new:",F12)
print("Outliers old:",outliers,", Outliers new:",outliers2)

plt.plot(X[:,0],X[:,1],'bx')
plt.axis([0,30,0,30])
plt.plot(X[outliers,0],X[outliers,1],'ro')
plt.show()
