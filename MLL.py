import numpy as np
import matplotlib.pyplot as plt

def seq(start,stop,step):
    sq = []
    for i in range(start,stop+1,step):
        sq.append(i)
    return sq

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def featureNormalize(X):
    # This function normalize X and returns the mean and sd
    mu = X.mean(axis=0)
    X_norm = np.subtract(X,mu)
    sigma = X_norm.std(axis=0,ddof=1)
    X_norm = np.divide(X_norm,sigma)
    returnobj = np.empty((X.shape[0]+2,X.shape[1]))
    returnobj[np.array(seq(0,X.shape[0]-1,1)),:] = X_norm
    returnobj[-2,:] = mu
    returnobj[-1,:] = sigma
    return returnobj

def LinearRegCost(X,y,theta,reg):
    # This functions find the cost and gradient with a linear
    # regression model
    m = y.shape[0]
    h = np.dot(X,theta)
    df = h-y
    J = ((1/(2*m))*np.dot(df.T,df))+(reg/(2*m))*sum(theta[1:]**2)
    grad = (1/m)*np.dot(X.T,df)
    grad[1:] += (reg/m)*theta[1:]
    
    #store results in a return object
    returnobj = np.empty(grad.shape[0]+1)
    returnobj[0] = J
    returnobj[1:] = grad
    return returnobj

def LogisticRegCost(X,y,theta,reg):
    # This functions find the cost and gradient with a logistic
    # regression model
    m = y.shape[0] # save length of y
    h = sigmoid(np.dot(X,theta))
    df = h-y
    J = ((1.0/m)*sum(-y*np.log(h)-(1.0-y)*np.log(1.0-h)))+(reg/(2.0*m))*sum(theta[1:]**2)
    grad = (1.0/m)*np.dot(X.T,df)
    grad[1:] += (reg/m)*theta[1:]

    #store results in a return object
    returnobj = np.empty(grad.shape[0]+1)
    returnobj[0] = J
    returnobj[1:] = grad
    return returnobj

def LogisticPredict(X,theta):
    p = sigmoid(np.dot(X,theta))
    p = p > 0.5
    return 1.0*p

def LinearPredict(X,theta):
    return np.dot(X,theta)

def gradientDescent(CostFun,X,y,theta,alpha,reg):
    maxit = 100 * theta.shape[0]
    
    for it in range(1,maxit):
        res = CostFun(X,y,theta,reg)
        J = res[0]
        grad = res[1:]

        theta = theta - alpha * grad
    
    return theta

def plotDecisionBoundary(X,y,theta):
    if theta.shape[0] < 3:
        thetaold = theta
        theta = np.ones(3)
        theta[1] = thetaold[1]
        theta[2] = thetaold[2]
        
    plt.plot(X[y==0,1],X[y==0,2],'ro')
    plt.plot(X[y==1,1],X[y==1,2],'bx')
    plotx = np.linspace(np.min(X[:,1])-2.0, np.max(X[:,1])+2.0, num=100)
    ploty = (-1.0/theta[2])*(theta[1]*plotx + theta[0])
    plt.plot(plotx,ploty)
    plt.axis([np.min(X[:,1])-1,np.max(X[:,1])+1,np.min(X[:,2])-1,np.max(X[:,2])+1])
    plt.show()














