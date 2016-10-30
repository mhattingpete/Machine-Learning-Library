import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


##############################################
##############################################
##                 classes                  ##
##############################################
##############################################

class costobj:
    def __init__(self,cost,gradient):
        self.J = cost
        self.grad = gradient

class normalizeX:
    def __init__(self,X_norm,mu,sigma):
        self.X = X_norm
        self.mean = mu
        self.sd = sigma

class centroid_obj:
    def __init__(self,centroids,idx):
        self.centroids = centroids
        self.idx = idx

class gaussian_obj:
    def __init__(self,mu,sigma2):
        self.mu = mu
        self.sigma2 = sigma2

class threshold_obj:
    def __init__(self,Epsilon,F1):
        self.epsilon = Epsilon
        self.F1 = F1

class anomaly_obj:
    def __init__(self,outlier_index,F1,epsilon):
        self.out_index = outlier_index
        self.F1 = F1
        self.epsilon = epsilon

class costFun_obj:
    def __init__(self,JFun,gradFun):
        self.JFun = JFun
        self.gradFun = gradFun


##############################################
##############################################
##               functions                  ##
##############################################
##############################################

def seq(start,stop,step):
    sq = []
    for i in range(start,stop+1,step):
        sq.append(i)
    return sq

def featureNormalize(X):
    # This function normalize X and returns the mean and sd
    mu = X.mean(axis=0)
    X_norm = np.subtract(X,mu)
    sigma = X_norm.std(axis=0,ddof=1)
    X_norm = np.divide(X_norm,sigma)
    return normalizeX(X_norm,mu,sigma)


##############################################
##          Optimization algorithms         ##
##############################################

def gradientDescent(CostFun,X,y,theta,alpha,reg):
    maxit = 100 * theta.shape[0]
    
    for it in range(1,maxit):
        res = CostFun(theta,X,y,reg)
        J = res.J
        grad = res.grad
        theta = theta - alpha * grad
        if np.linalg.norm(grad,np.inf) < 1.0e-10:
            break
        
    return theta

def fmincg(costFun,theta,*args):
    def JFun(theta):
        return costFun(theta,*args).J

    def gradFun(theta):
        return costFun(theta,*args).grad

    res = optimize.fmin_cg(JFun, theta, fprime=gradFun)
    return res

##############################################
##            Supervised learning           ##
##############################################

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

##############################################
##           Linear Regression              ##

def LinearRegCost(theta,X,y,reg):
    # This functions find the cost and gradient with a linear
    # regression model
    m = y.shape[0]
    h = np.dot(X,theta)
    df = h-y
    J = ((1/(2*m))*np.dot(df.T,df))+(reg/(2*m))*sum(theta[1:]**2)
    grad = (1/m)*np.dot(X.T,df)
    grad[1:] += (reg/m)*theta[1:]
    
    return costobj(J,grad)

def LinearPredict(X,theta):
    return np.dot(X,theta)

def plotDataFit(X,y,theta):
    # plots data fit from a linear model
    plt.plot(X[:,1],y,'bx')
    ploty = np.dot(X,theta)
    plt.plot(X[:,1],ploty)
    plt.axis([np.min(X[:,1])-1,np.max(X[:,1])+1,np.min(ploty)-1,np.max(ploty)+1])
    plt.show()

##############################################
##           Logistic Regression            ##
    
def LogisticRegCost(theta,X,y,reg):
    # This functions find the cost and gradient with a logistic
    # regression model
    m = y.shape[0] # save length of y
    h = sigmoid(np.dot(X,theta))
    df = h-y
    J = ((1.0/m)*sum(-y*np.log(h)-(1.0-y)*np.log(1.0-h)))+(reg/(2.0*m))*sum(theta[1:]**2)
    grad = (1.0/m)*np.dot(X.T,df)
    grad[1:] += (reg/m)*theta[1:]

    return costobj(J,grad)

def LogisticPredict(X,theta):
    p = sigmoid(np.dot(X,theta))
    p = p > 0.5
    return 1.0*p

def oneVsAllPredict(X,all_theta):
    return np.argmax(sigmoid(np.dot(X,all_theta)),axis=1)

def plotDecisionBoundary(X,y,theta):
    # plots decision boundary for logistic model
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

def oneVsAll(X,y,num_labels,reg,alpha):
    # estimate groups based on multiple labels
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((n,num_labels))

    initial_theta = np.zeros(n)
    for c in range(0,num_labels):
        theta = gradientDescent(LogisticRegCost,X,np.array(1.0*(y==c)),initial_theta,alpha,reg)
        all_theta[:,c] = theta

    return all_theta



##############################################
##           Unsupervised learning          ##
##############################################


##############################################
##           K-means clustering             ##

def findClosestCentroids(X,centroids):
    # finds the closest centroid for all points in X
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0],1))
    err = np.zeros((X.shape[0],K))

    for i in range(0,X.shape[0]):
        for j in range(0,K):
            err[i,j] = np.linalg.norm(X[i,:] - centroids[j,:],2)**2
        idx[i] = np.argmin(err[i,:])

    return idx.T

def computeCentroids(X,idx,K):
    # finds the new centroids based on the mean in each group
    n = X.shape[1]
    centroids = np.zeros((K,n))

    for i in range(0,K):
        for j in range(0,n):
            temp = (idx==i)*X[:,j]
            temp = temp[temp!=0]
            centroids[i,j] = np.mean(temp)

    return centroids

def runkMeans(X,initial_centroids,max_iters):
    # unsupervised algortihm for finding K groups
    m = X.shape[0]
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    for i in range(0,max_iters):
        idx = findClosestCentroids(X,centroids) # find the group of the centroids
        centroids = computeCentroids(X,idx,K) # find the new centroids

    return centroid_obj(centroids,idx)

        
##############################################
##           Anomaly Detection              ##

def estimateGaussian(X):
    # estimate gaussian parameters
    mu = np.mean(X,axis=0)
    sigma2 = np.var(X,axis=0)
    return gaussian_obj(mu,sigma2)

def multivariateGaussian(X,mu,sigma2):
    k = mu.shape[0]
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)

    X = np.subtract(X,mu)
    p = np.dot((2*np.pi)**(-k/2)*np.linalg.det(sigma2)**(-0.5),
    np.exp(-0.5*np.sum(np.multiply(np.dot(X,np.linalg.inv(sigma2)),X),axis=1)))
    return p

def selectThreshold(yval,pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    epsilon = np.linspace(np.min(pval),np.max(pval),1001)
    for i in range(0,1001):
        pred = pval < epsilon[i]
        tp = sum((pred == 1)*(yval == 1))
        fp = sum((pred == 1)*(yval == 0))
        fn = sum((pred == 0)*(yval == 1))

        if (tp+fp) == 0:
            prec = np.nan
        else:
            prec = tp/(tp+fp)
        if (tp + fn) == 0:
            rec = np.nan
        else:
            rec = tp/(tp+fn)
        if (prec + rec) == 0:
            F1 = np.nan
        else:
            F1 = (2*prec*rec)/(prec+rec)
            
        if F1 > bestF1:
            bestF1 = F1
            bestepsilon = epsilon[i]

    return threshold_obj(bestepsilon,bestF1)

def findAnomalies(X,Xval,yval):
    # A function that finds the anomalies for a dataset
    # and finds the best F1 and epsilon for the validation set.
    res = estimateGaussian(X)
    mu = res.mu
    sigma2 = res.sigma2
    p = multivariateGaussian(X,mu,sigma2)
    pval = multivariateGaussian(Xval,mu,sigma2)
    res = selectThreshold(yval,pval)
    epsilon = res.epsilon
    F1 = res.F1
    outliers = np.where(p<epsilon)[0]
    return anomaly_obj(outliers,F1,epsilon)

##############################################
##        Collaborative filtering           ##

def cofiCostFun(params, Y, R, num_users, num_movies, num_features, reg):
    X = np.reshape(params[0:((num_movies*num_features)-1)],(num_movies,num_features))
    Theta = np.reshape(params[(num_movies*num_features):],(num_movies,num_features))
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    J = (1/2)*sum(sum(R*(np.dot(X,Theta)-Y)**2))+(reg/2)*sum(sum(Theta**2))+(reg/2)*sum(sum(X**2))

    for i in range(0,num_movies):
        idx = np.where(R[i,:]==1)
        Theta_temp = Theta[idx,:]
        Y_temp = Y[i,idx]
        X_grad[i,:] = np.dot((np.dot(X[i,:],Theta_temp.T)-Y_temp),Theta_temp)+reg*X[i,:]
        
    for j in range(0,num_users):
            idx = np.where(R[:,j]==1)
            Y_temp = Y[idx,j]
            X_temp = X[idx,:]
            Theta_grad[i,:] = np.dot((np.dot(X_temp,Theta[j,:].T)-Y_temp).T,X_temp)+reg*Theta[j,:]

    grad = np.concatenate((np.reshape(X_grad,X_grad.size),np.reshape(Theta_grad,Theta_grad.size)))
    return costobj(J,grad)




