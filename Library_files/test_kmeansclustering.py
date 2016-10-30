import MLL as ml
import numpy as np

mdim = 5
ndim = 2
#X = np.random.rand(mdim,ndim)
X = np.array([[5.2017,22.9198],[11.7281,24.3165],[24.9414,20.1549],
              [24.1009,29.8406],[1.8141,21.6717],[11.9777,21.0622],
              [15.8063,23.7241],[12.5040,21.9812],[19.7058,24.8969],
              [18.8392,23.3949]])

K = 2
initial_centroids = np.array([[3,20],[30,20]])

print("X:")
print(X)
print("Centroids:")
print(initial_centroids)

idx = ml.findClosestCentroids(X,initial_centroids)

print("Centroid groups:")
print(idx)

centroids = ml.computeCentroids(X,idx,K)

print("New centroid positions:")
print(centroids)

max_iters = 10
res = ml.runkMeans(X,initial_centroids,max_iters)
centroids = res.centroids
idx = res.idx

print("Centroids:")
print(centroids)
print("idx:")
print(idx)






