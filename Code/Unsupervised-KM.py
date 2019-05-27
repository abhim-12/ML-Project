from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv('xclara.csv')
print(data.shape)
print(data.head())

f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

#function to compute Eucledian Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)   #Returns norm of matrix/vector

#Number of centroids
k = 3
C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
#Centroids Array
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print('Assumed Centroid Values: ')
print(C)

#Plotting values
plt.scatter(f1, f2, c='#050505', s=7)
#Plotting initial Centroid values over the clusters
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

C_old = np.zeros(C.shape)       #Generates a Matrix of zero's of shape C.shape
clusters = np.zeros(len(X))
error = dist(C, C_old, None)

#Run untill the centroid values do not change : error=0
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        #Finding the least distance
        cluster = np.argmin(distances)
        #Assign least distance to list clusters
        clusters[i] = cluster
    #Copying the values of the Centroid to C_old
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        #Find average of points
        C[i] = np.mean(points, axis=0)
    #Find new distance values
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
#Plot the new Values over the cluster
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()
print('New Centroid Values: ')
print(C)
