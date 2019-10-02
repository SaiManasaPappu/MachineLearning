import cv2
import random
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from numpy import linalg as LA
#np.random.seed(0)

def assign(d,n,x,K,S):
    temp = np.zeros((n,K))
    for k in range(K):
		temp[:,k]=(np.linalg.norm(x.T-S[:,k],axis=1))
    return np.argmin(temp,axis=1)

def nextS(d,n,x,k,cluster):
	nextC = np.zeros((d,k))
	for i in range(n):
		nextC[:,int(cluster[i])] = nextC[:,int(cluster[i])] + x[:,i]
	for i in range(k):
		nextC[:,i] = nextC[:,i]/np.sum(np.count_nonzero(cluster==i))
	return nextC

def printerror(n,x,cluster,S,iteration):
	error = 0
	for i in range(n):
		error = error + (LA.norm(S[:,int(cluster[i])]-x[:,i]))**2
	print("Iteration: "+str(iteration) + "    Error:  "+str(error))

def disp(d,n,x,k,S,cluster):
	if d==2:
		# randomly generate colors
		colors = [numpy.random.rand(3,) for i in range(k)]
		for i in range(n):
			plt.scatter(x[0,i],x[1,i],c=colors[int(cluster[i])])
		inver = (np.ones(3)-colors)
		for i in range(k):
			plt.scatter(S[0,:],S[1,:],c=inver)
		plt.grid()
		plt.show()
		
def k_means(d,n,x,k,epsilon):
	
	# intialisation of centroids
	# Taking random Samples from input itself
	# S is the set of all centroids, i.e, d x k  matrix
	S = np.zeros((d,k))
	for i in range(k):
		S[:,i] = np.mean(x[: , i*(n/k):(i+1)*(n/k)-1 ], axis = 1)
	#print(S)
	# cluster is a 1xn vector
	# ith entry respresent centroid index of x[i] in set S
	cluster = assign(d,n,x,k,S)
	nextC = nextS(d,n,x,k,cluster)
	iteration = 0
	#printerror(n,x,cluster,S,iteration)
	
	while (LA.norm(nextC - S))**2 >= epsilon :
		S = nextC
		iteration = iteration + 1
		cluster = assign(d,n,x,k,S)		
		#printerror(n,x,cluster,S,iteration)
		nextC = nextS(d,n,x,k,cluster)
	
	res = x
	for i in range(n):
		res[:,i]=S[:,int(cluster[i])]
	print("\nFinal Centroids\n")
	print(S)
	disp(d,n,x,k,S,cluster)
	return res
	

k = int(input("Enter k :  "))
epsilon = float(input("Enter epsilon:  "))
path = raw_input("Full image path :  ")

# reading and reshaping image
img = cv2.imread(str(path))
#str(path))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
vectorized = (img.reshape((-1,3))).T
vectorized = np.float32(vectorized)

n = len(vectorized[0,:])
d = 3

result = ((k_means(d,n,vectorized,k,epsilon)).T).reshape((img.shape))

figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result)
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()
