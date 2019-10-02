import cv2
import random
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from numpy import linalg as LA
import csv

np.random.seed(0)

def pca(d,n,xin):
	x = xin*1.0
	for i in range(d):
		mean_i = np.mean(x[i,:])
		x[i,:] = x[i,:] - mean_i
	cxx = (np.matmul(x,x.T))/n
	w, v = LA.eig(cxx)
	y = np.matmul(v.T,x)
	cyy = (np.matmul(y,y.T))/n
	print("cyy")
	print(cyy)
	var = np.zeros(d)
	for i in range(d):
		var[i] = np.std(y[i,:])
	plt.plot(var)
	plt.show()
	

d = int(input("Enter d :  "))
n = int(input("Enter n :  "))
path = raw_input("Full csv file path (dxn elements with no trailing empty cells) :  ")

x = []

with open(str(path))as f:
  data = csv.reader(f)
  for row in data:
        for element in row:
			x.append(float(element))
	
x = np.array(x)
x = x.reshape(d,n)
pca(d,n,x)
