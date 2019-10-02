import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv

np.random.seed(0)

def binomial(n,x):
	p = (np.count_nonzero(x == 1))/n
	x_generated = np.random.binomial(n,p,n)
	print("Binomial estimate: p = " + str(p)) 
	plot(n,x,x_generated,'binomial')
	
def poisson(n,x):
	lam = np.mean(x)
	if lam<=0:
		print("Poisson doesn't fit since lam <=0")
		return
	x_generated = numpy.random.poisson(lam, n)
	print("Poisson estimate: Lambda = " +str(lam))
	plot(n,x,x_generated,'poisson')
	
def exponential(n,x):
	theta = np.mean(x)
	if theta<=0:
		print("Exponential doesn't fit since theta <=0")
		return
	x_generated = numpy.random.exponential( theta, n)
	print("Exponential estimate: Theta = " + str(theta))
	plot(n,x,x_generated,'exponential')

def gaussian(n,x):
	mean = np.mean(x)
	stddev = np.std(x)
	x_generated = np.random.normal(mean, stddev, n)
	print("Gaussian estimates : Mean = " +str(mean) + " Standard Deviation = " + str(stddev))
	plot(n,x,x_generated,'gaussian')
	
def laplacian(n,x):
	mean = np.mean(x)
	mu = np.median(x)
	sigma = np.sqrt(2)*np.sum(np.absolute(np.array(x) -mean))/n
	x_generated = numpy.random.laplace(mu,sigma, n)
	print("Laplacian estimates : mu = " +str(mu)+" sigma = "+str(sigma))
	plot(n,x,x_generated,'laplacian')
	
	
def plot(n,x,y,s):	
	# plotting cdf curves of x and generated samples on same graph
	global count
	minval = min(x)
	maxval = max(x)
	len = 100
	x_val = np.linspace(minval, maxval, len)
	xy = np.zeros(len)
	yy = np.zeros(len)
	for i in range(len):
		xy[i] = np.count_nonzero(x<=x_val[i])
		yy[i] = np.count_nonzero(y<=x_val[i])
	plt.plot(x_val,yy,label=str(s+' generated'))
	if count == 0:
		plt.plot(x_val,xy,label='data')
		count = 1
		
	
count = 0



n = int(input("Enter n :  "))
path = raw_input("Full csv file path (containing 1 row, n columns with no trailing empty cells) :  ")
x = []

with open(str(path))as f:
  data = csv.reader(f)
  for row in data:
        for element in row:
			x.append(float(element))

x = np.array(x)

binomial(n,x)
poisson(n,x)
exponential(n,x)
gaussian(n,x)
laplacian(n,x)
plt.legend(loc='best')
plt.grid()
plt.show()
