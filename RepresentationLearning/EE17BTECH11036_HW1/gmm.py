import numpy.linalg as LA
import random
import numpy as np
import numpy.random
from numpy import linalg as LA
#np.random.seed(0)

def generate_data(dim,K,N,pi):
	mean = np.reshape(np.linspace(0,K*dim,K*dim),(K,dim))
	cov = np.array( [np.eye(dim) for _ in range(K)] )
	X = np.zeros((N*K,dim))
	for i in range(K):
		X[i*N:(i+1)*N,:] = np.random.multivariate_normal(mean[i,:],cov[i],N)
	return mean,cov,X

def init_params(K,dim):
	pi = np.random.uniform(0.1,0.9,K)
	pi = pi/sum(pi)
	mu = np.random.randn(K,dim)
	cov = np.array( [np.random.uniform(0.5,1.0)*np.eye(dim) for _ in range(K)] )
	return pi, mu, cov

def pdf(xn, mu, cov):
	return np.exp(-0.5 * np.matmul((xn-mu), np.dot(xn-mu , LA.inv(cov)) )) / np.sqrt(((2.0*np.pi)**len(xn)) * LA.det(cov) )

def likelihood(K, pi, mu, cov, X):
	N = X.shape[0]
	l = 0
	for n in range(N):
		temp = 0
		for k in range(K):
			temp = temp + pi[k]*pdf(X[n],mu[k,:],cov[k])
		l = l + np.log(temp)
	return l
			

def compute_gama(K, pi, mu, cov, X):
	N = X.shape[0]
	num = np.zeros((N,K))
	denom = np.zeros((N,1))
	for n in range(N):
		for k in range(K):
			num[n,k] = pi[k] * pdf(X[n],mu[k,:],cov[k])
			denom[n,0] = denom[n,0] + pi[k]*pdf(X[n],mu[k,:],cov[k])
	denom = np.matmul(denom, np.ones((1,K)))
	gama = num/denom
	return gama

def update_params(K, gama, X):
	N = X.shape[0]
	dim = X.shape[1]
	Nk = np.sum(gama, axis=0)
	mu_next = np.zeros((K,dim))
	cov_next = np.array([np.zeros((dim,dim)) for _ in range(K)])
	pi_next = Nk/N
	for k in range(K):
		mu_next[k,:] = np.dot(np.reshape(gama[:,k],(1,N)),X)/Nk[k]
		for n in range(N):
			cov_next[k] = cov_next[k] + (gama[n,k]/Nk[k]) * np.dot(np.reshape( X[n,:]-mu_next[k,:],(dim,1)), np.reshape( X[n,:]-mu_next[k,:],(1,dim)) )
	return pi_next, mu_next, cov_next



def gmm(K,dim,X,epsilon):
	pi, mu, cov = init_params(K,dim)
	current_likelihood = likelihood(K, pi, mu, cov, X)
	gama = compute_gama(K, pi, mu, cov, X)
	pi_next, mu_next, cov_next = update_params(K, gama, X)
	next_likelihood = likelihood(K, pi_next, mu_next, cov_next, X)
	iteration = 0

	while((next_likelihood - current_likelihood) > epsilon):
		print("Iteration: "+str(iteration)+"   Error: "+str(next_likelihood - current_likelihood)+"   Likelihood: "+str(current_likelihood))
		pi, mu, cov =  pi_next, mu_next, cov_next
		current_likelihood = next_likelihood
		gama = compute_gama(K, pi, mu, cov, X)
		pi_next, mu_next, cov_next = update_params(K, gama, X)
		next_likelihood = likelihood(K, pi_next, mu_next, cov_next, X)
		iteration+=1

	return pi, mu, cov

K = int(input("Enter K: "))
N_per_mixture = int(input("Samples per mixture : "))
dim = int(input("Dimension of each sample: "))
epsilon = float(input("Epsilon for convergence: "))

pi = np.ones(K)/K
mean,cov,X = generate_data(dim,K,N_per_mixture,pi)
pi_est, mu_est, cov_est = gmm(K,dim,X,epsilon)
print("=====================================================")
print("\nNOTE THAT THE ORDER IN WHICH PARAMETERS APPEAR MIGHT DIFFER\n")
print("\nEstimated means\n")
print(mu_est)
print("\nEstimated Covariances\n")
print(cov_est)
print("\nEstimated weights\n")
print(pi_est)



# def assign(d,n,x,K,S):
#     temp = np.zeros((n,K))
#     for k in range(K):
# 		temp[:,k]=(np.linalg.norm(x.T-S[:,k],axis=1))
#     return np.argmin(temp,axis=1)

# def nextS(d,n,x,k,cluster):
# 	nextC = np.zeros((d,k))
# 	for i in range(n):
# 		nextC[:,int(cluster[i])] = nextC[:,int(cluster[i])] + x[:,i]
# 	for i in range(k):
# 		nextC[:,i] = nextC[:,i]/np.sum(np.count_nonzero(cluster==i))
# 	return nextC

# def k_means(d,n,x,k,epsilon):
# 	S = np.zeros((d,k))
# 	for i in range(k):
# 		S[:,i] = np.mean(x[: , i*(n/k):(i+1)*(n/k)-1 ], axis = 1)
# 	cluster = assign(d,n,x,k,S)
# 	nextC = nextS(d,n,x,k,cluster)
	
# 	while (LA.norm(nextC - S))**2 >= epsilon :
# 		S = nextC
# 		cluster = assign(d,n,x,k,S)		
# 		nextC = nextS(d,n,x,k,cluster)
# 		return S
