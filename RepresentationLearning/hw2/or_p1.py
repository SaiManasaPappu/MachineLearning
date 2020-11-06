import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import numpy.linalg as LA

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
    
def sigmoid_gradient(x):
    return x*(1.0-x)

def initialize_parameters(n_x, n_h, n_y):
    alpha = np.random.randn(n_h, n_x)
    alpha_0 = np.zeros((n_h, 1))
    beta = np.random.randn(n_y, n_h)
    beta_0 = np.zeros((n_y, 1))
    return alpha, alpha_0, beta, beta_0
   
def forward_propagation(X,  alpha, alpha_0, beta, beta_0):
    Z = sigmoid( np.dot(alpha,X) + alpha_0 )
    Y_hat = sigmoid( np.dot(beta,Z) + beta_0 )
    return Z, Y_hat
    
def cost(Y_hat, Y):
    m = Y.shape[1]
    cst = (1.0/m)*LA.norm(Y-Y_hat)**2
    return cst
    
def backward_propagation(X, Y,  alpha, alpha_0, beta, beta_0):
    m = X.shape[1]
    Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
    dZ2 = Y_hat - Y
    d_beta = (1.0/m)*np.dot(dZ2,Z.T)
    d_beta_0 = (1.0/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(beta.T,dZ2),sigmoid_gradient(Z))
    d_alpha = (1.0/m)*np.dot(dZ1,X.T)
    d_alpha_0 = (1.0/m)*np.sum(dZ1,axis=1,keepdims=True)
    
    return d_alpha, d_alpha_0, d_beta, d_beta_0
    # d_alpha = np.zeros(alpha.shape)
    # d_alpha_0 = np.zeros(alpha_0.shape)
    # d_beta = np.zeros(beta.shape)
    # d_beta_0 = np.zeros(beta_0.shape)
    # Z, Y_hat = forward_propagation(X,  alpha, alpha_0, beta, beta_0)
    
    # for i in range(m):
    #     y_diff = Y_hat[:,i]-Y[:,i]
    #     temp = np.reshape(2.0*y_diff*sigmoid_gradient(Y_hat[:,i]),(n_y,1))
    #     # print("temp shape")
    #     # print(temp.shape)
    #     d_beta += np.dot(temp,np.reshape(Z[:,i],(1,len(Z[:,i]))))
    #     d_beta_0 += temp
    #     d_alpha += np.dot(np.dot(beta.T,temp),np.reshape(X[:,i],(1,n_x)))
    #     d_alpha_0 += np.dot(beta.T,temp)
    
    # return d_alpha, d_alpha_0, d_beta, d_beta_0
   
def update_parameters(alpha, alpha_0, beta, beta_0, d_alpha, d_alpha_0, d_beta, d_beta_0, learning_rate):
    alpha_next = alpha - learning_rate*d_alpha
    alpha_0_next = alpha_0 -  learning_rate*d_alpha_0
    beta_next = beta - learning_rate*d_beta
    beta_0_next = beta_0 - learning_rate*d_beta_0
    return alpha_next , alpha_0_next, beta_next,beta_0_next
    
def predict(x, alpha, alpha_0, beta, beta_0):
    Z, Y_hat = forward_propagation(x, alpha, alpha_0, beta, beta_0)
    return Y_hat>=0.5

def mlp(X,Y,n_x,n_h,n_y,learning_rate,iterations):
    alpha, alpha_0, beta, beta_0 = initialize_parameters(n_x, n_h, n_y)
    for i in range(iterations):
        Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
        if i%50 == 0:
            print("Iteration:"+str(i)+" Cost:" + str(cost(Y_hat,Y)))
        d_alpha, d_alpha_0, d_beta, d_beta_0 = backward_propagation(X, Y, alpha, alpha_0, beta, beta_0)
        alpha, alpha_0, beta, beta_0 = update_parameters(alpha, alpha_0, beta, beta_0,d_alpha, d_alpha_0, d_beta, d_beta_0, learning_rate)
    Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
    print("Iteration:"+str(iterations-1)+" Cost:" + str(cost(Y_hat,Y)))
    m_test = 100
    x = [[1,1,0,0],[1,0,1,0]]
    # x = np.random.randn(n_x,m_test)*0.1 + np.random.randint(2,size=(n_x,m_test))
    # y_true = np.reshape((x[0,:]>=0.5) + (x[1,:]>=0.5)==1,(1,m_test))
    print("test input:")
    print(x)
    print("y")
    print(predict(x, alpha, alpha_0, beta, beta_0))
    print("weights")
    print(alpha)
    print(alpha_0)
    print(beta)
    print(beta_0)
    # print("Accuracy:"+str(np.count_nonzero(y_true == predict(x, alpha, alpha_0, beta, beta_0))/(.01*m_test)) )



m = 100  # training set size
n_x = 2
n_h = 2
n_y = 1
learning_rate = 0.1
iterations = 1000

X_without_noise = np.zeros((2,m))
X_without_noise[:,0:m/4] += np.array([[1],[1]])
X_without_noise[:,m/4:m/2] += np.array([[1],[0]])
X_without_noise[:,m/2:3*m/4] += np.array([[0],[1]])
X = np.random.randn(n_x,m)*0.01 + X_without_noise

#print(X_without_noise)
# and is true when only both inputs are true
Y = np.reshape(((X_without_noise[0,:]==1) | (X_without_noise[1,:]==1)).astype(float),(1,m)) 
#print(Y) 
shape_X = X.shape
shape_Y = Y.shape

### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))

mlp(X,Y,n_x,n_h,n_y,learning_rate,iterations)
