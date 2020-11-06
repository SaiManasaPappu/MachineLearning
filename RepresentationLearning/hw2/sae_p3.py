import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import pickle

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
    
def cost(Z, Y_hat, lam, p, Y):
    m = Y.shape[1]
    z_mean = np.mean(Z,axis=1,keepdims=True)
    cst = (1.0/m)*LA.norm(Y-Y_hat)**2 + lam*np.sum(p*np.log(p/z_mean) + (1.0-p)*np.log((1.0-p)/(1.0-z_mean)))
    return cst
    
def backward_propagation(X, Y, lam, p, alpha, alpha_0, beta, beta_0):
    m = X.shape[1]
    Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
    dZ2 = Y_hat - Y
    d_beta = (1.0/m)*np.dot(dZ2,Z.T)
    d_beta_0 = (1.0/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(beta.T,dZ2),Z*(1.0-Z))
    z_mean = np.mean(Z,axis=1,keepdims=True)
    d_alpha = (1.0/m)*np.dot(dZ1,X.T) + lam*(1.0/m)*((-p/z_mean) + (1.0-p)/(1.0-z_mean))*np.dot(Z*(1.0-Z),X.T)
    d_alpha_0 = (1.0/m)*np.sum(dZ1,axis=1,keepdims=True) + lam*(-p/z_mean + (1.0-p)/(1.0-z_mean))*(1.0/m)*np.sum(Z*(1.0-Z),axis=1,keepdims=True)
    return d_alpha, d_alpha_0, d_beta, d_beta_0
   
def update_parameters(alpha, alpha_0, beta, beta_0, d_alpha, d_alpha_0, d_beta, d_beta_0, learning_rate):    
    alpha_next = alpha - learning_rate*d_alpha
    alpha_0_next = alpha_0 -  learning_rate*d_alpha_0
    beta_next = beta - learning_rate*d_beta
    beta_0_next = beta_0 - learning_rate*d_beta_0
    return alpha_next , alpha_0_next, beta_next,beta_0_next
    
def predict(x, alpha, alpha_0, beta, beta_0):
    Z, Y_hat = forward_propagation(x, alpha, alpha_0, beta, beta_0)
    return Y_hat>=0.5

def sparse_ae(X_train,Y_train,n_x,n_h,n_y,learning_rate,lam,p,iterations):
    global epochs
    global batch_size
    alpha, alpha_0, beta, beta_0 = initialize_parameters(n_x, n_h, n_y)
    for epoch in range(epochs):
        X = X_train[:,epoch*batch_size:(epoch+1)*batch_size]
        Y = Y_train[:,epoch*batch_size:(epoch+1)*batch_size]
        Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
        print("epoch:"+str(epoch)+" Cost:" + str(cost(Z, Y_hat, lam, p, Y)))
        for i in range(iterations):
            Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
            d_alpha, d_alpha_0, d_beta, d_beta_0 = backward_propagation(X, Y, lam, p, alpha, alpha_0, beta, beta_0)
            alpha, alpha_0, beta, beta_0 = update_parameters(alpha, alpha_0, beta, beta_0,d_alpha, d_alpha_0, d_beta, d_beta_0, learning_rate)
            Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
    Z, Y_hat = forward_propagation(X, alpha, alpha_0, beta, beta_0)
    print("Final cost:"+str(cost(Z, Y_hat, lam, p, Y)))
    
    x_test = np.reshape(train_imgs[1501],(n_x,1)) # m = 100, Hence 105th vector is unseen
    img = x_test.reshape((image_size,image_size))
    plt.imshow(img, cmap="Greys")
    Z, Y_hat = forward_propagation(x_test, alpha, alpha_0, beta, beta_0)
    # print("Z")
    # print(np.sum(Z)/n_h)
    plt.title("Actual Image")
    plt.figure()
    ae = Y_hat.reshape((image_size,image_size))
    plt.imshow(ae, cmap="Greys")
    plt.title("Y_hat")
    plt.show()

with open("pickled_mnist.pkl", "r") as fh:
    data = pickle.load(fh)

image_size = 14 # width and length
train_imgs = np.array(data[0])
print(train_imgs.shape)

X = train_imgs.T
Y = X
epochs = 5
batch_size = 250
n_x = X.shape[0]
n_h = 700 # n_x is 196
n_y = n_x
learning_rate = 1.0
lam = 0.3
p = 0.2
iterations = 200

print(X.shape)
print(Y.shape)

sparse_ae(X,Y,n_x,n_h,n_y,learning_rate,lam,p,iterations)
