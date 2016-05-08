import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

#writing activation functions and their derivatives
def sigmoid(z):
    return 1./(1+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def tanh(z):
    return 2*sigmoid(2*z)-1

def tanhPrime(z):
    return 1-tanh(z)**2

class Neural_Network(object):
    #the neural net will NOT preprocess the input data by adding ficticious dimension
    def __init__(self, n_in=456, n_hidden=200, n_out=2, \
        actFuncHid=tanh, actFuncPrimeHid=tanhPrime, actFuncOut=sigmoid, actFuncPrimeOut=sigmoidPrime):        
        #Define Neural Net size and function hyperparameters
        self.inputLayerSize = n_in
        self.hiddenLayerSize = n_hidden
        self.outputLayerSize = n_out
        self.actFuncHid = actFuncHid
        self.actFuncPrimeHid = actFuncPrimeHid
        self.actFuncOut = actFuncOut
        self.actFuncPrimeOut = actFuncPrimeOut
        
        #Initialize Weights (parameters)
        self.V = np.random.normal(0, .01, self.hiddenLayerSize*(self.inputLayerSize+1)).reshape(self.hiddenLayerSize, self.inputLayerSize+1)
        self.W = np.random.normal(0, .01, self.outputLayerSize*(self.hiddenLayerSize+1)).reshape(self.outputLayerSize, self.hiddenLayerSize+1)
        
    #assumes X already has the bias term
    def forward(self, X):
        #Propogate inputs though network
        if len(X.shape) == 1:
            self.z2 = np.dot(X, self.V.T)
            self.a2 = np.hstack((self.actFuncHid(self.z2), np.ones(1)))
            self.z3 = np.dot(self.a2, self.W.T)
            yHat = self.actFuncOut(self.z3)
            return yHat
        self.z2 = np.dot(X, self.V.T)
        self.a2 = np.hstack((self.actFuncHid(self.z2), np.ones(self.z2.shape[0]).reshape(self.z2.shape[0],1)))
        self.z3 = np.dot(self.a2, self.W.T)
        yHat = self.actFuncOut(self.z3)
        return yHat

    #outputs a single digit value, argmax of output layer
    #assumes X already has the bias term
    def predict(self, X):
        # if len(X.shape) == 1:
        #     X = np.hstack((X, np.ones(1)))
        # else:
        #     X = np.hstack((X, np.ones(X.shape[0]).reshape(X.shape[0],1)))
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=-1)

    def meanSquaredError(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*np.sum((y-self.yHat)**2)
        return J
        
    def meanSquaredPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        #assumes X is a vector (one sample)
        if len(X.shape) > 1:
            self.yHat = self.forward(X)
            if self.yHat.shape[1] == 1:
                self.yHat = self.yHat.reshape(self.yHat.shape[0])
                delta3 = np.multiply(-(y-self.yHat), self.actFuncPrimeOut(self.z3).reshape(self.z3.shape[0])).reshape(y.shape[0], 1)
            else:
                delta3 = np.multiply(-(y-self.yHat), self.actFuncPrimeOut(self.z3))
            dJdW = np.dot(delta3.T, self.a2)
            
            delta2 = np.dot(delta3, self.W)[:,:-1]*self.actFuncPrimeHid(self.z2)
            dJdV = np.dot(delta2.T, X)
            
            return dJdV, dJdW

        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.actFuncPrimeOut(self.z3))
        dJdW = np.outer(delta3.T, self.a2)
        
        delta2 = np.dot(delta3, self.W)[:-1]*self.actFuncPrimeHid(self.z2)
        dJdV = np.outer(delta2.T, X)
        
        return dJdV, dJdW

    def crossEntropyError(self, X, y):
        self.yHat = self.forward(X)
        J = -np.sum(y*np.log(self.yHat)+(1-y)*np.log(1-self.yHat))
        return J

    def crossEntropyPrime(self, X, y):
        if len(X.shape) > 1:
            self.yHat = self.forward(X)

            delta3 = -self.actFuncPrimeOut(self.z3)*(y/self.actFuncOut(self.z3)-(1-y)/(1-self.actFuncOut(self.z3)))
            dJdW = np.dot(delta3.T, self.a2)
            
            delta2 = np.dot(delta3, self.W)[:,:-1]*self.actFuncPrimeHid(self.z2)
            dJdV = np.dot(delta2.T, X)
            
            return dJdV, dJdW

        self.yHat = self.forward(X)

        delta3 = -self.actFuncPrimeOut(self.z3)*(y/self.actFuncOut(self.z3)-(1-y)/(1-self.actFuncOut(self.z3)))
        dJdW = np.outer(delta3.T, self.a2)
        
        delta2 = np.dot(delta3, self.W)[:-1]*self.actFuncPrimeHid(self.z2)
        dJdV = np.outer(delta2.T, X)
        
        return dJdV, dJdW

    def gradient_descent(self, x, y, costFuncPrime, epsilon=.01):
        dJdV, dJdW = costFuncPrime(x, y)
        self.V = self.V - epsilon*dJdV
        self.W = self.W - epsilon*dJdW

    def train(self, X, y, costFunc, costFuncPrime, epsilon=.01, convergence=.0000001, batch_size=1, decrease_epsilon=False):
        total_training_errors = []
        classification_accuracies = []
        past_w = float('inf')*np.ones_like(self.W)
        past_v = float('inf')*np.ones_like(self.V)
        counter = 0
        change_count = 0
        change_vals = float('inf')*np.ones(10)
        stats_constant = 1000
        while np.mean(change_vals) > convergence:
            #perform statistics and bookkeeping every 1000 iterations
            if counter % stats_constant == 0 and counter != 0:
                print "distance from convergence:", np.mean(change_vals)-convergence
                print "   epsilon =", epsilon
                np.save("V.npy", self.V)
                np.save("W.npy", self.W)
                if decrease_epsilon:
                    new_ep = epsilon-epsilon/(np.sqrt(counter))
                    if new_ep > 0:
                        epsilon = new_ep

            #keep track of how the weights are changing so we know when to terminate the training
            change = np.linalg.norm(self.W-past_w) + np.linalg.norm(self.V-past_v)
            change_vals[change_count] = change
            change_count += 1
            change_count = change_count % 10

            #performs the weight update w/gradient descent
            past_w = self.W
            past_v = self.V
            idx = np.random.choice(X.shape[0], batch_size, replace=False)
            x = X[idx]
            self.gradient_descent(x, y[idx], costFuncPrime, epsilon=epsilon)
            counter += 1
        #saves the determined weights and plots statistics
        np.save("V.npy", self.V)
        np.save("W.npy", self.W)

    def load_weights(self, arg_list=["V.npy", "W.npy"]):
        self.V = np.load(arg_list[0])
        self.W = np.load(arg_list[1])


#The following function is the converted 
#benchmark.m copied off piazza from Kunal Marwaha
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices
