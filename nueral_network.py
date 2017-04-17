import numpy as np
import pylab
import matplotlib.pyplot as plt

#Code inspired from Welch Lab's video series, Great content !
class Nueral_Network(object):
    def __init__(self,size_in,size_hidden,size_out,learning_rate=3,bias=False):
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.bias = bias
        self.weights = []
        self.num_hidden = 2
        self.learning_rate = learning_rate
        #TODO: Take care of bias=True case
        self.W1 = np.random.randn(self.size_in, self.size_hidden)
        self.W2 = np.random.randn(self.size_hidden, self.size_out)

    def forward(self):
        self.z2 = np.dot(self.X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def cost_function(self):
        return 0.5*(sum(self.y-self.yHat)**2)

    def sigmoid_prime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def backward(self):
        delta3 = np.multiply(-(self.y-self.yHat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(self.X.T, delta2)
        return dJdW1, dJdW2

    def fit(self,X,y,iterations = 50000,sampling=5000):
        self.X = X
        self.y = y
        self.learning = []
        self.iterations = iterations
        self.sampling = sampling
        for i in range(self.iterations):
            self.forward()
            dJdW1,dJdW2 = self.backward()
            self.W1 = self.W1 - self.learning_rate*dJdW1;
            self.W2 = self.W2 - self.learning_rate*dJdW2;
            if(i%self.sampling == 0):
                self.learning.append(self.cost_function())
        print "Done"

    def predict(self,X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        return self.sigmoid(self.z3)

    def plot_learning(self):
        plt.plot(np.arange(1,len(nn.learning)+1)*nn.sampling,nn.learning)
        pylab.show()
