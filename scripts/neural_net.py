import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y, nodes, lr):
        self.input      = x
        self.output     = np.zeros(y.shape)
        self.weights1   = np.random.normal(scale = 0.1, size =(self.input.shape[1]*nodes)).reshape(self.input.shape[1],nodes)
        self.weights2   = np.random.normal(scale = 0.1, size = (nodes * self.output.shape[1])).reshape(nodes, self.output.shape[1])
        self.y          = y
        self.lr = lr

    def feedforward(self):
        # add bias vector to each matrix here
        bias = 1
        self.layer1 = sigmoid(np.dot(np.insert(self.input, 0, bias, axis=1), np.insert(self.weights1, 0, bias, axis = 0)))
        self.output = sigmoid(np.dot(np.insert(self.layer1,0, bias, axis = 1), np.insert(self.weights2, 0, bias, axis = 0)))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 * self.lr
        self.weights2 += d_weights2 * self.lr

    def train(self, X,y):
        self.input = X
        self.y = y
        for i in range(100000):
            self.feedforward()
            self.backprop()
        #return(np.round(self.output))
