import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
""" to initialize neutal network, need to specifiy the input (x), labels(y), number
of hidden nodes, and the learning rate"""
class NeuralNetwork:

    def __init__(self, x, y, nodes, lr):
        # set input data and training labels
        np.random.seed(2408)
        self.input      = x
        self.y          = y
        # creating an empty output vector of the same shape as the labels
        self.output     = np.zeros(y.shape)
        # initialize weights with small random values (positive and negative)
        self.theta1   = np.random.normal(scale = 0.1, size =(self.input.shape[1]*nodes)).reshape(self.input.shape[1],nodes)
        self.theta2   = np.random.normal(scale = 0.1, size = (nodes * self.output.shape[1])).reshape(nodes, self.output.shape[1])
        self.lr = lr

    def feedforward(self):
        # add a bias row/column to the input and hidden layer
        # take the dot product of the input layer and the first set of weights
        # then take the sigmoid function of that to generate the hidden layer
        bias = 1
        self.hidden_layer = sigmoid(np.dot(np.insert(self.input, 0, bias, axis=1), np.insert(self.theta1, 0, bias, axis = 0)))
        self.output = sigmoid(np.dot(np.insert(self.hidden_layer, 0, bias, axis = 1), np.insert(self.theta2, 0, bias, axis = 0)))

    def backprop(self):
        self.error = ((self.output - self.y)**2).mean()
        # application of the chain rule to find derivative of the loss function with respect to the weights
        d_theta2 = np.dot(self.hidden_layer.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_theta1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.theta2.T) * sigmoid_derivative(self.hidden_layer)))

        # update the weights with the derivative of the loss function, multiply by the
        #learning rate to scale down
        self.theta1 += d_theta1 * self.lr
        self.theta2 += d_theta2 * self.lr

    # the training function will continue to propogate forward and back until
    # max iterations are reached or error falls below threshold
    # alternatively, the number of iterations can be specified (for testing)
    def train(self, X,y, iterations = False):
        self.input = X
        #self.errors = []
        self.y = y
        # stop after 1000 iterations or as soon as error falls below .03
        # see main.py for cutoff justification
        if iterations == False:
            for i in range(1000):
                self.feedforward()
                self.backprop()
                if self.error < 0.02:
                    break
        # otherwise iterate for specified number of cycles
        else:
            for i in range(iterations):
                self.feedforward()
                self.backprop()

    # to predict unknown data, reset input to new data and feedforward one time
    # return the output as prediction
    def predict(self, X, y = []):
        self.input = X
        self.feedforward()
        if len(y) == 0:
            return self.output
        else:
            error = ((self.output - y)**2).mean()
            return self.output, error


# this function will randomly split the data and labels into a training and testing set
def split_predict(X, y, nodes, lr, cycles = False):
    # split the datat into 70% training and 30% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # generate neural net
    nn = NeuralNetwork(x_train, y_train, nodes, lr)
    # train
    if cycles == False:
        nn.train(x_train, y_train)
    else:
        nn.train(x_train, y_train, iterations = cycles)
    model_error = (nn.error)
    # predict using the held out data and compare to the held out labels to get error
    prediction, validation_error = nn.predict(x_test, y_test)
    return model_error, validation_error

# running once is a little stochastic so this function will average the error
# from cross validating on 10 different splits
def mean_split_predict(X, y, nodes, lr, iterations = False):
    m_error = []
    v_error = []
    for j in range(10):
        m, v = split_predict(X, y, nodes, lr, cycles = iterations)
        m_error.append(m)
        v_error.append(v)
    return np.mean(m_error), np.mean(v_error)
