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
        self.error = ((self.output - self.y)**2).mean()
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 * self.lr
        self.weights2 += d_weights2 * self.lr

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
    def predict(self, X):
        self.input = X
        self.feedforward()
        return self.output


# this function will randomly split the data and labels into a training and testing set
def cross_validate(X, y, nodes, lr, cycles = False):
    model_error = []
    validation_error = []
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
    prediction = nn.predict(x_test)
    # compare the prediction to the known label to get the error of the prediction
    validation_error = ((prediction - y_test)**2).mean()
    return model_error, validation_error

# running once is a little stochastic so this function will average the error
# from cross validating on 10 different splits
def mean_cross_validate(X, y, nodes, lr, iterations = False):
    m_error = []
    v_error = []
    # cross-validate 10 times and save the model error and validation error
    for j in range(10):
        m, v = cross_validate(X, y, nodes, lr, cycles = iterations)
        m_error.append(m)
        v_error.append(v)
    # average the errors from the iterations
    return np.mean(m_error), np.mean(v_error)
