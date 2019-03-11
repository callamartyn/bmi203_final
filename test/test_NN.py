from scripts import neural_net
import numpy as np
# adding a test function to make sure I get a distance of 0 for self

def test_NN():
    X = np.identity(8)
    y = X
    nn = neural_net.NeuralNetwork(X, y, 3, .05)
    nn.train(X, y)
    out = np.round(nn.output)
    assert np.array_equiv(out, X)
