print('running main')
import numpy as np
from scripts import neural_net
print('neural net imported')

X = np.identity(8)
y = X
nn = neural_net.NeuralNetwork(X, y, 3, .05)
nn.train(X, y)
print(np.round(nn.output))
# out = np.round(nn.output)
# assert np.array_equiv(out, X)
# print(X)
# print(out)
