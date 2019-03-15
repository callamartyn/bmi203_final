from scripts import neural_net, io
import numpy as np
import random

# testing that training the neural net on the identity matrix will retrun
#the identity matrix
def test_NN():
    X = np.identity(8)
    y = X
    nn = neural_net.NeuralNetwork(X, y, 3, .05)
    nn.train(X, y, iterations = 100000)
    out = np.round(nn.output)
    assert np.array_equiv(out, X)

# testing that the sequence to binary function is correctly converting the sequences
def test_translation():
    testseq = 'TGGCAT'
    x = io.seq_to_binary(testseq)
    assert np.array_equiv(x, [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
       0., 0., 0., 0., 1., 0., 0.])

def test_downsample():
    l = [random.sample(range(100), 10) for x in range(20)]
    d = io.downsample(l, 6, 4)
    assert len(d) == 6
    assert [any(x in sl for sl in l) for x in d[0]]
